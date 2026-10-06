"""Mid-epoch resume regression using the real Torch sampler/dataset/optimizer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'ext/co-tracker'), str(ROOT / 'experiments')]
from cotracker.utils.resumable_loader import set_resumable_loader_epoch
from cotracker.utils.teacher_sampling import TeacherPairSampler
from recover_backbone_growth import choose_reusable, verify_schedule, verify_transition, recover
from audit_backbone_growth import records, file_sha, verify_reuse_provenance
from cotracker.models.core.cotracker.backbone_growth import PARAMETERS


class EpochDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.epoch = 0

    def __len__(self):
        return 40

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(20261005 + self.epoch * 1_000_003 + index)
        return index, torch.rand(4, generator=generator)


class Fabric194Wrapper:
    """The exact epoch-counter behavior from Lightning/Fabric 1.9.4."""
    def __init__(self):
        self.dataset = EpochDataset()
        self.sampler = torch.utils.data.DistributedSampler(self.dataset, num_replicas=1, rank=0, seed=0)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=1, sampler=self.sampler)
        self._num_iter_calls = 0

    def __iter__(self):
        self.sampler.set_epoch(self._num_iter_calls)
        self._num_iter_calls += 1
        yield from iter(self.loader)


def run_training(stop=170, checkpoint=None, fixed=True):
    torch.manual_seed(20261005)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.001, total_steps=270, pct_start=0.0, cycle_momentum=False)
    teacher = TeacherPairSampler(3, seed=20261005)
    loader = Fabric194Wrapper()
    step, epoch, start_batch = 0, 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        teacher.load_state_dict(checkpoint['teacher'])
        step, epoch, start_batch = checkpoint['step'], checkpoint['epoch'], checkpoint['batch']
        torch.set_rng_state(checkpoint['rng'])
    signatures = []
    while step < stop:
        loader.dataset.set_epoch(epoch)
        if fixed:
            set_resumable_loader_epoch(loader, epoch)
        for batch_index, (case, video) in enumerate(loader):
            if batch_index < start_batch:
                continue
            torch.manual_seed(20261005 + step)
            query = torch.rand(2)
            primary = teacher.sample().primary
            signatures.append((step, case.item(), hashlib.sha256(video.numpy().tobytes()).hexdigest(), query.tolist(), primary))
            optimizer.zero_grad()
            loss = (model(video).flatten() - primary - query.sum()).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step == stop:
                # Serialize/deserialize actual weights and optimizer buffers,
                # rather than retaining mutable references from the live run.
                import io
                stream = io.BytesIO()
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'teacher':teacher.state_dict(),
                            'step':step,'epoch':epoch,'batch':batch_index+1,'rng':torch.get_rng_state()}, stream)
                stream.seek(0)
                return signatures, torch.load(stream, weights_only=False)
        epoch += 1
        start_batch = 0


class BackboneResumeTests(unittest.TestCase):
    def test_mid_epoch_resume_matches_inputs_teachers_and_final_parameters(self):
        full, expected = run_training()
        old_full, old_expected = run_training(fixed=False)
        self.assertEqual(old_full, full)
        self.assertTrue(all(torch.equal(value, old_expected['model'][key]) for key,value in expected['model'].items()))
        for boundary in (40, 125, 146):
            with self.subTest(boundary=boundary):
                prefix, checkpoint = run_training(stop=boundary)
                suffix, resumed = run_training(checkpoint=checkpoint)
                self.assertEqual(prefix + suffix, full)
                self.assertTrue(all(torch.equal(value, resumed['model'][key]) for key,value in expected['model'].items()))
                self.assertEqual(expected['teacher'], resumed['teacher'])
                self.assertEqual(expected['scheduler'], resumed['scheduler'])
        _, checkpoint = run_training(stop=125)
        broken, _ = run_training(checkpoint=checkpoint, fixed=False)
        self.assertNotEqual(broken[0][1], full[125][1])

    def test_reuse_requires_two_matching_complete_arms(self):
        self.assertEqual(choose_reusable({'base':{0:1},'time6':{0:1},'space_time6':{0:2}}), ['base','time6'])
        with self.assertRaises(ValueError):
            choose_reusable({'base':{0:1},'time6':{0:2}})

    def test_original_schedule_detects_wrong_case_without_deleting_records(self):
        cases = [f'C_{i:03}' for i in range(40)]
        sampler = torch.utils.data.DistributedSampler(cases, num_replicas=1, rank=0, seed=0)
        teacher = TeacherPairSampler(3, seed=20261005)
        rows = {}
        for epoch in range(4):
            sampler.set_epoch(epoch)
            for i,index in enumerate(sampler):
                step = epoch * 40 + i
                rows[step] = tuple(json.dumps(v) for v in ([cases[index]],'query','video',teacher.sample().primary,None))
        verify_schedule(rows, {'sequence_len':10,'teacher_seed':20261005}, cases, 160)
        rows[125] = (json.dumps(['WRONG_CASE']),) + rows[125][1:]
        with self.assertRaisesRegex(ValueError, 'step 125'):
            verify_schedule(rows, {'sequence_len':10,'teacher_seed':20261005}, cases, 160)

    def test_transition_allows_only_exact_resume_fix(self):
        trainer = ROOT / 'ext/co-tracker/train_on_real_data.py'
        old = {'Schema':2, 'Steps':1000, 'Source':[{'File':'cotracker-algorithm/ext/co-tracker/train_on_real_data.py',
                'SHA256':'1531458d7c1da577e59cabc15fae4a4c6ec746311371c7cd722abbe0cd1dd5e4'}]}
        new = {'Schema':3,'Steps':1000,'RecoverySource':'old',
               'Source':[{'File':old['Source'][0]['File'],'SHA256':file_sha(trainer)}]}
        self.assertEqual(verify_transition(old, new, ROOT.parent), [old['Source'][0]['File']])
        new['Steps'] = 2000
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            verify_transition(old, new, ROOT.parent)

    def test_recovery_imports_eight_arms_preserves_old_files_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            old_root, new_root = Path(temp) / 'old', Path(temp) / 'new'
            old_root.mkdir(); new_root.mkdir()
            dump = lambda p,v: p.write_text(json.dumps(v), encoding='utf-8')
            cases = [f'C_{i:03}' for i in range(40)]
            trainer_path = 'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
            old = {'Schema':2,'Seeds':[0,1,2],'Steps':160,'SaveEverySteps':25,
                   'Source':[{'File':trainer_path,'SHA256':'1531458d7c1da577e59cabc15fae4a4c6ec746311371c7cd722abbe0cd1dd5e4'}],
                   'Dataset':[{'Case':case,'Split':'train-40','File':'C:\\data\\cases\\'+case+'\\images\\frames.mha'} for case in cases]}
            dump(old_root / 'frozen-run.json', old)
            new = dict(old, Schema=3, RecoverySource=str(old_root),
                       Source=[{'File':trainer_path,'SHA256':file_sha(ROOT.parent / trainer_path)}])
            dump(new_root / 'frozen-run.json', new)
            runtime = {'python':'3.10','torch':'2.13','torch_cuda':'12.9','pytorch_lightning':'1.9.4'}
            dump(old_root / 'runtime-check.json', runtime); dump(new_root / 'runtime-check.json', runtime)
            dump(old_root / 'training-image.json', {'Id':'sha256:'+'a'*64})
            for seed in (0,1,2):
                sampler = torch.utils.data.DistributedSampler(cases, num_replicas=1, rank=0, seed=0)
                teacher = TeacherPairSampler(3, seed=20261005+seed)
                entries = []
                for epoch in range(4):
                    sampler.set_epoch(epoch)
                    for i,index in enumerate(sampler):
                        entries.append({'step':epoch*40+i,'case_names':[cases[index]],'primary_index':teacher.sample().primary,
                                        'auxiliary_index':None,'queries_sha256':f'query-{seed}-{i}', 'video_sha256':f'video-{epoch}-{index}'})
                for arch in ('base','time6','space_time6'):
                    folder = old_root / 'train' / f'seed_{seed}' / arch
                    folder.mkdir(parents=True)
                    config = {'profile':f'seed_{seed}_{arch}','architecture':arch,'steps':160,'save_every_steps':25,
                              'seed':seed,'training_seed':20261005+seed,'paired_step_seed':20261005+seed,'teacher_seed':20261005+seed,
                              'source_protocol_sha256':file_sha(old_root / 'frozen-run.json').upper(),'dataset':'C:\\data\\cases',
                              'sequence_len':10,'traj_per_sample':384,'train_iters':4,'lr':0.00005,
                              'auxiliary_teacher_weight':0,'confidence_target_mode':'hard','mask_loss_weight':0}
                    dump(folder / 'run-config.json', config)
                    dump(folder / 'initialization-report.json', {'architecture':arch,'parameters':PARAMETERS[arch],
                         'initialization_outputs_exact':True,'initialization_max_abs_difference':[0,0,0]})
                    (folder / 'cotracker_three_final.pth').write_bytes(('checkpoint-'+arch).encode())
                    rows = [dict(entry, architecture=arch) for entry in entries]
                    if seed == 0 and arch == 'space_time6':
                        rows.append(dict(rows[125], case_names=['WRONG_CASE'],video_sha256='wrong-video'))
                    (folder / 'paired-steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            before = {str(p.relative_to(old_root)):file_sha(p) for p in old_root.rglob('*') if p.is_file()}
            # Real strict checkpoint loading is covered in test_backbone_growth;
            # here small payloads isolate recovery selection/copy/provenance.
            with patch('recover_backbone_growth.validate_checkpoint', return_value={}):
                proof = recover(old_root, new_root, ROOT.parent)
                self.assertEqual(len(proof['reused_arms']), 8)
                self.assertEqual(proof['retrain_arms'][0]['arm'], 'seed_0/space_time6')
                self.assertFalse((new_root / 'train/seed_0/space_time6').exists())
                self.assertEqual(recover(old_root, new_root, ROOT.parent), proof)
            after = {str(p.relative_to(old_root)):file_sha(p) for p in old_root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            imported = new_root / 'train/seed_0/base'
            cfg = json.loads((imported / 'reused-run-config.json').read_text())
            cfg['source_protocol_sha256'] = file_sha(new_root / 'frozen-run.json').upper()
            dump(imported / 'run-config.json', cfg)
            verify_reuse_provenance(imported)
            (imported / 'cotracker_three_final.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                verify_reuse_provenance(imported)


if __name__ == '__main__':
    unittest.main()
