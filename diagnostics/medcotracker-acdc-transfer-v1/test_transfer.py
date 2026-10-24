import copy
import ast
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import common as C
import external as E
import study as D
import train_transfer as TT

C.torch.set_num_threads(1)


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('COTRACKER_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        C.save(self.root / 'manifest.json', {'fixture': 1})

    def test_schedule_balanced_and_restart_independent(self):
        jobs = [E.schedule(i) for i in range(500)]
        self.assertEqual(jobs, [E.schedule(i) for i in range(500)])
        self.assertEqual([E.schedule(i) for i in range(125, 500)], jobs[125:])
        for epoch in range(6):
            self.assertEqual({i for e, i, _ in jobs[epoch*80:(epoch+1)*80]}, set(range(80)))
        self.assertEqual({s for _, _, s in jobs}, {0, 1, 2})

    def test_adapter_compiles_preserves_optimizer_and_loader(self):
        source = (C.APP / 'ext/co-tracker/train_on_real_data.py').read_text(encoding='utf-8')
        adapted = C.instrument(source)
        compile(adapted, 'adapted', 'exec')
        self.assertEqual(adapted.count('__initialize__(model, args)'), 1)
        self.assertEqual(adapted.count('if total_steps >= __stop_steps__:'), 2)
        optimizer = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fetch_optimizer')
        self.assertTrue(ast.get_source_segment(source, optimizer) in adapted)
        for begin, end in [('        resume_epoch = 0', '        enter_training_mode()')]:
            original = source[source.index(begin):]
            original = original[:original.index(end)]
            self.assertTrue(original in adapted)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            C.instrument(source.replace('architecture_report = warm_start(', 'architecture_report = changed('))

    def test_inventory_checks_disjointness_without_test_io(self):
        rows = []
        for split, count, prefix in [('train-40', 40, 'T'), ('validation-10', 10, 'V'), ('test-38', 38, 'X')]:
            for i in range(count):
                case = f'{prefix}_{i:03}'
                rows.append({'Split': split, 'Case': case, 'File': f'Z:\\unavailable\\{case}\\image.mha', 'SHA256': 'a'*64})
        ids = C.validate_inventory({'Dataset': rows})
        self.assertEqual(len(ids['test-38']), 38)
        self.assertEqual(len(C.input_files({'Dataset': rows}, 'train-40')), 40)
        with self.assertRaisesRegex(ValueError, 'No test'):
            C.input_files({'Dataset': rows}, 'test-38')
        rows[-1]['Case'] = 'T_000'
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            C.validate_inventory({'Dataset': rows})

    def test_marker_detects_tampering_and_escape(self):
        path = self.root / 'artifact.json'
        C.save(path, {'x': 1})
        C.publish(self.root, 'proof.json', [path])
        self.assertTrue(C.marker(self.root, 'proof.json')['complete'])
        C.save(path, {'x': 2})
        with self.assertRaisesRegex(ValueError, 'changed'): C.marker(self.root, 'proof.json')
        with self.assertRaises(ValueError): C.verify_files(self.root, {'../escape': 'a'*64})

    def test_extra_training_files_rejected(self):
        C.save(self.root / 'T_001/data.json', {'x': 1})
        files = {'T_001/data.json': C.digest(self.root / 'T_001/data.json'),
                 'manifest.json': C.digest(self.root / 'manifest.json')}
        C.verify_dataset(self.root, files, ['T_001'])
        C.save(self.root / 'T_001/extra.json', {})
        with self.assertRaisesRegex(ValueError, 'file inventory'): C.verify_dataset(self.root, files, ['T_001'])

    def test_external_driver_resume_and_completed_cache(self):
        torch = C.torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.2))
            def forward(self, video, **kwargs): return video.mean() * self.weight
        class Data:
            def set_epoch(self, epoch): self.epoch = epoch
            def __getitem__(self, index):
                return {'patient': 'patient%03d' % index, 'sampling': {'epoch': self.epoch},
                        'video': torch.ones(2, 3, 2, 2) * (index + 1) / 80}
        def new_model(*args): return Tiny(), {'trainable_names': ['weight']}
        interrupt = [False]
        def label(root, step, sample, weights):
            if interrupt[0] and step == 31: raise RuntimeError('SIMULATED INTERRUPTION')
            path = root / 'pilot/labels' / f'{step:04}.pth'
            value = {'queries': torch.zeros(1, 1, 3), 'tracks': torch.zeros(1, 2, 1, 2),
                     'visibility': torch.ones(1, 2, 1)}
            if not path.exists():
                C.atomic_torch(path, value); C.publish(root, f'pilot/labels/{step:04}.json', [path])
            return value, C.digest(path)
        audit = self.root / 'audit'
        C.save(audit / 'data-manifest.json', {})
        plan = {'weights_sha256': {}, 'external_manifest_sha256': C.digest(audit / 'data-manifest.json')}
        roots = [self.root / name for name in ('continuous', 'resumed')]
        for root in roots:
            C.save(root / 'manifest.json', {'fixture': 1})
            C.publish(root, 'smoke/verified.json', [])
        with ExitStack() as stack:
            stack.enter_context(patch.dict(E.RULE, external_steps=50))
            stack.enter_context(patch.object(E, 'gpu'))
            stack.enter_context(patch.object(E, 'checked_plan', return_value=plan))
            stack.enter_context(patch.object(E.S, 'ACDCCineDataset', return_value=Data()))
            stack.enter_context(patch.object(E.S, 'new_model', side_effect=new_model))
            stack.enter_context(patch.object(E, 'label_for', side_effect=label))
            stack.enter_context(patch.object(E.S, 'training_loss', side_effect=lambda result, target: result.square()))
            stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self: self))
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self: self))
            for name in ('reset_peak_memory_stats', 'synchronize'):
                stack.enter_context(patch.object(torch.cuda, name))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_reserved', return_value=0))
            E.train(roots[0], self.root, audit, self.root, 'A1')
            interrupt[0] = True
            with self.assertRaisesRegex(RuntimeError, 'SIMULATED'): E.train(roots[1], self.root, audit, self.root, 'A1')
            interrupt[0] = False
            E.train(roots[1], self.root, audit, self.root, 'A1')
            with patch.object(E.S, 'new_model', side_effect=AssertionError('Should reuse verified completion')):
                E.train(roots[1], self.root, audit, self.root, 'A1')
        states = [torch.load(r / 'pilot/external/A1/external-final.pth', weights_only=True)['model'] for r in roots]
        self.assertEqual(C.hashes(states[0]), C.hashes(states[1]))

    def test_resume_rotation_survives_interruption(self):
        plan = C.digest(self.root / 'manifest.json')
        def checkpoint(step):
            return {'route': 'A1', 'steps': step, 'rows': [{'step': i} for i in range(step)],
                    'manifest_sha256': plan, 'model': {'x': C.torch.tensor([step])}}
        E.save_resume(self.root, checkpoint(25))
        E.save_resume(self.root, checkpoint(50))
        self.assertEqual(E.load_resume(self.root, plan, 'A1')['steps'], 50)
        # Interruption between rotate and publish leaves previous as the sole committed state.
        (self.root / 'resume.pth').rename(self.root / 'uncommitted.pth')
        self.assertEqual(E.load_resume(self.root, plan, 'A1')['steps'], 25)
        with self.assertRaisesRegex(ValueError, 'Wrong'): E.load_resume(self.root, 'changed', 'A1')
        C.save(self.root / 'resume.pth', {'corrupt': True})
        self.assertEqual(E.load_resume(self.root, plan, 'A1')['steps'], 25)

    def test_optimizer_resume_equals_uninterrupted(self):
        torch = C.torch
        def model_optimizer():
            C.reset_rng(10)
            model = torch.nn.Linear(4, 2)
            return model, torch.optim.AdamW(model.parameters(), lr=.001)
        def steps(model, optimizer, start, stop):
            for step in range(start, stop):
                C.reset_rng(C.S.STUDENT_SEED + step)
                optimizer.zero_grad(set_to_none=True)
                model(torch.rand(3, 4)).square().mean().backward()
                optimizer.step()
        ref, opt = model_optimizer()
        steps(ref, opt, 0, 10)
        part, optimizer = model_optimizer()
        steps(part, optimizer, 0, 4)
        C.atomic_torch(self.root / 'state.pth', {'model': part.state_dict(), 'optimizer': optimizer.state_dict()})
        resumed, resumed_optimizer = model_optimizer()
        state = torch.load(self.root / 'state.pth', weights_only=True)
        resumed.load_state_dict(state['model']); resumed_optimizer.load_state_dict(state['optimizer'])
        steps(resumed, resumed_optimizer, 4, 10)
        self.assertEqual(C.S.state_hashes(ref), C.S.state_hashes(resumed))

    def test_cached_label_checks_video_and_payload(self):
        video = C.torch.zeros(2, 3, 4, 4)
        sample = {'patient': 'patient001', 'sampling': {'frames': [0, 1]}, 'video': video}
        signature = {'step': 0, 'patient': sample['patient'], 'sampling': sample['sampling'],
                     'video_sha256': C.S.tensor_sha(video[None]), 'slot': E.schedule(0)[2]}
        path = self.root / 'pilot/labels/0000.pth'
        C.atomic_torch(path, {'signature': signature})
        C.publish(self.root, 'pilot/labels/0000.json', [path])
        self.assertEqual(E.label_for(self.root, 0, sample, self.root)[0]['signature'], signature)
        sample['video'] = video + 1
        with self.assertRaisesRegex(ValueError, 'clip changed'): E.label_for(self.root, 0, sample, self.root)
        C.atomic_torch(path, {'signature': signature, 'changed': True})
        with self.assertRaisesRegex(ValueError, 'Artifact changed'): E.label_for(self.root, 0, sample, self.root)

    def test_selection_rejects_metric_tradeoffs(self):
        base = {'Arm': 'R1', 'DSC': .9, 'D98': .95, 'HD95': 4., 'CD': 1.5}
        a = dict(base, Arm='A1', DSC=.901, D98=.947)
        b = dict(base, Arm='B1', DSC=.902, HD95=4.021)
        c = dict(base, Arm='B2', DSC=.9001, HD95=3.99)
        self.assertEqual(C.selection([base, a, b, c]), ['B2'])
        self.assertEqual(C.selection([base, a, b]), [])

    def test_handoff_and_actual_initializer_preserve_adaptation(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        from cotracker.models.core.cotracker.feature_sidecar import attach_sidecar, assert_sidecar_training_mode
        torch = C.torch
        torch.manual_seed(8)
        original = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        state = copy.deepcopy(original.state_dict())
        fresh_hashes = []
        for arm in ('R1', 'B1', 'A1'):
            model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
            model.load_state_dict(state)
            if arm == 'B1':
                with torch.no_grad(): next(model.fnet.parameters()).add_(.01)
            if arm == 'A1':
                torch.manual_seed(C.S.STUDENT_SEED)
                attach_sidecar(model, 'mamba')
                with torch.no_grad(): model.updateformer.feature_sidecar.output.weight.fill_(.01)
            before = C.S.state_hashes(model)
            proof = C.S.configure_handoff(model, arm)
            after = C.S.state_hashes(model)
            self.assertTrue(all(after[n] == h for n, h in before.items()))
            if arm != 'A1': fresh_hashes.append(proof['sidecar_hashes'])
            path = self.root / 'smoke/init' / (arm + '.pth')
            C.atomic_torch(path, {'model': model.state_dict()})
            C.publish(self.root, f'smoke/init/{arm}.json', [path], state_hashes=after, source_sha256='a'*64)
            loaded = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
            report = TT.initialize(loaded, self.root, 'smoke', arm)
            self.assertTrue(report['strict_handoff_exact'])
            self.assertEqual(report['trainable_parameters'], 3377325)
            assert_sidecar_training_mode(loaded)
            self.assertEqual(C.S.state_hashes(loaded), after)
            if arm == 'A1': self.assertTrue(bool((loaded.updateformer.feature_sidecar.output.weight == .01).all()))
            del model, loaded
        self.assertEqual(fresh_hashes[0], fresh_hashes[1])

    def test_no_test_or_full_entry(self):
        runner = (C.PACKAGE / 'run_experiment.ps1').read_text()
        self.assertIn("ValidateSet('plan','smoke','pilot','all')", runner)
        self.assertNotIn("$Datasets['test-38']", runner)
        self.assertNotIn('target=/datasets/test', runner)
        self.assertNotIn('docker build', runner)
        self.assertIn("@('train-40','validation-10')", runner)

    def test_report_copies_official_aggregates_and_creates_output(self):
        from summarize_backbone_growth import FIELDS
        cases = [f'V_{i:03}' for i in range(10)]
        runtime = {'algorithm': 'fixture'}
        plan = {'trackrad_ids': {'validation-10': cases}, 'student_config': {'border_points': 1000}}
        C.save(self.root / 'reference-images.json', runtime)
        for arm in ('pretrained',) + C.ARMS:
            folder = self.root / 'pilot/validation-10' / arm
            C.save(folder / 'frozen-images.json', runtime)
            C.save(folder / 'evaluation-inputs.json', {'fixture': True})
            # Intentionally different from case averages: report must copy official aggregates.
            aggregates = dict(zip(FIELDS.values(), [.9 + (.001 if arm == 'A1' else 0), 4., 1.4, 1.5, .95]))
            C.save(folder / C.PROFILE / 'metrics.json', {'aggregates': {**aggregates, 'total_time': 123.},
                   'results': [{'case_id': case, **{f: .5 for f in FIELDS.values()}} for case in cases]})
            for case in cases:
                job = folder / C.PROFILE / 'checkpoint/jobs' / case
                output = job / 'output/images/mri-linac-series-targets/output.mha'
                output.parent.mkdir(parents=True)
                output.write_bytes(b'unit-test fixture, not a medical prediction')
                (job / '.complete').write_text('fixture')
                C.save(job / 'diagnostics.json', {'profile': C.PROFILE, 'config': plan['student_config'],
                       'model': {'parameters': 25385700 if arm == 'pretrained' else 28763025,
                                 'feature_sidecar': 'none' if arm == 'pretrained' else 'mamba',
                                 'frozen_pretrained': True, 'sidecar_runtime': C.RUNTIME},
                       'prediction': {'output_file_sha256': C.digest(output)},
                       'mechanism': {'sidecar_calls': 2, 'sidecar_residual_rms_max': .001}})
        with patch.object(D, 'checked_plan', return_value=plan), patch.object(D, 'audit'), patch.object(D, 'eval_bind'):
            D.report(self.root)
        dest = self.root / 'reports/validation-10'
        with (dest / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual(float(next(r for r in rows if r['Arm'] == 'R1')['DSC']), .9)
        self.assertEqual(C.read(dest / 'selection.json')['eligible_routes'], ['A1'])
        self.assertTrue(C.marker(self.root, 'summary.json')['complete'])


if __name__ == '__main__': unittest.main()
