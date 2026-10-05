"""Real-network identity, gradient and checkpoint checks for backbone depth."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'ext' / 'co-tracker'), str(ROOT / 'experiments')]
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.backbone_growth import (
    ARCHITECTURES, PARAMETERS, warm_start, grow_backbone, repair_audit_tail,
)
from audit_backbone_growth import records
from audit_backbone_growth import file_sha
from summarize_backbone_growth import summarize, PROFILE, FIELDS
from resources.model import setup_model


class BackboneGrowthTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_real_forward_identity_parameter_counts_and_new_layer_gradients(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
                state = {key:value.clone() for key,value in model.state_dict().items()}
                before_rng = torch.get_rng_state().clone()
                report = warm_start(model, state, architecture)
                self.assertEqual(report['parameters'], PARAMETERS[architecture])
                self.assertEqual(report['initialization_max_abs_difference'], [0.,0.,0.])
                self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
                # Old tensors remain byte-identical after adding new layers.
                self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key,value in state.items()))
                if architecture == 'base':
                    continue
                self.assertIsNot(model.updateformer.time_blocks[2].attn.to_q.weight,
                                 model.updateformer.time_blocks[3].attn.to_q.weight)
                called = []
                hooks = [block.register_forward_hook(lambda module, args, result, i=i: called.append(i))
                         for i,block in enumerate(model.updateformer.space_virtual_blocks)]
                tokens = torch.randn(1, 4, 3, 1110)
                prediction = model.updateformer(tokens)
                self.assertEqual(called, list(range(3 if architecture == 'time6' else 6)))
                for hook in hooks:
                    hook.remove()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                loss = (prediction - (prediction.detach() + .1)).square().mean()
                loss.backward()
                output = model.updateformer.time_blocks[3].attn.to_out.weight
                self.assertGreater(float(output.grad.abs().sum()), 0.)
                optimizer.step()
                self.assertGreater(float(output.detach().abs().sum()), 0.)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'grown.pth'
                    torch.save({'model':model.state_dict(),'backbone_architecture':architecture}, path)
                    restored = setup_model(checkpoint=str(path), device='cpu')
                    with torch.no_grad():
                        torch.testing.assert_close(model.eval().updateformer(tokens), restored.updateformer(tokens), rtol=0, atol=0)
                    self.assertEqual(restored.backbone_architecture, architecture)

    def test_audit_tail_recovery_and_changed_resumed_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'paired-steps.jsonl'
            row = {'step':0,'architecture':'base','case_names':['A_001'],
                   'queries_sha256':'q','video_sha256':'v','primary_index':0,'auxiliary_index':None}
            content = json.dumps(row) + '\n'
            path.write_text(content + '{"step":', encoding='utf-8')
            repair_audit_tail(path)
            self.assertEqual(path.read_text(), content)
            self.assertEqual(len(list(Path(directory).glob('*.tail-*.txt'))), 1)
            self.assertEqual(len(records(path, 1, 'base')), 1)
            row['video_sha256'] = 'different_clip'
            path.write_text(content + json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'resumed step input changed'):
                records(path, 1, 'base')

    def test_split_summary_checks_executed_architecture_and_keeps_paired_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / 'train'
            train.mkdir()
            checkpoints = {}
            for architecture in ARCHITECTURES:
                path = train / 'seed_0' / architecture / 'cotracker_three_final.pth'
                path.parent.mkdir(parents=True)
                path.write_bytes(architecture.encode())
                checkpoints[str(path.relative_to(train))] = file_sha(path)
            (train / 'training-audit.json').write_text(json.dumps({
                'complete':True,'seeds':[0],'steps':1,'final_checkpoints_sha256':checkpoints}))
            for name, architecture, offset in [('pretrained','base',0.), ('seed_0/base','base',.01),
                                               ('seed_0/time6','time6',.02), ('seed_0/space_time6','space_time6',.03)]:
                folder = root / 'validation-10' / name / PROFILE
                results = []
                for case_index in range(10):
                    case = 'A_' + str(case_index)
                    job = folder / 'checkpoint' / 'jobs' / case
                    output = job / 'output' / 'images' / 'mri-linac-series-targets' / 'output.mha'
                    output.parent.mkdir(parents=True)
                    output.write_bytes((name + case).encode())
                    (job / '.complete').touch()
                    (job / 'diagnostics.json').write_text(json.dumps({
                        'profile':PROFILE, 'model':{'backbone_architecture':architecture,'parameters':PARAMETERS[architecture]},
                        'prediction':{'output_file_sha256':file_sha(output)}}))
                    results.append({'case_id':case,**{field:.8+offset for field in FIELDS.values()}})
                (folder / 'metrics.json').write_text(json.dumps({'results':results,'aggregates':{
                    **{field:.8+offset for field in FIELDS.values()},'total_time':10.}}))
            with redirect_stdout(io.StringIO()):
                report = summarize(root,[0],1,'validation-10')
            self.assertEqual(report['cases'],10)
            self.assertIn('pretrained/', next(iter(report['metrics_sha256'])))
            text = (root / 'validation-10' / 'backbone-growth-paired-deltas.csv').read_text(encoding='utf-8-sig')
            self.assertIn('time6,base',text)
            self.assertIn('base,pretrained',text)
            diagnostic = next((root / 'validation-10' / 'seed_0/time6').rglob('diagnostics.json'))
            data = json.loads(diagnostic.read_text())
            data['model']['backbone_architecture'] = 'base'
            diagnostic.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'executed backbone'):
                summarize(root,[0],1,'validation-10')


if __name__ == '__main__':
    unittest.main()
