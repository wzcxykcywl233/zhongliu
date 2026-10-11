import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(ROOT/'ext'/'co-tracker')]
from training_phase_probe import PhaseProbe, InstrumentTrainer, instrument
from profile_sidecar_training_phases import training_arguments
from summarize_sidecar_runtime import isolated
from cotracker.models.core.cotracker.feature_sidecar import FeatureSidecar


class RuntimeDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_sha_verified_original_trainer_and_stop_only(self):
        path=ROOT/'ext/co-tracker/train_on_real_data.py'
        data=path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),'64c27699becba986ac39d0ecf1475d6a6edd86e0d2a502d82a2931532b7b8959')
        source=data.decode()
        _,counts=instrument(source,str(path))
        self.assertEqual(counts['early_stop'],2)
        self.assertEqual(counts['training_step'],1)
        tree=InstrumentTrainer().visit(ast.parse(source))
        optimizer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fetch_optimizer')
        original=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fetch_optimizer')
        self.assertEqual(ast.dump(optimizer),ast.dump(original))
        remaining=[n for n in ast.walk(tree) if isinstance(n,ast.Compare) and ast.unparse(n.left)=='total_steps'
                   and any(ast.unparse(c)=='args.num_steps' for c in n.comparators)]
        self.assertFalse(remaining)
        with self.assertRaisesRegex(ValueError,'timing anchor'):
            instrument(source.replace('self.backward(scaler.scale(loss))','print(loss)'),str(path))
        self.assertEqual(path.read_bytes(),data)

    def test_recipe_preserved_not_a_three_step_lr_schedule(self):
        args=training_arguments('mamba',Path('/results'),Path('/data'),Path('/checkpoints'))
        for flag,value in (('--num_steps','1000'),('--sequence_len','60'),('--traj_per_sample','128'),('--train_iters','4'),
                           ('--num_workers','0'),('--paired_step_seed','20261008'),('--confidence_target_mode','hard')):
            self.assertEqual(args[args.index(flag)+1],value)
        self.assertNotIn('--mixed_precision',args)
        self.assertNotIn('--trackrad_mask_supervision_weight',args)

    def test_nested_timings_exclusive_and_partial_failure(self):
        probe=PhaseProbe(3)
        with patch.object(probe,'sync'),patch.object(probe,'memory',return_value={'cuda':False}),patch('training_phase_probe.time.perf_counter',side_effect=[0.,1.,2.,5.,7.,8.]):
            probe.pending_data=.2
            with probe.step(0):
                with probe.stage('student'):
                    with probe.stage('sidecar'): pass
                probe.mark_update(1)
        phases=probe.rows[0]['phases']
        self.assertEqual(phases['sidecar']['exclusive_seconds'],3.)
        self.assertEqual(phases['student']['inclusive_seconds'],6.)
        self.assertEqual(phases['student']['exclusive_seconds'],3.)
        self.assertEqual(probe.rows[0]['wall_seconds_including_data'],8.2)
        with patch.object(probe,'sync'),patch.object(probe,'memory',return_value={'cuda':False}):
            with self.assertRaisesRegex(RuntimeError,'original'):
                with probe.step(1): raise RuntimeError('original')
        self.assertFalse(probe.rows[-1]['complete'])

    def test_installed_probe_does_not_change_rng_outputs_gradients_or_mode(self):
        torch.manual_seed(4)
        branch=FeatureSidecar('mamba',input_dim=8,hidden_dim=12,width=8,depth=1).train()
        with torch.no_grad(): branch.output.weight.normal_(std=.01)
        twin=copy.deepcopy(branch)
        h,r=torch.randn(1,2,5,12),torch.randn(1,2,5,8)
        expected=branch.residual(h,r);expected.square().sum().backward()
        state=torch.get_rng_state().clone()
        probe=PhaseProbe(1)
        original=FeatureSidecar.residual
        with patch.object(probe,'sync'),patch.object(probe,'memory',return_value={'cuda':False}),probe.installed():
            with probe.step(0):
                actual=twin.residual(h,r);actual.square().sum().backward();probe.mark_update(1)
        self.assertIs(FeatureSidecar.residual,original)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(twin.training)
        self.assertTrue(torch.equal(expected,actual))
        for p,q in zip(branch.parameters(),twin.parameters()): self.assertTrue(torch.equal(p.grad,q.grad))
        self.assertEqual(probe.rows[0]['phases']['sidecar_forward']['calls'],1)

    def test_timeout_not_measurement_and_no_automatic_selection(self):
        runtime={'backend':'reference','point_chunk_size':0,'gain':1,'max_correction_pixels':0,'checkpoint_blocks':False}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=[]
            for index,(status,code) in enumerate((('timeout',124),('killed',137))):
                name='probe'+str(index);(root/'isolated'/name).mkdir(parents=True)
                manifest.append({'name':name,'kind':'mamba','runtime':runtime})
                (root/'isolated'/name/'result.json').write_text(json.dumps({'kind':'mamba','runtime':runtime,
                    'complete':False,'status':status,'exit_code':code}))
            (root/'probe-manifest.json').write_text(json.dumps(manifest))
            isolated(root)
            report=json.loads((root/'isolated-summary.json').read_text())
            self.assertEqual(report['completed'],0)
            self.assertEqual(report['timeouts'],1);self.assertEqual(report['killed'],1)
            self.assertNotIn('selected',report)
            self.assertEqual(report['rows'][0]['Seconds'],'')


if __name__=='__main__': unittest.main()
