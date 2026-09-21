import importlib.util
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from dataclasses import fields
import torch
from test_forward_pass import resource_model, HierarchicalFakeCoTracker
from frame_backcheck import score_frames
from experiments import FRAME_BACKCHECK_EXPERIMENTS, EXPERIMENTS, ExperimentConfig
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('backcheck_analysis_test', ROOT / 'experiments' / 'analyze_frame_backcheck.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class FrameBackcheckTests(unittest.TestCase):
    def test_profiles_factor_iterations_and_observation_only(self):
        baseline = EXPERIMENTS['memory_control']
        for n in (2,4,6):
            off = EXPERIMENTS[f'backcheck_control_i{n}']
            on = EXPERIMENTS[f'backcheck_i{n}']
            self.assertEqual([f.name for f in fields(on) if getattr(on,f.name)!=getattr(off,f.name)], ['frame_backcheck'])
            self.assertEqual([f.name for f in fields(off) if getattr(off,f.name)!=getattr(baseline,f.name)], [] if n==2 else ['n_iterations'])
        manifest = json.loads((ROOT/'experiments'/'frame-backcheck-40-10-38.json').read_text())
        self.assertEqual(set(manifest['profiles']), set(FRAME_BACKCHECK_EXPERIMENTS))
        with self.assertRaises(ValueError):
            ExperimentConfig(frame_backcheck=True)

    def test_same_offset_matching_identity_features_and_shared_probe_count(self):
        torch.manual_seed(12)
        fmap = torch.rand(1,1,8,16,16)
        f = fmap.expand(-1,3,-1,-1,-1)
        q = torch.tensor([[[0.,24.,24.], [0.,32.,32.]]])
        p = q[:,None,:,1:].expand(-1,3,-1,-1).clone()
        history = (p+2, p+2.1, p.clone())  # second near-duplicate is rejected
        rows = score_frames(f,q,p,history,10)
        self.assertEqual([r['frame'] for r in rows],[11,12])
        for row in rows:
            self.assertEqual(row['accepted_probes'], 2)
            self.assertEqual(row['supported_points'], 2)
            for key in ('center','history','fixed','center_history','center_fixed'):
                self.assertAlmostEqual(row[key],1.,places=5)

    def test_missing_evidence_is_null_not_perfect_confidence(self):
        f = torch.ones(1,2,4,16,16)
        q = torch.tensor([[[0.,24.,24.]]])
        p = q[:,None,:,1:].expand(-1,2,-1,-1).clone()
        for previous in (p.clone(), p+100):
            rows = score_frames(f,q,p,(previous,p),0)
            self.assertEqual(rows[0]['supported_points'],0)
            self.assertIsNone(rows[0]['center_history'])
            self.assertIsNotNone(rows[0]['center_all'])

    def test_each_frame_uses_its_own_history(self):
        f = torch.ones(1,3,4,16,16)
        q = torch.tensor([[[0.,24.,24.]]])
        p = q[:,None,:,1:].expand(-1,3,-1,-1).clone()
        prior = p.clone()
        prior[:,1] += 2
        rows = score_frames(f,q,p,(prior,p),0)
        self.assertEqual(rows[0]['accepted_probes'],1)
        self.assertEqual(rows[1]['accepted_probes'],0)

    def test_hierarchical_merge_replaces_scores_and_preserves_predictions(self):
        args = dict(span=2,support_grid_size=0,n_iterations=2,dual_anchor_weight=.5,
                    occlusion_merge=True,device='cpu')
        video, queries = torch.zeros(1,12,1,64,64), torch.full((1,4,3),0.)
        queries[...,1:] = 8
        a = resource_model.hierarchical_forward_pass(HierarchicalFakeCoTracker(occluded_query_x=12),video,queries,**args)
        rows = []
        b = resource_model.hierarchical_forward_pass(HierarchicalFakeCoTracker(occluded_query_x=12),video,queries,frame_backcheck_rows=rows,**args)
        torch.testing.assert_close(a.trajectories,b.trajectories,rtol=0,atol=0)
        self.assertEqual(sorted(r['frame'] for r in rows),list(range(1,12)))
        self.assertEqual(len({r['frame'] for r in rows}),11)

    def test_real_model_history_callback_is_observation_only(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
                with self.subTest(device=device), torch.inference_mode():
                    torch.manual_seed(1)
                    model = CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60).to(device).eval()
                    video = torch.rand(1,3,3,64,64,device=device)*255
                    q = torch.tensor([[[0.,24.,24.],[0.,32.,32.]]],device=device)
                    base = model(video,q,iters=2)
                    capture = {}
                    def observer(history,features):
                        capture.update(history=history,features=features)
                    observed = model(video,q,iters=2,iteration_observer=observer)
                    for a,b in zip(base[:3],observed[:3]):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)
                    self.assertEqual(len(capture['history']),2)
                    rows = score_frames(capture['features'],q,observed[0],capture['history'],0)
                    self.assertEqual(len(rows),2)
        finally:
            torch.set_num_threads(old)

    def test_score_statistics_direction_and_undefined_auc(self):
        rows = [dict(center=s,dice=d) for s,d in zip([.1,.2,.8,.9],[.3,.4,.9,1.])]
        stats = analysis.statistics(rows,'center')
        self.assertAlmostEqual(stats['spearman_score_vs_dice'],1.)
        self.assertEqual(stats['auc_low_score_detects_bad'],1.)
        self.assertIsNone(analysis.statistics(rows[2:],'center')['auc_low_score_detects_bad'])
        self.assertIsNone(analysis.statistics([],'center')['spearman_score_vs_dice'])

    def test_offline_analysis_exports_and_rejects_changed_predictions(self):
        # Exercise all filesystem/schema plumbing without requiring SimpleITK
        # locally. Actual MHA decoding remains the standard evaluation library.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, results = root/'data', root/'results'
            dataset.mkdir()
            results.mkdir()
            cases = [f'A_{i:03}' for i in range(10)]
            arrays = {}
            metric_names = ('dice_similarity_coefficient','hausdorff_distance_95',
                            'surface_distance_average','center_distance','relative_d98_dose','total_time')
            def write_json(path, value):
                path.write_text(json.dumps(value), encoding='utf-8')
            for case in cases:
                target = dataset/case/'targets'
                target.mkdir(parents=True)
                label = target/f'{case}_labels.mha'
                label.touch()
                arrays[str(label)] = np.ones((2,2,3))
            for n in (2,4,6):
                for enabled in (False,True):
                    profile = f'backcheck_i{n}' if enabled else f'backcheck_control_i{n}'
                    folder = results/profile
                    folder.mkdir()
                    write_json(folder/'metrics.json', {'results':[{'case_id':c} for c in cases],
                               'aggregates':dict.fromkeys(metric_names,1.)})
                    for case in cases:
                        job = folder/'checkpoint'/'jobs'/case
                        image = job/'output'/'images'/'mri-linac-series-targets'/'output.mha'
                        image.parent.mkdir(parents=True)
                        image.write_bytes(b'fixture')
                        arrays[str(image)] = np.ones((2,2,3))
                        rows = [dict(frame=t, points=10, supported_points=10,
                                     center=.9,center_history=.8,center_fixed=.7) for t in (1,2)]
                        write_json(job/'diagnostics.json', {'profile':profile,
                            'config':{'n_iterations':n,'frame_backcheck':enabled},
                            'prediction':{'output_file_sha256':analysis.sha256(image),'array_sha256':'equal'},
                            'frame_backcheck':rows})
            fake_sitk = SimpleNamespace(ReadImage=lambda p: arrays[p],GetArrayFromImage=lambda x:x)
            with patch.dict('sys.modules', {'SimpleITK':fake_sitk}):
                analysis.analyze(dataset,results,'validation-10')
                summary = analysis.read_json(results/'backcheck-analysis-summary.json')
                self.assertEqual(summary['output_pairs'],30)
                self.assertEqual(summary['common_frames'],20)
                self.assertTrue((results/'iteration-performance-validation-10.csv').exists())
                path = results/'backcheck_i2'/'checkpoint'/'jobs'/cases[0]/'diagnostics.json'
                data = analysis.read_json(path)
                data['prediction']['array_sha256'] = 'changed'
                write_json(path,data)
                with self.assertRaisesRegex(ValueError,'Observer changed prediction'):
                    analysis.analyze(dataset,results,'validation-10')


if __name__ == '__main__':
    unittest.main()
