from dataclasses import fields, replace
import json
from pathlib import Path
import unittest
import torch
from experiments import EXPERIMENTS, FOURWAY_BACKCHECK_EXPERIMENTS
from frame_backcheck import score_fourway_frames, sample
from test_forward_pass import resource_model, HierarchicalFakeCoTracker
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


class FourwayBackcheckTests(unittest.TestCase):
    def test_profiles_manifest_and_guard(self):
        off = EXPERIMENTS['fourway_control_i2']
        on = EXPERIMENTS['fourway_backcheck_i2']
        self.assertEqual(off,EXPERIMENTS['memory_control'])
        self.assertEqual([f.name for f in fields(off) if getattr(off,f.name)!=getattr(on,f.name)],
                         ['frame_backcheck','frame_backcheck_fourway'])
        for kwargs in ({'n_iterations':4},{'frame_backcheck':False}):
            with self.assertRaises(ValueError):
                replace(on,**kwargs)
        manifest = json.loads((Path(__file__).resolve().parents[1]/'experiments'/'fourway-backcheck-40-10-38.json').read_text())
        self.assertEqual(set(manifest['profiles']),set(FOURWAY_BACKCHECK_EXPERIMENTS))

    def fixture(self):
        torch.manual_seed(7)
        f = torch.rand(1,3,8,16,16)
        q = torch.tensor([[[0.,24.,24.],[0.,32.,32.]]])
        p = q[:,None,:,1:].expand(-1,3,-1,-1).clone()
        previous = p + torch.tensor([2.,1.])
        return f,q,p,previous

    def test_exact_four_reflections_and_equal_center_neighborhood_weight(self):
        f,q,p,previous = self.fixture()
        saved = p.clone()
        rows = score_fourway_frames(f,q,p,(previous,p),10)
        self.assertEqual([r['frame'] for r in rows],[11,12])
        offsets = [(2,1),(-2,1),(-2,-1),(2,-1)]
        for t,row in enumerate(rows,1):
            self.assertEqual(row['accepted_probes'],8)
            self.assertEqual(row['supported_points'],2)
            expected = []
            for j,offset in enumerate(offsets):
                delta = p.new_tensor(offset)
                similarity = (sample(f[:,0],q[...,1:]+delta,4)*sample(f[:,t],p[:,t]+delta,4)).sum(-1)
                expected.append(similarity.mean().item())
                self.assertAlmostEqual(row[f'history_direction_{j}'],expected[-1],places=6)
            self.assertAlmostEqual(row['history_four'],sum(expected)/4,places=6)
            self.assertAlmostEqual(row['center_history_four'],.5*(row['center']+sum(expected)/4),places=6)
            self.assertAlmostEqual(row['center_history'],.5*(row['center']+expected[0]),places=6)
        torch.testing.assert_close(p,saved,rtol=0,atol=0)

    def test_degenerate_and_bounds_are_missing_not_high_confidence(self):
        f,q,p,_ = self.fixture()
        for offset in ((0,1),(.1,1),(0,0),(30,30)):
            row = score_fourway_frames(f,q,p,(p+p.new_tensor(offset),p),0)[0]
            self.assertEqual(row['supported_points'],0)
            self.assertIsNone(row['center_history_four'])
        q[...,1:] = 1
        p[:] = 1
        row = score_fourway_frames(f,q,p,(p+p.new_tensor([2,1]),p),0)[0]
        self.assertEqual(row['bounds_rejected_points'],2)
        self.assertEqual(row['supported_points'],0)

    def test_shared_point_population_and_frame_specific_history(self):
        f,q,p,previous = self.fixture()
        previous[:,:,1] = p[:,:,1]  # point1 unsupported; all scores use point0
        previous[:,2] = p[:,2]     # last target frame has no usable offset
        rows = score_fourway_frames(f,q,p,(previous,p),0)
        self.assertEqual(rows[0]['supported_points'],1)
        self.assertEqual(rows[1]['supported_points'],0)
        expected_center = (sample(f[:,0],q[:,:1,1:],4)*sample(f[:,1],p[:,1,:1],4)).sum(-1).item()
        self.assertAlmostEqual(rows[0]['center'],expected_center,places=6)
        self.assertIsNone(rows[1]['center'])

    def test_merged_segments_are_unique_and_do_not_change_predictions(self):
        args = dict(span=2,support_grid_size=0,n_iterations=2,dual_anchor_weight=.5,
                    occlusion_merge=True,device='cpu')
        video,queries = torch.zeros(1,12,1,64,64),torch.zeros(1,4,3)
        queries[...,1:] = 8
        a = resource_model.hierarchical_forward_pass(HierarchicalFakeCoTracker(occluded_query_x=12),video,queries,**args)
        rows=[]
        b = resource_model.hierarchical_forward_pass(HierarchicalFakeCoTracker(occluded_query_x=12),video,queries,
            frame_backcheck_rows=rows,frame_backcheck_fourway=True,**args)
        torch.testing.assert_close(a.trajectories,b.trajectories,rtol=0,atol=0)
        self.assertEqual(sorted(r['frame'] for r in rows),list(range(1,12)))

    def test_real_model_callback_cpu_and_cuda_when_available(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
                with torch.inference_mode(),self.subTest(device=device):
                    torch.manual_seed(1)
                    model=CoTrackerThreeOffline(stride=4,corr_radius=3,window_len=60).to(device).eval()
                    video=torch.rand(1,3,3,64,64,device=device)*255
                    q=torch.tensor([[[0.,24.,24.],[0.,32.,32.]]],device=device)
                    a=model(video,q,iters=2)
                    capture={}
                    def observe(history,features):
                        capture.update(history=history,features=features)
                    b=model(video,q,iters=2,iteration_observer=observe)
                    rows=score_fourway_frames(capture['features'],q,b[0],capture['history'],0)
                    self.assertEqual(len(rows),2)
                    for x,y in zip(a[:3],b[:3]):
                        torch.testing.assert_close(x,y,rtol=0,atol=0)
        finally:
            torch.set_num_threads(old)


if __name__ == '__main__':
    unittest.main()
