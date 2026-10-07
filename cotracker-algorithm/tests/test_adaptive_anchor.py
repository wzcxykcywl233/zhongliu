import unittest
from unittest.mock import patch
import torch
from resources.adaptive_anchor import last_stable_offsets
from resources.model import hierarchical_forward_pass, TrackingResult
from experiments import get_experiment_config


class AdaptiveAnchorTests(unittest.TestCase):
    def test_relative_drop_independent_per_point(self):
        v = torch.ones(1, 8, 2) * .99
        c = v.clone()
        c[:, 4:, 0] = .75
        q = torch.zeros(1, 2, dtype=torch.long)
        offsets, sustained, pending = last_stable_offsets(v, c, q)
        self.assertEqual(offsets.tolist(), [[3, 7]])
        self.assertEqual(sustained.tolist(), [[True, False]])
        self.assertFalse(pending.any())

    def test_one_frame_dip_recovers_and_late_query(self):
        v = torch.ones(1, 6, 2) * .9
        c = v.clone()
        v[:, 2, 0] = .3
        c[:, :3, 1] = 0
        q = torch.tensor([[0, 3]])
        self.assertEqual(last_stable_offsets(v, c, q)[0].tolist(), [[5, 5]])

    def test_terminal_dip_not_promoted(self):
        v = torch.ones(1, 5, 1)
        v[:, -1] = .1
        self.assertEqual(last_stable_offsets(v, torch.ones_like(v), torch.zeros(1, 1, dtype=torch.long))[0].item(), 3)

    def test_never_retroactively_use_recovery_after_drop(self):
        v = torch.ones(1, 8, 1)
        v[:, 3:5] = .1
        self.assertEqual(last_stable_offsets(v, torch.ones_like(v), torch.zeros(1, 1, dtype=torch.long))[0].item(), 2)

    def test_profiles_share_nonpolicy_controls(self):
        from dataclasses import asdict
        fixed = asdict(get_experiment_config('anchor_fixed')[1])
        for name in ('anchor_diagnostic', 'anchor_endpoint', 'anchor_rollback'):
            candidate = asdict(get_experiment_config(name)[1])
            self.assertEqual({k:v for k,v in fixed.items() if k != 'adaptive_anchor'},
                             {k:v for k,v in candidate.items() if k != 'adaptive_anchor'})

    def test_joint_mixed_queries_advance_and_bounded_fallback(self):
        calls = []
        video = torch.arange(51.).view(1, 51, 1, 1, 1)
        queries = torch.tensor([[[0., 2., 3.], [0., 4., 5.]]])
        def run(model, clip, q, **kwargs):
            start = int(clip[0,0,0,0,0])
            end = int(clip[0,-1,0,0,0])
            calls.append((start, end, q.clone()))
            frames = torch.arange(start, end + 1).view(1, -1, 1, 1).expand(1, -1, 2, 2).float()
            p = frames + queries[:, None, :, 1:]
            v = torch.ones(1, end-start+1, 2) * .99
            # Point0 is unreliable after absolute frame4; point1 stays healthy.
            v[:, max(0,4-start):, 0] = .1
            return TrackingResult(p, v, torch.ones_like(v)), None, None
        global_result = TrackingResult(torch.arange(51.).view(1,51,1,1).expand(1,51,2,2) + queries[:,None,:,1:],
                                       torch.ones(1,51,2), torch.ones(1,51,2))
        d = {}
        with patch('resources.model._run_single_clip', side_effect=run), patch('resources.model.forward_pass', return_value=global_result):
            out = hierarchical_forward_pass(None, video, queries, span=10, support_grid_size=0,
                dual_anchor_weight=.5, adaptive_anchor='rollback', anchor_max_age=30, diagnostics=d, device='cpu')
        self.assertEqual(out.trajectories.shape, (1,51,2,2))
        self.assertLessEqual(max(e-s for s,e,q in calls), 30)
        self.assertGreater(d['anchor_global_fallback_points'], 0)
        self.assertTrue(any(q[0,0,0] != q[0,1,0] for s,e,q in calls))
        self.assertTrue(torch.equal(out.trajectories[:,0],queries[...,1:]))

    def test_real_model_memory_and_diagnostic_identity(self):
        from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
        torch.set_num_threads(1)
        torch.manual_seed(19)
        model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60).eval()
        video = torch.rand(1, 11, 1, 64, 64) * 255
        queries = torch.tensor([[[0.,16.,16.], [0.,32.,32.], [0.,48.,48.]]])
        kwargs = dict(span=3, support_grid_size=0, n_iterations=2, original_feature_weight=.5,
                      dual_anchor_weight=.5, query_memory_mode='topk_confidence_diversity',
                      query_memory_slots=4, anchor_max_age=9, device='cpu')
        with torch.no_grad():
            fixed = hierarchical_forward_pass(model, video, queries, adaptive_anchor='fixed', **kwargs)
            diagnostic = hierarchical_forward_pass(model, video, queries, adaptive_anchor='diagnostic', **kwargs)
            rollback = hierarchical_forward_pass(model, video, queries, adaptive_anchor='rollback', **kwargs)
        for a,b in zip(fixed,diagnostic):
            self.assertTrue(torch.equal(a,b))
        self.assertTrue(all(torch.isfinite(x).all() for x in rollback))
        self.assertTrue(torch.equal(rollback.trajectories[:,0],queries[...,1:]))


if __name__ == '__main__':
    unittest.main()
