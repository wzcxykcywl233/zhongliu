import math
import unittest
import torch
from test_forward_pass import resource_model, HierarchicalFakeCoTracker
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


class StateFake(HierarchicalFakeCoTracker):
    def __init__(self, hide_at=None):
        super().__init__()
        self.states = []
        self.hide_at = hide_at

    def __call__(self, **kwargs):
        v = kwargs.pop('initial_visibility_logits', None)
        c = kwargs.pop('initial_confidence_logits', None)
        x = float(kwargs['queries'][0, 0, 1])
        self.states.append((x, kwargs['video'].shape[1], v, c))
        result = super().__call__(**kwargs)
        result[1].fill_(.9 - .025 * x)
        result[2].fill_(.7 - .025 * x)
        if self.hide_at == x and kwargs['video'].shape[1] == 3:
            result[1][:, 1:] = 0.
        return result


class QueryStateTests(unittest.TestCase):
    def run_fake(self, mode, merge=False):
        model = StateFake(hide_at=4. if merge else None)
        diagnostics = {}
        result = resource_model.hierarchical_forward_pass(
            model, torch.zeros(1, 11, 1, 16, 16), torch.zeros(1, 4, 3),
            span=2, support_grid_size=0, n_iterations=2,
            query_state_inheritance=mode, occlusion_merge=merge,
            diagnostics=diagnostics, device='cpu')
        return model, result, diagnostics

    def test_probability_conversion_clipping_and_scopes(self):
        p = torch.tensor([[0., .5, 1., .8]], requires_grad=True)
        full = resource_model.inherited_state_logits(p, 11, 'vc', 10)
        self.assertFalse(full.requires_grad)
        self.assertTrue(torch.isfinite(full).all())
        torch.testing.assert_close(full.sigmoid()[:, 0], p.detach().clamp(1e-4, 1-1e-4))
        decay = resource_model.inherited_state_logits(p, 11, 'vc_decay', 10)
        torch.testing.assert_close(decay[:, 10], full[:, 0] * math.exp(-1))
        anchor = resource_model.inherited_state_logits(p, 11, 'vc_query', 10)
        torch.testing.assert_close(anchor[:, 0], full[:, 0])
        self.assertEqual(torch.count_nonzero(anchor[:, 1:]), 0)
        for bad in (float('nan'), -0.1, 1.1):
            with self.assertRaises(ValueError):
                resource_model.inherited_state_logits(torch.tensor([[bad]]), 2, 'vc', 10)

    def test_first_segment_and_v_c_ablations(self):
        for mode in ('none','v','c','vc','vc_decay','vc_query'):
            with self.subTest(mode=mode):
                model, result, diagnostics = self.run_fake(mode)
                self.assertIsNone(model.states[0][2])
                self.assertIsNone(model.states[0][3])
                v, c = model.states[1][2:]
                self.assertEqual(v is not None, mode in ('v','vc','vc_decay','vc_query'))
                self.assertEqual(c is not None, mode in ('c','vc','vc_decay','vc_query'))
                if v is not None:
                    torch.testing.assert_close(v[:, 0].sigmoid(), torch.full((1, 4), .9))
                if c is not None:
                    torch.testing.assert_close(c[:, 0].sigmoid(), torch.full((1, 4), .7))
                self.assertEqual(diagnostics['state_inherited_segments'], 0 if mode == 'none' else 4)
                self.assertEqual(result.trajectories.shape[1], 11)

    def test_merge_reuses_start_prior_not_failed_endpoint(self):
        model, _, diagnostics = self.run_fake('vc', merge=True)
        attempts = [record for record in model.states if record[0] == 2.]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0][1], 3)
        self.assertEqual(attempts[1][1], 7)
        for record in attempts:
            torch.testing.assert_close(record[2][:, 0].sigmoid(), torch.full((1, 4), .9))
            torch.testing.assert_close(record[3][:, 0].sigmoid(), torch.full((1, 4), .7))
        self.assertEqual(diagnostics['occlusion_merges'], 1)

    def test_global_branch_unmodified_and_source_is_fused_endpoint(self):
        model = StateFake()
        resource_model.hierarchical_forward_pass(
            model, torch.zeros(1, 5, 1, 16, 16), torch.zeros(1, 4, 3),
            span=2, support_grid_size=0, dual_anchor_weight=.5,
            query_state_inheritance='vc', device='cpu')
        self.assertEqual(model.states[0][1], 5)  # full global clip
        self.assertIsNone(model.states[0][2])
        self.assertIsNone(model.states[1][2])  # first local clip
        torch.testing.assert_close(model.states[2][2][:, 0].sigmoid(), torch.full((1, 4), .9))

    def test_real_model_zero_init_identical_and_nonzero_reaches_transformer(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
                with self.subTest(device=device), torch.inference_mode():
                    torch.manual_seed(42)
                    model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60).to(device).eval()
                    video = torch.rand(1, 3, 3, 64, 64, device=device) * 255
                    queries = torch.tensor([[[0.,16.,16.],[0.,32.,32.]]], device=device)
                    inputs = []
                    handle = model.updateformer.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
                    base = model(video, queries, iters=1)
                    # CPU initial states must be safely moved to the model device.
                    zero = model(video, queries, iters=1, initial_visibility_logits=torch.zeros(1,3,2), initial_confidence_logits=torch.zeros(1,3,2))
                    for a, b in zip(base[:3], zero[:3]):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    warm = model(video, queries, iters=1, initial_visibility_logits=torch.full((1,3,2), 1.5), initial_confidence_logits=torch.full((1,3,2), -.5))
                    delta = inputs[-1] - inputs[0]
                    torch.testing.assert_close(delta[..., 0], torch.full_like(delta[..., 0], 1.5))
                    torch.testing.assert_close(delta[..., 1], torch.full_like(delta[..., 1], -.5))
                    torch.testing.assert_close(delta[..., 2:], torch.zeros_like(delta[..., 2:]), rtol=0, atol=0)
                    self.assertTrue(all(torch.isfinite(t).all() for t in warm[:3]))
                    self.assertFalse(torch.equal(base[2], warm[2]))
                    handle.remove()
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
