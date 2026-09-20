"""Exercise retrieval with real CoTracker feature shapes on CPU (and CUDA if available)."""
import sys
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "resources"), str(ROOT / "ext" / "co-tracker")]
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from query_memory import DynamicQueryMemoryBank
from chain_diagnostics import ChainTrace
from typing import NamedTuple

class Memory(NamedTuple):
    track: tuple
    support: tuple

class RealFeatureIntegrationTests(unittest.TestCase):
    def test_real_forward_with_callable_memory_and_patch_weights(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                with self.subTest(device=device), torch.inference_mode():
                    torch.manual_seed(42)
                    model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60).to(device).eval()
                    video = torch.rand(1, 3, 3, 64, 64, device=device) * 255
                    queries = torch.tensor([[[0., 16., 16.], [0., 32., 32.]]], device=device)
                    out = model(video, queries, iters=1, return_query_feature_memory=True)
                    memory = Memory(tuple(out[4][0]), tuple(out[4][1]))
                    for refinement in ("pointwise_write", "pointwise_fusion", "current_retrieval"):
                        bank = DynamicQueryMemoryBank("topk_confidence_diversity", 4, refinement=refinement)
                        bank.add(0, memory, 1.)
                        bank.add(10, memory, .75, torch.tensor([[1., .5]], device=device))
                        resolver = lambda feature: bank.fused_memory(feature, torch.ones(1, 2, device=device))
                        resolved = resolver(memory.track[0])
                        direct = model(video, queries, iters=1, query_feature_memory=resolved, query_feature_weight=.5)
                        callback = model(video, queries, iters=1, query_feature_memory=resolver, query_feature_weight=.5)
                        trace = ChainTrace(sample_points=2)
                        observed = model(video, queries, iters=1, query_feature_memory=resolver,
                                         query_feature_weight=.5, feature_observer=trace.observer('segment'))
                        for a, b in zip(callback[:3], observed[:3]):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        self.assertEqual(trace.data['segment/feature/0/input_support'].shape[:2], (3, 2))
                        self.assertIn('segment/feature/3/memory_track', trace.data)
                        for a, b in zip(direct[:3], callback[:3]):
                            self.assertTrue(torch.isfinite(b).all())
                            torch.testing.assert_close(a, b)
        finally:
            torch.set_num_threads(previous_threads)

if __name__ == '__main__':
    unittest.main()
