import importlib.util
from pathlib import Path
from typing import NamedTuple
import unittest

import torch


ALGORITHM_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "query_memory_standalone",
    ALGORITHM_DIR / "resources" / "query_memory.py",
)
query_memory = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(query_memory)


class Memory(NamedTuple):
    track: tuple[torch.Tensor, ...]
    support: tuple[torch.Tensor, ...]


def make_memory(x: float, y: float) -> Memory:
    feature = torch.tensor([[[x, y]]], dtype=torch.float32)
    return Memory((feature,), (feature[:, None],))


class DynamicQueryMemoryTests(unittest.TestCase):
    def test_pointwise_admission_keeps_good_point_despite_low_mean(self):
        bank = query_memory.DynamicQueryMemoryBank("topk_confidence_diversity", 4, refinement="pointwise_write")
        feature = torch.tensor([[[1., 0.], [1., 0.]]])
        original = Memory((feature,), (feature[:, None].expand(-1, 49, -1, -1),))
        changed = torch.tensor([[[0.6, 0.8], [0., 1.]]])
        new = Memory((changed,), (changed[:, None].expand(-1, 49, -1, -1),))
        bank.add(0, original, 1.)
        self.assertTrue(bank.add(10, new, 0.45, torch.tensor([[0.9, 0.]])))
        result = bank.fused_memory()
        self.assertGreater(float(result.track[0][0, 0, 1]), 0)
        torch.testing.assert_close(result.track[0][0, 1], original.track[0][0, 1])
        torch.testing.assert_close(result.support[0][0, :, 1], original.support[0][0, :, 1])

    def test_pointwise_fusion_changes_weights_without_changing_admission(self):
        banks = [query_memory.DynamicQueryMemoryBank("topk_confidence_diversity", 4,
                 min_similarity=-1, refinement=refinement) for refinement in ("none", "pointwise_fusion")]
        def memory(features):
            f = torch.tensor([features], dtype=torch.float32)
            return Memory((f,), (f[:, None],))
        for bank in banks:
            bank.add(0, memory([[1, 0], [1, 0]]), 1)
            bank.add(10, memory([[0.8, 0.6], [0.8, 0.6]]), 0.7, torch.tensor([[1., .4]]))
            bank.add(20, memory([[0.8, -0.6], [0.8, -0.6]]), 0.7, torch.tensor([[.4, 1.]]))
        self.assertEqual(banks[0].frame_indices, banks[1].frame_indices)
        result = banks[1].fused_memory().track[0]
        self.assertGreater(result[0, 0, 1], 0)
        self.assertLess(result[0, 1, 1], 0)
        self.assertFalse(torch.allclose(result, banks[0].fused_memory().track[0]))

    def test_retrieval_uses_current_feature_and_falls_back_for_low_reliability(self):
        bank = query_memory.DynamicQueryMemoryBank("topk_confidence_diversity", 4, refinement="current_retrieval", min_similarity=-1)
        bank.add(0, make_memory(1, 0), 1)
        bank.add(10, make_memory(.8, .6), 1)
        bank.add(20, make_memory(.8, -.6), 1)
        positive = bank.fused_memory(torch.tensor([[[.8, .6]]]), torch.ones(1, 1)).track[0]
        negative = bank.fused_memory(torch.tensor([[[.8, -.6]]]), torch.ones(1, 1)).track[0]
        fallback = bank.fused_memory(torch.tensor([[[.8, .6]]]), torch.zeros(1, 1)).track[0]
        self.assertGreater(positive[0, 0, 1], 0)
        self.assertLess(negative[0, 0, 1], 0)
        self.assertAlmostEqual(float(fallback[0, 0, 1]), 0, places=6)

    def test_cycle_rejection_preserves_original_feature(self):
        bank = query_memory.DynamicQueryMemoryBank("topk_confidence_diversity", 4, refinement="cycle_write")
        bank.add(0, make_memory(1, 0), 1)
        self.assertFalse(bank.add(10, make_memory(.8, .6), 1, cycle_valid=torch.zeros(1, 1, dtype=torch.bool)))
        self.assertEqual(bank.frame_indices, (0,))

    def test_recent_slot_retains_lower_quality_latest_anchor(self):
        bank = query_memory.DynamicQueryMemoryBank("topk_confidence_diversity", 4, refinement="recent_slot")
        bank.add(0, make_memory(1, 0), 1)
        for i in range(1, 5):
            bank.add(i, make_memory(1, 0), 1 if i < 4 else .5)
        self.assertEqual(len(bank.entries), 4)
        self.assertIn(0, bank.frame_indices)
        self.assertIn(4, bank.frame_indices)

    def test_contour_guard_only_repairs_isolated_low_confidence_point(self):
        tracks = torch.zeros(1, 3, 8, 2)
        tracks[:, 1:] = 1
        tracks[0, 1, 3, 0] = 9
        reliability = torch.ones(1, 3, 8)
        reliability[0, 1, 3] = .2
        diagnostics = {}
        result = query_memory.contour_motion_guard(tracks, reliability, diagnostics)
        self.assertEqual(float(result[0, 1, 3, 0]), 7.)
        self.assertEqual(diagnostics["memory_contour_corrected_points"], 1)
        torch.testing.assert_close(result[:, 0], tracks[:, 0])
        torch.testing.assert_close(result[:, 2], tracks[:, 2])

    def test_latest_preserves_original_and_newest_reliable_anchor(self) -> None:
        diagnostics = {}
        bank = query_memory.DynamicQueryMemoryBank(
            "latest", 2, min_reliability=0.5, diagnostics=diagnostics
        )
        self.assertTrue(bank.add(0, make_memory(1.0, 0.0), 1.0))
        self.assertFalse(bank.add(5, make_memory(1.0, 0.0), 0.2))
        self.assertTrue(bank.add(10, make_memory(1.0, 0.0), 0.8))
        self.assertTrue(bank.add(20, make_memory(1.0, 0.0), 0.9))

        self.assertEqual(bank.frame_indices, (0, 20))
        self.assertEqual(diagnostics["query_memory_writes_rejected_reliability"], 1)
        self.assertEqual(diagnostics["query_memory_evictions"], 1)

    def test_topk_retains_highest_quality_anchors(self) -> None:
        bank = query_memory.DynamicQueryMemoryBank("topk_confidence", 3)
        bank.add(0, make_memory(1.0, 0.0), 1.0)
        bank.add(1, make_memory(1.0, 0.0), 0.6)
        bank.add(2, make_memory(1.0, 0.0), 0.9)
        bank.add(3, make_memory(1.0, 0.0), 0.8)
        self.assertEqual(bank.frame_indices, (0, 2, 3))

    def test_diversity_can_prefer_a_distinct_anchor(self) -> None:
        bank = query_memory.DynamicQueryMemoryBank(
            "topk_confidence_diversity",
            2,
            min_similarity=-1.0,
            diversity_weight=0.25,
        )
        bank.add(0, make_memory(1.0, 0.0), 1.0)
        bank.add(1, make_memory(0.99, 0.1), 1.0)
        bank.add(2, make_memory(0.0, 1.0), 0.95)
        self.assertEqual(bank.frame_indices, (0, 2))

    def test_fusion_keeps_original_floor_and_normalizes_features(self) -> None:
        diagnostics = {}
        bank = query_memory.DynamicQueryMemoryBank(
            "latest",
            2,
            min_similarity=-1.0,
            original_floor=0.3,
            diagnostics=diagnostics,
        )
        bank.add(0, make_memory(1.0, 0.0), 1.0)
        bank.add(10, make_memory(0.0, 1.0), 1.0)
        fused = bank.fused_memory()

        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(fused.track[0])), 1.0, places=6
        )
        self.assertGreater(float(fused.track[0][0, 0, 1]), 0.0)
        self.assertAlmostEqual(diagnostics["query_memory_original_weight_sum"], 0.3)
        self.assertEqual(diagnostics["query_memory_slots_used_max"], 2)

    def test_similarity_rejection_is_audited(self) -> None:
        diagnostics = {}
        bank = query_memory.DynamicQueryMemoryBank(
            "topk_confidence",
            3,
            min_similarity=0.5,
            diagnostics=diagnostics,
        )
        bank.add(0, make_memory(1.0, 0.0), 1.0)
        self.assertFalse(bank.add(10, make_memory(0.0, 1.0), 1.0))
        self.assertEqual(bank.frame_indices, (0,))
        self.assertEqual(diagnostics["query_memory_writes_rejected_similarity"], 1)


if __name__ == "__main__":
    unittest.main()
