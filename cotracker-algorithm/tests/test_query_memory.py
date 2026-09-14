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
