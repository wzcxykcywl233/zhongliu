import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_COTRACKER = ALGORITHM_ROOT / "ext" / "co-tracker"
sys.path.insert(0, str(EXTERNAL_COTRACKER))
PATH = ALGORITHM_ROOT / "resources" / "long_fusion.py"
spec = importlib.util.spec_from_file_location("long_fusion_standalone", PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class Result:
    def __init__(self, trajectories, visibility, confidence):
        self.trajectories = trajectories
        self.visibility = visibility
        self.confidence = confidence


class LongFusionTests(unittest.TestCase):
    def make_results(self):
        local_tracks = torch.zeros(1, 6, 4, 2)
        global_tracks = torch.ones(1, 6, 4, 2) * 4
        reliability = torch.ones(1, 6, 4)
        return (
            Result(local_tracks, reliability, reliability),
            Result(global_tracks, reliability, reliability),
        )

    def test_feature_sequence_has_one_token_per_frame(self) -> None:
        local, global_result = self.make_results()
        similarity = torch.ones(1, 6, 4)
        features = module.build_long_fusion_features(
            local, global_result, similarity, similarity, (384, 512)
        )
        self.assertEqual(features.shape, (1, 6, module.LONG_FUSION_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all())

    def test_mlp_has_no_temporal_communication(self) -> None:
        model = module.FrameMLPFusionGate()
        features = torch.randn(1, 7, module.LONG_FUSION_FEATURE_DIM)
        before = model(features)
        changed = features.clone()
        changed[:, 0] += 10
        after = model(changed)
        self.assertTrue(torch.equal(before[:, 1:], after[:, 1:]))

    def test_equal_reliability_uses_gate_weight_and_preserves_query(self) -> None:
        local, global_result = self.make_results()
        fused, global_weight = module.apply_long_fusion_gate(
            local,
            global_result,
            torch.zeros(1, 6),
        )
        self.assertTrue(torch.equal(fused.trajectories[:, 0], local.trajectories[:, 0]))
        self.assertTrue(
            torch.allclose(fused.trajectories[:, 1:], torch.full((1, 5, 4, 2), 2.0))
        )
        self.assertTrue(torch.equal(global_weight, torch.full((1, 6), 0.5)))

    def test_mamba_communicates_across_time(self) -> None:
        model = module.MambaLongFusionGate(model_dim=12, state_dim=4, blocks=1)
        features = torch.randn(1, 7, module.LONG_FUSION_FEATURE_DIM)
        before = model(features)
        changed = features.clone()
        changed[:, 0] += 10
        after = model(changed)
        self.assertFalse(torch.equal(before[:, 1:], after[:, 1:]))

    def test_gate_checkpoint_round_trip(self) -> None:
        for kind in ("mlp", "mamba"):
            model = module.make_long_fusion_gate(kind)
            checkpoint = {
                "kind": kind,
                "model": model.state_dict(),
                "feature_mean": torch.zeros(module.LONG_FUSION_FEATURE_DIM),
                "feature_std": torch.ones(module.LONG_FUSION_FEATURE_DIM),
            }
            path = Path(__file__).with_name(f".{kind}-gate-test.pth")
            try:
                torch.save(checkpoint, path)
                loaded, mean, std = module.load_long_fusion_gate(path, kind, "cpu")
                self.assertEqual(type(loaded), type(model))
                self.assertEqual(mean.shape, (module.LONG_FUSION_FEATURE_DIM,))
                self.assertTrue(torch.equal(std, torch.ones_like(std)))
            finally:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
