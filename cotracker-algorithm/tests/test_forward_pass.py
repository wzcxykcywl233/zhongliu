import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ALGORITHM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALGORITHM_DIR))
sys.path.insert(0, str(ALGORITHM_DIR / "ext" / "co-tracker"))

spec = importlib.util.spec_from_file_location(
    "cotracker_resource_model_standalone",
    ALGORITHM_DIR / "resources" / "model.py",
)
resource_model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(resource_model)


class FakeCoTracker:
    def __call__(self, *, video, queries, iters, is_train):
        del iters, is_train
        batch, keyframes = video.shape[:2]
        points = queries.shape[1]
        coordinates = (
            torch.arange(keyframes, dtype=video.dtype)
            .view(1, keyframes, 1, 1)
            .expand(batch, keyframes, points, 2)
            .clone()
        )
        visibility = torch.ones((batch, keyframes, points))
        confidence = torch.full((batch, keyframes, points), 0.75)
        return coordinates, visibility, confidence, None


class ForwardPassTests(unittest.TestCase):
    def test_keyframe_mode_restores_full_shapes_and_query_anchor(self) -> None:
        video = torch.zeros((1, 6, 1, 8, 8))
        queries = torch.tensor([[[0.0, 10.0, 20.0]]])
        result = resource_model.forward_pass(
            FakeCoTracker(),
            video,
            queries,
            support_grid_size=0,
            temporal_stride=2,
            device="cpu",
        )

        self.assertEqual(result.trajectories.shape, (1, 6, 1, 2))
        self.assertEqual(result.visibility.shape, (1, 6, 1))
        self.assertEqual(result.confidence.shape, (1, 6, 1))
        self.assertTrue(
            torch.allclose(
                result.trajectories[0, :, 0, 0],
                torch.tensor([10.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
            )
        )


if __name__ == "__main__":
    unittest.main()
