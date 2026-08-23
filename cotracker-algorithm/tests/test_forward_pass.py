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


class HierarchicalFakeCoTracker:
    def __init__(self, occluded_query_x=None) -> None:
        self.calls = []
        self.occluded_query_x = occluded_query_x

    def __call__(
        self,
        *,
        video,
        queries,
        iters,
        is_train,
        query_feature_memory=None,
        query_feature_weight=0.0,
        return_query_feature_memory=False,
    ):
        del iters, is_train
        self.calls.append(
            {
                "frames": video.shape[1],
                "query_x": float(queries[0, 0, 1]),
                "has_memory": query_feature_memory is not None,
                "feature_weight": query_feature_weight,
            }
        )
        batch, frames = video.shape[:2]
        points = queries.shape[1]
        offsets = torch.arange(frames, dtype=video.dtype).view(1, frames, 1, 1)
        coordinates = queries[:, None, :, 1:3] + offsets
        visibility = torch.ones((batch, frames, points), dtype=video.dtype)
        if self.occluded_query_x is not None and float(queries[0, 0, 1]) == float(
            self.occluded_query_x
        ):
            visibility[:, 1:] = 0.0
        confidence = torch.ones_like(visibility)
        result = (coordinates, visibility, confidence, None)
        if return_query_feature_memory:
            track = torch.ones((batch, points, 4), dtype=video.dtype)
            support = torch.ones((batch, 1, points, 4), dtype=video.dtype)
            return result + (([track], [support]),)
        return result


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

    def test_hierarchical_mode_reanchors_at_each_span_endpoint(self) -> None:
        model = HierarchicalFakeCoTracker()
        video = torch.zeros((1, 6, 1, 8, 8))
        queries = torch.tensor([[[0.0, 10.0, 20.0]]])
        result = resource_model.hierarchical_forward_pass(
            model,
            video,
            queries,
            span=2,
            support_grid_size=0,
            device="cpu",
        )
        self.assertEqual([call["frames"] for call in model.calls], [3, 3, 2])
        self.assertTrue(
            torch.allclose(
                result.trajectories[0, :, 0, 0],
                torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            )
        )

    def test_original_feature_memory_is_captured_once_and_reused(self) -> None:
        model = HierarchicalFakeCoTracker()
        video = torch.zeros((1, 5, 1, 8, 8))
        queries = torch.tensor([[[0.0, 1.0, 2.0]]])
        resource_model.hierarchical_forward_pass(
            model,
            video,
            queries,
            span=2,
            support_grid_size=0,
            original_feature_weight=0.5,
            device="cpu",
        )
        self.assertFalse(model.calls[0]["has_memory"])
        self.assertTrue(model.calls[1]["has_memory"])
        self.assertEqual(model.calls[1]["feature_weight"], 0.5)

    def test_occlusion_detector_requires_continuous_hidden_points(self) -> None:
        visibility = torch.tensor(
            [[[1.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.1, 0.7]]]
        )
        self.assertTrue(resource_model.segment_is_occluded(visibility, 0.5, 0.5))
        self.assertFalse(resource_model.segment_is_occluded(visibility, 0.5, 1.0))

    def test_occluded_middle_level_is_replaced_by_three_span_match(self) -> None:
        model = HierarchicalFakeCoTracker(occluded_query_x=2.0)
        video = torch.zeros((1, 7, 1, 8, 8))
        queries = torch.tensor([[[0.0, 0.0, 0.0]]])
        result = resource_model.hierarchical_forward_pass(
            model,
            video,
            queries,
            span=2,
            support_grid_size=0,
            occlusion_merge=True,
            occlusion_visibility_threshold=0.5,
            occlusion_point_fraction=0.5,
            device="cpu",
        )
        self.assertEqual([call["frames"] for call in model.calls], [3, 3, 7])
        self.assertTrue(torch.all(result.visibility == 1.0))


if __name__ == "__main__":
    unittest.main()
