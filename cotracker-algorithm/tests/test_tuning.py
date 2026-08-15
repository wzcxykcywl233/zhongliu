import unittest

import torch

from tuning import build_validity_mask, interpolate_keyframes, temporal_median_smooth


class TuningTests(unittest.TestCase):
    def test_keyframe_interpolation_uses_actual_indices(self) -> None:
        values = torch.tensor([[[0.0], [20.0], [50.0]]])
        result = interpolate_keyframes(values, torch.tensor([0, 2, 5]), 6)
        self.assertTrue(
            torch.allclose(
                result,
                torch.tensor(
                    [[[0.0], [10.0], [20.0], [30.0], [40.0], [50.0]]]
                ),
            )
        )

    def test_temporal_median_removes_single_frame_spike(self) -> None:
        trajectories = torch.zeros((1, 5, 1, 2))
        trajectories[:, 2] = 100.0
        result = temporal_median_smooth(trajectories, 3)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_no_thresholds_preserve_baseline_path(self) -> None:
        scores = torch.rand((1, 2, 3))
        self.assertIsNone(build_validity_mask(scores, scores, None, None))

    def test_visibility_and_confidence_are_combined(self) -> None:
        visibility = torch.tensor([[[0.9, 0.4, 0.9]]])
        confidence = torch.tensor([[[0.9, 0.9, 0.2]]])
        result = build_validity_mask(visibility, confidence, 0.5, 0.5)
        self.assertEqual(result.tolist(), [[[True, False, False]]])


if __name__ == "__main__":
    unittest.main()
