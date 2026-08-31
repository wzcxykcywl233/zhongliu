import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "ext" / "co-tracker"
sys.path.insert(0, str(UPSTREAM_ROOT))

from cotracker.models.core.cotracker.losses import sequence_prob_loss  # noqa: E402


class ConfidenceTargetLossTests(unittest.TestCase):
    @staticmethod
    def _loss_for_distances(distances, predictions, **kwargs):
        tracks = [[torch.tensor([[[distance, 0.0] for distance in distances]])]]
        targets = [torch.zeros((1, 1, len(distances), 2))]
        confidence = [[torch.tensor([[predictions]], dtype=torch.float32)]]
        visibility = [torch.ones((1, 1, len(distances)))]
        return sequence_prob_loss(
            tracks,
            confidence,
            targets,
            visibility,
            **kwargs,
        )

    def test_hard_mode_matches_original_twelve_pixel_rule(self):
        predictions = [0.9, 0.9, 0.1]
        loss = self._loss_for_distances(
            [11.9, 12.0, 12.1],
            predictions,
            target_mode="hard",
            expected_dist_thresh=12.0,
        )
        expected_targets = torch.tensor([[1.0, 1.0, 0.0]])
        expected = torch.nn.functional.binary_cross_entropy(
            torch.tensor([predictions]), expected_targets
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_linear_soft_8_16_targets(self):
        targets = [1.0, 0.75, 0.5, 0.25, 0.0]
        loss = self._loss_for_distances(
            [8.0, 10.0, 12.0, 14.0, 16.0],
            targets,
            target_mode="linear_soft",
            soft_inner_radius=8.0,
            soft_outer_radius=16.0,
        )
        expected = torch.nn.functional.binary_cross_entropy(
            torch.tensor([targets]), torch.tensor([targets])
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_linear_soft_6_18_midpoint_is_half(self):
        loss = self._loss_for_distances(
            [6.0, 12.0, 18.0],
            [1.0, 0.5, 0.0],
            target_mode="linear_soft",
            soft_inner_radius=6.0,
            soft_outer_radius=18.0,
        )
        expected = torch.nn.functional.binary_cross_entropy(
            torch.tensor([[1.0, 0.5, 0.0]]),
            torch.tensor([[1.0, 0.5, 0.0]]),
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_invalid_soft_radii_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "inner < outer"):
            self._loss_for_distances(
                [12.0],
                [0.5],
                target_mode="linear_soft",
                soft_inner_radius=16.0,
                soft_outer_radius=8.0,
            )


if __name__ == "__main__":
    unittest.main()
