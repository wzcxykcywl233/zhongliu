import importlib.util
from pathlib import Path
import unittest

import torch


ALGORITHM_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mask_supervision_standalone",
    ALGORITHM_DIR
    / "ext"
    / "co-tracker"
    / "cotracker"
    / "utils"
    / "mask_supervision.py",
)
mask_supervision = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mask_supervision)


class MaskSupervisionTests(unittest.TestCase):
    @staticmethod
    def _example():
        masks = torch.zeros((1, 2, 1, 7, 7), dtype=torch.float32)
        masks[0, 0, 0, 2:5, 1:4] = 1.0
        masks[0, 1, 0, 2:5, 2:5] = 1.0
        queries = torch.tensor([[[0.0, 1.0, 3.0], [0.0, 6.0, 6.0]]])
        correct = torch.tensor(
            [[[[1.0, 3.0], [6.0, 6.0]], [[2.0, 3.0], [6.0, 6.0]]]]
        )
        wrong = correct.clone()
        wrong[0, 1, 0, 0] = 1.0
        return masks, queries, correct, wrong

    def test_correct_membership_has_lower_loss(self) -> None:
        masks, queries, correct, wrong = self._example()
        correct_loss, metrics = mask_supervision.mask_membership_consistency_loss(
            masks, queries, correct, blur_radius=0
        )
        wrong_loss, _ = mask_supervision.mask_membership_consistency_loss(
            masks, queries, wrong, blur_radius=0
        )
        self.assertLess(float(correct_loss), float(wrong_loss))
        self.assertEqual(metrics["inside_fraction"], 0.5)
        self.assertEqual(metrics["eligible_points"], 2.0)

    def test_loss_backpropagates_to_student_coordinates(self) -> None:
        masks, queries, _, wrong = self._example()
        wrong.requires_grad_(True)
        loss, _ = mask_supervision.mask_membership_consistency_loss(
            masks, queries, wrong, blur_radius=1
        )
        loss.backward()
        self.assertIsNotNone(wrong.grad)
        self.assertGreater(float(wrong.grad.abs().sum()), 0.0)

    def test_valid_mask_can_exclude_every_prediction(self) -> None:
        masks, queries, correct, _ = self._example()
        correct.requires_grad_(True)
        loss, metrics = mask_supervision.mask_membership_consistency_loss(
            masks,
            queries,
            correct,
            valid=torch.zeros((1, 2, 2), dtype=torch.bool),
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(metrics["eligible_points"], 0.0)


if __name__ == "__main__":
    unittest.main()
