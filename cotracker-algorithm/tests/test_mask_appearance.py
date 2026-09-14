import importlib.util
from pathlib import Path
import unittest

import torch


ALGORITHM_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mask_appearance_standalone",
    ALGORITHM_DIR / "resources" / "mask_appearance.py",
)
mask_appearance = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mask_appearance)


class MaskAppearanceTests(unittest.TestCase):
    def test_shift_uses_zero_fill(self) -> None:
        mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.bool)
        shifted = mask_appearance._shift_without_wrap(mask, 0, -1)
        self.assertFalse(bool(shifted.any()))

    def test_better_mask_prototype_triggers_bounded_translation(self) -> None:
        masks = torch.zeros((1, 2, 12, 12), dtype=torch.bool)
        masks[:, :, 4:8, 4:8] = True
        query_mask = masks[:, :1].clone()
        features = torch.zeros((1, 2, 2, 3, 3), dtype=torch.float32)
        features[0, 0, 0] = 1.0
        features[0, 1, 1] = 1.0
        features[0, 1, 0, 1, 2] = 1.0
        features[0, 1, 1, 1, 2] = 0.0
        diagnostics = {}

        refined = mask_appearance.refine_masks_by_appearance(
            masks,
            query_mask,
            features,
            radius_pixels=4,
            feature_stride=4,
            minimum_similarity_gain=0.01,
            displacement_penalty=0.0,
            diagnostics=diagnostics,
        )

        expected = torch.zeros((12, 12), dtype=torch.bool)
        expected[4:8, 8:12] = True
        self.assertTrue(torch.equal(refined[0, 1], expected))
        self.assertEqual(diagnostics["mask_appearance_corrected_frames"], 1)
        self.assertEqual(diagnostics["mask_appearance_shift_pixels_sum"], 4.0)

    def test_minimum_gain_prevents_unnecessary_motion(self) -> None:
        masks = torch.zeros((1, 2, 12, 12), dtype=torch.bool)
        masks[:, :, 4:8, 4:8] = True
        features = torch.ones((1, 2, 2, 3, 3), dtype=torch.float32)
        diagnostics = {}
        refined = mask_appearance.refine_masks_by_appearance(
            masks,
            masks[:, :1],
            features,
            radius_pixels=4,
            feature_stride=4,
            minimum_similarity_gain=0.01,
            diagnostics=diagnostics,
        )
        self.assertTrue(torch.equal(refined, masks))
        self.assertEqual(diagnostics["mask_appearance_corrected_frames"], 0)


if __name__ == "__main__":
    unittest.main()
