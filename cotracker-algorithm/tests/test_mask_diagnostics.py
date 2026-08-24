import importlib.util
from pathlib import Path
import unittest

import torch

ALGORITHM_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "trackrad_tap_to_seg_standalone",
    ALGORITHM_DIR / "resources" / "tap_to_seg.py",
)
tap_to_seg = importlib.util.module_from_spec(spec)
assert spec.loader is not None
try:
    spec.loader.exec_module(tap_to_seg)
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    tap_to_seg = None
convert_points_to_mask = (
    None if tap_to_seg is None else tap_to_seg.convert_points_to_mask
)


@unittest.skipIf(tap_to_seg is None, "OpenCV is provided by the Pixi environment")
class MaskDiagnosticTests(unittest.TestCase):
    def test_validity_fallback_is_counted(self) -> None:
        points = torch.tensor(
            [[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]]
        )
        diagnostics = {}
        mask = convert_points_to_mask(
            points,
            (8, 8),
            validity=torch.tensor([True, False, False, False]),
            diagnostics=diagnostics,
        )
        self.assertTrue(bool(mask.any()))
        self.assertEqual(diagnostics["validity_total_points"], 4)
        self.assertEqual(diagnostics["validity_rejected_points"], 3)
        self.assertEqual(diagnostics["validity_fallback_frames"], 1)

    def test_morphology_and_component_changes_are_counted(self) -> None:
        points = torch.tensor(
            [[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]]
        )
        diagnostics = {}
        convert_points_to_mask(
            points,
            (8, 8),
            morph_close_kernel=3,
            keep_largest_component=True,
            diagnostics=diagnostics,
        )
        self.assertIn("morph_close_changed_pixels", diagnostics)
        self.assertIn("component_total_foreground_components", diagnostics)
        self.assertIn("largest_component_changed_pixels", diagnostics)


if __name__ == "__main__":
    unittest.main()
