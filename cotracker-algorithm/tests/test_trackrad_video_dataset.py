import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


# The bounds helper does not need CoTrackerData, but the dataset module imports
# it for __getitem__. Supply a minimal module so this unit test stays isolated.
utils_module = types.ModuleType("cotracker.datasets.utils")
utils_module.CoTrackerData = object
sys.modules.setdefault("cotracker", types.ModuleType("cotracker"))
sys.modules.setdefault("cotracker.datasets", types.ModuleType("cotracker.datasets"))
sys.modules["cotracker.datasets.utils"] = utils_module

MODULE = (
    Path(__file__).parents[1]
    / "ext"
    / "co-tracker"
    / "cotracker"
    / "datasets"
    / "trackrad_video_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("trackrad_video_dataset", MODULE)
trackrad_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trackrad_dataset
SPEC.loader.exec_module(trackrad_dataset)


class TrackRADVideoDatasetTests(unittest.TestCase):
    def test_bounds_are_deterministic_for_large_arrays(self):
        array = np.linspace(-20, 200, 2_000_003, dtype=np.float32).reshape(1, 1, -1)
        first = trackrad_dataset.TrackRADVideoDataset._robust_intensity_bounds(
            array, max_samples=100_000
        )
        second = trackrad_dataset.TrackRADVideoDataset._robust_intensity_bounds(
            array, max_samples=100_000
        )
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])

    def test_constant_array_gets_nonzero_range(self):
        lower, upper = (
            trackrad_dataset.TrackRADVideoDataset._robust_intensity_bounds(
                np.ones((3, 4, 5), dtype=np.float32)
            )
        )
        self.assertEqual(lower, 1.0)
        self.assertEqual(upper, 2.0)


if __name__ == "__main__":
    unittest.main()
