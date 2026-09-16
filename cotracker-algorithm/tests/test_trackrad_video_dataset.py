import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch


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

    def test_clip_indices_can_pair_frames_and_masks(self):
        dataset = object.__new__(trackrad_dataset.TrackRADVideoDataset)
        dataset.seq_len = 4
        dataset.random_frame_rate = False
        dataset.sampling_seed = None
        dataset.epoch = 0
        indices = dataset._sample_indices(2)
        self.assertTrue(torch.equal(indices, torch.tensor([0, 1, 1, 0])))

    def test_seeded_clip_sampling_is_independent_of_global_rng(self):
        dataset = object.__new__(trackrad_dataset.TrackRADVideoDataset)
        dataset.seq_len = 10
        dataset.random_frame_rate = True
        dataset.sampling_seed = 20260916
        dataset.epoch = 7

        torch.manual_seed(1)
        first = dataset._sample_indices(80, sample_index=3)
        torch.manual_seed(99999)
        second = dataset._sample_indices(80, sample_index=3)
        self.assertTrue(torch.equal(first, second))

        dataset.set_epoch(8)
        third = dataset._sample_indices(80, sample_index=3)
        self.assertFalse(torch.equal(first, third))


if __name__ == "__main__":
    unittest.main()
