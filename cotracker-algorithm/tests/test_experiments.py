from dataclasses import fields
import unittest

from experiments import BASELINE, EXPERIMENTS, ExperimentConfig, get_experiment_config


class ExperimentTests(unittest.TestCase):
    def test_baseline_matches_original_submission(self) -> None:
        self.assertEqual(BASELINE, ExperimentConfig())
        self.assertEqual(BASELINE.border_points, 1000)
        self.assertEqual(BASELINE.support_grid_size, 10)
        self.assertEqual(BASELINE.n_iterations, 4)

    def test_each_profile_changes_exactly_one_field(self) -> None:
        names = [field.name for field in fields(ExperimentConfig)]
        for profile, config in EXPERIMENTS.items():
            if profile == "baseline":
                continue
            changed = [
                name
                for name in names
                if getattr(config, name) != getattr(BASELINE, name)
            ]
            self.assertEqual(len(changed), 1, (profile, changed))

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown COTRACKER_EXPERIMENT"):
            get_experiment_config("does-not-exist")


if __name__ == "__main__":
    unittest.main()
