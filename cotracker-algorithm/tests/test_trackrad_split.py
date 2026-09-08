import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "create_trackrad_40_10_split.py"
SPEC = importlib.util.spec_from_file_location("create_trackrad_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrackRADSplitTests(unittest.TestCase):
    def test_largest_remainder_allocation_is_exact(self) -> None:
        allocation = MODULE.allocate_validation_counts(
            {"abdomen": 14, "pelvis": 3, "thorax": 8}, 5
        )
        self.assertEqual(sum(allocation.values()), 5)
        self.assertTrue(allocation["abdomen"] >= allocation["pelvis"])

    def test_split_is_deterministic_and_has_fixed_cohort_quotas(self) -> None:
        cases = []
        for cohort, count in (("A", 25), ("B", 15), ("C", 10)):
            for index in range(count):
                cases.append(
                    {
                        "case": f"{cohort}_{index:03d}",
                        "cohort": cohort,
                        "scanned_region": ("abdomen", "pelvis", "thorax")[index % 3],
                    }
                )

        training_a, validation_a = MODULE.select_split(cases, "fixed-seed")
        training_b, validation_b = MODULE.select_split(cases, "fixed-seed")
        self.assertEqual(training_a, training_b)
        self.assertEqual(validation_a, validation_b)
        self.assertEqual(len(training_a), 40)
        self.assertEqual(len(validation_a), 10)
        validation_cohorts = {}
        for cohort in "ABC":
            validation_cohorts[cohort] = sum(
                case["cohort"] == cohort for case in validation_a
            )
        self.assertEqual(validation_cohorts, {"A": 5, "B": 3, "C": 2})
        self.assertFalse(
            {case["case"] for case in training_a}
            & {case["case"] for case in validation_a}
        )


if __name__ == "__main__":
    unittest.main()

