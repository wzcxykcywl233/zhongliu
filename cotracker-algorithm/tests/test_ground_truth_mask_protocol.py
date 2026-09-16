import json
from pathlib import Path
import unittest


ALGORITHM_DIR = Path(__file__).resolve().parents[1]


class GroundTruthMaskProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = (
            ALGORITHM_DIR
            / "experiments"
            / "ground-truth-mask-supervision-40-10-38.json"
        )
        cls.protocol = json.loads(path.read_text(encoding="utf-8"))

    def test_split_and_selection_are_frozen(self) -> None:
        protocol = self.protocol["protocol"]
        self.assertEqual(protocol["training_cases"], 40)
        self.assertEqual(protocol["validation_cases"], 10)
        self.assertEqual(protocol["test_cases"], 38)
        self.assertTrue(protocol["test_is_not_used_for_selection"])
        self.assertEqual(self.protocol["selection"]["split"], "validation-10")

    def test_only_mask_weight_varies(self) -> None:
        self.assertEqual(
            self.protocol["profiles"],
            {
                "control": 0.0,
                "mask_w0025": 0.025,
                "mask_w005": 0.05,
                "mask_w010": 0.1,
            },
        )
        teacher = self.protocol["primary_teacher_sampling"]
        self.assertFalse(teacher["auxiliary_teacher"])
        self.assertTrue(teacher["teacher_sequence_paired_across_profiles"])
        self.assertTrue(teacher["training_case_sequence_paired_across_profiles"])
        self.assertTrue(teacher["query_sequence_paired_across_profiles"])
        self.assertFalse(self.protocol["training"]["soft_labels"])
        self.assertEqual(self.protocol["training"]["paired_step_seed"], 20260915)
        self.assertIn("global-step", self.protocol["training"]["resume_protocol"])


if __name__ == "__main__":
    unittest.main()
