import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


AUDIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ext"
    / "co-tracker"
    / "audit_confidence_head_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("confidence_checkpoint_audit", AUDIT_PATH)
AUDIT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT_MODULE)


class ConfidenceHeadCheckpointAuditTests(unittest.TestCase):
    def _write_pair(self, forbidden_change: bool = False):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        base_path = root / "base.pth"
        tuned_path = root / "tuned.pth"
        base = {
            "encoder.weight": torch.ones(2, 2),
            "updateformer.vis_conf_head.weight": torch.ones(2, 3),
            "updateformer.vis_conf_head.bias": torch.ones(2),
        }
        tuned = {key: value.clone() for key, value in base.items()}
        tuned["updateformer.vis_conf_head.weight"][1, 0] += 0.25
        tuned["updateformer.vis_conf_head.bias"][1] -= 0.5
        if forbidden_change:
            tuned["encoder.weight"][0, 0] += 1
        torch.save(base, base_path)
        torch.save(tuned, tuned_path)
        return temporary, base_path, tuned_path

    def test_accepts_only_confidence_row_changes(self):
        temporary, base_path, tuned_path = self._write_pair()
        try:
            result = AUDIT_MODULE.audit(base_path, tuned_path)
        finally:
            temporary.cleanup()
        self.assertTrue(result["passed"])
        self.assertEqual(result["changed_elements"], 2)

    def test_rejects_any_other_parameter_change(self):
        temporary, base_path, tuned_path = self._write_pair(forbidden_change=True)
        try:
            with self.assertRaisesRegex(AssertionError, "forbidden parameter"):
                AUDIT_MODULE.audit(base_path, tuned_path)
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
