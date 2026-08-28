import importlib.util
from pathlib import Path
import sys
import unittest


MODULE = (
    Path(__file__).parents[1]
    / "ext"
    / "co-tracker"
    / "cotracker"
    / "utils"
    / "teacher_sampling.py"
)
SPEC = importlib.util.spec_from_file_location("teacher_sampling", MODULE)
teacher_sampling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = teacher_sampling
SPEC.loader.exec_module(teacher_sampling)


class TeacherPairSamplerTests(unittest.TestCase):
    def test_baseline_has_only_primary_teacher(self):
        sampler = teacher_sampling.TeacherPairSampler(3, auxiliary_weight=0.0)
        self.assertIsNone(sampler.sample().auxiliary)

    def test_auxiliary_teacher_is_distinct(self):
        sampler = teacher_sampling.TeacherPairSampler(4, auxiliary_weight=0.1, seed=4)
        for _ in range(100):
            selected = sampler.sample()
            self.assertNotEqual(selected.primary, selected.auxiliary)

    def test_sampler_state_resumes_exact_sequence(self):
        first = teacher_sampling.TeacherPairSampler(4, auxiliary_weight=0.1, seed=9)
        for _ in range(7):
            first.sample()
        state = first.state_dict()
        expected = [first.sample() for _ in range(10)]

        resumed = teacher_sampling.TeacherPairSampler(4, auxiliary_weight=0.1, seed=999)
        resumed.load_state_dict(state)
        self.assertEqual(expected, [resumed.sample() for _ in range(10)])

    def test_same_teacher_control(self):
        sampler = teacher_sampling.TeacherPairSampler(
            3, auxiliary_weight=0.1, same_teacher_control=True
        )
        for _ in range(10):
            selected = sampler.sample()
            self.assertEqual(selected.primary, selected.auxiliary)

    def test_normalized_loss(self):
        blended = teacher_sampling.blend_teacher_losses(10.0, 2.0, 0.1)
        self.assertAlmostEqual(blended, 9.2)
        self.assertEqual(
            teacher_sampling.blend_teacher_losses(10.0, None, 0.0), 10.0
        )


if __name__ == "__main__":
    unittest.main()
