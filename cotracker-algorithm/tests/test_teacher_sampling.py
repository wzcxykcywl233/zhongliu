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
        first = teacher_sampling.TeacherPairSampler(
            4, auxiliary_weight=0.1, seed=9, auxiliary_seed=19
        )
        for _ in range(7):
            first.sample()
        state = first.state_dict()
        expected = [first.sample() for _ in range(10)]

        resumed = teacher_sampling.TeacherPairSampler(
            4, auxiliary_weight=0.1, seed=999, auxiliary_seed=1999
        )
        resumed.load_state_dict(state)
        self.assertEqual(expected, [resumed.sample() for _ in range(10)])

    def test_same_teacher_control(self):
        sampler = teacher_sampling.TeacherPairSampler(
            3, auxiliary_weight=0.1, same_teacher_control=True
        )
        for _ in range(10):
            selected = sampler.sample()
            self.assertEqual(selected.primary, selected.auxiliary)

    def test_primary_sequence_is_identical_across_auxiliary_weights(self):
        samplers = [
            teacher_sampling.TeacherPairSampler(
                3,
                auxiliary_weight=weight,
                seed=71,
                auxiliary_seed=81,
            )
            for weight in (0.0, 0.1, 0.15, 0.2, 0.25)
        ]
        sequences = [[sampler.sample().primary for _ in range(100)] for sampler in samplers]
        self.assertTrue(all(sequence == sequences[0] for sequence in sequences[1:]))

    def test_auxiliary_sequence_is_identical_across_nonzero_weights(self):
        samplers = [
            teacher_sampling.TeacherPairSampler(
                3,
                auxiliary_weight=weight,
                seed=71,
                auxiliary_seed=81,
            )
            for weight in (0.1, 0.15, 0.2, 0.25)
        ]
        sequences = [
            [(choice.primary, choice.auxiliary) for choice in (sampler.sample() for _ in range(100))]
            for sampler in samplers
        ]
        self.assertTrue(all(sequence == sequences[0] for sequence in sequences[1:]))

    def test_normalized_loss(self):
        blended = teacher_sampling.blend_teacher_losses(10.0, 2.0, 0.1)
        self.assertAlmostEqual(blended, 9.2)
        self.assertEqual(
            teacher_sampling.blend_teacher_losses(10.0, None, 0.0), 10.0
        )

    def test_same_teacher_control_is_an_exact_no_op(self):
        primary = object()
        self.assertIs(
            teacher_sampling.blend_teacher_losses(
                primary,
                primary,
                0.2,
                same_teacher_control=True,
            ),
            primary,
        )


if __name__ == "__main__":
    unittest.main()
