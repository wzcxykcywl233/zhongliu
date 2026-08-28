"""Deterministic primary/auxiliary teacher sampling for pseudo-label training."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class TeacherSelection:
    """Teacher indices used for one optimizer batch."""

    primary: int
    auxiliary: int | None


class TeacherPairSampler:
    """Sample the baseline teacher and an optional low-weight tutor teacher.

    A private random-number generator makes the sequence independent from data
    augmentation.  Its state is serializable in the training checkpoint, so an
    interrupted run resumes the exact teacher sequence.
    """

    def __init__(
        self,
        teacher_count: int,
        auxiliary_weight: float = 0.0,
        seed: int = 0,
        same_teacher_control: bool = False,
    ) -> None:
        if teacher_count < 1:
            raise ValueError("teacher_count must be positive")
        if not 0.0 <= auxiliary_weight <= 0.5:
            raise ValueError("auxiliary_weight must be in [0, 0.5]")
        if auxiliary_weight > 0.0 and teacher_count < 2 and not same_teacher_control:
            raise ValueError("an auxiliary teacher requires at least two teachers")
        self.teacher_count = teacher_count
        self.auxiliary_weight = auxiliary_weight
        self.same_teacher_control = same_teacher_control
        self._rng = random.Random(seed)

    def sample(self) -> TeacherSelection:
        primary = self._rng.randrange(self.teacher_count)
        if self.auxiliary_weight == 0.0:
            return TeacherSelection(primary=primary, auxiliary=None)
        if self.same_teacher_control:
            return TeacherSelection(primary=primary, auxiliary=primary)

        # Draw uniformly from every index except the primary teacher.
        auxiliary = self._rng.randrange(self.teacher_count - 1)
        if auxiliary >= primary:
            auxiliary += 1
        return TeacherSelection(primary=primary, auxiliary=auxiliary)

    def state_dict(self) -> dict[str, Any]:
        return {
            "teacher_count": self.teacher_count,
            "auxiliary_weight": self.auxiliary_weight,
            "same_teacher_control": self.same_teacher_control,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = (
            self.teacher_count,
            self.auxiliary_weight,
            self.same_teacher_control,
        )
        actual = (
            state["teacher_count"],
            state["auxiliary_weight"],
            state["same_teacher_control"],
        )
        if actual != expected:
            raise ValueError(
                "teacher sampler checkpoint does not match this experiment: "
                f"checkpoint={actual}, current={expected}"
            )
        self._rng.setstate(state["rng_state"])


def blend_teacher_losses(primary_loss, auxiliary_loss, auxiliary_weight: float):
    """Blend two losses without increasing total teacher-loss strength."""

    if not 0.0 <= auxiliary_weight <= 0.5:
        raise ValueError("auxiliary_weight must be in [0, 0.5]")
    if auxiliary_loss is None or auxiliary_weight == 0.0:
        return primary_loss
    return (1.0 - auxiliary_weight) * primary_loss + auxiliary_weight * auxiliary_loss
