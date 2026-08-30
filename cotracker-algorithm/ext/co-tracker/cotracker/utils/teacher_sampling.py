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

    Independent private random-number generators keep primary and auxiliary
    sequences paired across weights and independent from data augmentation.
    Both states are serializable, so an interrupted run resumes exactly.
    """

    def __init__(
        self,
        teacher_count: int,
        auxiliary_weight: float = 0.0,
        seed: int = 0,
        auxiliary_seed: int | None = None,
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
        self._primary_rng = random.Random(seed)
        self._auxiliary_rng = random.Random(
            seed ^ 0x9E3779B9 if auxiliary_seed is None else auxiliary_seed
        )

    def sample(self) -> TeacherSelection:
        primary = self._primary_rng.randrange(self.teacher_count)
        if self.auxiliary_weight == 0.0:
            return TeacherSelection(primary=primary, auxiliary=None)
        if self.same_teacher_control:
            return TeacherSelection(primary=primary, auxiliary=primary)

        # Draw uniformly from every index except the primary teacher.
        auxiliary = self._auxiliary_rng.randrange(self.teacher_count - 1)
        if auxiliary >= primary:
            auxiliary += 1
        return TeacherSelection(primary=primary, auxiliary=auxiliary)

    def state_dict(self) -> dict[str, Any]:
        return {
            "teacher_count": self.teacher_count,
            "auxiliary_weight": self.auxiliary_weight,
            "same_teacher_control": self.same_teacher_control,
            "primary_rng_state": self._primary_rng.getstate(),
            "auxiliary_rng_state": self._auxiliary_rng.getstate(),
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
        if "primary_rng_state" not in state or "auxiliary_rng_state" not in state:
            raise ValueError(
                "legacy coupled teacher-sampler checkpoints cannot be resumed "
                "as a controlled retest"
            )
        self._primary_rng.setstate(state["primary_rng_state"])
        self._auxiliary_rng.setstate(state["auxiliary_rng_state"])


def blend_teacher_losses(
    primary_loss,
    auxiliary_loss,
    auxiliary_weight: float,
    same_teacher_control: bool = False,
):
    """Blend two losses without increasing total teacher-loss strength."""

    if not 0.0 <= auxiliary_weight <= 0.5:
        raise ValueError("auxiliary_weight must be in [0, 0.5]")
    if same_teacher_control or auxiliary_loss is None or auxiliary_weight == 0.0:
        return primary_loss
    return (1.0 - auxiliary_weight) * primary_loss + auxiliary_weight * auxiliary_loss
