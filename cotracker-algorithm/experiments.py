"""Named CoTracker experiments for TrackRAD.

The default profile intentionally reproduces the original submission.  Every
single-point profile changes exactly one inference or mask-reconstruction
setting.  Combination profiles are defined separately and only combine
settings that were supported by the completed single-point ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ExperimentConfig:
    border_points: int = 1000
    support_grid_size: int = 10
    n_iterations: int = 4
    visibility_threshold: float | None = None
    confidence_threshold: float | None = None
    temporal_median_window: int = 1
    morph_close_kernel: int = 1
    keep_largest_component: bool = False
    lock_first_mask: bool = False
    temporal_stride: int = 1
    hierarchical_span: int = 0
    original_feature_weight: float = 0.0
    occlusion_merge: bool = False
    occlusion_visibility_threshold: float = 0.5
    occlusion_point_fraction: float = 0.5
    dual_anchor_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.border_points < 3:
            raise ValueError("border_points must be at least 3")
        if self.support_grid_size < 0:
            raise ValueError("support_grid_size must be non-negative")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be positive")
        for name, threshold in (
            ("visibility_threshold", self.visibility_threshold),
            ("confidence_threshold", self.confidence_threshold),
        ):
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in (
            ("temporal_median_window", self.temporal_median_window),
            ("morph_close_kernel", self.morph_close_kernel),
        ):
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.temporal_stride < 1:
            raise ValueError("temporal_stride must be positive")
        if self.hierarchical_span < 0:
            raise ValueError("hierarchical_span must be non-negative")
        for name, value in (
            ("original_feature_weight", self.original_feature_weight),
            ("occlusion_visibility_threshold", self.occlusion_visibility_threshold),
            ("occlusion_point_fraction", self.occlusion_point_fraction),
            ("dual_anchor_weight", self.dual_anchor_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.occlusion_merge and self.hierarchical_span == 0:
            raise ValueError("occlusion_merge requires hierarchical_span")
        if self.original_feature_weight > 0 and self.hierarchical_span == 0:
            raise ValueError("original feature memory requires hierarchical_span")
        if self.dual_anchor_weight > 0 and self.hierarchical_span == 0:
            raise ValueError("dual anchor fusion requires hierarchical_span")


BASELINE = ExperimentConfig()

SINGLE_POINT_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline": BASELINE,
    "points_500": ExperimentConfig(border_points=500),
    "points_1500": ExperimentConfig(border_points=1500),
    "support_grid_0": ExperimentConfig(support_grid_size=0),
    "support_grid_15": ExperimentConfig(support_grid_size=15),
    "iterations_2": ExperimentConfig(n_iterations=2),
    "iterations_6": ExperimentConfig(n_iterations=6),
    "visibility_0_5": ExperimentConfig(visibility_threshold=0.5),
    "confidence_0_5": ExperimentConfig(confidence_threshold=0.5),
    "temporal_median_3": ExperimentConfig(temporal_median_window=3),
    "morph_close_3": ExperimentConfig(morph_close_kernel=3),
    "largest_component": ExperimentConfig(keep_largest_component=True),
    "lock_first_mask": ExperimentConfig(lock_first_mask=True),
    "keyframe_stride_2": ExperimentConfig(temporal_stride=2),
}

COMBINATION_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "grid0_iterations2": ExperimentConfig(
        support_grid_size=0,
        n_iterations=2,
    ),
    "grid0_stride2": ExperimentConfig(
        support_grid_size=0,
        temporal_stride=2,
    ),
    "grid0_iterations2_stride2": ExperimentConfig(
        support_grid_size=0,
        n_iterations=2,
        temporal_stride=2,
    ),
}


HIERARCHICAL_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "hierarchical_d10": ExperimentConfig(
        hierarchical_span=10,
    ),
    "hierarchical_d10_original_feat_05": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
    ),
    "hierarchical_d10_occlusion_merge": ExperimentConfig(
        hierarchical_span=10,
        occlusion_merge=True,
    ),
    "hierarchical_d10_dual_anchor": ExperimentConfig(
        hierarchical_span=10,
        dual_anchor_weight=0.5,
    ),
    "hierarchical_full": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
    ),
    "hierarchical_d10_original_feat_025": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.25,
    ),
    "hierarchical_d10_original_feat_075": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.75,
    ),
    "hierarchical_full_d5": ExperimentConfig(
        hierarchical_span=5,
        original_feature_weight=0.5,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
    ),
    "hierarchical_full_d15": ExperimentConfig(
        hierarchical_span=15,
        original_feature_weight=0.5,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
    ),
}

EXPERIMENTS: dict[str, ExperimentConfig] = {
    **SINGLE_POINT_EXPERIMENTS,
    **COMBINATION_EXPERIMENTS,
    **HIERARCHICAL_EXPERIMENTS,
}


def get_experiment_config(name: str | None = None) -> tuple[str, ExperimentConfig]:
    """Return a validated profile selected explicitly or through the environment."""

    selected = name or os.environ.get("COTRACKER_EXPERIMENT", "baseline")
    try:
        return selected, EXPERIMENTS[selected]
    except KeyError as exc:
        available = ", ".join(EXPERIMENTS)
        raise ValueError(
            f"Unknown COTRACKER_EXPERIMENT={selected!r}. Available: {available}"
        ) from exc
