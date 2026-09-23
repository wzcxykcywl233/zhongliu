"""Named CoTracker experiments for TrackRAD.

The default profile intentionally reproduces the original submission.  Every
single-point profile changes exactly one inference or mask-reconstruction
setting.  Combination profiles are defined separately and only combine
settings that were supported by the completed single-point ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    feature_gate_distance: float = 0.0
    feature_similarity_threshold: float = 0.5
    feature_revalidate_radius: int = 0
    mamba_refiner: bool = False
    mamba_replace_time_attention: bool = False
    long_fusion_gate: str = "none"
    query_memory_mode: str = "none"
    query_memory_slots: int = 0
    query_memory_min_reliability: float = 0.5
    query_memory_min_similarity: float = 0.5
    query_memory_original_floor: float = 0.3
    query_memory_diversity_weight: float = 0.25
    query_memory_refinement: str = "none"
    query_state_inheritance: str = "none"
    query_state_decay_tau: float = 1.0
    frame_backcheck: bool = False
    frame_backcheck_fourway: bool = False
    rotation_backcheck: bool = False
    mask_appearance_radius: int = 0
    mask_appearance_min_gain: float = 0.01
    mask_appearance_displacement_penalty: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.query_state_decay_tau <= 100.0:
            raise ValueError('query_state_decay_tau must be in (0,100]')
        if self.rotation_backcheck and (not self.frame_backcheck or self.frame_backcheck_fourway or self.n_iterations != 2):
            raise ValueError('rotation observation requires backcheck, two iterations and no four-way probes')
        if self.frame_backcheck_fourway and (not self.frame_backcheck or self.n_iterations != 2):
            raise ValueError('four-way backcheck requires enabled observation and exactly 2 iterations')
        if self.frame_backcheck and (self.hierarchical_span <= 0 or self.support_grid_size != 0 or self.n_iterations < 2
                                    or self.temporal_median_window != 1 or self.query_state_inheritance != 'none'
                                    or self.long_fusion_gate != 'none' or self.mask_appearance_radius != 0
                                    or self.query_memory_refinement != 'none'):
            raise ValueError('backcheck requires unmodified hierarchical grid0 trajectories and >=2 iterations')
        if self.query_state_inheritance not in {"none", "v", "c", "vc", "vc_decay", "vc_query"}:
            raise ValueError("unsupported query state inheritance")
        if self.query_state_inheritance != "none" and (self.hierarchical_span <= 0 or self.support_grid_size != 0):
            raise ValueError("query state inheritance requires hierarchical matching and grid0")
        if self.query_memory_refinement not in {"none", "pointwise_write", "pointwise_fusion", "current_retrieval", "cycle_write", "contour_guard", "recent_slot"}:
            raise ValueError("unsupported memory refinement")
        if self.query_memory_refinement != "none" and (
            self.query_memory_mode != "topk_confidence_diversity"
            or self.support_grid_size != 0 or self.query_memory_slots != 4
        ):
            raise ValueError("memory refinement requires diverse four-slot memory and grid0")
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
        if self.feature_gate_distance < 0:
            raise ValueError("feature_gate_distance must be non-negative")
        if not -1.0 <= self.feature_similarity_threshold <= 1.0:
            raise ValueError("feature_similarity_threshold must be in [-1, 1]")
        if self.feature_revalidate_radius < 0:
            raise ValueError("feature_revalidate_radius must be non-negative")
        if self.mamba_refiner and self.mamba_replace_time_attention:
            raise ValueError("select exactly one Mamba integration strategy")
        if self.long_fusion_gate not in {"none", "mlp", "mamba"}:
            raise ValueError("long_fusion_gate must be none, mlp, or mamba")
        if self.long_fusion_gate != "none" and self.dual_anchor_weight == 0:
            raise ValueError("long fusion gate requires dual anchor tracking")
        if self.query_memory_mode not in {
            "none",
            "latest",
            "topk_confidence",
            "topk_confidence_diversity",
        }:
            raise ValueError("unsupported query memory mode")
        if self.query_memory_mode == "none" and self.query_memory_slots != 0:
            raise ValueError("query memory slots require an active query memory mode")
        if self.query_memory_mode != "none":
            if self.hierarchical_span == 0:
                raise ValueError("query memory requires hierarchical tracking")
            if self.original_feature_weight == 0:
                raise ValueError("query memory requires query feature weighting")
            if self.query_memory_slots < 2:
                raise ValueError("query memory requires at least two slots")
        for name, value in (
            ("query_memory_min_reliability", self.query_memory_min_reliability),
            ("query_memory_original_floor", self.query_memory_original_floor),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 <= self.query_memory_min_similarity <= 1.0:
            raise ValueError("query_memory_min_similarity must be in [-1, 1]")
        if self.query_memory_diversity_weight < 0:
            raise ValueError("query_memory_diversity_weight must be non-negative")
        if self.mask_appearance_radius < 0:
            raise ValueError("mask_appearance_radius must be non-negative")
        if self.mask_appearance_min_gain < 0:
            raise ValueError("mask_appearance_min_gain must be non-negative")
        if self.mask_appearance_displacement_penalty < 0:
            raise ValueError(
                "mask_appearance_displacement_penalty must be non-negative"
            )
        if self.mask_appearance_radius > 0:
            if self.hierarchical_span == 0 or self.dual_anchor_weight == 0:
                raise ValueError(
                    "mask appearance validation requires hierarchical dual anchor tracking"
                )
            if self.long_fusion_gate != "none":
                raise ValueError(
                    "mask appearance validation and long fusion gate are separate studies"
                )
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
        if self.feature_gate_distance > 0 and self.dual_anchor_weight == 0:
            raise ValueError("feature gating requires dual anchor fusion")
        if self.feature_gate_distance > 0 and self.original_feature_weight == 0:
            raise ValueError("feature gating requires original feature memory")
        if self.feature_revalidate_radius > 0 and self.feature_gate_distance == 0:
            raise ValueError("feature revalidation requires feature gating")


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
    "hierarchical_full_feature_gate": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        feature_gate_distance=4.0,
    ),
    "hierarchical_full_feature_revalidate_r4": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        feature_gate_distance=4.0,
        feature_similarity_threshold=0.5,
        feature_revalidate_radius=4,
    ),
    "hierarchical_feat05_grid0": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
    ),
    "hierarchical_feat05_iterations2": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        n_iterations=2,
    ),
    "hierarchical_feat05_grid0_iterations2": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
    ),
    "hierarchical_full_grid0": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
    ),
    "hierarchical_full_iterations2": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
    ),
    "hierarchical_full_grid0_iterations2": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
    ),
    "hierarchical_full_grid0_mamba": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        mamba_refiner=True,
    ),
    "hierarchical_full_grid0_mamba_replacement": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        mamba_replace_time_attention=True,
    ),
    "hierarchical_full_grid0_iterations2_mlp_gate": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        long_fusion_gate="mlp",
    ),
    "hierarchical_full_grid0_iterations2_mamba_gate": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        long_fusion_gate="mamba",
    ),
    "hierarchical_full_grid0_iterations2_memory_latest": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        query_memory_mode="latest",
        query_memory_slots=2,
    ),
    "hierarchical_full_grid0_iterations2_memory_topk": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        query_memory_mode="topk_confidence",
        query_memory_slots=4,
    ),
    "hierarchical_full_grid0_iterations2_memory_topk_diverse": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        query_memory_mode="topk_confidence_diversity",
        query_memory_slots=4,
    ),
    "hierarchical_full_grid0_iterations2_memory_topk_mask_appearance_r4": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        query_memory_mode="topk_confidence",
        query_memory_slots=4,
        mask_appearance_radius=4,
    ),
    "hierarchical_full_grid0_iterations2_memory_topk_mask_appearance_r8": ExperimentConfig(
        hierarchical_span=10,
        original_feature_weight=0.5,
        support_grid_size=0,
        n_iterations=2,
        occlusion_merge=True,
        dual_anchor_weight=0.5,
        query_memory_mode="topk_confidence",
        query_memory_slots=4,
        mask_appearance_radius=8,
    ),
}

# Audit-only duplicate of the baseline.  Comparing its per-case output hash with
# ``baseline`` detects nondeterministic inference before tiny ablation deltas are
# interpreted as real effects.
AUDIT_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "baseline_repeat": BASELINE,
}

MEMORY_REFINEMENT_EXPERIMENTS = {
    "memory_control": HIERARCHICAL_EXPERIMENTS["hierarchical_full_grid0_iterations2_memory_topk_diverse"],
    **{
        "memory_" + refinement: replace(
            HIERARCHICAL_EXPERIMENTS["hierarchical_full_grid0_iterations2_memory_topk_diverse"],
            query_memory_refinement=refinement,
        )
        for refinement in ("pointwise_write", "pointwise_fusion", "current_retrieval", "cycle_write", "contour_guard", "recent_slot")
    },
}

QUERY_STATE_EXPERIMENTS = {
    "pvc_control": MEMORY_REFINEMENT_EXPERIMENTS["memory_control"],
    **{"pvc_" + mode: replace(MEMORY_REFINEMENT_EXPERIMENTS["memory_control"], query_state_inheritance=mode)
       for mode in ("v", "c", "vc", "vc_decay", "vc_query")},
}

FRAME_BACKCHECK_EXPERIMENTS = {
    **{'backcheck_control_i' + str(n): replace(MEMORY_REFINEMENT_EXPERIMENTS['memory_control'], n_iterations=n)
       for n in (2, 4, 6)},
    **{'backcheck_i' + str(n): replace(MEMORY_REFINEMENT_EXPERIMENTS['memory_control'], n_iterations=n, frame_backcheck=True)
       for n in (2, 4, 6)},
}

FOURWAY_BACKCHECK_EXPERIMENTS = {
    'fourway_control_i2': MEMORY_REFINEMENT_EXPERIMENTS['memory_control'],
    'fourway_backcheck_i2': replace(MEMORY_REFINEMENT_EXPERIMENTS['memory_control'],
                                  frame_backcheck=True, frame_backcheck_fourway=True),
}

ROTATION_BACKCHECK_EXPERIMENTS = {
    'rotation_control_i2': MEMORY_REFINEMENT_EXPERIMENTS['memory_control'],
    'rotation_history_i2': replace(MEMORY_REFINEMENT_EXPERIMENTS['memory_control'],
                                 frame_backcheck=True, rotation_backcheck=True),
}

from parameter_retune import build_retune_catalog
RETUNE_EXPERIMENTS, RETUNE_CATALOG = build_retune_catalog(MEMORY_REFINEMENT_EXPERIMENTS['memory_control'])

EXPERIMENTS: dict[str, ExperimentConfig] = {
    **RETUNE_EXPERIMENTS,
    **SINGLE_POINT_EXPERIMENTS,
    **COMBINATION_EXPERIMENTS,
    **HIERARCHICAL_EXPERIMENTS,
    **AUDIT_EXPERIMENTS,
    **MEMORY_REFINEMENT_EXPERIMENTS,
    **QUERY_STATE_EXPERIMENTS,
    **FRAME_BACKCHECK_EXPERIMENTS,
    **FOURWAY_BACKCHECK_EXPERIMENTS,
    **ROTATION_BACKCHECK_EXPERIMENTS,
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
