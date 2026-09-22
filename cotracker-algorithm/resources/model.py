from datetime import datetime
import logging
import os
from time import perf_counter
from pathlib import Path
from typing import NamedTuple

import torch
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline

from tuning import interpolate_keyframes

try:
    from .query_memory import DynamicQueryMemoryBank, contour_motion_guard
except ImportError:  # pragma: no cover - standalone test/module loading
    from query_memory import DynamicQueryMemoryBank, contour_motion_guard

logger = logging.getLogger(__name__)


def _diagnostic_add(
    diagnostics: dict[str, int | float] | None,
    key: str,
    value: int | float,
) -> None:
    """Accumulate a numeric diagnostic without affecting submission inference."""
    if diagnostics is not None:
        diagnostics[key] = diagnostics.get(key, 0) + value


def _diagnostic_max(
    diagnostics: dict[str, int | float] | None,
    key: str,
    value: int | float,
) -> None:
    if diagnostics is not None:
        diagnostics[key] = max(diagnostics.get(key, value), value)


class TrackingResult(NamedTuple):
    trajectories: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor


class QueryFeatureMemory(NamedTuple):
    track: tuple[torch.Tensor, ...]
    support: tuple[torch.Tensor, ...]


class LongFusionContext(NamedTuple):
    hierarchical: TrackingResult
    global_result: TrackingResult
    hierarchical_similarity: torch.Tensor
    global_similarity: torch.Tensor


class MaskAppearanceContext(NamedTuple):
    tracking: TrackingResult
    frame_features: torch.Tensor


def setup_model(
    checkpoint: str = "cotracker3_offline",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    mamba_replace_time_attention: bool = False,
    mamba_refiner: bool = False,
) -> CoTrackerThreeOffline:
    ts_start = datetime.now()
    checkpoint = os.environ.get("COTRACKER_CHECKPOINT", checkpoint)
    if checkpoint == "cotracker3_offline":
        local_source = Path(
            os.environ.get(
                "COTRACKER_SOURCE_DIR",
                Path(__file__).resolve().parents[1] / "ext" / "co-tracker",
            )
        )
        if not local_source.is_dir():
            raise FileNotFoundError(f"Local CoTracker source not found: {local_source}")
        model = torch.hub.load(
            str(local_source),
            "cotracker3_offline",
            source="local",
        ).model
    else:
        model = CoTrackerThreeOffline(stride=4, corr_radius=3, window_len=60)
        if mamba_replace_time_attention:
            from cotracker.models.core.cotracker.mamba_time import (
                replace_updateformer_time_attention,
            )

            replace_updateformer_time_attention(model)
        elif mamba_refiner:
            from cotracker.models.core.cotracker.mamba_time import (
                attach_trajectory_mamba_refiner,
            )

            attach_trajectory_mamba_refiner(model)
        logger.info(
            "Configured trained checkpoint architecture: mamba_refiner=%s, mamba_time_replacement=%s",
            mamba_refiner,
            mamba_replace_time_attention,
        )
        if checkpoint is not None:
            with open(checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location="cpu")
                # Handle different checkpoint formats
                if isinstance(state_dict, dict):
                    if "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    elif "model" in state_dict:
                        state_dict = state_dict["model"]
            model.load_state_dict(state_dict)

    if mamba_replace_time_attention and checkpoint == "cotracker3_offline":
        raise ValueError(
            "Mamba time replacement requires a trained checkpoint override"
        )
    if mamba_refiner and checkpoint == "cotracker3_offline":
        raise ValueError("Mamba trajectory refiner requires a trained checkpoint override")

    model.eval()
    model.to(device)

    ts_end = datetime.now()
    logger.info(f"Time taken to load model: {ts_end - ts_start}")

    return model


def _add_support_grid(
    video: torch.Tensor,
    queries: torch.Tensor,
    support_grid_size: int,
) -> tuple[torch.Tensor, int]:
    batch, original_points, _ = queries.shape
    if support_grid_size <= 0:
        return queries, original_points

    height, width = video.shape[-2:]
    padding_x, padding_y = width / support_grid_size, height / support_grid_size
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(
            padding_y,
            height - padding_y,
            support_grid_size,
            device=queries.device,
            dtype=queries.dtype,
        ),
        torch.linspace(
            padding_x,
            width - padding_x,
            support_grid_size,
            device=queries.device,
            dtype=queries.dtype,
        ),
        indexing="ij",
    )
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    grid_points = grid_points.unsqueeze(0).expand(batch, -1, -1)
    grid_points = torch.cat(
        [
            torch.zeros(
                batch,
                grid_points.shape[1],
                1,
                device=queries.device,
                dtype=queries.dtype,
            ),
            grid_points,
        ],
        dim=2,
    )
    return torch.cat([queries, grid_points], dim=1), original_points


def inherited_state_logits(probability, frames, mode, span):
    """Boundary probabilities -> inference logits; decay towards neutral 0.5."""
    if probability.ndim != 2 or not bool(torch.isfinite(probability).all()):
        raise ValueError("boundary probability must be finite B,N")
    if bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("boundary probability outside [0,1]")
    if frames < 1 or span <= 0:
        raise ValueError("invalid state initialization span")
    base = torch.logit(probability.detach().float().clamp(1e-4, 1 - 1e-4))
    offsets = torch.arange(frames, device=base.device).float()
    if mode == "vc_decay":
        scale = torch.exp(-offsets / span)
    elif mode == "vc_query":
        scale = (offsets == 0).float()
    elif mode in {"v", "c", "vc"}:
        scale = torch.ones_like(offsets)
    else:
        raise ValueError("unsupported inheritance mode")
    return base[:, None, :] * scale[None, :, None]


def _run_single_clip(
    model: CoTrackerThreeOffline,
    video: torch.Tensor,
    queries: torch.Tensor,
    support_grid_size: int = 10,
    n_iterations: int = 4,
    temporal_stride: int = 1,
    query_feature_memory: QueryFeatureMemory | None = None,
    query_feature_weight: float = 0.0,
    return_query_feature_memory: bool = False,
    return_frame_features: bool = False,
    feature_observer=None,
    initial_visibility_logits=None,
    initial_confidence_logits=None,
    iteration_observer=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[TrackingResult, QueryFeatureMemory | None, torch.Tensor | None]:
    """
    Performs a single forward pass of provided batched samples through the model.
    Requires TAP data.

    Args:
        model: CoTracker model in eval mode.
        video: Tensor of shape (B, T, D=1, H, W); will be expanded to 3 channels.
        queries: Tensor of shape (B, N, 3) with (t, x, y) per point.
        support_grid_size: If >0, adds a regular grid of support points.
        n_iterations: Number of refinement iterations. CoTracker default is 4.
        temporal_stride: Run CoTracker on every Nth frame and linearly recover
            intermediate trajectories. The final frame is always a keyframe.
        device: Compute device.

    Returns:
        TrackingResult with trajectories, visibility and confidence.
    """
    if temporal_stride < 1:
        raise ValueError("temporal_stride must be positive")

    video = video.to(device)
    queries = queries.to(device)
    boundary_queries = queries
    queries, original_N = _add_support_grid(video, queries, support_grid_size)

    output_length = video.shape[1]
    key_indices = torch.arange(0, output_length, temporal_stride, device=device)
    if int(key_indices[-1]) != output_length - 1:
        key_indices = torch.cat(
            [key_indices, key_indices.new_tensor([output_length - 1])]
        )
    model_video = video.index_select(1, key_indices)

    model_kwargs = {
        "video": model_video.expand(-1, -1, 3, -1, -1),
        "queries": queries,
        "iters": n_iterations,
        "is_train": False,
    }
    for name, initial in (("initial_visibility_logits", initial_visibility_logits),
                          ("initial_confidence_logits", initial_confidence_logits)):
        if initial is not None:
            if initial.shape != (video.shape[0], output_length, original_N):
                raise ValueError("initial state must match the full clip and boundary queries")
            if queries.shape[1] != original_N:
                raise ValueError("inherited state does not support added grid points")
            model_kwargs[name] = initial.to(device).index_select(1, key_indices)
    if query_feature_memory is not None:
        model_kwargs["query_feature_memory"] = query_feature_memory if callable(query_feature_memory) else (
            query_feature_memory.track,
            query_feature_memory.support,
        )
        model_kwargs["query_feature_weight"] = query_feature_weight
    if return_query_feature_memory:
        model_kwargs["return_query_feature_memory"] = True
    if return_frame_features:
        model_kwargs["return_frame_features"] = True
    if feature_observer is not None:
        model_kwargs["feature_observer"] = feature_observer
    if iteration_observer is not None:
        model_kwargs["iteration_observer"] = iteration_observer
    out = model(**model_kwargs)

    # Always use original_N to ensure consistent shapes with ground truth
    pred_trajectory = out[0][:, :, :original_N, :]
    pred_visibility = out[1][:, :, :original_N]
    pred_confidence = out[2][:, :, :original_N]

    if temporal_stride > 1:
        pred_trajectory = interpolate_keyframes(
            pred_trajectory, key_indices, output_length
        )
        pred_visibility = interpolate_keyframes(
            pred_visibility, key_indices, output_length
        )
        pred_confidence = interpolate_keyframes(
            pred_confidence, key_indices, output_length
        )

    # optional: overwrite first-timestep preds with query
    B, T, N, _ = pred_trajectory.shape
    queries_t = boundary_queries[..., 0].to(torch.int64)

    for b in range(B):
        for n in range(N):
            frame_idx = queries_t[b, n]
            # Overwrite prediction with exact query point
            pred_trajectory[b, frame_idx, n, :2] = boundary_queries[b, n, 1:3]

    memory = None
    extra_index = 4
    if return_query_feature_memory:
        raw_memory = out[extra_index]
        memory = QueryFeatureMemory(tuple(raw_memory[0]), tuple(raw_memory[1]))
        extra_index += 1

    frame_features = None
    if return_frame_features:
        frame_features = out[extra_index]

    return (
        TrackingResult(pred_trajectory, pred_visibility, pred_confidence),
        memory,
        frame_features,
    )


def forward_pass(
    model: CoTrackerThreeOffline,
    video: torch.Tensor,
    queries: torch.Tensor,
    support_grid_size: int = 10,
    n_iterations: int = 4,
    temporal_stride: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TrackingResult:
    """Run the original full-span CoTracker inference path."""
    result, _, _ = _run_single_clip(
        model=model,
        video=video,
        queries=queries,
        support_grid_size=support_grid_size,
        n_iterations=n_iterations,
        temporal_stride=temporal_stride,
        device=device,
    )
    return result


def fuse_tracking_results(
    local: TrackingResult,
    global_result: TrackingResult,
    global_weight: float,
) -> TrackingResult:
    """Fuse local and original-anchor predictions with normalized reliability weights."""
    if not 0.0 <= global_weight <= 1.0:
        raise ValueError("global_weight must be in [0, 1]")
    if local.trajectories.shape != global_result.trajectories.shape:
        raise ValueError("local and global trajectories must have matching shapes")

    local_score = local.visibility * local.confidence
    global_score = global_result.visibility * global_result.confidence
    local_weight = (1.0 - global_weight) * local_score
    weighted_global = global_weight * global_score
    denominator = local_weight + weighted_global
    fallback = (
        (1.0 - global_weight) * local.trajectories
        + global_weight * global_result.trajectories
    )
    fused_trajectories = torch.where(
        denominator[..., None] > 1e-6,
        (
            local_weight[..., None] * local.trajectories
            + weighted_global[..., None] * global_result.trajectories
        )
        / denominator.clamp_min(1e-6)[..., None],
        fallback,
    )
    # The segment starts at an explicitly supplied anchor. Keeping it exact avoids
    # a discontinuity between adjacent matching levels.
    fused_trajectories[:, 0] = local.trajectories[:, 0]
    fused_visibility = (
        (1.0 - global_weight) * local.visibility
        + global_weight * global_result.visibility
    )
    fused_confidence = (
        (1.0 - global_weight) * local.confidence
        + global_weight * global_result.confidence
    )
    return TrackingResult(fused_trajectories, fused_visibility, fused_confidence)


def sample_trajectory_features(
    model: CoTrackerThreeOffline,
    frame_features: torch.Tensor,
    trajectories: torch.Tensor,
) -> torch.Tensor:
    """Sample normalized level-zero CoTracker features at per-frame positions."""
    batch, frames, points, _ = trajectories.shape
    queried_frames = (
        torch.arange(frames, device=trajectories.device)
        .view(1, frames, 1)
        .expand(batch, frames, points)
        .reshape(batch, frames * points)
    )
    queried_coords = trajectories.reshape(batch, frames * points, 2) / model.stride
    sampled, _ = model.get_track_feat(
        frame_features,
        queried_frames,
        queried_coords,
        support_radius=0,
    )
    return torch.nn.functional.normalize(
        sampled[:, 0].reshape(batch, frames, points, -1),
        dim=-1,
    )


def build_validation_reference(
    original_memory: QueryFeatureMemory,
    current_memory: QueryFeatureMemory,
    point_count: int,
    original_weight: float,
) -> torch.Tensor:
    """Build the same original/current identity feature used by local tracking."""
    original = original_memory.track[0]
    current = current_memory.track[0]
    if original.ndim == 4:
        original = original[:, 0]
    if current.ndim == 4:
        current = current[:, 0]
    original = original[:, :point_count]
    current = current[:, :point_count]
    return torch.nn.functional.normalize(
        original_weight * original + (1.0 - original_weight) * current,
        dim=-1,
    )


def feature_gate_tracking_results(
    local: TrackingResult,
    global_result: TrackingResult,
    local_similarity: torch.Tensor,
    global_similarity: torch.Tensor,
    global_weight: float,
    distance_threshold: float,
    diagnostics: dict[str, int | float] | None = None,
) -> tuple[TrackingResult, torch.Tensor, torch.Tensor]:
    """Use appearance reliability to select a branch when positions disagree."""
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive")
    fused = fuse_tracking_results(local, global_result, global_weight)
    disagreement = torch.linalg.vector_norm(
        local.trajectories - global_result.trajectories,
        dim=-1,
    )
    gate_mask = disagreement > distance_threshold
    gate_mask[:, 0] = False

    local_reliability = local.visibility * local.confidence
    global_reliability = global_result.visibility * global_result.confidence
    local_quality = (
        (1.0 - global_weight)
        * local_reliability
        * ((local_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
    )
    global_quality = (
        global_weight
        * global_reliability
        * ((global_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
    )
    choose_global = global_quality > local_quality
    _diagnostic_add(diagnostics, "feature_gate_candidates", int(gate_mask.sum().item()))
    _diagnostic_add(
        diagnostics,
        "feature_gate_choose_global",
        int((gate_mask & choose_global).sum().item()),
    )
    _diagnostic_add(
        diagnostics,
        "feature_gate_choose_local",
        int((gate_mask & ~choose_global).sum().item()),
    )
    selected_trajectories = torch.where(
        choose_global[..., None],
        global_result.trajectories,
        local.trajectories,
    )
    selected_visibility = torch.where(
        choose_global, global_result.visibility, local.visibility
    )
    selected_confidence = torch.where(
        choose_global, global_result.confidence, local.confidence
    )
    selected_similarity = torch.where(
        choose_global, global_similarity, local_similarity
    )

    trajectories = torch.where(
        gate_mask[..., None], selected_trajectories, fused.trajectories
    )
    visibility = torch.where(gate_mask, selected_visibility, fused.visibility)
    confidence = torch.where(gate_mask, selected_confidence, fused.confidence)
    trajectories[:, 0] = local.trajectories[:, 0]
    return (
        TrackingResult(trajectories, visibility, confidence),
        gate_mask,
        selected_similarity,
    )


def revalidate_in_feature_neighborhood(
    model: CoTrackerThreeOffline,
    frame_features: torch.Tensor,
    result: TrackingResult,
    reference_features: torch.Tensor,
    revalidate_mask: torch.Tensor,
    radius: int,
    image_shape: tuple[int, int],
    diagnostics: dict[str, int | float] | None = None,
) -> tuple[TrackingResult, torch.Tensor]:
    """Search a bounded image-space neighborhood for a better feature match."""
    if radius < 1:
        raise ValueError("radius must be positive")
    height, width = image_shape
    base_trajectories = result.trajectories
    trajectories = base_trajectories.clone()
    reference = reference_features[:, None]
    best_features = sample_trajectory_features(
        model, frame_features, base_trajectories
    )
    best_score = (best_features * reference).sum(dim=-1)
    corrected = torch.zeros_like(revalidate_mask)

    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            if offset_x == 0 and offset_y == 0:
                continue
            squared_distance = offset_x * offset_x + offset_y * offset_y
            if squared_distance > radius * radius:
                continue
            candidate = base_trajectories.clone()
            candidate[..., 0] = (candidate[..., 0] + offset_x).clamp(0, width - 1)
            candidate[..., 1] = (candidate[..., 1] + offset_y).clamp(0, height - 1)
            candidate_features = sample_trajectory_features(
                model, frame_features, candidate
            )
            candidate_score = (candidate_features * reference).sum(dim=-1)
            # A small motion prior avoids moving to an equally similar but more
            # distant pixel. Feature evidence still dominates within radius 4.
            candidate_score = candidate_score - 0.01 * (
                squared_distance / float(radius * radius)
            )
            improve = revalidate_mask & (candidate_score > best_score)
            trajectories = torch.where(
                improve[..., None], candidate, trajectories
            )
            best_score = torch.where(improve, candidate_score, best_score)
            corrected |= improve

    trajectories[:, 0] = result.trajectories[:, 0]
    corrected[:, 0] = False
    correction_distance = torch.linalg.vector_norm(
        trajectories - base_trajectories,
        dim=-1,
    )
    _diagnostic_add(
        diagnostics,
        "feature_revalidate_candidates",
        int(revalidate_mask.sum().item()),
    )
    _diagnostic_add(
        diagnostics,
        "feature_revalidate_corrected",
        int(corrected.sum().item()),
    )
    _diagnostic_add(
        diagnostics,
        "feature_revalidate_distance_sum",
        float(correction_distance[corrected].sum().item()),
    )
    if bool(corrected.any().item()):
        _diagnostic_max(
            diagnostics,
            "feature_revalidate_distance_max",
            float(correction_distance[corrected].max().item()),
        )
    return (
        TrackingResult(trajectories, result.visibility, result.confidence),
        corrected,
    )


def segment_is_occluded(
    visibility: torch.Tensor,
    threshold: float,
    point_fraction: float,
) -> bool:
    """Return true when enough points stay invisible for an entire matching level."""
    if visibility.ndim != 3:
        raise ValueError("visibility must have shape (B, T, N)")
    if visibility.shape[1] <= 1:
        return False
    continuously_hidden = (visibility[:, 1:] < threshold).all(dim=1)
    hidden_fraction = continuously_hidden.float().mean(dim=1)
    return bool((hidden_fraction >= point_fraction).any().item())


def hierarchical_forward_pass(
    model: CoTrackerThreeOffline,
    video: torch.Tensor,
    queries: torch.Tensor,
    span: int,
    support_grid_size: int = 10,
    n_iterations: int = 4,
    original_feature_weight: float = 0.0,
    occlusion_merge: bool = False,
    occlusion_visibility_threshold: float = 0.5,
    occlusion_point_fraction: float = 0.5,
    dual_anchor_weight: float = 0.0,
    feature_gate_distance: float = 0.0,
    feature_similarity_threshold: float = 0.5,
    feature_revalidate_radius: int = 0,
    query_memory_mode: str = "none",
    query_memory_slots: int = 0,
    query_memory_min_reliability: float = 0.5,
    query_memory_min_similarity: float = 0.5,
    query_memory_original_floor: float = 0.3,
    query_memory_diversity_weight: float = 0.25,
    query_memory_refinement: str = "none",
    query_state_inheritance: str = "none",
    frame_backcheck_rows=None,
    frame_backcheck_fourway=False,
    chain_trace=None,
    diagnostics: dict[str, int | float] | None = None,
    return_long_fusion_context: bool = False,
    return_mask_appearance_context: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TrackingResult | LongFusionContext | MaskAppearanceContext:
    """Track short levels, re-anchor at endpoints, and optionally merge occlusions."""
    if span < 1:
        raise ValueError("span must be positive")
    if feature_gate_distance < 0:
        raise ValueError("feature_gate_distance must be non-negative")
    if not -1.0 <= feature_similarity_threshold <= 1.0:
        raise ValueError("feature_similarity_threshold must be in [-1, 1]")
    if feature_revalidate_radius < 0:
        raise ValueError("feature_revalidate_radius must be non-negative")
    if feature_gate_distance > 0 and dual_anchor_weight <= 0:
        raise ValueError("feature gating requires dual anchor tracking")
    if feature_gate_distance > 0 and original_feature_weight <= 0:
        raise ValueError("feature gating requires original feature memory")
    if feature_revalidate_radius > 0 and feature_gate_distance <= 0:
        raise ValueError("feature revalidation requires feature gating")
    if query_memory_mode != "none" and original_feature_weight <= 0:
        raise ValueError("query memory requires query feature weighting")
    if query_memory_mode == "none" and query_memory_slots != 0:
        raise ValueError("query memory slots require an active memory mode")
    if return_long_fusion_context and return_mask_appearance_context:
        raise ValueError("select exactly one hierarchical feature context")
    if video.ndim != 5 or queries.ndim != 3:
        raise ValueError("unexpected video or query shape")

    video = video.to(device)
    queries = queries.to(device).clone()
    queries[..., 0] = 0
    batch, total_frames = video.shape[:2]
    point_count = queries.shape[1]
    if diagnostics is not None:
        diagnostics["hierarchical_span"] = span
        diagnostics["hierarchical_point_count"] = point_count
        diagnostics["hierarchical_total_frames"] = total_frames
    if total_frames == 1:
        result, _, frame_features = _run_single_clip(
            model,
            video,
            queries,
            support_grid_size=support_grid_size,
            n_iterations=n_iterations,
            return_frame_features=return_mask_appearance_context,
            device=device,
        )
        if return_mask_appearance_context:
            if frame_features is None:
                raise RuntimeError("mask appearance context requires frame features")
            return MaskAppearanceContext(result, frame_features)
        return result

    global_result = None
    global_memory = None
    global_frame_features = None
    if dual_anchor_weight > 0:
        if return_long_fusion_context or return_mask_appearance_context:
            global_result, global_memory, global_frame_features = _run_single_clip(
                model,
                video,
                queries,
                support_grid_size=support_grid_size,
                n_iterations=n_iterations,
                return_query_feature_memory=return_long_fusion_context,
                return_frame_features=True,
                device=device,
            )
        else:
            global_result = forward_pass(
                model,
                video,
                queries,
                support_grid_size=support_grid_size,
                n_iterations=n_iterations,
                device=device,
            )

    trajectories = torch.zeros(
        (batch, total_frames, point_count, 2), device=device, dtype=queries.dtype
    )
    visibility = torch.zeros(
        (batch, total_frames, point_count), device=device, dtype=queries.dtype
    )
    confidence = torch.zeros_like(visibility)
    covered = torch.zeros(total_frames, device=device, dtype=torch.bool)

    original_memory = None
    current_queries = queries
    previous_start = None
    previous_queries = None
    start = 0
    level = 0
    validation_enabled = feature_gate_distance > 0
    query_memory = None
    if query_memory_mode != "none":
        query_memory = DynamicQueryMemoryBank(
            mode=query_memory_mode,
            capacity=query_memory_slots,
            min_reliability=query_memory_min_reliability,
            min_similarity=query_memory_min_similarity,
            original_floor=query_memory_original_floor,
            diversity_weight=query_memory_diversity_weight,
            diagnostics=diagnostics,
            refinement=query_memory_refinement,
        )
        if diagnostics is not None:
            diagnostics["query_memory_enabled"] = 1
            diagnostics["query_memory_capacity"] = query_memory_slots
            for key in (
                "query_memory_write_candidates",
                "query_memory_writes_accepted",
                "query_memory_writes_rejected_reliability",
                "query_memory_writes_rejected_similarity",
                "query_memory_duplicate_frames",
                "query_memory_evictions",
                "query_memory_fusions",
                "query_memory_slots_used_sum",
                "query_memory_slots_used_max",
                "query_memory_bank_size_max",
                "query_memory_original_weight_sum",
                "memory_point_candidates", "memory_point_rejected", "memory_point_accepted",
                "memory_point_fusion_reads", "memory_original_only_points", "memory_weight_changed_points",
                "memory_retrieval_point_reads", "memory_retrieval_fallback_points",
                "memory_cycle_runs", "memory_cycle_checked_points", "memory_cycle_rejected_points",
                "memory_cycle_error_sum", "memory_cycle_seconds",
                "memory_contour_checked_points", "memory_contour_corrected_points",
                "memory_contour_correction_sum", "memory_recent_slot_prunes",
            ):
                diagnostics[key] = 0

    if query_state_inheritance not in {"none", "v", "c", "vc", "vc_decay", "vc_query"}:
        raise ValueError("unsupported query state inheritance")
    if query_state_inheritance != "none" and support_grid_size != 0:
        raise ValueError("query state inheritance requires grid0")
    # Preserve the exact prior that accompanied each query. A merged retry must
    # not use overwritten boundary predictions or the failed segment endpoint.
    boundary_states = {}
    backcheck_pending = {}
    if frame_backcheck_rows is not None and (batch != 1 or support_grid_size != 0 or n_iterations < 2):
        raise ValueError("frame backcheck requires batch1, grid0 and >=2 iterations")
    for field in ("state_inherited_segments", "state_v_values", "state_c_values",
                  "state_v_abs_logit_sum", "state_c_abs_logit_sum",
                  "state_v_nonzero_values", "state_c_nonzero_values",
                  "state_v_high_prior_points", "state_c_high_prior_points",
                  "state_v_low_prior_points", "state_c_low_prior_points"):
        _diagnostic_add(diagnostics, field, 0)

    def remember_boundary(frame, result):
        if query_state_inheritance != "none":
            boundary_states[frame] = (result.visibility[:, -1].detach().clone(),
                                      result.confidence[:, -1].detach().clone())

    def run_level(level_start, level_end, level_queries):
        nonlocal original_memory
        trace_key = chain_trace.begin_level(level_start, level_end, level_queries) if chain_trace is not None else None
        _diagnostic_add(diagnostics, "hierarchical_level_runs", 1)
        _diagnostic_add(
            diagnostics,
            "hierarchical_level_frame_visits",
            level_end - level_start + 1,
        )
        history_memory = (
            query_memory.fused_memory()
            if query_memory is not None and query_memory_refinement != "current_retrieval"
            else original_memory
        )
        if query_memory is not None and query_memory_refinement == "current_retrieval" and query_memory.entries:
            point_reliability = visibility[:, level_start] * confidence[:, level_start]
            if level_start == 0:
                point_reliability = torch.ones_like(point_reliability)
            history_memory = lambda feature: query_memory.fused_memory(feature, point_reliability)
        capture_memory = (
            (original_feature_weight > 0 or validation_enabled)
            and original_memory is None
        )
        return_memory = (
            query_memory is not None or capture_memory or validation_enabled
        )
        state_kwargs = {}
        backcheck_capture = {}
        if frame_backcheck_rows is not None:
            def observe_iterations(history, features):
                backcheck_capture['history'] = history
                backcheck_capture['features'] = features
            state_kwargs['iteration_observer'] = observe_iterations
        if query_state_inheritance != "none" and level_start > 0:
            prior_v, prior_c = boundary_states[level_start]
            _diagnostic_add(diagnostics, "state_inherited_segments", 1)
            for kind, probability in (("v", prior_v), ("c", prior_c)):
                if query_state_inheritance in {"v", "c"} and kind != query_state_inheritance:
                    continue
                initial = inherited_state_logits(probability, level_end - level_start + 1, query_state_inheritance, span)
                field = "initial_visibility_logits" if kind == "v" else "initial_confidence_logits"
                state_kwargs[field] = initial
                _diagnostic_add(diagnostics, "state_" + kind + "_values", initial.numel())
                _diagnostic_add(diagnostics, "state_" + kind + "_abs_logit_sum", float(initial.abs().sum().item()))
                _diagnostic_add(diagnostics, "state_" + kind + "_nonzero_values", int((initial.abs() > 1e-6).sum().item()))
                _diagnostic_add(diagnostics, "state_" + kind + "_high_prior_points", int((probability > .99).sum().item()))
                _diagnostic_add(diagnostics, "state_" + kind + "_low_prior_points", int((probability < .01).sum().item()))
        result, captured, frame_features = _run_single_clip(
            model,
            video[:, level_start : level_end + 1],
            level_queries,
            support_grid_size=support_grid_size,
            n_iterations=n_iterations,
            query_feature_memory=history_memory,
            query_feature_weight=original_feature_weight,
            return_query_feature_memory=return_memory,
            return_frame_features=validation_enabled,
            feature_observer=(chain_trace.observer(trace_key) if chain_trace is not None else None),
            **state_kwargs,
            device=device,
        )
        if chain_trace is not None:
            chain_trace.tensor(trace_key + "/local_trajectory", result.trajectories)
        current_memory = captured
        if captured is not None and original_memory is None:
            original_memory = captured
        if global_result is not None:
            global_slice = TrackingResult(
                global_result.trajectories[:, level_start : level_end + 1],
                global_result.visibility[:, level_start : level_end + 1],
                global_result.confidence[:, level_start : level_end + 1],
            )
            if chain_trace is not None:
                local_score = (1.0 - dual_anchor_weight) * result.visibility * result.confidence
                global_score = dual_anchor_weight * global_slice.visibility * global_slice.confidence
                weight = torch.where(local_score + global_score > 1e-6,
                                     local_score / (local_score + global_score).clamp_min(1e-6),
                                     1.0 - dual_anchor_weight)
                chain_trace.tensor(trace_key + "/local_fusion_weight", weight)
                chain_trace.tensor(trace_key + "/global_trajectory", global_slice.trajectories)
            if validation_enabled:
                if current_memory is None or original_memory is None:
                    raise RuntimeError("feature validation memory was not returned")
                if frame_features is None:
                    raise RuntimeError("frame features were not returned")
                reference = build_validation_reference(
                    original_memory,
                    current_memory,
                    point_count,
                    original_feature_weight,
                )
                local_features = sample_trajectory_features(
                    model, frame_features, result.trajectories
                )
                global_features = sample_trajectory_features(
                    model, frame_features, global_slice.trajectories
                )
                local_similarity = (local_features * reference[:, None]).sum(dim=-1)
                global_similarity = (
                    global_features * reference[:, None]
                ).sum(dim=-1)
                disagreement = torch.linalg.vector_norm(
                    result.trajectories - global_slice.trajectories,
                    dim=-1,
                )
                disagreement_mask = torch.ones_like(disagreement, dtype=torch.bool)
                disagreement_mask[:, 0] = False
                measured_disagreement = disagreement[disagreement_mask]
                _diagnostic_add(
                    diagnostics,
                    "dual_anchor_point_frames",
                    int(measured_disagreement.numel()),
                )
                _diagnostic_add(
                    diagnostics,
                    "dual_anchor_disagreement_sum",
                    float(measured_disagreement.sum().item()),
                )
                if measured_disagreement.numel() > 0:
                    _diagnostic_max(
                        diagnostics,
                        "dual_anchor_disagreement_max",
                        float(measured_disagreement.max().item()),
                    )
                result, gate_mask, selected_similarity = feature_gate_tracking_results(
                    result,
                    global_slice,
                    local_similarity,
                    global_similarity,
                    dual_anchor_weight,
                    feature_gate_distance,
                    diagnostics,
                )
                logger.info(
                    "Feature gate selected one branch for %d point-frames",
                    int(gate_mask.sum().item()),
                )
                if feature_revalidate_radius > 0:
                    revalidate_mask = gate_mask & (
                        torch.maximum(local_similarity, global_similarity)
                        < feature_similarity_threshold
                    )
                    _diagnostic_add(
                        diagnostics,
                        "feature_both_below_threshold",
                        int(revalidate_mask.sum().item()),
                    )
                    result, corrected = revalidate_in_feature_neighborhood(
                        model,
                        frame_features,
                        result,
                        reference,
                        revalidate_mask,
                        feature_revalidate_radius,
                        tuple(video.shape[-2:]),
                        diagnostics,
                    )
                    logger.info(
                        "Feature neighborhood corrected %d point-frames",
                        int(corrected.sum().item()),
                    )
            else:
                disagreement = torch.linalg.vector_norm(
                    result.trajectories - global_slice.trajectories,
                    dim=-1,
                )
                disagreement_mask = torch.ones_like(disagreement, dtype=torch.bool)
                disagreement_mask[:, 0] = False
                measured_disagreement = disagreement[disagreement_mask]
                _diagnostic_add(
                    diagnostics,
                    "dual_anchor_point_frames",
                    int(measured_disagreement.numel()),
                )
                _diagnostic_add(
                    diagnostics,
                    "dual_anchor_disagreement_sum",
                    float(measured_disagreement.sum().item()),
                )
                if measured_disagreement.numel() > 0:
                    _diagnostic_max(
                        diagnostics,
                        "dual_anchor_disagreement_max",
                        float(measured_disagreement.max().item()),
                    )
                result = fuse_tracking_results(
                    result, global_slice, dual_anchor_weight
                )
        if chain_trace is not None:
            chain_trace.tensor(trace_key + "/fused_trajectory", result.trajectories)
        if frame_backcheck_rows is not None:
            try:
                from .frame_backcheck import score_frames, score_fourway_frames
            except ImportError:
                from frame_backcheck import score_frames, score_fourway_frames
            scorer = score_fourway_frames if frame_backcheck_fourway else score_frames
            backcheck_pending[(level_start, level_end)] = scorer(
                backcheck_capture['features'], level_queries, result.trajectories,
                backcheck_capture['history'], level_start, model.stride)
        return result, captured

    def commit_query_memory(level_start, result, captured):
        if query_memory is None or captured is None:
            return
        if query_memory_refinement == "cycle_write" and level_start in query_memory.frame_indices:
            _diagnostic_add(diagnostics, "query_memory_duplicate_frames", 1)
            return
        if level_start == 0:
            reliability = 1.0
            point_reliability = torch.ones_like(result.visibility[:, 0])
        elif bool(covered[level_start].item()):
            point_reliability = visibility[:, level_start] * confidence[:, level_start]
            reliability = float(
                (
                    visibility[:, level_start]
                    * confidence[:, level_start]
                ).mean().item()
            )
        else:
            point_reliability = result.visibility[:, 0] * result.confidence[:, 0]
            reliability = float(
                (result.visibility[:, 0] * result.confidence[:, 0]).mean().item()
            )
        cycle_valid = None
        if query_memory_refinement == "cycle_write" and level_start > 0:
            cycle_start = max(0, level_start - span)
            if not bool(covered[cycle_start : level_start + 1].all()):
                raise RuntimeError("cycle check requires committed forward trajectory")
            reverse_queries = torch.cat([
                torch.zeros_like(queries[..., :1]), trajectories[:, level_start]
            ], dim=-1)
            if video.is_cuda:
                torch.cuda.synchronize(video.device)
            cycle_started = perf_counter()
            reverse, _, _ = _run_single_clip(
                model, video[:, cycle_start : level_start + 1].flip(1), reverse_queries,
                support_grid_size=support_grid_size, n_iterations=n_iterations, device=device,
            )
            error = torch.linalg.vector_norm(reverse.trajectories[:, -1] - trajectories[:, cycle_start], dim=-1)
            cycle_valid = error <= 2.0
            if video.is_cuda:
                torch.cuda.synchronize(video.device)
            _diagnostic_add(diagnostics, "memory_cycle_seconds", perf_counter() - cycle_started)
            _diagnostic_add(diagnostics, "memory_cycle_runs", 1)
            _diagnostic_add(diagnostics, "memory_cycle_checked_points", error.numel())
            _diagnostic_add(diagnostics, "memory_cycle_rejected_points", int((~cycle_valid).sum().item()))
            _diagnostic_add(diagnostics, "memory_cycle_error_sum", float(error.sum().item()))
        accepted = query_memory.add(level_start, captured, reliability, point_reliability, cycle_valid)
        logger.info(
            "Query memory anchor frame=%d reliability=%.4f accepted=%s bank=%s",
            level_start,
            reliability,
            accepted,
            query_memory.frame_indices,
        )

    def store_level(level_start, level_end, result):
        if frame_backcheck_rows is not None:
            # Discard superseded segments on merge. Each nonquery frame has one
            # final score, attached to the anchor that predicted that frame.
            frame_backcheck_rows[:] = [r for r in frame_backcheck_rows if not level_start < r['frame'] <= level_end]
            frame_backcheck_rows.extend(backcheck_pending[(level_start, level_end)])
            backcheck_pending.clear()
        trajectories[:, level_start : level_end + 1] = result.trajectories
        visibility[:, level_start : level_end + 1] = result.visibility
        confidence[:, level_start : level_end + 1] = result.confidence
        covered[level_start : level_end + 1] = True

    while start < total_frames - 1:
        end = min(start + span, total_frames - 1)
        logger.info("Hierarchical matching level %d: frames %d-%d", level, start, end)
        level_result, level_memory = run_level(start, end, current_queries)

        should_merge = (
            occlusion_merge
            and end < total_frames - 1
            and segment_is_occluded(
                level_result.visibility,
                occlusion_visibility_threshold,
                occlusion_point_fraction,
            )
        )
        if occlusion_merge:
            _diagnostic_add(diagnostics, "occlusion_merge_checks", 1)
        if should_merge:
            _diagnostic_add(diagnostics, "occlusion_merges", 1)
            merge_start = previous_start if previous_start is not None else start
            merge_queries = previous_queries if previous_queries is not None else current_queries
            merge_end = min(end + span, total_frames - 1)
            logger.info(
                "Occlusion merge: replacing frames %d-%d with one matching level",
                merge_start,
                merge_end,
            )
            merged_result, merged_memory = run_level(
                merge_start, merge_end, merge_queries
            )
            commit_query_memory(merge_start, merged_result, merged_memory)
            store_level(merge_start, merge_end, merged_result)
            start = merge_end
            remember_boundary(start, merged_result)
            current_queries = torch.cat(
                [
                    torch.zeros_like(merge_queries[..., :1]),
                    merged_result.trajectories[:, -1],
                ],
                dim=-1,
            )
            previous_start = None
            previous_queries = None
            level += 1
            continue

        commit_query_memory(start, level_result, level_memory)
        store_level(start, end, level_result)
        previous_start = start
        previous_queries = current_queries.clone()
        start = end
        remember_boundary(start, level_result)
        current_queries = torch.cat(
            [
                torch.zeros_like(current_queries[..., :1]),
                level_result.trajectories[:, -1],
            ],
            dim=-1,
        )
        level += 1

    if not bool(covered.all().item()):
        raise RuntimeError("hierarchical tracker did not cover every frame")

    if query_memory_refinement == "contour_guard":
        trajectories = contour_motion_guard(trajectories, visibility * confidence, diagnostics)
    hierarchical = TrackingResult(trajectories, visibility, confidence)
    if return_mask_appearance_context:
        if global_frame_features is None:
            raise RuntimeError("mask appearance context requires global frame features")
        return MaskAppearanceContext(hierarchical, global_frame_features)
    if not return_long_fusion_context:
        return hierarchical
    if global_result is None or global_memory is None or global_frame_features is None:
        raise RuntimeError("long fusion context requires global tracking features")

    reference = global_memory.track[0]
    if reference.ndim == 4:
        reference = reference[:, 0]
    reference = torch.nn.functional.normalize(reference[:, :point_count], dim=-1)
    hierarchical_features = sample_trajectory_features(
        model, global_frame_features, hierarchical.trajectories
    )
    hierarchical_similarity = (hierarchical_features * reference[:, None]).sum(dim=-1)
    del hierarchical_features
    global_features = sample_trajectory_features(
        model, global_frame_features, global_result.trajectories
    )
    global_similarity = (global_features * reference[:, None]).sum(dim=-1)
    return LongFusionContext(
        hierarchical,
        global_result,
        hierarchical_similarity,
        global_similarity,
    )
