from datetime import datetime
import logging
import os
from pathlib import Path
from typing import NamedTuple

import torch
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline

from tuning import interpolate_keyframes

logger = logging.getLogger(__name__)


class TrackingResult(NamedTuple):
    trajectories: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor


class QueryFeatureMemory(NamedTuple):
    track: tuple[torch.Tensor, ...]
    support: tuple[torch.Tensor, ...]


def setup_model(
    checkpoint: str = "cotracker3_offline",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CoTrackerThreeOffline:
    ts_start = datetime.now()
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[TrackingResult, QueryFeatureMemory | None]:
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
    if query_feature_memory is not None:
        model_kwargs["query_feature_memory"] = (
            query_feature_memory.track,
            query_feature_memory.support,
        )
        model_kwargs["query_feature_weight"] = query_feature_weight
    if return_query_feature_memory:
        model_kwargs["return_query_feature_memory"] = True
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
    if return_query_feature_memory:
        raw_memory = out[4]
        memory = QueryFeatureMemory(tuple(raw_memory[0]), tuple(raw_memory[1]))

    return TrackingResult(pred_trajectory, pred_visibility, pred_confidence), memory


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
    result, _ = _run_single_clip(
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TrackingResult:
    """Track short levels, re-anchor at endpoints, and optionally merge occlusions."""
    if span < 1:
        raise ValueError("span must be positive")
    if video.ndim != 5 or queries.ndim != 3:
        raise ValueError("unexpected video or query shape")

    video = video.to(device)
    queries = queries.to(device).clone()
    queries[..., 0] = 0
    batch, total_frames = video.shape[:2]
    point_count = queries.shape[1]
    if total_frames == 1:
        result, _ = _run_single_clip(
            model,
            video,
            queries,
            support_grid_size=support_grid_size,
            n_iterations=n_iterations,
            device=device,
        )
        return result

    global_result = None
    if dual_anchor_weight > 0:
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

    def run_level(level_start, level_end, level_queries):
        nonlocal original_memory
        capture_memory = original_feature_weight > 0 and original_memory is None
        result, captured = _run_single_clip(
            model,
            video[:, level_start : level_end + 1],
            level_queries,
            support_grid_size=support_grid_size,
            n_iterations=n_iterations,
            query_feature_memory=original_memory,
            query_feature_weight=original_feature_weight,
            return_query_feature_memory=capture_memory,
            device=device,
        )
        if captured is not None:
            original_memory = captured
        if global_result is not None:
            global_slice = TrackingResult(
                global_result.trajectories[:, level_start : level_end + 1],
                global_result.visibility[:, level_start : level_end + 1],
                global_result.confidence[:, level_start : level_end + 1],
            )
            result = fuse_tracking_results(result, global_slice, dual_anchor_weight)
        return result

    def store_level(level_start, level_end, result):
        trajectories[:, level_start : level_end + 1] = result.trajectories
        visibility[:, level_start : level_end + 1] = result.visibility
        confidence[:, level_start : level_end + 1] = result.confidence
        covered[level_start : level_end + 1] = True

    while start < total_frames - 1:
        end = min(start + span, total_frames - 1)
        logger.info("Hierarchical matching level %d: frames %d-%d", level, start, end)
        level_result = run_level(start, end, current_queries)

        should_merge = (
            occlusion_merge
            and end < total_frames - 1
            and segment_is_occluded(
                level_result.visibility,
                occlusion_visibility_threshold,
                occlusion_point_fraction,
            )
        )
        if should_merge:
            merge_start = previous_start if previous_start is not None else start
            merge_queries = previous_queries if previous_queries is not None else current_queries
            merge_end = min(end + span, total_frames - 1)
            logger.info(
                "Occlusion merge: replacing frames %d-%d with one matching level",
                merge_start,
                merge_end,
            )
            merged_result = run_level(merge_start, merge_end, merge_queries)
            store_level(merge_start, merge_end, merged_result)
            start = merge_end
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

        store_level(start, end, level_result)
        previous_start = start
        previous_queries = current_queries.clone()
        start = end
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

    return TrackingResult(trajectories, visibility, confidence)
