"""Differentiable TrackRAD mask-membership supervision for point tracks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _normalize_tracks(tracks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    normalized = tracks.clone()
    normalized[..., 0] = 2.0 * normalized[..., 0] / max(width - 1, 1) - 1.0
    normalized[..., 1] = 2.0 * normalized[..., 1] / max(height - 1, 1) - 1.0
    return normalized


def _sample_video_maps(maps: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
    """Sample ``(B,T,1,H,W)`` maps at ``(B,T,N,2)`` pixel coordinates."""
    if maps.ndim != 5 or maps.shape[2] != 1:
        raise ValueError("maps must have shape (B,T,1,H,W)")
    if tracks.ndim != 4 or tracks.shape[:2] != maps.shape[:2]:
        raise ValueError("tracks must have shape (B,T,N,2)")
    batch, frames, _, height, width = maps.shape
    points = tracks.shape[2]
    grid = _normalize_tracks(tracks, height, width).reshape(
        batch * frames, points, 1, 2
    )
    sampled = F.grid_sample(
        maps.reshape(batch * frames, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(batch, frames, points)


def mask_membership_consistency_loss(
    segmentation: torch.Tensor,
    queries: torch.Tensor,
    tracks: torch.Tensor,
    valid: torch.Tensor | None = None,
    blur_radius: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep each tracked point's target membership consistent across frames.

    A query receives an inside/outside identity from the real mask at its query
    frame.  Student coordinates are sampled against smoothed real masks in all
    other frames.  Positive and negative query classes are averaged separately
    so background points cannot overwhelm small targets.
    """
    if segmentation.ndim != 5 or segmentation.shape[2] != 1:
        raise ValueError("segmentation must have shape (B,T,1,H,W)")
    if queries.ndim != 3 or queries.shape[-1] != 3:
        raise ValueError("queries must have shape (B,N,3)")
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("tracks must have shape (B,T,N,2)")
    if tracks.shape[0] != segmentation.shape[0] or tracks.shape[1] != segmentation.shape[1]:
        raise ValueError("track and segmentation batch/time shapes must match")
    if queries.shape[0] != tracks.shape[0] or queries.shape[1] != tracks.shape[2]:
        raise ValueError("query and track point shapes must match")
    if blur_radius < 0:
        raise ValueError("blur_radius must be non-negative")

    masks = (segmentation > 0.5).float()
    batch, frames, _, height, width = masks.shape
    query_frames = queries[..., 0].round().long().clamp(0, frames - 1)
    query_x = queries[..., 1].round().long().clamp(0, width - 1)
    query_y = queries[..., 2].round().long().clamp(0, height - 1)
    batch_indices = torch.arange(batch, device=masks.device)[:, None]
    membership = masks[
        batch_indices, query_frames, 0, query_y, query_x
    ].bool()

    flat_masks = masks.reshape(batch * frames, 1, height, width)
    if blur_radius > 0:
        kernel = 2 * blur_radius + 1
        flat_masks = F.avg_pool2d(
            flat_masks, kernel_size=kernel, stride=1, padding=blur_radius
        )
    soft_masks = flat_masks.reshape(batch, frames, 1, height, width)
    epsilon = 1e-4
    probabilities = _sample_video_maps(soft_masks, tracks).clamp(
        epsilon, 1 - epsilon
    )
    targets = membership[:, None, :].expand_as(probabilities).float()
    losses = F.binary_cross_entropy(probabilities, targets, reduction="none")
    # Bound the auxiliary objective to [0, 1] so its configured weight remains
    # comparable to the existing teacher loss across cases and target sizes.
    losses = losses / -math.log(epsilon)

    frame_indices = torch.arange(frames, device=masks.device)[None, :, None]
    eligible = frame_indices != query_frames[:, None, :]
    if valid is not None:
        if valid.shape != losses.shape:
            raise ValueError("valid must have shape (B,T,N)")
        eligible = eligible & valid.bool()

    class_losses = []
    for class_value in (True, False):
        selected = eligible & (membership[:, None, :] == class_value)
        if bool(selected.any().item()):
            class_losses.append(losses[selected].mean())
    if not class_losses:
        loss = tracks.sum() * 0.0
    else:
        loss = torch.stack(class_losses).mean()

    metrics = {
        "loss": float(loss.detach().cpu()),
        "inside_fraction": float(membership.float().mean().detach().cpu()),
        "eligible_points": float(eligible.sum().detach().cpu()),
    }
    return loss, metrics
