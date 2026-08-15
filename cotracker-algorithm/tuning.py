"""Torch-only helpers shared by the single-point tuning profiles."""

from __future__ import annotations

import torch


def interpolate_keyframes(
    values: torch.Tensor,
    key_indices: torch.Tensor,
    output_length: int,
) -> torch.Tensor:
    """Linearly interpolate ``(B, K, ...)`` values at arbitrary key indices."""

    if values.ndim < 2:
        raise ValueError("values must have batch and time dimensions")
    if key_indices.ndim != 1 or values.shape[1] != key_indices.numel():
        raise ValueError("key_indices must match the values time dimension")
    if output_length < 1:
        raise ValueError("output_length must be positive")
    if key_indices.numel() == 0 or int(key_indices[0]) != 0:
        raise ValueError("key_indices must start at zero")
    if int(key_indices[-1]) != output_length - 1:
        raise ValueError("key_indices must include the final output frame")
    if torch.any(key_indices[1:] <= key_indices[:-1]):
        raise ValueError("key_indices must be strictly increasing")

    output = values.new_empty((values.shape[0], output_length, *values.shape[2:]))
    for interval in range(key_indices.numel() - 1):
        start = int(key_indices[interval])
        end = int(key_indices[interval + 1])
        steps = end - start + 1
        alpha_shape = (1, steps, *([1] * (values.ndim - 2)))
        alpha = torch.linspace(
            0.0,
            1.0,
            steps,
            device=values.device,
            dtype=values.dtype,
        ).view(alpha_shape)
        left = values[:, interval].unsqueeze(1)
        right = values[:, interval + 1].unsqueeze(1)
        output[:, start : end + 1] = left * (1.0 - alpha) + right * alpha

    if key_indices.numel() == 1:
        output[:] = values[:, :1]
    return output


def temporal_median_smooth(
    trajectories: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Median-filter trajectories along time without changing their shape."""

    if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape (B, T, N, 2)")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window == 1 or trajectories.shape[1] == 1:
        return trajectories

    radius = window // 2
    smoothed = trajectories.clone()
    for frame in range(trajectories.shape[1]):
        start = max(0, frame - radius)
        end = min(trajectories.shape[1], frame + radius + 1)
        smoothed[:, frame] = trajectories[:, start:end].median(dim=1).values
    return smoothed


def build_validity_mask(
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    visibility_threshold: float | None,
    confidence_threshold: float | None,
) -> torch.Tensor | None:
    """Combine optional CoTracker visibility and confidence thresholds."""

    if visibility.shape != confidence.shape:
        raise ValueError("visibility and confidence must have matching shapes")
    if visibility_threshold is None and confidence_threshold is None:
        return None

    valid = torch.ones_like(visibility, dtype=torch.bool)
    if visibility_threshold is not None:
        valid &= visibility >= visibility_threshold
    if confidence_threshold is not None:
        valid &= confidence >= confidence_threshold
    return valid
