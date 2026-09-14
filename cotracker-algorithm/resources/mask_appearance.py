"""Mask-level appearance validation using frozen CoTracker frame features."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _add(
    diagnostics: dict[str, int | float] | None,
    key: str,
    value: int | float,
) -> None:
    if diagnostics is not None:
        diagnostics[key] = diagnostics.get(key, 0) + value


def _maximum(
    diagnostics: dict[str, int | float] | None,
    key: str,
    value: int | float,
) -> None:
    if diagnostics is not None:
        diagnostics[key] = max(diagnostics.get(key, value), value)


def _shift_without_wrap(mask: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate ``(H, W)`` mask with zero fill instead of circular wrapping."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    height, width = mask.shape
    shifted = torch.zeros_like(mask)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    if source_y1 <= source_y0 or source_x1 <= source_x0:
        return shifted
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    shifted[target_y0:target_y1, target_x0:target_x1] = mask[
        source_y0:source_y1, source_x0:source_x1
    ]
    return shifted


def _masked_prototype(
    feature_map: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    if feature_map.ndim != 3 or mask.ndim != 2:
        raise ValueError("expected feature map (C,H,W) and mask (H,W)")
    if feature_map.shape[-2:] != mask.shape:
        raise ValueError("feature map and mask spatial shapes must match")
    selected = feature_map[:, mask]
    if selected.shape[1] == 0:
        return None
    return F.normalize(selected.mean(dim=1), dim=0)


def refine_masks_by_appearance(
    masks: torch.Tensor,
    query_mask: torch.Tensor,
    frame_features: torch.Tensor,
    radius_pixels: int,
    feature_stride: int = 4,
    minimum_similarity_gain: float = 0.01,
    displacement_penalty: float = 0.01,
    diagnostics: dict[str, int | float] | None = None,
) -> torch.Tensor:
    """Select a conservative whole-mask translation using target appearance.

    The first-frame ground-truth mask defines one normalized CoTracker feature
    prototype.  Later masks are translated on the feature grid and accepted only
    when their penalized prototype similarity improves by the configured margin.
    """
    if masks.ndim != 4 or masks.shape[0] != 1:
        raise ValueError("masks must have shape (1,T,H,W)")
    if query_mask.ndim != 4 or query_mask.shape[:2] != (1, 1):
        raise ValueError("query_mask must have shape (1,1,H,W)")
    if frame_features.ndim != 5 or frame_features.shape[:2] != masks.shape[:2]:
        raise ValueError("frame_features must have shape (1,T,C,Hf,Wf)")
    if radius_pixels < 1:
        raise ValueError("radius_pixels must be positive")
    if feature_stride < 1:
        raise ValueError("feature_stride must be positive")
    if minimum_similarity_gain < 0.0 or displacement_penalty < 0.0:
        raise ValueError("appearance gain and displacement penalty must be non-negative")

    feature_height, feature_width = frame_features.shape[-2:]
    masks_low = F.adaptive_max_pool2d(
        masks.float(), (feature_height, feature_width)
    ).bool()
    query_low = F.adaptive_max_pool2d(
        query_mask.float(), (feature_height, feature_width)
    ).bool()
    query_prototype = _masked_prototype(frame_features[0, 0], query_low[0, 0])
    if query_prototype is None:
        raise ValueError("query mask is empty at CoTracker feature resolution")

    radius_cells = max(1, math.ceil(radius_pixels / feature_stride))
    corrected = masks.clone()
    _add(diagnostics, "mask_appearance_frames", masks.shape[1] - 1)
    _add(diagnostics, "mask_appearance_candidate_evaluations", 0)
    _add(diagnostics, "mask_appearance_corrected_frames", 0)
    _add(diagnostics, "mask_appearance_shift_pixels_sum", 0.0)
    _add(diagnostics, "mask_appearance_similarity_gain_sum", 0.0)
    _add(diagnostics, "mask_appearance_similarity_gain_max", 0.0)

    for frame_index in range(1, masks.shape[1]):
        base_mask = masks_low[0, frame_index]
        base_area = int(base_mask.sum().item())
        if base_area == 0:
            _add(diagnostics, "mask_appearance_empty_frames", 1)
            continue
        base_prototype = _masked_prototype(
            frame_features[0, frame_index], base_mask
        )
        if base_prototype is None:
            continue
        base_similarity = float(torch.dot(base_prototype, query_prototype).item())
        candidate_masks = []
        candidate_shifts = []
        candidate_penalties = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx == 0 and dy == 0:
                    continue
                candidate_masks.append(_shift_without_wrap(base_mask, dy, dx))
                candidate_shifts.append((dy, dx))
                candidate_penalties.append(
                    displacement_penalty * math.hypot(dx, dy) / radius_cells
                )

        _add(
            diagnostics,
            "mask_appearance_candidate_evaluations",
            len(candidate_masks),
        )
        candidate_tensor = torch.stack(candidate_masks).float()
        candidate_areas = candidate_tensor.sum(dim=(1, 2))
        feature_sums = torch.einsum(
            "khw,chw->kc",
            candidate_tensor,
            frame_features[0, frame_index].float(),
        )
        candidate_prototypes = F.normalize(feature_sums, dim=1)
        similarities = candidate_prototypes @ query_prototype.float()
        penalties = similarities.new_tensor(candidate_penalties)
        objectives = similarities - penalties
        # Reject translations clipped by the image boundary.
        objectives = objectives.masked_fill(candidate_areas != base_area, -torch.inf)
        best_index = int(objectives.argmax().item())
        best_objective = float(objectives[best_index].item())
        best_similarity = float(similarities[best_index].item())
        best_shift = candidate_shifts[best_index]

        gain = best_objective - base_similarity
        if not math.isfinite(best_objective) or gain < minimum_similarity_gain:
            continue

        dy_pixels = best_shift[0] * feature_stride
        dx_pixels = best_shift[1] * feature_stride
        corrected[0, frame_index] = _shift_without_wrap(
            masks[0, frame_index], dy_pixels, dx_pixels
        )
        _add(diagnostics, "mask_appearance_corrected_frames", 1)
        _add(
            diagnostics,
            "mask_appearance_shift_pixels_sum",
            math.hypot(dx_pixels, dy_pixels),
        )
        raw_gain = best_similarity - base_similarity
        _add(diagnostics, "mask_appearance_similarity_gain_sum", raw_gain)
        _maximum(diagnostics, "mask_appearance_similarity_gain_max", raw_gain)

    return corrected
