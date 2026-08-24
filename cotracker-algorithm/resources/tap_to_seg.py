import torch
import numpy as np
import cv2


def convert_points_to_mask(
    points: torch.Tensor,  # (N, 2)
    target_shape: tuple[int, int],  # H, W
    validity: torch.Tensor | None = None,
    morph_close_kernel: int = 1,
    keep_largest_component: bool = False,
    diagnostics: dict[str, int | float] | None = None,
) -> torch.Tensor:
    """Convert a set of points to a segmentation mask.
    Uses cv2.fillPoly to convert a set of boundary-aligned points to a mask.

    Args:
        points: Tensor containing boundary point coordinates of shape (N, 2).
        target_shape: Target dimensions of the segmentation mask (H, W).

    Returns:
        Tensor containing the segmentation mask of shape (H, W) with True values at the points.
    """
    if validity is not None:
        if validity.ndim != 1 or validity.shape[0] != points.shape[0]:
            raise ValueError("validity must have shape (N,)")
        filtered = points[validity]
        if diagnostics is not None:
            diagnostics["validity_total_points"] = (
                diagnostics.get("validity_total_points", 0) + points.shape[0]
            )
            diagnostics["validity_rejected_points"] = (
                diagnostics.get("validity_rejected_points", 0)
                + points.shape[0]
                - filtered.shape[0]
            )
        # A polygon needs at least three vertices. Falling back keeps failures
        # local to the threshold profile rather than producing an empty mask.
        if filtered.shape[0] >= 3:
            points = filtered
        elif diagnostics is not None:
            diagnostics["validity_fallback_frames"] = (
                diagnostics.get("validity_fallback_frames", 0) + 1
            )

    if morph_close_kernel < 1 or morph_close_kernel % 2 == 0:
        raise ValueError("morph_close_kernel must be a positive odd integer")

    # initialize empty mask, cv2 can't handle bool
    mask = np.zeros(target_shape, dtype=np.uint8)

    cv2.fillPoly(
        img=mask,
        pts=[points.cpu().numpy().astype(np.int32)],
        color=1,
    )

    if morph_close_kernel > 1:
        before = mask.copy()
        kernel = np.ones((morph_close_kernel, morph_close_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if diagnostics is not None:
            changed = int(np.count_nonzero(before != mask))
            diagnostics["morph_close_changed_pixels"] = (
                diagnostics.get("morph_close_changed_pixels", 0) + changed
            )
            diagnostics["morph_close_changed_frames"] = (
                diagnostics.get("morph_close_changed_frames", 0)
                + int(changed > 0)
            )

    if keep_largest_component and mask.any():
        before = mask.copy()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if diagnostics is not None:
            diagnostics["component_total_foreground_components"] = (
                diagnostics.get("component_total_foreground_components", 0)
                + count
                - 1
            )
            diagnostics["component_multicomponent_frames"] = (
                diagnostics.get("component_multicomponent_frames", 0)
                + int(count > 2)
            )
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)
        if diagnostics is not None:
            changed = int(np.count_nonzero(before != mask))
            diagnostics["largest_component_changed_pixels"] = (
                diagnostics.get("largest_component_changed_pixels", 0) + changed
            )
            diagnostics["largest_component_changed_frames"] = (
                diagnostics.get("largest_component_changed_frames", 0)
                + int(changed > 0)
            )

    return torch.from_numpy(mask).bool()


def convert_point_trajectory_to_mask_sequence(
    point_trajectory: torch.Tensor,  # (B, T, N, 2)
    video_shape: tuple[int, int],
    validity: torch.Tensor | None = None,
    morph_close_kernel: int = 1,
    keep_largest_component: bool = False,
    diagnostics: dict[str, int | float] | None = None,
) -> torch.Tensor:
    """Convert a series of points (point trajectories) to a sequence of segmentation masks.
    Converts points at each time step to a mask and stacks results.

    Args:
        point_trajectory: Tensor containing boundary point coordinates of shape (B, T, N, 2).
        video_shape: Target dimensions of the segmentation mask (H, W).

    Returns:
        Tensor containing the segmentation mask of shape (B, T, H, W) with True values at the points.
    """
    # for each time step, convert the set of points to a mask
    B, T, N, _ = point_trajectory.shape
    assert B == 1
    if validity is not None and validity.shape != (B, T, N):
        raise ValueError(f"validity must have shape {(B, T, N)}")

    masks = [
        convert_points_to_mask(
            point_trajectory[0, t],
            video_shape,
            validity=None if validity is None else validity[0, t],
            morph_close_kernel=morph_close_kernel,
            keep_largest_component=keep_largest_component,
            diagnostics=diagnostics,
        )
        for t in range(T)
    ]
    # add batch dim
    return torch.stack(masks, dim=0).unsqueeze(0)
