import torch
import numpy as np
import cv2


def convert_points_to_mask(
    points: torch.Tensor,  # (N, 2)
    target_shape: tuple[int, int],  # H, W
    validity: torch.Tensor | None = None,
    morph_close_kernel: int = 1,
    keep_largest_component: bool = False,
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
        # A polygon needs at least three vertices. Falling back keeps failures
        # local to the threshold profile rather than producing an empty mask.
        if filtered.shape[0] >= 3:
            points = filtered

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
        kernel = np.ones((morph_close_kernel, morph_close_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if keep_largest_component and mask.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)

    return torch.from_numpy(mask).bool()


def convert_point_trajectory_to_mask_sequence(
    point_trajectory: torch.Tensor,  # (B, T, N, 2)
    video_shape: tuple[int, int],
    validity: torch.Tensor | None = None,
    morph_close_kernel: int = 1,
    keep_largest_component: bool = False,
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
        )
        for t in range(T)
    ]
    # add batch dim
    return torch.stack(masks, dim=0).unsqueeze(0)
