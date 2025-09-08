import torch
import numpy as np
import cv2


def convert_points_to_mask(
    points: torch.Tensor,  # (N, 2)
    target_shape: tuple[int, int],  # H, W
) -> torch.Tensor:
    """Convert a set of points to a segmentation mask.
    Uses cv2.fillPoly to convert a set of boundary-aligned points to a mask.

    Args:
        points: Tensor containing boundary point coordinates of shape (N, 2).
        target_shape: Target dimensions of the segmentation mask (H, W).

    Returns:
        Tensor containing the segmentation mask of shape (H, W) with True values at the points.
    """
    # initialize empty mask, cv2 can't handle bool
    mask = np.zeros(target_shape, dtype=np.uint8)

    cv2.fillPoly(
        img=mask,
        pts=[points.cpu().numpy().astype(np.int32)],
        color=1,
    )

    return torch.from_numpy(mask).bool()


def convert_point_trajectory_to_mask_sequence(
    point_trajectory: torch.Tensor,  # (B, T, N, 2)
    video_shape: tuple[int, int],
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

    masks = [
        convert_points_to_mask(point_trajectory[0, t], video_shape) for t in range(T)
    ]
    # add batch dim
    return torch.stack(masks, dim=0).unsqueeze(0)
