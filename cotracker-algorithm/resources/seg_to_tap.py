import numpy as np
import torch
from scipy.interpolate import splev, splprep
from skimage import measure


def _get_border_points(mask: torch.Tensor) -> torch.Tensor:
    contours = measure.find_contours(mask.numpy())
    if len(contours) == 0:
        return torch.empty(0, 2)

    # use only maximum contour
    contour = max(contours, key=len)

    # skimage returns (row, col), we want to return (x, y)
    return torch.from_numpy(contour)[:, [1, 0]]


def _sample_uniform_points(
    points: torch.Tensor, num_samples: int = 1000
) -> torch.Tensor:
    if len(points) == 0:
        return torch.empty(0, 2)

    # compute spline
    ck, u = splprep(points.cpu().numpy().T, s=0, per=True)

    # sample points uniformly from the spline
    u_new = torch.linspace(0, 1, num_samples)
    points_sampled = torch.tensor(np.array(splev(u_new, ck))).T

    return points_sampled


def convert_mask_to_points(
    seg_mask: torch.Tensor,  # (B, D, H, W)
    n_border_points: int,
    add_t0: bool = True,
) -> torch.Tensor:
    """Processes a single frame's segmentation (with D=1) to extract 3D points and padding mask.

    Args:
        seg_mask: Segmentation tensor for a single frame (B=1, D=1,H, W).
        n_border_points: Number of points to sample.
        add_t0: Whether to add a t=0 dimension to the sampled points. Use for cotracker
            inference.

    Returns:
        Tensor containing sampled boundary point coordinates of shape (B=1,n_border_points, 2). If add_t0 is True, the tensor has shape (B=1, n_border_points, 3), where the second dimension is (0, x, y).
    """
    assert seg_mask.ndim == 4 and seg_mask.shape[0] == 1 and seg_mask.shape[1] == 1
    mask = seg_mask[0, 0]

    # get contour coordinates
    border_pts_tensor = _get_border_points(mask)
    if border_pts_tensor.shape[0] == 0:
        raise ValueError("No boundary detected in segmentation mask")

    # sample points uniformly from the boundary
    boundary_points = _sample_uniform_points(
        border_pts_tensor, num_samples=n_border_points
    )
    assert boundary_points.shape == (n_border_points, 2), (
        f"Expected N * 2 tensor, but got {boundary_points.shape}"
    )

    # add t=0 information to cotracker query
    if add_t0:
        z_filler = torch.zeros((n_border_points, 1))
        boundary_points = torch.cat([z_filler, boundary_points], dim=-1)  # (0, x, y)

    # float32 needed for cotracker inference
    boundary_points = boundary_points.to(torch.float32)

    # add batch dim
    return boundary_points.unsqueeze(0)
