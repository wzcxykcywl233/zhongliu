from datetime import datetime
import logging

import torch
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline

logger = logging.getLogger(__name__)


def setup_model(
    checkpoint: str = "cotracker3_offline",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CoTrackerThreeOffline:
    ts_start = datetime.now()
    if checkpoint == "cotracker3_offline":
        model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline"
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


def forward_pass(
    model: CoTrackerThreeOffline,
    video: torch.Tensor,
    queries: torch.Tensor,
    support_grid_size: int = 10,
    n_iterations: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Performs a single forward pass of provided batched samples through the model.
    Requires TAP data.

    Args:
        model: CoTracker model in eval mode.
        video: Tensor of shape (B, T, D=1, H, W); will be expanded to 3 channels.
        queries: Tensor of shape (B, N, 3) with (t, x, y) per point.
        support_grid_size: If >0, adds a regular grid of support points.
        n_iterations: Number of refinement iterations. CoTracker default is 4.
        device: Compute device.

    Returns:
        Tensor of shape (B, T, N, 2) with (x, y) trajectories per time step.
    """
    B, N, _ = queries.shape
    original_N = N  # store original number of query points

    if support_grid_size > 0:
        # generate support_grid_size ^2 points evenly spaced across the image
        h, w = video.shape[-2:]
        padding_x, padding_y = w / support_grid_size, h / support_grid_size
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(padding_y, h - padding_y, support_grid_size),
            torch.linspace(padding_x, w - padding_x, support_grid_size),
            indexing="ij",
        )

        # Reshape and combine x,y coordinates
        grid_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        grid_pts = grid_pts.unsqueeze(0).expand(B, -1, -1)

        # Add time dimension (zeros) and concatenate with queries
        grid_pts = torch.cat(
            [torch.zeros(B, grid_pts.shape[1], 1), grid_pts],
            dim=2,
        )
        queries = torch.cat([queries, grid_pts], dim=1)

    video = video.to(device)
    queries = queries.to(device)

    out = model(
        video=video.expand(-1, -1, 3, -1, -1),
        queries=queries,
        iters=n_iterations,
        is_train=False,
    )

    # Always use original_N to ensure consistent shapes with ground truth
    pred_trajectory = out[0][:, :, :original_N, :]
    # we ignore visibility here

    # optional: overwrite first-timestep preds with query
    B, T, N, _ = pred_trajectory.shape
    queries_t = queries[..., 0].to(torch.int64)  # Query frame indices

    for b in range(B):
        for n in range(N):
            frame_idx = queries_t[b, n]
            # Overwrite prediction with exact query point
            pred_trajectory[b, frame_idx, n, :2] = queries[b, n, 1:3]

    return pred_trajectory
