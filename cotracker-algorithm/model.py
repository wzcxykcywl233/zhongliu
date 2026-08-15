"""
Edit this file to implement your algorithm.

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 3D numpy array of shape (W, H, 1) containing the MRI linac target (single-channel mask).

Returns:
- numpy.ndarray: A 3D numpy array of shape (W, H, T) containing the predicted target masks per frame.
"""

import logging
import numpy as np
import torch

import resources
from experiments import get_experiment_config
from tuning import build_validity_mask, temporal_median_smooth

logger = logging.getLogger(__name__)


def run_algorithm(
    frames: np.ndarray,
    target: np.ndarray,
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:
    """
    Implement your algorithm here.

    Args:
    - frames (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
    - target (numpy.ndarray): A 2D numpy array of shape (W, H, 1) containing the MRI linac target.
    - frame_rate (float): The frame rate of the MRI linac series.
    - magnetic_field_strength (float): The magnetic field strength of the MRI linac series.
    - scanned_region (str): The scanned region of the MRI linac series.
    """

    # Step 1: Input format conversion ------------------------------------------------
    experiment_name, config = get_experiment_config()
    logger.info("CoTracker experiment: %s (%s)", experiment_name, config)

    W, H, T = frames.shape
    ORIGINAL_SHAPE = (H, W)  # Store original dimensions for coordinate rescaling
    COTRACKER_SHAPE = (384, 512)

    # Convert frames from (W, H, T) to (B=1, T, D=1, H, W)
    logger.info("input data:")
    logger.info(f"\tframes.shape={frames.shape}")
    logger.info(f"\ttarget.shape={target.shape}")
    frames_tensor = torch.from_numpy(frames).float()
    frames_tensor = frames_tensor.permute(2, 1, 0)  # (T, H, W)
    frames_tensor = frames_tensor.unsqueeze(0).unsqueeze(2)  # (B=1, T, D=1, H, W)

    # Convert target from (W, H, 1) to (B=1, D=1, H, W)
    target_tensor = torch.from_numpy(target).float()
    target_tensor = target_tensor.permute(1, 0, 2)  # (H, W, 1)
    target_tensor = target_tensor.permute(2, 0, 1)  # (1, H, W)
    target_tensor = target_tensor.unsqueeze(0)  # (B=1, D=1, H, W)

    logger.info("after parsing:")
    logger.info(f"\tframes_tensor.shape={frames_tensor.shape}")
    logger.info(f"\ttarget_tensor.shape={target_tensor.shape}")

    # Step 2: Model loading -----------------------------------------------------------
    model = resources.setup_model()

    # Step 3: Create BatchData object -------------------------------------------------
    video = frames_tensor
    query_mask = target_tensor

    # Step 4: Preprocessing -----------------------------------------------------------
    with torch.no_grad():
        # Store original data for later restoration
        video = resources.reshape_video(video, target_shape=COTRACKER_SHAPE)
        query = resources.reshape_video(query_mask, target_shape=COTRACKER_SHAPE)

        logger.info("after reshaping:")
        logger.info(f"\tvideo.shape={video.shape}")
        logger.info(f"\tquery.shape={query.shape}")

        assert video.shape[0] == 1 and video.shape[2:] == (1,) + COTRACKER_SHAPE, (
            f"Expected video shape (B=1, D=1, H, W) with (H,W)={COTRACKER_SHAPE}, got {tuple(video.shape)}"
        )
        assert query.shape == (1, 1) + COTRACKER_SHAPE, (
            f"Expected query shape (1, 1, H, W) with (H,W)={COTRACKER_SHAPE}, got {tuple(query.shape)}"
        )

        # extract points from query mask
        queries = resources.convert_mask_to_points(
            seg_mask=query,
            n_border_points=config.border_points,
        )

        tracking = resources.forward_pass(
            model=model,
            video=video,
            queries=queries,
            support_grid_size=config.support_grid_size,
            n_iterations=config.n_iterations,
            temporal_stride=config.temporal_stride,
        )
        logger.info("forward pass output:")
        logger.info(f"\tprediction.shape={tracking.trajectories.shape}")

        trajectories = temporal_median_smooth(
            tracking.trajectories,
            config.temporal_median_window,
        )
        # Every TrackRAD query is made at t=0. Preserve the exact annotation
        # after optional temporal filtering.
        trajectories[:, 0] = queries[:, :, 1:3].to(trajectories.device)

        validity = build_validity_mask(
            tracking.visibility,
            tracking.confidence,
            config.visibility_threshold,
            config.confidence_threshold,
        )

        prediction = resources.convert_point_trajectory_to_mask_sequence(
            trajectories,
            video_shape=COTRACKER_SHAPE,
            validity=validity,
            morph_close_kernel=config.morph_close_kernel,
            keep_largest_component=config.keep_largest_component,
        )  # should now be B, T, C, H, W
        logger.info("after TAP->SEG conversion:")
        logger.info(f"\tprediction.shape={prediction.shape}")

        # convert back to original shape
        prediction = resources.reshape_video(prediction, target_shape=ORIGINAL_SHAPE)
        assert prediction.shape == (1, T, H, W), f"{prediction.shape} != {(1, T, H, W)}"

        if config.lock_first_mask:
            prediction[:, 0] = target_tensor[:, 0] > 0.5

        # remove batch
        prediction = prediction[0]

        # bring into output np format
        prediction = prediction.cpu().numpy().transpose(2, 1, 0)  # W, H, T

    return prediction
