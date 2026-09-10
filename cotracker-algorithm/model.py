"""
Edit this file to implement your algorithm.

The file must contain a function called `run_algorithm` that takes two arguments:
- `frames` (numpy.ndarray): A 3D numpy array of shape (W, H, T) containing the MRI linac series.
- `target` (numpy.ndarray): A 3D numpy array of shape (W, H, 1) containing the MRI linac target (single-channel mask).

Returns:
- numpy.ndarray: A 3D numpy array of shape (W, H, T) containing the predicted target masks per frame.
"""

from dataclasses import asdict
import hashlib
import json
import logging
import os
import numpy as np
import torch

import resources
from experiments import get_experiment_config
from tuning import build_validity_mask, temporal_median_smooth

logger = logging.getLogger(__name__)

DIAGNOSTIC_PREFIX = "TRACKRAD_DIAGNOSTICS_JSON="


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
    diagnostics_enabled = os.environ.get("TRACKRAD_DIAGNOSTICS", "0") == "1"
    diagnostics: dict[str, object] | None = None
    numeric_diagnostics: dict[str, int | float] | None = None
    if diagnostics_enabled:
        config_dict = asdict(config)
        config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        numeric_diagnostics = {}
        diagnostics = {
            "schema_version": 1,
            "profile": experiment_name,
            "config": config_dict,
            "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            "mechanism": numeric_diagnostics,
        }
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
    model = resources.setup_model(
        mamba_replace_time_attention=config.mamba_replace_time_attention,
        mamba_refiner=config.mamba_refiner,
    )

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
        if numeric_diagnostics is not None:
            numeric_diagnostics["query_points"] = int(queries.shape[1])
            numeric_diagnostics["input_frames"] = int(T)
            numeric_diagnostics["support_grid_size"] = config.support_grid_size
            numeric_diagnostics["n_iterations"] = config.n_iterations
            numeric_diagnostics["temporal_stride"] = config.temporal_stride
            numeric_diagnostics["mamba_refiner_enabled"] = int(
                config.mamba_refiner
            )
            numeric_diagnostics["mamba_time_replacement_enabled"] = int(
                config.mamba_replace_time_attention
            )

        if config.hierarchical_span > 0:
            tracking_output = resources.hierarchical_forward_pass(
                model=model,
                video=video,
                queries=queries,
                span=config.hierarchical_span,
                support_grid_size=config.support_grid_size,
                n_iterations=config.n_iterations,
                original_feature_weight=config.original_feature_weight,
                occlusion_merge=config.occlusion_merge,
                occlusion_visibility_threshold=(
                    config.occlusion_visibility_threshold
                ),
                occlusion_point_fraction=config.occlusion_point_fraction,
                dual_anchor_weight=config.dual_anchor_weight,
                feature_gate_distance=config.feature_gate_distance,
                feature_similarity_threshold=config.feature_similarity_threshold,
                feature_revalidate_radius=config.feature_revalidate_radius,
                diagnostics=numeric_diagnostics,
                return_long_fusion_context=config.long_fusion_gate != "none",
            )
            if config.long_fusion_gate == "none":
                tracking = tracking_output
            else:
                gate_checkpoint = os.environ.get("COTRACKER_FUSION_GATE_CHECKPOINT")
                if not gate_checkpoint:
                    raise ValueError(
                        "long fusion gate profile requires COTRACKER_FUSION_GATE_CHECKPOINT"
                    )
                gate_features = resources.build_long_fusion_features(
                    tracking_output.hierarchical,
                    tracking_output.global_result,
                    tracking_output.hierarchical_similarity,
                    tracking_output.global_similarity,
                    COTRACKER_SHAPE,
                )
                # ``video`` remains the original CPU tensor because the tracking
                # helper transfers its own inputs.  The returned trajectories and
                # derived gate features live on the actual inference device.
                gate_device = gate_features.device
                gate, feature_mean, feature_std = resources.load_long_fusion_gate(
                    gate_checkpoint,
                    config.long_fusion_gate,
                    gate_device,
                )
                gate_logits = gate((gate_features - feature_mean) / feature_std)
                tracking, gate_weights = resources.apply_long_fusion_gate(
                    tracking_output.hierarchical,
                    tracking_output.global_result,
                    gate_logits,
                )
                if numeric_diagnostics is not None:
                    numeric_diagnostics["long_fusion_gate_frames"] = int(
                        gate_weights.numel()
                    )
                    numeric_diagnostics["long_fusion_global_weight_mean"] = float(
                        gate_weights.mean().item()
                    )
                    numeric_diagnostics["long_fusion_global_weight_min"] = float(
                        gate_weights.min().item()
                    )
                    numeric_diagnostics["long_fusion_global_weight_max"] = float(
                        gate_weights.max().item()
                    )
        else:
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
        if numeric_diagnostics is not None:
            smoothing_distance = torch.linalg.vector_norm(
                trajectories - tracking.trajectories,
                dim=-1,
            )
            changed = smoothing_distance > 1e-6
            numeric_diagnostics["temporal_median_changed_point_frames"] = int(
                changed.sum().item()
            )
            numeric_diagnostics["temporal_median_distance_sum"] = float(
                smoothing_distance[changed].sum().item()
            )
            numeric_diagnostics["temporal_median_distance_max"] = (
                float(smoothing_distance[changed].max().item())
                if bool(changed.any().item())
                else 0.0
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
        if numeric_diagnostics is not None:
            numeric_diagnostics["visibility_min"] = float(
                tracking.visibility.min().item()
            )
            numeric_diagnostics["visibility_mean"] = float(
                tracking.visibility.mean().item()
            )
            numeric_diagnostics["confidence_min"] = float(
                tracking.confidence.min().item()
            )
            numeric_diagnostics["confidence_mean"] = float(
                tracking.confidence.mean().item()
            )
            if config.visibility_threshold is not None:
                numeric_diagnostics["visibility_below_threshold"] = int(
                    (tracking.visibility < config.visibility_threshold).sum().item()
                )
            if config.confidence_threshold is not None:
                numeric_diagnostics["confidence_below_threshold"] = int(
                    (tracking.confidence < config.confidence_threshold).sum().item()
                )

        prediction = resources.convert_point_trajectory_to_mask_sequence(
            trajectories,
            video_shape=COTRACKER_SHAPE,
            validity=validity,
            morph_close_kernel=config.morph_close_kernel,
            keep_largest_component=config.keep_largest_component,
            diagnostics=numeric_diagnostics,
        )  # should now be B, T, C, H, W
        logger.info("after TAP->SEG conversion:")
        logger.info(f"\tprediction.shape={prediction.shape}")

        # convert back to original shape
        prediction = resources.reshape_video(prediction, target_shape=ORIGINAL_SHAPE)
        assert prediction.shape == (1, T, H, W), f"{prediction.shape} != {(1, T, H, W)}"

        if config.lock_first_mask:
            if numeric_diagnostics is not None:
                numeric_diagnostics["lock_first_mask_changed_pixels"] = int(
                    (
                        prediction[:, 0]
                        != (target_tensor[:, 0] > 0.5)
                    ).sum().item()
                )
            prediction[:, 0] = target_tensor[:, 0] > 0.5

        # remove batch
        prediction = prediction[0]

        # bring into output np format
        prediction = prediction.cpu().numpy().transpose(2, 1, 0)  # W, H, T

        if diagnostics is not None and numeric_diagnostics is not None:
            diagnostics["prediction"] = {
                "shape": list(prediction.shape),
                "foreground_voxels": int(np.count_nonzero(prediction)),
                "array_sha256": hashlib.sha256(
                    np.ascontiguousarray(prediction).tobytes()
                ).hexdigest(),
            }
            print(
                DIAGNOSTIC_PREFIX
                + json.dumps(diagnostics, sort_keys=True, separators=(",", ":")),
                flush=True,
            )

    return prediction
