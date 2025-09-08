import matplotlib.pyplot as plt
from data.batch import BatchData, BatchPrediction

import numpy as np


def visualize_batchdata(
    batch: BatchData,
    predictions: BatchPrediction | None = None,
    z_slice: int = 0,
    max_points: int | None = None,
    max_t: int | None = None,
    batch_index: int = 0,
    mask_alpha: float = 0.5,
    line_alpha: float = 0.2,
) -> plt.Figure:
    """Generates a matplotlib figure visualizing BatchData and optional BatchPrediction.

    Displays video frames at a specific z-slice with overlaid trajectories and segmentations.

    Args:
        batch: The BatchData object containing ground truth information.
        predictions: Optional BatchPrediction object with model outputs.
        z_slice: The depth slice index (z-coordinate) to visualize.
        max_points: Maximum number of trajectory points to display. If None, all points are shown.
        max_t: Maximum number of time steps (frames) to display. If None, all frames are shown.
        batch_index: The index within the batch to visualize.

    Returns:
        A matplotlib Figure object.
    """
    if batch.video.shape[0] > 1:
        print(
            f"Warning: Batch size is {batch.video.shape[0]}. Visualizing index {batch_index}."
        )
    elif batch.video.shape[0] == 0:
        raise ValueError("BatchData contains no samples.")

    b = batch_index

    video = batch.video[b]  # (T, D, H, W)
    T_orig, D, H, W = video.shape
    T = min(T_orig, max_t) if max_t is not None else T_orig

    # Ground Truth
    gt_traj = (
        batch.trajectory[b] if batch.trajectory is not None else None
    )  # (T, N, 3) (x,y,z)
    gt_seg = (
        batch.segmentation[b] if batch.segmentation is not None else None
    )  # (T, D, H, W)
    query_pts = (
        batch.query_points[b] if batch.query_points is not None else None
    )  # (N, 4) (t, x, y, z)
    query_seg = (
        batch.query_segmentation[b] if batch.query_segmentation is not None else None
    )  # (D, H, W)
    # Predictions
    pred_traj = (
        predictions.pred_trajectory[b]
        if predictions and predictions.pred_trajectory is not None
        else None
    )
    pred_seg = (
        predictions.pred_segmentation[b]
        if predictions and predictions.pred_segmentation is not None
        else None
    )

    # determine number of points to plot
    if gt_traj is not None:
        N = gt_traj.shape[1]
    elif pred_traj is not None:
        N = pred_traj.shape[1]
    elif query_pts is not None:
        N = query_pts.shape[0]
    else:
        N = 0

    if N == 0:
        print("Warning: No trajectory or query points found to visualize.")

    # select subset of points to plot
    point_indices = np.arange(N)
    if max_points is not None and N > max_points:
        point_indices = np.random.choice(N, max_points, replace=False)
        N = max_points  # Update N for color mapping

    colors = plt.cm.rainbow(np.linspace(0, 1, N))
    point_idx_to_color = {idx: colors[i] for i, idx in enumerate(point_indices)}

    n_cols = min(5, T)
    n_rows = (T + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False
    )
    axes_flat = axes.flatten()

    # hide unused subplots
    for i in range(T, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for t in range(T):
        ax = axes_flat[t]
        ax.set_title(f"Frame {t}, Z={z_slice}")
        ax.axis("off")

        # plot video frame -------------------------------------------------------------
        ax.imshow(video[t, z_slice].cpu().numpy(), cmap="gray", origin="lower")

        # plot segmentations -----------------------------------------------------------
        if gt_seg is not None and t < gt_seg.shape[0]:
            gt_mask_slice = gt_seg[t, z_slice].cpu().numpy()
            gt_masked = np.ma.masked_where(gt_mask_slice == 0, gt_mask_slice)
            ax.imshow(
                gt_masked,
                cmap="Blues",
                alpha=mask_alpha,
                origin="lower",
                vmin=0,
                vmax=1,
                label="GT Seg" if t == 0 else "",
            )

        if pred_seg is not None and t < pred_seg.shape[0]:
            pred_mask_slice = pred_seg[t, z_slice].cpu().numpy()
            pred_masked = np.ma.masked_where(pred_mask_slice == 0, pred_mask_slice)
            ax.imshow(
                pred_masked,
                cmap="Oranges",
                alpha=mask_alpha,
                origin="lower",
                vmin=0,
                vmax=1,
                label="Pred Seg" if t == 0 else "",
            )

        if query_seg is not None and t == 0:  # Query seg only at t=0
            query_mask_slice = query_seg[z_slice].cpu().numpy()
            query_masked = np.ma.masked_where(query_mask_slice == 0, query_mask_slice)
            ax.imshow(
                query_masked,
                cmap="Greens",
                alpha=mask_alpha,
                origin="lower",
                vmin=0,
                vmax=1,
            )

        plotted_query_indices = set()

        for n_idx_orig in point_indices:  # Iterate using original indices
            color = point_idx_to_color[n_idx_orig]

            # Plot true trajectory points ----------------------------------------------
            if (
                gt_traj is not None
                and t < gt_traj.shape[0]
                and n_idx_orig < gt_traj.shape[1]
            ):
                point_xyz = gt_traj[t, n_idx_orig].cpu().numpy()
                if point_xyz[2] == z_slice:
                    ax.plot(
                        point_xyz[0],
                        point_xyz[1],
                        marker="x",
                        color=color,
                        markersize=1,
                        linestyle="None",
                        label="True Trajectory"
                        if n_idx_orig == point_indices[0]
                        else "",
                    )
                    # Plot line from previous GT point
                    if t > 0 and gt_traj is not None:
                        prev_point_xyz = gt_traj[t - 1, n_idx_orig].cpu().numpy()
                        if prev_point_xyz[2] == z_slice:
                            ax.plot(
                                [prev_point_xyz[0], point_xyz[0]],
                                [prev_point_xyz[1], point_xyz[1]],
                                "-",
                                color=color,
                                linewidth=1,
                                alpha=line_alpha,
                            )

            # Plot predicted trajectory points -----------------------------------------
            if (
                pred_traj is not None
                and t < pred_traj.shape[0]
                and n_idx_orig < pred_traj.shape[1]
            ):
                point_xyz = pred_traj[t, n_idx_orig].cpu().numpy()
                if point_xyz[2] == z_slice:
                    ax.plot(
                        point_xyz[0],
                        point_xyz[1],
                        marker="+",
                        color=color,
                        markersize=1,
                        linestyle="None",
                        label="Predicted Trajectory"
                        if n_idx_orig == point_indices[0]
                        else "",
                    )
                    # Plot line from previous Pred point
                    if t > 0 and pred_traj is not None:
                        prev_point_xyz = pred_traj[t - 1, n_idx_orig].cpu().numpy()
                        if prev_point_xyz[2] == z_slice:
                            ax.plot(
                                [prev_point_xyz[0], point_xyz[0]],
                                [prev_point_xyz[1], point_xyz[1]],
                                "--",
                                color=color,
                                linewidth=1,
                                alpha=line_alpha,
                            )  # Dashed line for prediction

            # Plot query points --------------------------------------------------------
            if query_pts is not None and n_idx_orig < query_pts.shape[0]:
                qpt = query_pts[n_idx_orig].cpu().numpy()  # (t, x, y, z)
                q_t, q_x, q_y, q_z = qpt

                if (
                    int(q_t) == t
                    and int(q_z) == z_slice
                    and n_idx_orig not in plotted_query_indices
                ):
                    ax.plot(
                        q_x,
                        q_y,
                        marker="x",
                        color="magenta",
                        markersize=2,
                        linestyle="None",
                        label="Query Pt" if not plotted_query_indices else "",
                    )
                    plotted_query_indices.add(n_idx_orig)

    # Add a single legend for the entire figure
    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )  # Adjust layout to prevent overlap with legend/title
    fig.suptitle(f"Sample: {batch.identifier[b]}", fontsize=14)

    return fig
