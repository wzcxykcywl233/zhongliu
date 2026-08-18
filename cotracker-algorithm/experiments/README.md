# CoTracker3 single-point tuning for TrackRAD

These profiles isolate one inference-time variable at a time. The default
`baseline` profile reproduces the original submission: 1000 contour points,
a 10x10 support grid, four refinement iterations, no visibility filtering,
and direct polygon rasterization.

Select a profile with `COTRACKER_EXPERIMENT`. Unknown names fail immediately
instead of silently falling back to the baseline.

| Profile | Only changed variable | Motivation |
|---|---|---|
| `points_500` | 500 contour points | Speed/memory sensitivity |
| `points_1500` | 1500 contour points | Denser medical boundary sampling |
| `support_grid_0` | No support grid | Remove non-target context and reduce compute |
| `support_grid_15` | 15x15 support grid | Stronger global motion context |
| `iterations_2` | Two refinements | Low-latency operating point |
| `iterations_6` | Six refinements | Accuracy-oriented refinement |
| `visibility_0_5` | Visibility >= 0.5 | Use the CoTracker3 visibility head |
| `confidence_0_5` | Confidence >= 0.5 | Remove uncertain polygon vertices |
| `temporal_median_3` | Three-frame median | Cine-MRI temporal stability prior |
| `morph_close_3` | 3x3 closing | Repair small mask boundary gaps |
| `largest_component` | Keep largest component | Single-target anatomical prior |
| `lock_first_mask` | Copy annotated first mask | Preserve the exact supplied reference |
| `keyframe_stride_2` | Track every second frame | K-Track-inspired speed/accuracy test |

The visibility/confidence profiles preserve point order and fall back to the
unfiltered polygon if fewer than three vertices survive. The keyframe profile
always evaluates the first and final frames and linearly interpolates point
coordinates, visibility, and confidence at intermediate frames.

Run selected profiles from WSL/Linux:

```bash
cd /path/to/trackrad-cotracker-submission
python cotracker-algorithm/experiments/run_ablation.py \
  --dataset-dir ./dataset/labeled \
  --profiles baseline points_500 visibility_0_5 temporal_median_3
```

Each profile writes a complete log and the parsed TrackRAD `metrics.json` to
`ablation-results/`. Compare DSC (higher), HD95/MASD/center distance (lower),
relative D98 (higher), and runtime. Do not combine profiles until the
single-point table has identified beneficial changes.

Design references:

- CoTracker3: pseudo-labelled real-video training and explicit visibility /
  confidence prediction.
- K-Track: keyframe execution with motion-based recovery of intermediate
  frames; `keyframe_stride_2` is a deliberately simpler first ablation.
- TrackRAD: contour-point CoTracker3 baseline with medical segmentation
  reconstruction and unified challenge metrics.

CoWTracker, PointSt3R, DiT features, and DINOv2 fusion are intentionally not
mixed into this first matrix: each changes the representation or architecture
and needs a separate training protocol rather than a controlled inference
ablation.

## Combination validation after the single-point ablation

The 50-case public-data ablation identified `support_grid_0` as the only
profile that improved all five accuracy metrics while reducing runtime.
`iterations_2` provided a balanced speed gain, and `keyframe_stride_2`
provided the strongest latency reduction.  The following second-stage
profiles combine only those supported changes:

| Profile | Combined settings | Validation goal |
|---|---|---|
| `grid0_iterations2` | No support grid + 2 refinements | Balanced accuracy and speed |
| `grid0_stride2` | No support grid + stride 2 | Fast tracking with improved target context |
| `grid0_iterations2_stride2` | No support grid + 2 refinements + stride 2 | Maximum speed candidate |

Run only the second-stage validation:

```bash
python cotracker-algorithm/experiments/run_ablation.py \
  --dataset-dir ./dataset/trackrad2025_labeled_training_data \
  --profiles grid0_iterations2 grid0_stride2 grid0_iterations2_stride2
```
