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

## Hierarchical re-anchoring matrix

These profiles run full CoTracker3 inference inside ten-frame matching levels
and use each reliable endpoint as the next query anchor:

| Profile | Added mechanism |
|---|---|
| `hierarchical_d10` | Position-only re-anchoring every 10 frames |
| `hierarchical_d10_original_feat_05` | 0.5 immutable original-feature memory |
| `hierarchical_d10_occlusion_merge` | Merge a persistently occluded level with its neighbors |
| `hierarchical_d10_dual_anchor` | Reliability-normalized global/local position fusion |
| `hierarchical_full` | Feature memory + occlusion merge + dual anchor |

The resumable launcher commits each case atomically and appends live logs. Run
it under `nohup`; launching the same command again after an interruption skips
completed cases and profiles:

```bash
nohup bash scripts/run_hierarchical_experiments.sh \
  /path/to/trackrad2025_labeled_training_data \
  /path/to/hierarchical-results > hierarchical-launcher.log 2>&1 &
tail -f /path/to/hierarchical-results/supervisor.log
```

This matrix uses fixed pretrained weights. Its checkpoint unit is a completed
case, not an optimizer step.

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

## Combining hierarchical feature memory with the best base settings

The full audit identified `grid0_iterations2` as the best balanced base
configuration, while `hierarchical_d10_original_feat_05` achieved the highest
DSC.  These six profiles keep the fixed CoTracker3 weights,
`hierarchical_span=10`, `original_feature_weight=0.5`, and
`temporal_stride=1`, then test support-grid removal and two refinements both
separately and together:

| Profile | Grid | Iterations | Occlusion merge | Dual anchor | Goal |
|---|---:|---:|---|---:|---|
| `hierarchical_feat05_grid0` | 0 | 4 | no | 0 | Feature memory plus no grid |
| `hierarchical_feat05_iterations2` | 10 | 2 | no | 0 | Feature memory plus two refinements |
| `hierarchical_feat05_grid0_iterations2` | 0 | 2 | no | 0 | Lightweight combined candidate |
| `hierarchical_full_grid0` | 0 | 4 | yes | 0.5 | Full hierarchy plus no grid |
| `hierarchical_full_iterations2` | 10 | 2 | yes | 0.5 | Full hierarchy plus two refinements |
| `hierarchical_full_grid0_iterations2` | 0 | 2 | yes | 0.5 | Fully combined candidate |

Run all six with per-case crash recovery and diagnostics:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  "C:\zhongliu\zhongliu-tuning\scripts\run_hierarchical_combinations_resumable.ps1"
```

Results are written to `hierarchical-combination-results`.  Re-running the
same command skips atomically committed cases.  Stride 2, feature gating, and
neighborhood revalidation stay disabled so the combined effects remain
attributable.
