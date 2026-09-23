# CoTracker3 for Tumor Tracking 

### Six independent query-memory refinements

Run `scripts/run_memory_refinement_resumable.ps1` for the diverse-memory control
and six fixed inference-only ablations, on validation-10 then test-38.
The queue freezes source/weight/data hashes and resumes completed cases after interruption.
See [the protocol and commands](cotracker-algorithm/experiments/MEMORY_REFINEMENT.zh-CN.md).

Run from the root of this repository:
```
git submodule init
git submodule update

export TORCH_HOME=${PWD}/cotracker-algorithm/torch
cd cotracker-algorithm
pixi run python download_model.py
```

## Audit the TrackRAD2025 dataset

The PowerShell audit is read-only with respect to the dataset. It checks case
structure, metadata JSON files, MHA headers, frame counts, and image/label
dimensions, then writes CSV and JSON summaries:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\audit_trackrad_dataset.ps1 `
  -DatasetDir "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data" `
  -OutputDir ".\dataset-audit"
```

The generated files are `dataset-summary.json`, `dataset-cases.csv`,
`dataset-mha-files.csv`, and `dataset-issues.csv`.

## Run real-mask-assisted CoTracker training

The ground-truth-mask study keeps the original one-random-teacher-per-batch
training path and adds only a differentiable target-membership loss from the
40-case training masks. It evaluates all weights on validation-10, freezes the
selected weight, and evaluates only that candidate plus its paired control on
test-38. Its second protocol version derives clip and query randomness from the
epoch, sample, and global step, so training and evaluation resume without
changing the paired sequence after interruption.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_ground_truth_mask_supervision_resumable.ps1
```

The design and result locations are documented in
`cotracker-algorithm/experiments/GROUND_TRUTH_MASK_SUPERVISION.zh-CN.md`.

### Download and evaluate the 38-case public test set

The public test release consists of the 8-case preliminary set and the 30-case
final set with privacy-restricted cohort D excluded. On the remote Windows
machine, first download and audit both subsets. The script resumes interrupted
downloads and creates a combined 38-case directory using NTFS hard links, so it
does not duplicate the MHA payloads:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\download_trackrad_public_test_resumable.ps1
```

Then evaluate the original baseline and the eleven pre-registered directions
selected from the completed 50-case ablation table. The frozen queue is stored
in `cotracker-algorithm/experiments/public-test-38-profiles.json`. Per-case
outputs and per-profile metrics are checkpointed, so the same command safely
resumes after interruption:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_public_test_38_resumable.ps1
```

The queue may also be started before the download finishes. It waits for the
combined 38-case audit to pass and then launches the same resumable evaluation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\queue_public_test_38_when_ready.ps1
```

Results are written to `public-test-38-results`, including
`public-test-38-results.csv` and
`public-test-38-deltas-vs-baseline.csv`. This public test set must not be used
for further parameter selection or training.

### Establish the 40/10/38 protocol

Create a deterministic patient-level 40/10 split from the 50 labeled training
cases while keeping the 38 public test cases separate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\prepare_trackrad_40_10_38.ps1
```

The split preserves fixed A/B/C validation quotas (5/3/2) and approximately
stratifies anatomical regions within each cohort. It uses hard links and writes
the complete assignment to `trackrad-40-10-split.json`.

The existing non-trainable inference profiles do not consume the 40-case
training split. Re-evaluate their fixed queue on the 10-case validation split
with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_validation_10_resumable.ps1
```

Future trainable methods such as a Mamba adapter must train only on the 40-case
split, select settings only on the 10-case validation split, and use the
38-case set only for a final locked evaluation.

### Run the conservative and aggressive Mamba experiments

The dedicated queue trains two exploratory Mamba-style variants from the same
frozen CoTracker checkpoint: a conservative trajectory residual adapter and an
aggressive replacement of UpdateFormer temporal-attention blocks. Both use one
fixed offline teacher, hard labels, no auxiliary teacher, and the 40/10/38
protocol. Training is checkpointed every 25 steps and evaluation is committed
per case:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_mamba_40_10_38_resumable.ps1 `
  -NumSteps 1000 `
  -SaveEverySteps 25
```

Rerun the same command after an interruption. The 38-case table and direct
deltas against `hierarchical_full_grid0` are written below
`protocol-40-10-38\mamba-experiments\test-38`. The frozen design and reporting
rules are documented in
`cotracker-algorithm/experiments/MAMBA_EXPERIMENTS.zh-CN.md`.

The second Mamba study keeps CoTracker intact and learns only a full-case
hierarchical/global fusion gate from the 40-case ground-truth masks. It includes
an Oracle upper-bound audit and a per-frame MLP control so that any improvement
can be attributed to temporal modeling:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_long_fusion_mamba_resumable.ps1 `
  -TrainingSteps 500 `
  -SaveEverySteps 25
```

See `cotracker-algorithm/experiments/LONG_FUSION_MAMBA.zh-CN.md` for the frozen
design and output locations.

### Run the parameter-free dynamic query-memory study

This study keeps the CoTracker backbone and the established
`hierarchical_full_grid0_iterations2` configuration fixed. It permanently
retains the first query anchor and compares a latest reliable anchor, a
reliability-ranked Top-K memory, and a reliability-plus-diversity Top-K memory.
No model training, teacher model, or soft labels are used.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_dynamic_query_memory_resumable.ps1
```

The queue audits the 40/10/38 split, evaluates the fixed profiles on validation
10 and test 38, records memory admission/fusion diagnostics, and resumes at the
per-case boundary after interruption. See
`cotracker-algorithm/experiments/DYNAMIC_QUERY_MEMORY.zh-CN.md` for the frozen
design.

### Run mask-level appearance validation

This parameter-free follow-up keeps the validation-selected Top-K query memory
and uses frozen CoTracker frame features to validate each predicted mask against
the first-frame target appearance. It compares conservative four-pixel and
eight-pixel whole-mask translation searches while preserving mask shape and
area:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_mask_appearance_validation_resumable.ps1
```

The 10-case validation and 38-case test runs are checkpointed per case. Design
details and required activation diagnostics are in
`cotracker-algorithm/experiments/MASK_APPEARANCE_VALIDATION.zh-CN.md`.

### Run the supplementary 38-case queue

The remaining registered inference profiles and the controlled auxiliary-
teacher profiles have separate frozen manifests. Soft confidence labels are
disabled. Teacher models train on the 40-case split and are evaluated, but
never trained, on the 38-case public test set. Because the public-test results
have already been inspected, these runs are supplementary comparisons rather
than a new blind model-selection test.

The following command runs or resumes the complete queue. It retains per-step
training checkpoints, per-case prediction checkpoints, and append-only logs:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_remaining_public_test_38_resumable.ps1
```

Use `-SkipInference`, `-SkipTeacherTraining`, or `-SkipTeacherEvaluation` to
resume only a selected stage. The inference results are merged into
`public-test-38-results`; hard-label teacher training and evaluation are stored
under `protocol-40-10-38`.

### Diagnose memory features, trajectories, and masks

Run `scripts/run_memory_chain_diagnostics.ps1` for an observation-only paired
38-case diagnostic queue (control, repeated control, pointwise fusion, current
retrieval). It records sampled pyramid features, full trajectories and masks,
checks repeatability and prior outputs, and resumes per case/profile using
hashed checkpoints. It does not train or access later-frame ground-truth masks.
See [the diagnostic design and remote commands](cotracker-algorithm/experiments/MEMORY_CHAIN_DIAGNOSTICS.zh-CN.md).

### Inherit visibility/confidence across dynamic query anchors

`scripts/run_query_state_inheritance_resumable.ps1` runs six frozen inference
profiles: fixed memory control, V-only, C-only, V+C, time-decayed V+C, and
query-frame-only V+C. It uses original weights, no training, the existing
40/10/38 protocol, case-level resume, frozen inputs and mechanism audits.
Use the `test-38` CSV for performance reporting, not the validation CSV.
See [design, risks and remote commands](cotracker-algorithm/experiments/QUERY_STATE_INHERITANCE.zh-CN.md).

### Per-frame historical-offset backcheck and iteration sweep

`scripts/run_frame_backcheck_resumable.ps1` runs 2/4/6 iterations with scoring
off/on, keeping the diverse memory baseline and original weights. Each target
frame compares center, history-offset and equal-count fixed-neighborhood
evidence without changing predictions. The queue checks output equality,
resumes per case, evaluates validation 10 and test 38 separately, then runs
label-aware offline score analysis. Copy performance from
`test-38/iteration-performance-test-38.csv` (scoring-off groups).
See [the frozen experiment design](cotracker-algorithm/experiments/FRAME_BACKCHECK.zh-CN.md).
# Four-way per-frame backcheck (observation only)

Fixed two-iteration diverse-memory control; compare center, single historical
offset, four mirrored historical offsets, and fixed four-way offsets on identical
supported points. Two inference profiles compute four scores without position,
confidence, or memory feedback. Run `scripts/run_fourway_backcheck_resumable.ps1`;
results are isolated under `protocol-40-10-38/fourway-backcheck`.
See [四向回查实验说明](cotracker-algorithm/experiments/FOURWAY_BACKCHECK.zh-CN.md).
# History-guided rotation alignment (diagnostic only)

Run `scripts/run_rotation_backcheck_resumable.ps1` for fixed two-iteration
control versus history-guided angle estimation and raw-patch re-encoding.
No four-way expansion and no position/state feedback. Validation-10 and test-38
outputs stay separate under `protocol-40-10-38/rotation-backcheck`.
See [旋转对齐诊断说明](cotracker-algorithm/experiments/ROTATION_BACKCHECK.zh-CN.md).

# Best-mainline and VC-decay candidate diagnostics

Run `scripts/diagnose_best_candidates.ps1` to analyze the existing
`query-state-inheritance/test-38` outputs without training or rerunning inference.
It compares official case metrics, time/segment-offset error patterns,
ground-truth centroid-alignment counterfactuals, contour reconstruction losses,
and worst-frame overlays. CPU-only, offline, frozen-runtime, case-resumable;
source predictions are mounted read-only. Ground-truth probes are diagnostics,
not deployable improvements or official performance numbers.
See [主线与候选诊断设计及运行说明](cotracker-algorithm/experiments/BEST_CANDIDATE_DIAGNOSTICS.zh-CN.md).
# 当前最优方法参数复调（40 / 10 / 38）

入口：`scripts/run_parameter_retune_resumable.ps1`。默认仅运行验证搜索：63组首轮、最多20组交互组合、冻结入围复核；之后显式 `-Stage test` 才启动38例正式评估。断电重跑同一命令续接，不进行训练。

详细范围、对照、筛选规则和输出位置见 [参数复调协议](cotracker-algorithm/experiments/PARAMETER_RETUNE.zh-CN.md)。
