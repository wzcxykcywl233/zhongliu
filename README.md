# CoTracker3 for Tumor Tracking 

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
