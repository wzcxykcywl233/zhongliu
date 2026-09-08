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

Then evaluate the original baseline and all six pre-registered hierarchical
combinations. Per-case outputs and per-profile metrics are checkpointed, so the
same command safely resumes after interruption:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_public_test_38_resumable.ps1
```

Results are written to `public-test-38-results`, including
`public-test-38-results.csv` and
`public-test-38-deltas-vs-baseline.csv`. This public test set must not be used
for further parameter selection or training.
