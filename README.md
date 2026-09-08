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
