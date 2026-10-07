[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [ValidateSet('smoke','validation','test')][string]$Stage = 'smoke',
    [string]$TrainingRuntimeImage = 'trackrad-cotracker-backbone-growth:latest'
)
$ErrorActionPreference = 'Stop'
# Run sequentially on one GPU. Each child maintains its independent fingerprint,
# lock, atomic checkpoints, transcript and official split-specific report.
if ($Stage -eq 'smoke') {
    & (Join-Path $RepoRoot 'scripts\run_feature_sidecar_resumable.ps1') -RepoRoot $RepoRoot -Stage smoke -TrainingRuntimeImage $TrainingRuntimeImage
} else {
    & (Join-Path $RepoRoot 'scripts\run_adaptive_anchor_resumable.ps1') -RepoRoot $RepoRoot -Stage $Stage
    & (Join-Path $RepoRoot 'scripts\run_feature_sidecar_resumable.ps1') -RepoRoot $RepoRoot -Stage $Stage -TrainingRuntimeImage $TrainingRuntimeImage
}
