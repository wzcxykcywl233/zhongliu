[CmdletBinding()]
param(
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$TrainingResultsRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-modefix-v3',
    [string]$ResultsRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\sidecar-stability-v5\degradation',
    [ValidateSet('validation','test')][string]$Stage='validation'
)
$ErrorActionPreference='Stop'
Write-Host 'Host path repair only: same old images/weights/data/configs; preserve original frozen manifest and legacy attempts. NO TRAINING.'
& (Join-Path $RepoRoot 'scripts\run_sidecar_speed_safety_resumable.ps1') -RepoRoot $RepoRoot `
    -TrainingResultsRoot $TrainingResultsRoot -ResultsRoot $ResultsRoot -Stage $Stage `
    -ReferenceDiagnosisOnly -RepairOutputPaths
