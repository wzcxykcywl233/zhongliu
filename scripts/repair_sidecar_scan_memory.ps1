[CmdletBinding()]
param(
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$TrainingResultsRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-modefix-v3',
    [string]$ResultsRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\sidecar-compact-scan-v4',
    [ValidateSet('proof','training')][string]$Stage='proof'
)
$ErrorActionPreference='Stop'
$MappedStage=if ($Stage -eq 'proof') {'profile'} else {'training'}
& (Join-Path $RepoRoot 'scripts\run_sidecar_runtime_diagnostics.ps1') -RepoRoot $RepoRoot `
    -TrainingResultsRoot $TrainingResultsRoot -ResultsRoot $ResultsRoot -Stage $MappedStage `
    -ProbeSteps 3 -CompactScan
