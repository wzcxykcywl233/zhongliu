[CmdletBinding()]
param(
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$TrainingResultsRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-modefix-v3',
    [string]$ResultsRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\sidecar-stability-v5',
    [ValidateSet('proof','stability','validation','test')][string]$Stage='proof'
)
$ErrorActionPreference='Stop'
if ($Stage -in @('validation','test')) {
    # Old 1000-step weights only; completely independent of diagnostic weights.
    & (Join-Path $RepoRoot 'scripts\run_sidecar_speed_safety_resumable.ps1') -RepoRoot $RepoRoot `
        -TrainingResultsRoot $TrainingResultsRoot -ResultsRoot (Join-Path $ResultsRoot 'degradation') `
        -Stage $Stage -ReferenceDiagnosisOnly
} else {
    $Mapped=if ($Stage -eq 'proof') {'profile'} else {'training'}
    & (Join-Path $RepoRoot 'scripts\run_sidecar_runtime_diagnostics.ps1') -RepoRoot $RepoRoot `
        -TrainingResultsRoot $TrainingResultsRoot -ResultsRoot (Join-Path $ResultsRoot 'runtime') `
        -Stage $Mapped -CompactScan -CompactSoak -ProbeSteps 3 -TrainingTimeoutSeconds 1800
}
