[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-initfix-v2',
    [ValidateSet('smoke','train','validation','test')][string]$Stage = 'smoke',
    [string]$TrainingRuntimeImage = 'trackrad-cotracker-backbone-growth:latest'
)
$ErrorActionPreference = 'Stop'
# Keep v1 fingerprints, logs and images intact. The corrected source belongs to
# a new study, so no old cache is accepted or migrated by this repair launcher.
$LegacyRoot = [IO.Path]::GetFullPath((Join-Path $RepoRoot 'protocol-40-10-38\feature-sidecar-v1')).TrimEnd('\')
$TargetRoot = [IO.Path]::GetFullPath($ResultsRoot).TrimEnd('\')
if ($TargetRoot -eq $LegacyRoot -or $TargetRoot -eq ($LegacyRoot + '-smoke') -or
    $TargetRoot.StartsWith($LegacyRoot + '\',[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Use a separate new ResultsRoot; v1 results must remain unchanged.'
}
& (Join-Path $RepoRoot 'scripts\run_feature_sidecar_resumable.ps1') `
    -RepoRoot $RepoRoot -ResultsRoot $ResultsRoot -Stage $Stage -TrainingRuntimeImage $TrainingRuntimeImage
