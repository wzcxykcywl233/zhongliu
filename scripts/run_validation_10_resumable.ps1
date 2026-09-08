[CmdletBinding()]
param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\validation-10-results"
)

$ErrorActionPreference = "Stop"
$profileManifest = Join-Path $Repository `
    "cotracker-algorithm\experiments\public-test-38-profiles.json"
if (-not (Test-Path -LiteralPath $profileManifest -PathType Leaf)) {
    throw "Profile manifest does not exist: $profileManifest"
}
$manifest = Get-Content -LiteralPath $profileManifest -Raw | ConvertFrom-Json
$profiles = @($manifest.profiles | ForEach-Object { $_.name })

$caseCount = @(Get-ChildItem -LiteralPath $Dataset -Directory -Force).Count
if ($caseCount -ne 10) {
    throw "Validation dataset contains $caseCount cases; expected 10"
}

& (Join-Path $PSScriptRoot "audit_trackrad_dataset.ps1") `
    -DatasetDir $Dataset `
    -OutputDir (Join-Path $Results "dataset-audit") `
    -ExpectedCaseCount 10

& (Join-Path $PSScriptRoot "run_hierarchical_followup_resumable.ps1") `
    -Repository $Repository `
    -Dataset $Dataset `
    -Results $Results `
    -Profiles $profiles

& (Join-Path $PSScriptRoot "summarize_profile_results.ps1") `
    -Results $Results `
    -Prefix "validation-10"

