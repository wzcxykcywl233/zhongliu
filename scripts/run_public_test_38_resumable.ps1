[CmdletBinding()]
param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\public-test-38-results"
)

$ErrorActionPreference = "Stop"

$manifestPath = Join-Path $Repository `
    "cotracker-algorithm\experiments\public-test-38-profiles.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Public-test profile manifest does not exist: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$Profiles = @($manifest.profiles | ForEach-Object { $_.name })
if ($Profiles.Count -eq 0 -or $Profiles[0] -ne "baseline") {
    throw "Public-test profile manifest must start with baseline: $manifestPath"
}
if ([int]$manifest.expected_cases -ne 38) {
    throw "Public-test profile manifest does not specify 38 expected cases"
}

if (-not (Test-Path -LiteralPath $Dataset -PathType Container)) {
    throw "Combined public test dataset does not exist: $Dataset"
}
$caseCount = @(Get-ChildItem -LiteralPath $Dataset -Directory -Force).Count
if ($caseCount -ne [int]$manifest.expected_cases) {
    throw "Public test dataset contains $caseCount cases; expected $($manifest.expected_cases)"
}

$auditScript = Join-Path $PSScriptRoot "audit_trackrad_dataset.ps1"
$auditOutput = Join-Path $Results "dataset-audit"
& $auditScript `
    -DatasetDir $Dataset `
    -OutputDir $auditOutput `
    -ExpectedCaseCount ([int]$manifest.expected_cases)
$audit = Get-Content -LiteralPath (Join-Path $auditOutput "dataset-summary.json") -Raw |
    ConvertFrom-Json
if (-not $audit.StructureAuditPassed) {
    throw "Public test dataset failed structural audit: $auditOutput"
}

& (Join-Path $PSScriptRoot "run_hierarchical_followup_resumable.ps1") `
    -Repository $Repository `
    -Dataset $Dataset `
    -Results $Results `
    -Profiles $Profiles

& (Join-Path $PSScriptRoot "summarize_public_test_38.ps1") `
    -Results $Results
