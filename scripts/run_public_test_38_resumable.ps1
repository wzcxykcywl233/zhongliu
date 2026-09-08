[CmdletBinding()]
param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\public-test-38-results"
)

$ErrorActionPreference = "Stop"

$Profiles = @(
    "baseline",
    "hierarchical_feat05_grid0",
    "hierarchical_feat05_iterations2",
    "hierarchical_feat05_grid0_iterations2",
    "hierarchical_full_grid0",
    "hierarchical_full_iterations2",
    "hierarchical_full_grid0_iterations2"
)

if (-not (Test-Path -LiteralPath $Dataset -PathType Container)) {
    throw "Combined public test dataset does not exist: $Dataset"
}
$caseCount = @(Get-ChildItem -LiteralPath $Dataset -Directory -Force).Count
if ($caseCount -ne 38) {
    throw "Public test dataset contains $caseCount cases; expected 38"
}

$auditScript = Join-Path $PSScriptRoot "audit_trackrad_dataset.ps1"
$auditOutput = Join-Path $Results "dataset-audit"
& $auditScript `
    -DatasetDir $Dataset `
    -OutputDir $auditOutput `
    -ExpectedCaseCount 38
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

