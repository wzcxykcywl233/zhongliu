[CmdletBinding()]
param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$DatasetRoot = "C:\zhongliu\trackrad2025-main\dataset",
    [string]$LabeledSource = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$PublicTest = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$Seed = "trackrad-40-10-v1"
)

$ErrorActionPreference = "Stop"
$splitScript = Join-Path $PSScriptRoot "create_trackrad_40_10_split.py"
$auditScript = Join-Path $PSScriptRoot "audit_trackrad_dataset.ps1"
$auditRoot = Join-Path $Repository "dataset-audit-40-10-38"
$training = Join-Path $DatasetRoot "trackrad2025_labeled_train_40"
$validation = Join-Path $DatasetRoot "trackrad2025_labeled_validation_10"
$manifest = Join-Path $DatasetRoot "trackrad-40-10-split.json"

foreach ($path in @($Repository, $DatasetRoot, $LabeledSource, $PublicTest)) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        throw "Required directory does not exist: $path"
    }
}

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    & $python.Source -3 $splitScript `
        --source $LabeledSource `
        --output-root $DatasetRoot `
        --manifest $manifest `
        --seed $Seed
}
else {
    $python = Get-Command python -ErrorAction Stop
    & $python.Source $splitScript `
        --source $LabeledSource `
        --output-root $DatasetRoot `
        --manifest $manifest `
        --seed $Seed
}
if ($LASTEXITCODE -ne 0) {
    throw "40/10 split creation failed with exit code $LASTEXITCODE"
}

$splits = @(
    @{ Name = "train-40"; Path = $training; Cases = 40 },
    @{ Name = "validation-10"; Path = $validation; Cases = 10 },
    @{ Name = "test-38"; Path = $PublicTest; Cases = 38 }
)
foreach ($split in $splits) {
    $output = Join-Path $auditRoot $split.Name
    & $auditScript `
        -DatasetDir $split.Path `
        -OutputDir $output `
        -ExpectedCaseCount $split.Cases
    $summary = Get-Content -LiteralPath (Join-Path $output "dataset-summary.json") -Raw |
        ConvertFrom-Json
    if (-not $summary.StructureAuditPassed) {
        throw "$($split.Name) failed its structural audit: $output"
    }
}

Write-Host ""
Write-Host "TrackRAD 40/10/38 protocol is ready."
Write-Host "Train:      $training"
Write-Host "Validation: $validation"
Write-Host "Test:       $PublicTest"
Write-Host "Manifest:   $manifest"
Write-Host "Audit:      $auditRoot"

