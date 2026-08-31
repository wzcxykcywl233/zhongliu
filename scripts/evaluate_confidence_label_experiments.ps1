[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-training-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-evaluation-results"
)

$ErrorActionPreference = "Stop"
$Profiles = @(
    "confidence_hard_12",
    "confidence_soft_8_16",
    "confidence_soft_6_18"
)
$Evaluator = Join-Path $PSScriptRoot "evaluate_random_tutor_models_native.ps1"
& $Evaluator `
    -RepoRoot $RepoRoot `
    -DatasetDir $DatasetDir `
    -TrainingRoot $TrainingRoot `
    -EvaluationRoot $EvaluationRoot `
    -Profiles $Profiles

if ($LASTEXITCODE -ne 0) { throw "Confidence-label evaluation failed" }
Write-Host "All confidence-label evaluations completed: $EvaluationRoot"
