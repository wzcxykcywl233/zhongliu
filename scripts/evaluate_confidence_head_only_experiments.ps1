[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\confidence-head-only-training-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-head-only-evaluation-results"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
$Evaluator = Join-Path $PSScriptRoot "evaluate_random_tutor_models_native.ps1"
& $Evaluator `
    -RepoRoot $RepoRoot `
    -DatasetDir $DatasetDir `
    -TrainingRoot $TrainingRoot `
    -EvaluationRoot $EvaluationRoot `
    -Profiles @(
        "original_baseline",
        "confidence_head_hard_12",
        "confidence_head_soft_6_18"
    )

if ($LASTEXITCODE -ne 0) { throw "Confidence-head-only evaluation failed" }
Write-Host "All confidence-head-only evaluations completed: $EvaluationRoot"
