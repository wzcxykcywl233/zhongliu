[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\confidence-updateformer-training-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-updateformer-evaluation-results"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
& (Join-Path $PSScriptRoot "evaluate_random_tutor_models_native.ps1") `
    -RepoRoot $RepoRoot `
    -DatasetDir $DatasetDir `
    -TrainingRoot $TrainingRoot `
    -EvaluationRoot $EvaluationRoot `
    -Profiles @(
        "original_baseline",
        "confidence_head_soft_6_18",
        "confidence_updateformer_soft_6_18"
    )

if ($LASTEXITCODE -ne 0) { throw "Confidence-UpdateFormer evaluation failed" }
Write-Host "All confidence-UpdateFormer evaluations completed: $EvaluationRoot"
