[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$SeedZeroEvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-evaluation-results",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-controlled-retest-training",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-controlled-retest-evaluation"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

$Profiles = @("confidence_hard_12", "confidence_soft_6_18")
foreach ($Profile in $Profiles) {
    $SeedZeroMetrics = Join-Path $SeedZeroEvaluationRoot "$Profile\evaluation\metrics.json"
    if (-not (Test-Path -LiteralPath $SeedZeroMetrics -PathType Leaf)) {
        throw "Missing completed seed 0 metrics: $SeedZeroMetrics"
    }
}

$Evaluator = Join-Path $PSScriptRoot "evaluate_random_tutor_models_native.ps1"
foreach ($Seed in 1, 2) {
    $SeedTrainingRoot = Join-Path $TrainingRoot "seed_$Seed"
    $SeedEvaluationRoot = Join-Path $EvaluationRoot "seed_$Seed"
    Write-Host "===== EVALUATE CONFIDENCE RETEST seed=$Seed ====="
    & $Evaluator `
        -RepoRoot $RepoRoot `
        -DatasetDir $DatasetDir `
        -TrainingRoot $SeedTrainingRoot `
        -EvaluationRoot $SeedEvaluationRoot `
        -Profiles $Profiles
}

Write-Host "All additional confidence-label evaluations completed: $EvaluationRoot"
Write-Host "Seed 0 reused from: $SeedZeroEvaluationRoot"
