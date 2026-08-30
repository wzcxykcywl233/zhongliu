[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-controlled-retest-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-controlled-retest-evaluation",
    [int[]]$Seeds = @(0, 1, 2)
)

$ErrorActionPreference = "Stop"
$Evaluator = Join-Path $RepoRoot "scripts\evaluate_random_tutor_models_native.ps1"
$Profiles = @(
    "baseline_single_teacher",
    "random_tutor_w010",
    "random_tutor_w015",
    "random_tutor_w020",
    "random_tutor_w025",
    "same_teacher_control_w020"
)

foreach ($Seed in $Seeds) {
    Write-Host "===== EVALUATE CONTROLLED RETEST seed=$Seed ====="
    & $Evaluator `
        -RepoRoot $RepoRoot `
        -DatasetDir $DatasetDir `
        -TrainingRoot (Join-Path $TrainingRoot "seed_$Seed") `
        -EvaluationRoot (Join-Path $EvaluationRoot "seed_$Seed") `
        -Profiles $Profiles
    if ($LASTEXITCODE -ne 0) {
        throw "Controlled retest evaluation seed $Seed failed with exit code $LASTEXITCODE"
    }
}

Write-Host "All controlled random-tutor evaluations completed: $EvaluationRoot"
