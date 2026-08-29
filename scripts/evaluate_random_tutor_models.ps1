[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-training-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-evaluation-results",
    [string[]]$Profiles = @()
)

$NativeRunner = Join-Path $PSScriptRoot "evaluate_random_tutor_models_native.ps1"
$Arguments = @{
    RepoRoot = $RepoRoot
    DatasetDir = $DatasetDir
    TrainingRoot = $TrainingRoot
    EvaluationRoot = $EvaluationRoot
}
if ($Profiles.Count -gt 0) { $Arguments.Profiles = $Profiles }
& $NativeRunner @Arguments
