[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-training-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-evaluation-results",
    [string[]]$Profiles = @()
)

$ErrorActionPreference = "Stop"
foreach ($Path in @($RepoRoot, $DatasetDir, $TrainingRoot)) {
    if (-not (Test-Path $Path)) { throw "Path not found: $Path" }
}
New-Item -ItemType Directory -Force -Path $EvaluationRoot | Out-Null

function Convert-ToWslPath([string]$WindowsPath) {
    $Converted = & wsl.exe -d Ubuntu -- wslpath -a $WindowsPath
    if ($LASTEXITCODE -ne 0) { throw "wslpath failed: $WindowsPath" }
    return $Converted.Trim()
}

$WslRepo = Convert-ToWslPath $RepoRoot
$Arguments = @(
    "-d", "Ubuntu", "--", "bash",
    "$WslRepo/scripts/evaluate_random_tutor_models.sh",
    $WslRepo,
    (Convert-ToWslPath $DatasetDir),
    (Convert-ToWslPath $TrainingRoot),
    (Convert-ToWslPath $EvaluationRoot)
)
$Arguments += $Profiles
& wsl.exe @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE; rerun to resume completed cases."
}
