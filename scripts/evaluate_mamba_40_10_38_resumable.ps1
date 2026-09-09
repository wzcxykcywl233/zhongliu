[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$ValidationDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10",
    [string]$TestDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\mamba-training",
    [string]$ProtocolRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\mamba-experiments"
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $RepoRoot "scripts\run_hierarchical_followup_resumable.ps1"
$Summarizer = Join-Path $RepoRoot "scripts\summarize_profile_results.ps1"
$Profiles = @(
    "baseline",
    "hierarchical_full_grid0",
    "hierarchical_full_grid0_mamba",
    "hierarchical_full_grid0_mamba_replacement"
)
$Checkpoints = @{
    hierarchical_full_grid0_mamba = Join-Path $TrainingRoot "hierarchical_full_grid0_mamba\cotracker_three_final.pth"
    hierarchical_full_grid0_mamba_replacement = Join-Path $TrainingRoot "hierarchical_full_grid0_mamba_replacement\cotracker_three_final.pth"
}
if (@(Get-ChildItem -LiteralPath $ValidationDataset -Directory -Force).Count -ne 10) {
    throw "Expected 10 validation cases: $ValidationDataset"
}
if (@(Get-ChildItem -LiteralPath $TestDataset -Directory -Force).Count -ne 38) {
    throw "Expected 38 public-test cases: $TestDataset"
}
foreach ($Checkpoint in $Checkpoints.Values) {
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Missing trained Mamba checkpoint: $Checkpoint"
    }
}

foreach ($Split in @(
    @{ Name = "validation-10"; Dataset = $ValidationDataset; Prefix = "mamba-validation-10" },
    @{ Name = "test-38"; Dataset = $TestDataset; Prefix = "mamba-test-38" }
)) {
    $Results = Join-Path $ProtocolRoot $Split.Name
    New-Item -ItemType Directory -Force -Path $Results | Out-Null
    foreach ($Profile in $Profiles) {
        $Arguments = @{
            Repository = $RepoRoot
            Dataset = $Split.Dataset
            Results = $Results
            Profiles = @($Profile)
            RequireDiagnostics = $true
        }
        if ($Checkpoints.ContainsKey($Profile)) {
            $Arguments.ModelCheckpoint = $Checkpoints[$Profile]
        }
        & $Runner @Arguments
    }
    & $Summarizer -Results $Results -Prefix $Split.Prefix
    & (Join-Path $RepoRoot "scripts\summarize_mamba_results.ps1") `
        -Results $Results `
        -Prefix $Split.Prefix
}

Write-Host "Mamba validation and 38-case test evaluation completed: $ProtocolRoot"
