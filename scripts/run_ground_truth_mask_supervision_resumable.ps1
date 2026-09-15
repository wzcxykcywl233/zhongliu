[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$TrainDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40",
    [string]$ValidationDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10",
    [string]$TestDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\gt-mask",
    [int]$NumSteps = 1000
)

$ErrorActionPreference = "Stop"
$Profiles = @("control", "mask_w0025", "mask_w005", "mask_w010")
$InferenceProfile = "hierarchical_full_grid0_iterations2_memory_topk"
$TrainingRoot = Join-Path $ResultsRoot "train"
$Runner = Join-Path $RepoRoot "scripts\run_hierarchical_followup_resumable.ps1"
$TrainingScript = Join-Path $RepoRoot "scripts\run_ground_truth_mask_training_resumable.ps1"
$AuditScript = Join-Path $RepoRoot "scripts\audit_ground_truth_mask_training.ps1"
$Summarizer = Join-Path $RepoRoot "scripts\summarize_ground_truth_mask_supervision.ps1"
$QueueLog = Join-Path $ResultsRoot "queue.log"

foreach ($Expected in @(
    @{ Path = $TrainDataset; Count = 40 },
    @{ Path = $ValidationDataset; Count = 10 },
    @{ Path = $TestDataset; Count = 38 }
)) {
    if (-not (Test-Path -LiteralPath $Expected.Path -PathType Container)) {
        throw "Dataset is missing: $($Expected.Path)"
    }
    $Count = @(Get-ChildItem -LiteralPath $Expected.Path -Directory).Count
    if ($Count -ne $Expected.Count) {
        throw "Expected $($Expected.Count) cases, found $Count in $($Expected.Path)"
    }
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$TranscriptStarted = $false
try {
    Start-Transcript -Path $QueueLog -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Ground-truth-mask queue started"

    & $TrainingScript `
        -RepoRoot $RepoRoot `
        -TrainDataset $TrainDataset `
        -ResultsRoot $TrainingRoot `
        -NumSteps $NumSteps
    & $AuditScript -TrainingRoot $TrainingRoot -ExpectedSteps $NumSteps

    foreach ($Profile in $Profiles) {
        $Checkpoint = Join-Path $TrainingRoot "$Profile\cotracker_three_final.pth"
        $ProfileResults = Join-Path $ResultsRoot "val\$Profile"
        & $Runner `
            -Repository $RepoRoot `
            -Dataset $ValidationDataset `
            -Results $ProfileResults `
            -Profiles @($InferenceProfile) `
            -ModelCheckpoint $Checkpoint `
            -RequireDiagnostics
    }
    & $Summarizer `
        -ResultsRoot $ResultsRoot `
        -Split "validation-10" `
        -Profiles $Profiles `
        -InferenceProfile $InferenceProfile

    $Selection = Get-Content -LiteralPath (Join-Path $ResultsRoot "selection.json") -Raw |
        ConvertFrom-Json
    $TestProfiles = @("control", [string]$Selection.selected_profile) |
        Select-Object -Unique
    foreach ($Profile in $TestProfiles) {
        $Checkpoint = Join-Path $TrainingRoot "$Profile\cotracker_three_final.pth"
        $ProfileResults = Join-Path $ResultsRoot "test\$Profile"
        & $Runner `
            -Repository $RepoRoot `
            -Dataset $TestDataset `
            -Results $ProfileResults `
            -Profiles @($InferenceProfile) `
            -ModelCheckpoint $Checkpoint `
            -RequireDiagnostics
    }
    & $Summarizer `
        -ResultsRoot $ResultsRoot `
        -Split "test-38" `
        -Profiles $TestProfiles `
        -InferenceProfile $InferenceProfile
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Ground-truth-mask queue completed"
}
catch {
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] FAILED: $($_.Exception.Message)"
    throw
}
finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
