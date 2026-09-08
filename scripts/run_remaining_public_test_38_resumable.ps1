[CmdletBinding()]
param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$TrainDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40",
    [string]$TestDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$InferenceResults = "C:\zhongliu\zhongliu-tuning\public-test-38-results",
    [string]$TeacherTrainingResults = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\random-tutor-hard-label-training",
    [string]$TeacherEvaluationResults = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\random-tutor-hard-label-test-38",
    [int[]]$Seeds = @(0, 1, 2),
    [int]$NumSteps = 2000,
    [int]$SaveEverySteps = 25,
    [switch]$SkipInference,
    [switch]$SkipTeacherTraining,
    [switch]$SkipTeacherEvaluation
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$ProtocolRoot = Join-Path $Repository "protocol-40-10-38"
$QueueLog = Join-Path $ProtocolRoot "remaining-public-test-38-queue.log"
$InferenceManifestPath = Join-Path $Repository `
    "cotracker-algorithm\experiments\public-test-38-remaining-profiles.json"
$TeacherManifestPath = Join-Path $Repository `
    "cotracker-algorithm\experiments\random-tutor-hard-label-test-38.json"

New-Item -ItemType Directory -Force -Path $ProtocolRoot | Out-Null

function Write-QueueMessage {
    param([string]$Message)
    $Timestamp = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:ss.fffzzz")
    $Line = "[$Timestamp] $Message"
    Write-Host $Line
    [System.IO.File]::AppendAllText($QueueLog, $Line + "`n", $Utf8NoBom)
}

function Assert-CaseCount {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$Expected,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Name dataset does not exist: $Path"
    }
    $Actual = @(Get-ChildItem -LiteralPath $Path -Directory -Force).Count
    if ($Actual -ne $Expected) {
        throw "$Name dataset contains $Actual cases; expected $Expected"
    }
}

foreach ($Path in @($InferenceManifestPath, $TeacherManifestPath)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Experiment manifest does not exist: $Path"
    }
}

$InferenceManifest = Get-Content -LiteralPath $InferenceManifestPath -Raw |
    ConvertFrom-Json
$TeacherManifest = Get-Content -LiteralPath $TeacherManifestPath -Raw |
    ConvertFrom-Json

if ([int]$InferenceManifest.expected_cases -ne 38 -or
    [int]$TeacherManifest.evaluation_cases -ne 38) {
    throw "Both experiment manifests must target the 38-case public test set"
}
if ([bool]$TeacherManifest.soft_confidence_labels -or
    $TeacherManifest.confidence_target_mode -ne "hard") {
    throw "Teacher manifest must explicitly disable soft labels and use hard targets"
}
if ((@($TeacherManifest.seeds) -join ',') -ne (@($Seeds) -join ',')) {
    throw "Requested seeds differ from the frozen teacher manifest"
}
if ([int]$TeacherManifest.num_steps -ne $NumSteps -or
    [int]$TeacherManifest.save_every_steps -ne $SaveEverySteps) {
    throw "Requested training schedule differs from the frozen teacher manifest"
}

Assert-CaseCount -Path $TestDataset -Expected 38 -Name "Public test"
if (-not $SkipTeacherTraining) {
    Assert-CaseCount -Path $TrainDataset -Expected 40 -Name "Training"
}

Write-QueueMessage "Remaining experiment queue started or resumed"
Write-QueueMessage "Inference dataset: $TestDataset"
Write-QueueMessage "Teacher training dataset: $TrainDataset"
Write-QueueMessage "Teacher evaluation dataset: $TestDataset"
Write-QueueMessage "Soft confidence labels: disabled; confidence target mode: hard"

if (-not $SkipInference) {
    $Profiles = @($InferenceManifest.profiles | ForEach-Object { $_.name })
    $BaselineMetrics = Join-Path $InferenceResults "baseline\metrics.json"
    if (-not (Test-Path -LiteralPath $BaselineMetrics -PathType Leaf)) {
        $Profiles = @("baseline") + $Profiles
        Write-QueueMessage "Baseline metrics are absent; baseline was prepended"
    }
    Write-QueueMessage "Running supplementary inference profiles: $($Profiles -join ', ')"
    & (Join-Path $PSScriptRoot "run_hierarchical_followup_resumable.ps1") `
        -Repository $Repository `
        -Dataset $TestDataset `
        -Results $InferenceResults `
        -Profiles $Profiles
    & (Join-Path $PSScriptRoot "summarize_public_test_38.ps1") `
        -Results $InferenceResults
    & (Join-Path $PSScriptRoot "audit_public_test_38_baseline_repeat.ps1") `
        -Results $InferenceResults `
        -ExpectedCases 38
    Write-QueueMessage "Supplementary inference profiles completed"
}
else {
    Write-QueueMessage "Supplementary inference queue skipped by request"
}

if (-not $SkipTeacherTraining) {
    New-Item -ItemType Directory -Force -Path $TeacherTrainingResults | Out-Null
    $FrozenManifestPath = Join-Path $TeacherTrainingResults "protocol-manifest.json"
    $FrozenManifest = ConvertTo-Json -InputObject $TeacherManifest -Depth 30
    if (Test-Path -LiteralPath $FrozenManifestPath -PathType Leaf) {
        $ExistingManifest = Get-Content -LiteralPath $FrozenManifestPath -Raw |
            ConvertFrom-Json |
            ConvertTo-Json -Depth 30
        if ($ExistingManifest -ne $FrozenManifest) {
            throw "Teacher protocol differs from the existing resumable run"
        }
    }
    else {
        $TemporaryManifest = "$FrozenManifestPath.tmp"
        [System.IO.File]::WriteAllText(
            $TemporaryManifest,
            $FrozenManifest + "`n",
            $Utf8NoBom
        )
        Move-Item -LiteralPath $TemporaryManifest -Destination $FrozenManifestPath
    }
    Write-QueueMessage "Starting or resuming paired hard-label teacher training"
    & (Join-Path $PSScriptRoot "run_random_tutor_controlled_retest_resumable.ps1") `
        -RepoRoot $Repository `
        -DatasetDir $TrainDataset `
        -ResultsRoot $TeacherTrainingResults `
        -Seeds $Seeds `
        -NumSteps $NumSteps `
        -SaveEverySteps $SaveEverySteps
    Write-QueueMessage "Paired hard-label teacher training completed"
}
else {
    Write-QueueMessage "Teacher training skipped by request"
}

if (-not $SkipTeacherEvaluation) {
    Write-QueueMessage "Starting or resuming teacher-model evaluation on 38 public test cases"
    & (Join-Path $PSScriptRoot "evaluate_random_tutor_controlled_retest.ps1") `
        -RepoRoot $Repository `
        -DatasetDir $TestDataset `
        -TrainingRoot $TeacherTrainingResults `
        -EvaluationRoot $TeacherEvaluationResults `
        -Seeds $Seeds
    & (Join-Path $PSScriptRoot "summarize_random_tutor_controlled_retest.ps1") `
        -EvaluationRoot $TeacherEvaluationResults
    Write-QueueMessage "Teacher-model evaluation and summary completed"
}
else {
    Write-QueueMessage "Teacher-model evaluation skipped by request"
}

Write-QueueMessage "All requested remaining experiments completed"
