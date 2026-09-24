[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\parameter-retune',
    [string]$TrainDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40',
    [string]$ValidationDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$TestDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38',
    [ValidateSet('plan','validation','test')][string]$Stage = 'validation',
    [string]$PlannerImage = 'python:3.11-slim',
    [string]$ReuseResultsRoot = ''
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if ($ReuseResultsRoot) {
    $ReuseResultsRoot = (Resolve-Path -LiteralPath $ReuseResultsRoot).Path.TrimEnd('\')
    $NewRootPath = [IO.Path]::GetFullPath($ResultsRoot).TrimEnd('\')
    if ($NewRootPath -eq $ReuseResultsRoot -or $NewRootPath.StartsWith($ReuseResultsRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $ReuseResultsRoot.StartsWith($NewRootPath+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Reuse requires separate, non-nested old and new result directories.'
    }
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$ResultsRoot = (Resolve-Path -LiteralPath $ResultsRoot).Path
$QueueLock = [IO.File]::Open((Join-Path $ResultsRoot '.retune.lock'),'OpenOrCreate','ReadWrite','None')
$TranscriptStarted = $false
$SourceLock = $null
function Invoke-Plan([string]$PlanStage) {
    $Prior = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker run --rm --pull never --network none --platform linux/amd64 `
            --mount "type=bind,source=$RepoRoot\cotracker-algorithm,target=/app,readonly" `
            --mount "type=bind,source=$ResultsRoot,target=/results" `
            --entrypoint python $PlannerImage /app/experiments/plan_parameter_retune.py `
            --root /results --stage $PlanStage 2>&1 | ForEach-Object { Write-Host $_ }
        $Status = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Prior }
    if ($Status -ne 0) { throw "Planner failed ($Status). No subsequent stage was started." }
}
function Invoke-Stage([string]$Name,[string]$SplitStage) {
    $Reuse = if ($Name -eq 'single') { $ReuseResultsRoot } else { '' }
    $ImageReference = ''
    if ($Name -ne 'single') {
        $ImageReference = Join-Path $ResultsRoot 'single\validation-10\frozen-images.json'
    }
    & (Join-Path $RepoRoot 'scripts\run_memory_refinement_resumable.ps1') `
        -RepoRoot $RepoRoot -TrainDataset $TrainDataset -ValidationDataset $ValidationDataset `
        -TestDataset $TestDataset -ResultsRoot (Join-Path $ResultsRoot $Name) `
        -Stage $SplitStage -ManifestRelative (Join-Path $ResultsRoot "$Name.json") `
        -ResultPrefix "retune-$Name" -ReferenceImagesPath $ImageReference `
        -RetuneReuseRoot $Reuse -ReusePlannerImage $PlannerImage
}
try {
    if ($ReuseResultsRoot) {
        $SourceLock = [IO.File]::Open((Join-Path $ReuseResultsRoot '.retune.lock'),'Open','Read','None')
    }
    Start-Transcript -Path (Join-Path $ResultsRoot 'queue.log') -Append | Out-Null
    $TranscriptStarted = $true
    Invoke-Plan 'single'
    if ($Stage -eq 'plan') {
        Write-Host "Plan ready: $ResultsRoot\single.json; no inference performed."
    } elseif ($Stage -eq 'validation') {
        Invoke-Stage 'single' 'validation'
        Invoke-Plan 'combination'
        Invoke-Stage 'combination' 'validation'
        Invoke-Plan 'final'
        Invoke-Stage 'final' 'validation'
        Invoke-Plan 'verify-final'
        Write-Host 'Validation and confirmation complete. Final shortlist frozen. Run -Stage test explicitly for the 38-case report.'
    } else {
        if (-not (Test-Path -LiteralPath (Join-Path $ResultsRoot 'final.json'))) { throw 'Run validation first.' }
        # Recheck validation evidence. Never select parameters using test metrics.
        Invoke-Plan 'final'
        Invoke-Plan 'verify-final'
        Invoke-Stage 'final' 'test'
        Invoke-Plan 'audit-test'
        Write-Host "38-case final report: $ResultsRoot\final\test-38"
    }
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    $QueueLock.Dispose()
    if ($SourceLock) { $SourceLock.Dispose() }
}
