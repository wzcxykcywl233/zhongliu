param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\hierarchical-results",
    [string[]]$Profiles = @(
        "hierarchical_d10",
        "hierarchical_d10_original_feat_05",
        "hierarchical_d10_occlusion_merge",
        "hierarchical_d10_dual_anchor",
        "hierarchical_full"
    )
)

$ErrorActionPreference = "Stop"

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$SummaryPath = Join-Path $Results "summary.json"
$ExpectedCases = @(Get-ChildItem -LiteralPath $Dataset -Directory).Count

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )

    $Temporary = "$Path.tmp"
    $Json = ConvertTo-Json -InputObject $Value -Depth 100
    [System.IO.File]::WriteAllText($Temporary, $Json + "`n", $Utf8NoBom)
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

$Summary = [ordered]@{}
if (Test-Path -LiteralPath $SummaryPath -PathType Leaf) {
    $Existing = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
    foreach ($Property in $Existing.PSObject.Properties) {
        $Summary[$Property.Name] = $Property.Value
    }
}

Write-Host "Dataset cases: $ExpectedCases"

foreach ($Profile in $Profiles) {
    $ProfileDir = Join-Path $Results $Profile
    $Checkpoint = Join-Path $ProfileDir "checkpoint"
    $Jobs = Join-Path $Checkpoint "jobs"
    $Evaluation = Join-Path $Checkpoint "evaluation"
    $MetricsPath = Join-Path $ProfileDir "metrics.json"
    $EvaluationMetrics = Join-Path $Evaluation "metrics.json"
    $PredictionsPath = Join-Path $Checkpoint "predictions.json"
    $LogPath = Join-Path $ProfileDir "finalize.log"

    Write-Host "`n===== $Profile ====="

    if (Test-Path -LiteralPath $MetricsPath -PathType Leaf) {
        $Metrics = Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json
        $Summary[$Profile] = $Metrics
        Write-AtomicJson -Path $SummaryPath -Value $Summary
        Write-Host "Completed metrics checkpoint found; skipping."
        continue
    }

    $CaseDirs = @(Get-ChildItem -LiteralPath $Jobs -Directory | Sort-Object Name)
    $Predictions = New-Object System.Collections.Generic.List[object]

    foreach ($CaseDir in $CaseDirs) {
        $Complete = Join-Path $CaseDir.FullName ".complete"
        $Prediction = Join-Path $CaseDir.FullName "prediction.json"
        $Output = Join-Path $CaseDir.FullName "output\images\mri-linac-series-targets\output.mha"

        if (-not (Test-Path -LiteralPath $Complete -PathType Leaf)) {
            throw "$Profile/$($CaseDir.Name) is missing .complete"
        }
        if (-not (Test-Path -LiteralPath $Prediction -PathType Leaf)) {
            throw "$Profile/$($CaseDir.Name) is missing prediction.json"
        }
        if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) {
            throw "$Profile/$($CaseDir.Name) is missing output.mha"
        }
        if ((Get-Item -LiteralPath $Output).Length -le 0) {
            throw "$Profile/$($CaseDir.Name) has an empty output.mha"
        }

        $Predictions.Add((Get-Content -LiteralPath $Prediction -Raw | ConvertFrom-Json))
    }

    if ($Predictions.Count -ne $ExpectedCases) {
        throw "$Profile has $($Predictions.Count) valid cases; expected $ExpectedCases"
    }

    Write-AtomicJson -Path $PredictionsPath -Value $Predictions
    New-Item -ItemType Directory -Force -Path $Evaluation | Out-Null
    Write-Host "Prepared $($Predictions.Count) completed predictions"

    $DockerArguments = @(
        "run", "--rm",
        "--platform=linux/amd64",
        "--network", "none",
        "--gpus", "all",
        "--mount", "type=bind,source=$Checkpoint,target=/input,readonly",
        "--mount", "type=bind,source=$Evaluation,target=/output",
        "--mount", "type=bind,source=$Dataset,target=/opt/app/ground_truth,readonly",
        "trackrad-evaluation"
    )

    & docker @DockerArguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $DockerExitCode = $LASTEXITCODE
    if ($DockerExitCode -ne 0) {
        throw "$Profile evaluation failed with exit code $DockerExitCode"
    }
    if (-not (Test-Path -LiteralPath $EvaluationMetrics -PathType Leaf)) {
        throw "$Profile evaluation did not produce metrics.json"
    }

    $Metrics = Get-Content -LiteralPath $EvaluationMetrics -Raw | ConvertFrom-Json
    Write-AtomicJson -Path $MetricsPath -Value $Metrics
    $Summary[$Profile] = $Metrics
    Write-AtomicJson -Path $SummaryPath -Value $Summary
    Write-Host "Metrics committed: $MetricsPath"
}

Write-Host "`nAll hierarchical metrics are complete: $SummaryPath"
