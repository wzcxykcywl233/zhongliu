[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-training-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-evaluation-results",
    [string[]]$Profiles = @(
        "baseline_single_teacher",
        "random_tutor_w0025",
        "random_tutor_w005",
        "random_tutor_w010",
        "random_tutor_w020",
        "same_teacher_control_w010"
    )
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$AlgorithmDir = Join-Path $RepoRoot "cotracker-algorithm"
$EvaluationDir = Join-Path $RepoRoot "evaluation"
$RunnerLog = Join-Path $EvaluationRoot "runner.log"
$LockPath = Join-Path $EvaluationRoot ".runner.lock"

foreach ($Path in @($RepoRoot, $AlgorithmDir, $EvaluationDir, $DatasetDir, $TrainingRoot)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Path not found: $Path" }
}
New-Item -ItemType Directory -Force -Path $EvaluationRoot | Out-Null

function Write-RunMessage {
    param([string]$Message)
    $Timestamp = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:ss.fffzzz")
    $Line = "[$Timestamp] $Message"
    Write-Host $Line
    [System.IO.File]::AppendAllText($RunnerLog, $Line + "`n", $Utf8NoBom)
}

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

function Invoke-DockerLogged {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & docker @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath -Append |
            Tee-Object -FilePath $RunnerLog -Append |
            ForEach-Object { Write-Host $_ }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
    return $ExitCode
}

try {
    $LockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "Another random-tutor evaluator is using $EvaluationRoot"
}

try {
    Write-RunMessage "Native random-tutor evaluation started"
    Write-RunMessage "Profiles: $($Profiles -join ', ')"

    $BuildLog = Join-Path $EvaluationRoot "build.log"
    $Status = Invoke-DockerLogged -LogPath $BuildLog -Arguments @(
        "build", $AlgorithmDir,
        "--platform=linux/amd64",
        "--tag", "trackrad-algorithm-cotracker-algorithm"
    )
    if ($Status -ne 0) { throw "Algorithm image build failed with exit code $Status" }

    $Status = Invoke-DockerLogged -LogPath $BuildLog -Arguments @(
        "build", $EvaluationDir,
        "--platform=linux/amd64",
        "--tag", "trackrad-evaluation"
    )
    if ($Status -ne 0) { throw "Evaluation image build failed with exit code $Status" }

    $CaseFolders = @(Get-ChildItem -LiteralPath $DatasetDir -Directory | Sort-Object Name)
    if ($CaseFolders.Count -eq 0) { throw "No cases found in $DatasetDir" }

    foreach ($Profile in $Profiles) {
        $Checkpoint = Join-Path $TrainingRoot "$Profile\cotracker_three_final.pth"
        if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf) -or
            (Get-Item -LiteralPath $Checkpoint).Length -le 0) {
            throw "Missing completed model: $Checkpoint"
        }

        $ProfileDir = Join-Path $EvaluationRoot $Profile
        $Attempts = Join-Path $ProfileDir ".attempts"
        $Jobs = Join-Path $ProfileDir "jobs"
        $MetricsDir = Join-Path $ProfileDir "evaluation"
        $MetricsPath = Join-Path $MetricsDir "metrics.json"
        New-Item -ItemType Directory -Force -Path $Attempts, $Jobs, $MetricsDir | Out-Null

        Write-RunMessage "===== EVALUATE $Profile ====="
        if (Test-Path -LiteralPath $MetricsPath -PathType Leaf) {
            Write-RunMessage "Metrics checkpoint found; skipping $Profile"
            continue
        }

        foreach ($CaseFolder in $CaseFolders) {
            $CaseId = $CaseFolder.Name
            $CompletedDir = Join-Path $Jobs $CaseId
            $CompletedPrediction = Join-Path $CompletedDir "prediction.json"
            $CompletedOutput = Join-Path $CompletedDir "output\images\mri-linac-series-targets\output.mha"
            $ValidCheckpoint =
                (Test-Path -LiteralPath $CompletedPrediction -PathType Leaf) -and
                (Test-Path -LiteralPath $CompletedOutput -PathType Leaf) -and
                ((Get-Item -LiteralPath $CompletedOutput).Length -gt 0)

            if ($ValidCheckpoint) {
                Write-RunMessage "Checkpoint hit: $Profile/$CaseId"
                continue
            }

            if (Test-Path -LiteralPath $CompletedDir) {
                $Recovered = Join-Path $Attempts "$CaseId-incomplete-$([guid]::NewGuid())"
                Move-Item -LiteralPath $CompletedDir -Destination $Recovered
                Write-RunMessage "Moved incomplete checkpoint: $Recovered"
            }

            $AttemptDir = Join-Path $Attempts "$CaseId-$([guid]::NewGuid())"
            $AttemptOutput = Join-Path $AttemptDir "output"
            $CaseLog = Join-Path $AttemptDir "case.log"
            New-Item -ItemType Directory -Force -Path $AttemptOutput | Out-Null

            Write-RunMessage "Running: $Profile/$CaseId"
            $StartedAt = [DateTimeOffset]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ")
            $Status = Invoke-DockerLogged -LogPath $CaseLog -Arguments @(
                "run", "--rm",
                "--platform=linux/amd64",
                "--network", "none",
                "--gpus", "all",
                "--env", "COTRACKER_EXPERIMENT=baseline",
                "--env", "COTRACKER_CHECKPOINT=/opt/checkpoint/model.pth",
                "--mount", "type=bind,source=$Checkpoint,target=/opt/checkpoint/model.pth,readonly",
                "--mount", "type=bind,source=$($CaseFolder.FullName)\frame-rate.json,target=/input/frame-rate.json,readonly",
                "--mount", "type=bind,source=$($CaseFolder.FullName)\b-field-strength.json,target=/input/b-field-strength.json,readonly",
                "--mount", "type=bind,source=$($CaseFolder.FullName)\scanned-region.json,target=/input/scanned-region.json,readonly",
                "--mount", "type=bind,source=$($CaseFolder.FullName)\images\${CaseId}_frames.mha,target=/input/images/mri-linacs/${CaseId}_frames.mha,readonly",
                "--mount", "type=bind,source=$($CaseFolder.FullName)\targets\${CaseId}_first_label.mha,target=/input/images/mri-linac-target/target.mha,readonly",
                "--mount", "type=bind,source=$AttemptOutput,target=/output",
                "trackrad-algorithm-cotracker-algorithm"
            )
            if ($Status -ne 0) { throw "$Profile/$CaseId failed with exit code $Status" }
            $CompletedAt = [DateTimeOffset]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ")

            $Output = Join-Path $AttemptOutput "images\mri-linac-series-targets\output.mha"
            if (-not (Test-Path -LiteralPath $Output -PathType Leaf) -or
                (Get-Item -LiteralPath $Output).Length -le 0) {
                throw "$Profile/$CaseId did not produce a valid output.mha"
            }

            $Prediction = [ordered]@{
                pk = "jobs/$CaseId"
                inputs = @(
                    @{ value = 8; interface = @{ slug = "frame-rate" } },
                    @{ value = 1.5; interface = @{ slug = "magnetic-field-strength" } },
                    @{ value = "abdomen"; interface = @{ slug = "scanned-region" } },
                    @{ image = @{ name = "mri-linac-target.mha" }; interface = @{ slug = "mri-linac-target"; relative_path = "images/mri-linac-target" } },
                    @{ image = @{ name = "$CaseId.mha" }; interface = @{ slug = "mri-linac-series"; relative_path = "images/mri-linacs" } }
                )
                status = "Succeeded"
                outputs = @(
                    @{ image = @{ name = "output.mha" }; interface = @{ slug = "mri-linac-series-targets"; relative_path = "images/mri-linac-series-targets" } }
                )
                started_at = $StartedAt
                completed_at = $CompletedAt
            }
            Write-AtomicJson -Path (Join-Path $AttemptDir "prediction.json") -Value $Prediction
            [System.IO.File]::WriteAllText(
                (Join-Path $AttemptDir ".complete"),
                $CompletedAt + "`n",
                $Utf8NoBom
            )
            Move-Item -LiteralPath $AttemptDir -Destination $CompletedDir
            Write-RunMessage "Checkpoint committed: $Profile/$CaseId"
        }

        $Predictions = @(
            Get-ChildItem -LiteralPath $Jobs -Directory |
                Sort-Object Name |
                ForEach-Object {
                    Get-Content -LiteralPath (Join-Path $_.FullName "prediction.json") -Raw |
                        ConvertFrom-Json
                }
        )
        if ($Predictions.Count -ne $CaseFolders.Count) {
            throw "$Profile has $($Predictions.Count) predictions; expected $($CaseFolders.Count)"
        }
        Write-AtomicJson -Path (Join-Path $ProfileDir "predictions.json") -Value $Predictions
        Write-RunMessage "Prepared $($Predictions.Count) completed predictions: $Profile"

        $EvaluationLog = Join-Path $ProfileDir "evaluation.log"
        $Status = Invoke-DockerLogged -LogPath $EvaluationLog -Arguments @(
            "run", "--rm",
            "--platform=linux/amd64",
            "--network", "none",
            "--gpus", "all",
            "--mount", "type=bind,source=$ProfileDir,target=/input,readonly",
            "--mount", "type=bind,source=$MetricsDir,target=/output",
            "--mount", "type=bind,source=$DatasetDir,target=/opt/app/ground_truth,readonly",
            "trackrad-evaluation"
        )
        if ($Status -ne 0) { throw "$Profile evaluation failed with exit code $Status" }
        if (-not (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
            throw "$Profile evaluation did not produce metrics.json"
        }
        Write-RunMessage "Metrics committed: $MetricsPath"
    }

    Write-RunMessage "All random-tutor evaluations completed"
}
finally {
    if ($null -ne $LockStream) { $LockStream.Dispose() }
}
