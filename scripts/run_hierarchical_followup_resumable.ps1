param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\hierarchical-followup-results",
    [string[]]$Profiles = @(
        "hierarchical_d10_original_feat_025",
        "hierarchical_d10_original_feat_075",
        "hierarchical_full_d5",
        "hierarchical_full_d15"
    ),
    [switch]$RequireDiagnostics,
    [switch]$FreezeImages,
    [switch]$UsePinnedImages,
    [string]$ReferenceImagesPath = '',
    [string]$ModelCheckpoint = "",
    [string]$FusionGateCheckpoint = ""
)

$ErrorActionPreference = "Stop"

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$AlgorithmDir = Join-Path $Repository "cotracker-algorithm"
$EvaluationDir = Join-Path $Repository "evaluation"
$Finalizer = Join-Path $Repository "scripts\finalize_hierarchical_results.ps1"
$RunnerLog = Join-Path $Results "runner.log"
$LockPath = Join-Path $Results ".runner.lock"

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
    # Windows PowerShell 5 wraps native stderr as ErrorRecord objects. Docker
    # BuildKit writes ordinary progress to stderr, so temporarily keep those
    # records non-terminating and decide success exclusively from the process
    # exit code.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & docker @Arguments 2>&1 |
            Tee-Object -FilePath $LogPath -Append |
            Tee-Object -FilePath $RunnerLog -Append |
            ForEach-Object { Write-Host $_ }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    return $ExitCode
}

New-Item -ItemType Directory -Force -Path $Results | Out-Null

try {
    $LockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "Another follow-up experiment runner is using $Results"
}

try {
    Write-RunMessage "Follow-up experiments started"
    Write-RunMessage "Profiles: $($Profiles -join ', ')"
    Write-RunMessage "Dataset: $Dataset"
    Write-RunMessage "Results: $Results"
    if (-not [string]::IsNullOrWhiteSpace($ModelCheckpoint)) {
        if (-not (Test-Path -LiteralPath $ModelCheckpoint -PathType Leaf) -or
            (Get-Item -LiteralPath $ModelCheckpoint).Length -le 0) {
            throw "Model checkpoint is missing or empty: $ModelCheckpoint"
        }
        $ModelCheckpoint = (Resolve-Path -LiteralPath $ModelCheckpoint).Path
        Write-RunMessage "Model checkpoint: $ModelCheckpoint"
    }
    if (-not [string]::IsNullOrWhiteSpace($FusionGateCheckpoint)) {
        if (-not (Test-Path -LiteralPath $FusionGateCheckpoint -PathType Leaf) -or
            (Get-Item -LiteralPath $FusionGateCheckpoint).Length -le 0) {
            throw "Fusion gate checkpoint is missing or empty: $FusionGateCheckpoint"
        }
        $FusionGateCheckpoint = (Resolve-Path -LiteralPath $FusionGateCheckpoint).Path
        Write-RunMessage "Fusion gate checkpoint: $FusionGateCheckpoint"
    }

    $IdentityPath = Join-Path $Results 'frozen-images.json'
    $PinnedIdentity = $null
    if ($UsePinnedImages -and $FreezeImages) {
        if (Test-Path -LiteralPath $IdentityPath -PathType Leaf) {
            $PinnedIdentity = Get-Content -LiteralPath $IdentityPath -Raw | ConvertFrom-Json
            Write-RunMessage "Using this stage's frozen image IDs; skipping Docker builds"
        } elseif ($ReferenceImagesPath) {
            $PinnedIdentity = Get-Content -LiteralPath $ReferenceImagesPath -Raw | ConvertFrom-Json
            Write-RunMessage "Using first stage's frozen image IDs; skipping Docker builds"
        }
    }
    if ($null -eq $PinnedIdentity) {
        $BuildLog = Join-Path $Results "build.log"
        $Status = Invoke-DockerLogged -LogPath $BuildLog -Arguments @(
            "build", $AlgorithmDir,
            "--platform=linux/amd64",
            "--tag", "trackrad-algorithm-cotracker-algorithm"
        )
        if ($Status -ne 0) {
            throw "Algorithm image build failed with exit code $Status"
        }

        $Status = Invoke-DockerLogged -LogPath $BuildLog -Arguments @(
            "build", $EvaluationDir,
            "--platform=linux/amd64",
            "--tag", "trackrad-evaluation"
        )
        if ($Status -ne 0) {
            throw "Evaluation image build failed with exit code $Status"
        }
    }

    if ($FreezeImages) {
        $ImageIdentity = [ordered]@{}
        foreach ($Image in @('trackrad-algorithm-cotracker-algorithm', 'trackrad-evaluation')) {
            $ExpectedId = if ($null -ne $PinnedIdentity) { [string]$PinnedIdentity.$Image } else { '' }
            if ($null -ne $PinnedIdentity -and $ExpectedId -notmatch '^sha256:[0-9a-f]{64}$') {
                throw "Invalid frozen image ID: $Image"
            }
            $InspectTarget = if ($ExpectedId) { $ExpectedId } else { $Image }
            $ImageId = & docker image inspect --format '{{.Id}}' $InspectTarget
            if ($LASTEXITCODE -ne 0) { throw "Frozen image unavailable: $InspectTarget. Keep existing results; do not substitute a tag." }
            if ($ExpectedId -and [string]$ImageId -ne $ExpectedId) { throw "Frozen image identity mismatch: $Image" }
            $ImageIdentity[$Image] = [string]$ImageId
        }
        if ($ReferenceImagesPath) {
            $ReferenceImages = Get-Content -LiteralPath $ReferenceImagesPath -Raw | ConvertFrom-Json
            foreach ($Image in $ImageIdentity.Keys) {
                if ($ReferenceImages.$Image -ne $ImageIdentity[$Image]) {
                    throw "Image differs from the first search stage: $Image. Do not mix runtimes."
                }
            }
        }
        if (Test-Path -LiteralPath $IdentityPath) {
            $PreviousImages = Get-Content -LiteralPath $IdentityPath -Raw | ConvertFrom-Json
            foreach ($Image in $ImageIdentity.Keys) {
                if ($PreviousImages.$Image -ne $ImageIdentity[$Image]) {
                    throw "Docker image changed since checkpoint: $Image. Use a new results directory."
                }
            }
        } else {
            Write-AtomicJson -Path $IdentityPath -Value $ImageIdentity
        }
    }
    $AlgorithmImage = if ($UsePinnedImages -and $FreezeImages) { $ImageIdentity['trackrad-algorithm-cotracker-algorithm'] } else { 'trackrad-algorithm-cotracker-algorithm' }
    $EvaluationImage = if ($UsePinnedImages -and $FreezeImages) { $ImageIdentity['trackrad-evaluation'] } else { 'trackrad-evaluation' }

    $Failures = New-Object System.Collections.Generic.List[string]
    $CaseFolders = @(Get-ChildItem -LiteralPath $Dataset -Directory | Sort-Object Name)

    foreach ($Profile in $Profiles) {
        try {
            $ProfileDir = Join-Path $Results $Profile
            $Checkpoint = Join-Path $ProfileDir "checkpoint"
            $Attempts = Join-Path $Checkpoint ".attempts"
            $Jobs = Join-Path $Checkpoint "jobs"
            $MetricsPath = Join-Path $ProfileDir "metrics.json"
            New-Item -ItemType Directory -Force -Path $Attempts, $Jobs | Out-Null

            Write-RunMessage "===== $Profile ====="
            if (Test-Path -LiteralPath $MetricsPath -PathType Leaf) {
                Write-RunMessage "$Profile metrics checkpoint found; skipping profile"
                continue
            }

            foreach ($CaseFolder in $CaseFolders) {
                $CaseId = $CaseFolder.Name
                $CompletedDir = Join-Path $Jobs $CaseId
                $CompletedMarker = Join-Path $CompletedDir ".complete"
                $CompletedPrediction = Join-Path $CompletedDir "prediction.json"
                $CompletedOutput = Join-Path $CompletedDir "output\images\mri-linac-series-targets\output.mha"
                $CompletedDiagnostics = Join-Path $CompletedDir "diagnostics.json"

                $ValidCheckpoint =
                    (Test-Path -LiteralPath $CompletedMarker -PathType Leaf) -and
                    (Test-Path -LiteralPath $CompletedPrediction -PathType Leaf) -and
                    (Test-Path -LiteralPath $CompletedOutput -PathType Leaf) -and
                    ((Get-Item -LiteralPath $CompletedOutput).Length -gt 0) -and
                    ((-not $RequireDiagnostics) -or
                        (Test-Path -LiteralPath $CompletedDiagnostics -PathType Leaf))

                if ($ValidCheckpoint) {
                    Write-RunMessage "Checkpoint hit: $Profile/$CaseId"
                    continue
                }

                if (Test-Path -LiteralPath $CompletedDir) {
                    # Keep transient paths below the legacy Windows MAX_PATH
                    # boundary.  Long profile names plus a full GUID can exceed
                    # 260 characters even though Docker successfully writes the
                    # bind-mounted output.
                    $RecoveryId = ([guid]::NewGuid().ToString("N")).Substring(0, 8)
                    $Recovered = Join-Path $Attempts "$CaseId-incomplete-$RecoveryId"
                    Move-Item -LiteralPath $CompletedDir -Destination $Recovered
                    Write-RunMessage "Moved incomplete checkpoint: $Recovered"
                }

                $AttemptId = ([guid]::NewGuid().ToString("N")).Substring(0, 8)
                $AttemptDir = Join-Path $Attempts "$CaseId-$AttemptId"
                $AttemptOutput = Join-Path $AttemptDir "output"
                $CaseLog = Join-Path $AttemptDir "case.log"
                New-Item -ItemType Directory -Force -Path $AttemptOutput | Out-Null

                Write-RunMessage "Running: $Profile/$CaseId"
                $StartedAt = [DateTimeOffset]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ")
                $DockerArguments = @(
                    "run", "--rm",
                    "--platform=linux/amd64",
                    "--network", "none",
                    "--gpus", "all",
                    "--env", "COTRACKER_EXPERIMENT=$Profile",
                    "--mount", "type=bind,source=$($CaseFolder.FullName)\frame-rate.json,target=/input/frame-rate.json,readonly",
                    "--mount", "type=bind,source=$($CaseFolder.FullName)\b-field-strength.json,target=/input/b-field-strength.json,readonly",
                    "--mount", "type=bind,source=$($CaseFolder.FullName)\scanned-region.json,target=/input/scanned-region.json,readonly",
                    "--mount", "type=bind,source=$($CaseFolder.FullName)\images\${CaseId}_frames.mha,target=/input/images/mri-linacs/${CaseId}_frames.mha,readonly",
                    "--mount", "type=bind,source=$($CaseFolder.FullName)\targets\${CaseId}_first_label.mha,target=/input/images/mri-linac-target/target.mha,readonly",
                    "--mount", "type=bind,source=$AttemptOutput,target=/output",
                    $AlgorithmImage
                )
                if (-not [string]::IsNullOrWhiteSpace($ModelCheckpoint)) {
                    $ImageIndex = $DockerArguments.Count - 1
                    $DockerArguments = @(
                        $DockerArguments[0..($ImageIndex - 1)] +
                        @(
                            "--env", "COTRACKER_CHECKPOINT=/opt/checkpoint/model.pth",
                            "--mount", "type=bind,source=$ModelCheckpoint,target=/opt/checkpoint/model.pth,readonly"
                        ) +
                        $DockerArguments[$ImageIndex]
                    )
                }
                if (-not [string]::IsNullOrWhiteSpace($FusionGateCheckpoint)) {
                    $ImageIndex = $DockerArguments.Count - 1
                    $DockerArguments = @(
                        $DockerArguments[0..($ImageIndex - 1)] +
                        @(
                            "--env", "COTRACKER_FUSION_GATE_CHECKPOINT=/opt/fusion-gate/gate.pth",
                            "--mount", "type=bind,source=$FusionGateCheckpoint,target=/opt/fusion-gate/gate.pth,readonly"
                        ) +
                        $DockerArguments[$ImageIndex]
                    )
                }
                if ($RequireDiagnostics) {
                    $ImageIndex = $DockerArguments.Count - 1
                    $DockerArguments = @(
                        $DockerArguments[0..($ImageIndex - 1)] +
                        @("--env", "TRACKRAD_DIAGNOSTICS=1") +
                        $DockerArguments[$ImageIndex]
                    )
                }
                $Status = Invoke-DockerLogged `
                    -LogPath $CaseLog `
                    -Arguments $DockerArguments
                if ($Status -ne 0) {
                    throw "$Profile/$CaseId failed with exit code $Status"
                }
                $CompletedAt = [DateTimeOffset]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ")

                $Output = Join-Path $AttemptOutput "images\mri-linac-series-targets\output.mha"
                if (-not (Test-Path -LiteralPath $Output -PathType Leaf) -or
                    (Get-Item -LiteralPath $Output).Length -le 0) {
                    throw "$Profile/$CaseId did not produce a valid output.mha"
                }

                if ($RequireDiagnostics) {
                    $Prefix = "TRACKRAD_DIAGNOSTICS_JSON="
                    $DiagnosticLines = @(
                        Get-Content -LiteralPath $CaseLog |
                            Where-Object { $_.StartsWith($Prefix) }
                    )
                    if ($DiagnosticLines.Count -ne 1) {
                        throw "$Profile/$CaseId produced $($DiagnosticLines.Count) diagnostic records; expected 1"
                    }
                    try {
                        $Diagnostic = $DiagnosticLines[0].Substring($Prefix.Length) |
                            ConvertFrom-Json
                    }
                    catch {
                        throw "$Profile/$CaseId produced invalid diagnostic JSON: $($_.Exception.Message)"
                    }
                    if ($Diagnostic.profile -ne $Profile) {
                        throw "$Profile/$CaseId reported profile '$($Diagnostic.profile)'"
                    }
                    if ($Profile -like "*_gate") {
                        $GateFrames = $Diagnostic.mechanism.long_fusion_gate_frames
                        if ($null -eq $GateFrames -or [int]$GateFrames -le 0) {
                            throw "$Profile/$CaseId did not report an active long-fusion gate"
                        }
                    }
                    if ($Profile -like "*_memory_*" -or $Profile -like "memory_*") {
                        $MemoryFusions = $Diagnostic.mechanism.query_memory_fusions
                        $MemoryWrites = $Diagnostic.mechanism.query_memory_writes_accepted
                        if ($null -eq $MemoryFusions -or [int]$MemoryFusions -le 0) {
                            throw "$Profile/$CaseId did not report active query-memory fusion"
                        }
                        if ($null -eq $MemoryWrites -or [int]$MemoryWrites -le 0) {
                            throw "$Profile/$CaseId did not report an accepted query-memory anchor"
                        }
                    }
                    if ($Profile -like "*_mask_appearance_*") {
                        $AppearanceFrames = $Diagnostic.mechanism.mask_appearance_frames
                        $AppearanceCandidates = $Diagnostic.mechanism.mask_appearance_candidate_evaluations
                        if ($null -eq $AppearanceFrames -or [int]$AppearanceFrames -le 0) {
                            throw "$Profile/$CaseId did not report mask appearance frames"
                        }
                        if ($null -eq $AppearanceCandidates -or [int]$AppearanceCandidates -le 0) {
                            throw "$Profile/$CaseId did not activate mask appearance candidates"
                        }
                    }
                    $OutputHash = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash.ToLowerInvariant()
                    $Diagnostic.prediction |
                        Add-Member -NotePropertyName output_file_sha256 `
                            -NotePropertyValue $OutputHash -Force
                    Write-AtomicJson `
                        -Path (Join-Path $AttemptDir "diagnostics.json") `
                        -Value $Diagnostic
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

            & $Finalizer -Repository $Repository -Dataset $Dataset -Results $Results -Profiles @($Profile) -EvaluationImage $EvaluationImage
            Write-RunMessage "Metrics committed: $Profile"
        }
        catch {
            $Failures.Add($Profile)
            Write-RunMessage "FAILED ${Profile}: $($_.Exception.Message)"
        }
    }

    if ($Failures.Count -gt 0) {
        throw "Failed profiles: $($Failures -join ', ')"
    }
    Write-RunMessage "All follow-up experiments completed"
}
finally {
    if ($null -ne $LockStream) {
        $LockStream.Dispose()
    }
}
