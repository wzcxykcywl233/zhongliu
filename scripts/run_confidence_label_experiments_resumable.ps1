[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-training-results",
    [int]$NumSteps = 1000,
    [int]$SaveEverySteps = 25,
    [ValidateRange(0, 16)]
    [int]$DataLoaderWorkers = 0,
    [string]$DockerShmSize = "4g",
    [int]$TrainingSeed = 0,
    [int]$TeacherSeed = 20260831,
    [switch]$ContinueOnFailure,
    [string[]]$Profiles = @(
        "confidence_hard_12",
        "confidence_soft_8_16",
        "confidence_soft_6_18"
    )
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

if ($NumSteps -lt 1) { throw "NumSteps must be positive" }
if ($SaveEverySteps -lt 1) { throw "SaveEverySteps must be positive" }
if (-not (Test-Path -LiteralPath $DatasetDir -PathType Container)) {
    throw "Dataset not found: $DatasetDir"
}

$Definitions = @{
    confidence_hard_12 = [ordered]@{
        target_mode = "hard"
        hard_threshold = 12.0
        inner_radius = 8.0
        outer_radius = 16.0
    }
    confidence_soft_8_16 = [ordered]@{
        target_mode = "linear_soft"
        hard_threshold = 12.0
        inner_radius = 8.0
        outer_radius = 16.0
    }
    confidence_soft_6_18 = [ordered]@{
        target_mode = "linear_soft"
        hard_threshold = 12.0
        inner_radius = 6.0
        outer_radius = 18.0
    }
}

foreach ($Profile in $Profiles) {
    if (-not $Definitions.ContainsKey($Profile)) {
        throw "Unknown confidence-label profile: $Profile"
    }
}

$CheckpointDir = Join-Path $RepoRoot "training-checkpoints"
$RequiredCheckpoints = @(
    "scaled_offline.pth",
    "baseline_online.pth",
    "baseline_offline.pth",
    "cotracker2v1.pth"
)
foreach ($Name in $RequiredCheckpoints) {
    $Path = Join-Path $CheckpointDir $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Path. Run scripts\prepare_random_tutor_checkpoints.ps1 first."
    }
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$LockPath = Join-Path $ResultsRoot ".runner.lock"
try {
    $LockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "Another confidence-label runner is using $ResultsRoot"
}

function Write-AtomicText {
    param([string]$Path, [string]$Text)
    $Temporary = "$Path.tmp"
    [System.IO.File]::WriteAllText($Temporary, $Text, $Utf8NoBom)
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
            ForEach-Object { Write-Host $_ }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
    return $ExitCode
}

try {
    $SelectedProfiles = @(
        foreach ($Profile in $Profiles) {
            [ordered]@{ name = $Profile; target = $Definitions[$Profile] }
        }
    )
    $CheckpointHashes = @(
        foreach ($Name in $RequiredCheckpoints) {
            $Path = Join-Path $CheckpointDir $Name
            [ordered]@{
                name = $Name
                sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
            }
        }
    )
    $Design = [ordered]@{
        schema = 1
        objective = "Compare hard and piecewise-linear confidence targets"
        dataset = (Resolve-Path -LiteralPath $DatasetDir).Path
        num_steps = $NumSteps
        save_every_steps = $SaveEverySteps
        training_seed = $TrainingSeed
        teacher_seed = $TeacherSeed
        auxiliary_teacher_weight = 0.0
        confidence_loss_weight = 1.0
        batch_size = 1
        sequence_len = 10
        trajectories_per_sample = 384
        train_iterations = 4
        learning_rate = 0.00005
        profiles = $SelectedProfiles
        checkpoint_hashes = $CheckpointHashes
    }
    $DesignJson = ConvertTo-Json -InputObject $Design -Depth 20
    $DesignPath = Join-Path $ResultsRoot "confidence-label-design.json"
    if (Test-Path -LiteralPath $DesignPath -PathType Leaf) {
        $Existing = Get-Content -LiteralPath $DesignPath -Raw |
            ConvertFrom-Json |
            ConvertTo-Json -Depth 20 -Compress
        $Requested = $DesignJson | ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress
        if ($Existing -ne $Requested) {
            throw "The saved experiment design differs from this command. Use a new ResultsRoot."
        }
    }
    else {
        Write-AtomicText -Path $DesignPath -Text ($DesignJson + "`n")
    }

    $AlgorithmDir = Join-Path $RepoRoot "cotracker-algorithm"
    $Image = "trackrad-cotracker-confidence-label-training"
    $BuildLog = Join-Path $ResultsRoot "docker-build.log"
    $BuildStatus = Invoke-DockerLogged -LogPath $BuildLog -Arguments @(
        "build", "--progress=plain", "--platform=linux/amd64",
        "--file", (Join-Path $AlgorithmDir "Dockerfile.training"),
        "--tag", $Image,
        $AlgorithmDir
    )
    if ($BuildStatus -ne 0) {
        throw "Training image build failed with exit code $BuildStatus"
    }

    $Failures = @()
    foreach ($Profile in $Profiles) {
        $Definition = $Definitions[$Profile]
        $ProfileDir = Join-Path $ResultsRoot $Profile
        New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
        $FinalCheckpoint = Join-Path $ProfileDir "cotracker_three_final.pth"
        if ((Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -and
            (Get-Item -LiteralPath $FinalCheckpoint).Length -gt 0) {
            Write-Host "SKIP completed profile: $Profile"
            continue
        }

        $Log = Join-Path $ProfileDir "training.log"
        "===== START $Profile $(Get-Date -Format o) =====" |
            Tee-Object -FilePath $Log -Append
        $Arguments = @(
            "run", "--rm", "--gpus", "all",
            "--shm-size", $DockerShmSize,
            "--mount", "type=bind,source=$DatasetDir,target=/data,readonly",
            "--mount", "type=bind,source=$CheckpointDir,target=/checkpoints,readonly",
            "--mount", "type=bind,source=$ProfileDir,target=/results",
            "--workdir", "/opt/app/ext/co-tracker",
            $Image,
            "--batch_size", "1",
            "--num_workers", "$DataLoaderWorkers",
            "--num_steps", "$NumSteps",
            "--ckpt_path", "/results",
            "--model_name", "cotracker_three",
            "--experiment_name", $Profile,
            "--sequence_len", "10",
            "--traj_per_sample", "384",
            "--train_iters", "4",
            "--save_freq", "1000000",
            "--save_every_n_steps", "$SaveEverySteps",
            "--keep_last_checkpoints", "3",
            "--save_every_n_epoch", "1",
            "--trackrad_data_dir", "/data",
            "--skip_evaluation",
            "--offline_model",
            "--restore_ckpt", "/checkpoints/scaled_offline.pth",
            "--teacher_types", "cotracker2v1", "online_cotracker_three", "offline_cotracker_three",
            "--teacher_cotracker2_ckpt", "/checkpoints/cotracker2v1.pth",
            "--teacher_online_ckpt", "/checkpoints/baseline_online.pth",
            "--teacher_offline_ckpt", "/checkpoints/baseline_offline.pth",
            "--auxiliary_teacher_weight", "0",
            "--seed", "$TrainingSeed",
            "--teacher_seed", "$TeacherSeed",
            "--auxiliary_teacher_seed", "20260901",
            "--teacher_log_every", "1",
            "--lr", "0.00005",
            "--supervise_confidence",
            "--confidence_loss_weight", "1.0",
            "--confidence_target_mode", "$($Definition.target_mode)",
            "--confidence_distance_threshold", "$($Definition.hard_threshold)",
            "--confidence_inner_radius", "$($Definition.inner_radius)",
            "--confidence_outer_radius", "$($Definition.outer_radius)"
        )

        $RunStatus = Invoke-DockerLogged -LogPath $Log -Arguments $Arguments
        if ($RunStatus -ne 0) {
            $Failures += $Profile
            "===== FAILED $Profile exit=$RunStatus $(Get-Date -Format o) =====" |
                Tee-Object -FilePath $Log -Append
            if (-not $ContinueOnFailure) {
                throw "Profile $Profile failed. Fix the first failure and rerun to resume."
            }
            continue
        }
        "===== DONE $Profile $(Get-Date -Format o) =====" |
            Tee-Object -FilePath $Log -Append
    }

    if ($Failures.Count -gt 0) {
        throw "Failed profiles: $($Failures -join ', '). Rerun the same command to resume."
    }

    & (Join-Path $PSScriptRoot "audit_confidence_label_experiments.ps1") `
        -ResultsRoot $ResultsRoot `
        -NumSteps $NumSteps `
        -Profiles $Profiles
    if ($LASTEXITCODE -ne 0) { throw "Confidence-label audit failed" }
    Write-Host "All confidence-label training profiles completed: $ResultsRoot"
}
finally {
    if ($null -ne $LockStream) { $LockStream.Dispose() }
}
