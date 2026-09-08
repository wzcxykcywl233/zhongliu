[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-training-results",
    [int]$NumSteps = 2000,
    [int]$SaveEverySteps = 25,
    [ValidateRange(0, 16)]
    [int]$DataLoaderWorkers = 0,
    [string]$DockerShmSize = "4g",
    [int]$TrainingSeed = 0,
    [int]$PrimaryTeacherSeed = 20260829,
    [int]$AuxiliaryTeacherSeed = 20260830,
    [switch]$ContinueOnFailure,
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
if (-not (Test-Path $DatasetDir)) { throw "Dataset not found: $DatasetDir" }
$CheckpointDir = Join-Path $RepoRoot "training-checkpoints"
$Required = @(
    "scaled_offline.pth",
    "baseline_online.pth",
    "baseline_offline.pth",
    "cotracker2v1.pth"
)
foreach ($Name in $Required) {
    $Path = Join-Path $CheckpointDir $Name
    if (-not (Test-Path $Path)) {
        throw "Missing $Path. Run scripts\prepare_random_tutor_checkpoints.ps1 first."
    }
}

$Definitions = @{
    baseline_single_teacher = @{ Weight = 0.0; Same = $false }
    random_tutor_w0025 = @{ Weight = 0.025; Same = $false }
    random_tutor_w005 = @{ Weight = 0.05; Same = $false }
    random_tutor_w010 = @{ Weight = 0.10; Same = $false }
    random_tutor_w015 = @{ Weight = 0.15; Same = $false }
    random_tutor_w020 = @{ Weight = 0.20; Same = $false }
    random_tutor_w025 = @{ Weight = 0.25; Same = $false }
    same_teacher_control_w010 = @{ Weight = 0.10; Same = $true }
    same_teacher_control_w020 = @{ Weight = 0.20; Same = $true }
}
foreach ($Profile in $Profiles) {
    if (-not $Definitions.ContainsKey($Profile)) {
        throw "Unknown profile: $Profile"
    }
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$Image = "trackrad-cotracker-random-tutor-training"
$AlgorithmDir = Join-Path $RepoRoot "cotracker-algorithm"
$BuildLog = Join-Path $ResultsRoot "docker-build.log"

$PreviousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& docker build --progress=plain --platform=linux/amd64 `
    --file (Join-Path $AlgorithmDir "Dockerfile.training") `
    --tag $Image $AlgorithmDir 2>&1 |
    Tee-Object -FilePath $BuildLog -Append
$BuildExit = $LASTEXITCODE
$ErrorActionPreference = $PreviousPreference
if ($BuildExit -ne 0) { throw "Training image build failed with exit code $BuildExit" }

$Failures = @()
foreach ($Profile in $Profiles) {
    $Definition = $Definitions[$Profile]
    $ProfileDir = Join-Path $ResultsRoot $Profile
    New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
    if (Test-Path (Join-Path $ProfileDir "cotracker_three_final.pth")) {
        Write-Host "SKIP completed profile: $Profile"
        continue
    }

    $Log = Join-Path $ProfileDir "training.log"
    "===== START $Profile $(Get-Date -Format o) =====" | Tee-Object -FilePath $Log -Append
    $DockerArguments = @(
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
        "--save_freq", "200",
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
        "--auxiliary_teacher_weight", "$($Definition.Weight)",
        "--seed", "$TrainingSeed",
        "--teacher_seed", "$PrimaryTeacherSeed",
        "--auxiliary_teacher_seed", "$AuxiliaryTeacherSeed",
        "--teacher_log_every", "1",
        "--confidence_target_mode", "hard",
        "--lr", "0.00005"
    )
    if ($Definition.Same) { $DockerArguments += "--same_teacher_control" }

    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker @DockerArguments 2>&1 | Tee-Object -FilePath $Log -Append
    $RunExit = $LASTEXITCODE
    $ErrorActionPreference = $PreviousPreference
    if ($RunExit -ne 0) {
        $Failures += $Profile
        "===== FAILED $Profile exit=$RunExit $(Get-Date -Format o) =====" |
            Tee-Object -FilePath $Log -Append
        if (-not $ContinueOnFailure) {
            throw "Profile $Profile failed with exit code $RunExit. Fix the first failure, then rerun to resume."
        }
        continue
    }
    "===== DONE $Profile $(Get-Date -Format o) =====" |
        Tee-Object -FilePath $Log -Append
}

if ($Failures.Count -gt 0) {
    throw "Failed profiles: $($Failures -join ', '). Re-run the same command to resume."
}
Write-Host "All random-tutor training profiles completed: $ResultsRoot"
