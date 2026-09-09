[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\mamba-training",
    [int]$NumSteps = 1000,
    [int]$SaveEverySteps = 25,
    [int]$TrainingSeed = 20260910,
    [int]$TeacherSeed = 20260910,
    [int]$DataLoaderWorkers = 0,
    [string]$DockerShmSize = "6g"
)

$ErrorActionPreference = "Stop"
$CheckpointDir = Join-Path $RepoRoot "training-checkpoints"
$BaseCheckpoint = Join-Path $CheckpointDir "scaled_offline.pth"
$TeacherCheckpoint = Join-Path $CheckpointDir "baseline_offline.pth"
foreach ($Path in @($RepoRoot, $DatasetDir, $BaseCheckpoint, $TeacherCheckpoint)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required path missing: $Path" }
}
if (@(Get-ChildItem -LiteralPath $DatasetDir -Directory -Force).Count -ne 40) {
    throw "Mamba training requires exactly 40 training cases: $DatasetDir"
}

$Definitions = @(
    @{ Name = "hierarchical_full_grid0_mamba"; Flag = "--mamba_trajectory_refiner" },
    @{ Name = "hierarchical_full_grid0_mamba_replacement"; Flag = "--mamba_time_replacement" }
)
$AlgorithmDir = Join-Path $RepoRoot "cotracker-algorithm"
$Image = "trackrad-cotracker-mamba-training"
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$BuildLog = Join-Path $ResultsRoot "docker-build.log"

$PreviousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& docker build --progress=plain --platform=linux/amd64 `
    --file (Join-Path $AlgorithmDir "Dockerfile.training") `
    --tag $Image $AlgorithmDir 2>&1 | Tee-Object -FilePath $BuildLog -Append
$BuildExit = $LASTEXITCODE
$ErrorActionPreference = $PreviousPreference
if ($BuildExit -ne 0) { throw "Mamba training image build failed: $BuildExit" }

foreach ($Definition in $Definitions) {
    $Profile = $Definition.Name
    $ProfileDir = Join-Path $ResultsRoot $Profile
    $Final = Join-Path $ProfileDir "cotracker_three_final.pth"
    New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
    if (Test-Path -LiteralPath $Final -PathType Leaf) {
        Write-Host "SKIP completed profile: $Profile"
        continue
    }
    $Log = Join-Path $ProfileDir "training.log"
    "===== START $Profile $(Get-Date -Format o) =====" | Tee-Object -FilePath $Log -Append
    $Arguments = @(
        "run", "--rm", "--gpus", "all", "--shm-size", $DockerShmSize,
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
        "--sequence_len", "20",
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
        "--teacher_types", "offline_cotracker_three",
        "--teacher_offline_ckpt", "/checkpoints/baseline_offline.pth",
        "--auxiliary_teacher_weight", "0",
        "--seed", "$TrainingSeed",
        "--teacher_seed", "$TeacherSeed",
        "--teacher_log_every", "1",
        "--confidence_target_mode", "hard",
        "--lr", "0.00005",
        $Definition.Flag
    )
    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousPreference
    if ($ExitCode -ne 0) {
        throw "$Profile failed with exit code $ExitCode; rerun this command to resume"
    }
    if (-not (Test-Path -LiteralPath $Final -PathType Leaf)) {
        throw "$Profile finished without final checkpoint: $Final"
    }
    "===== DONE $Profile $(Get-Date -Format o) =====" | Tee-Object -FilePath $Log -Append
}

Write-Host "All Mamba training profiles completed: $ResultsRoot"
