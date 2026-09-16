[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$TrainDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\gt-mask-paired-v2\train",
    [int]$NumSteps = 1000,
    [int]$SaveEverySteps = 25,
    [ValidateRange(0, 16)][int]$DataLoaderWorkers = 0,
    [string]$DockerShmSize = "4g",
    [int]$TrainingSeed = 20260915,
    [int]$TeacherSeed = 20260915
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Profiles = [ordered]@{
    control = 0.0
    mask_w0025 = 0.025
    mask_w005 = 0.05
    mask_w010 = 0.10
}

if ($NumSteps -lt 1) { throw "NumSteps must be positive" }
if (-not (Test-Path -LiteralPath $TrainDataset -PathType Container)) {
    throw "Training dataset is missing: $TrainDataset"
}
$Cases = @(Get-ChildItem -LiteralPath $TrainDataset -Directory | Sort-Object Name)
if ($Cases.Count -ne 40) { throw "Expected 40 training cases, found $($Cases.Count)" }
foreach ($Case in $Cases) {
    $Label = Join-Path $Case.FullName "targets\$($Case.Name)_labels.mha"
    if (-not (Test-Path -LiteralPath $Label -PathType Leaf)) {
        throw "Full training label is missing: $Label"
    }
}

$CheckpointDir = Join-Path $RepoRoot "training-checkpoints"
foreach ($Name in @(
    "scaled_offline.pth",
    "baseline_online.pth",
    "baseline_offline.pth",
    "cotracker2v1.pth"
)) {
    $Path = Join-Path $CheckpointDir $Name
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing training checkpoint: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$AlgorithmDir = Join-Path $RepoRoot "cotracker-algorithm"
$Image = "trackrad-cotracker-gt-mask-training"
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
foreach ($Entry in $Profiles.GetEnumerator()) {
    $Profile = $Entry.Key
    $Weight = [double]$Entry.Value
    $ProfileDir = Join-Path $ResultsRoot $Profile
    New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
    $FinalCheckpoint = Join-Path $ProfileDir "cotracker_three_final.pth"
    if ((Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -and
        (Get-Item -LiteralPath $FinalCheckpoint).Length -gt 0) {
        Write-Host "SKIP completed profile: $Profile"
        continue
    }

    $Config = [ordered]@{
        schema_version = 2
        profile = $Profile
        mask_supervision_weight = $Weight
        mask_blur_radius_pixels = 4
        num_steps = $NumSteps
        training_seed = $TrainingSeed
        paired_step_seed = $TrainingSeed
        teacher_seed = $TeacherSeed
        teacher_types = @(
            "cotracker2v1",
            "online_cotracker_three",
            "offline_cotracker_three"
        )
        auxiliary_teacher_weight = 0.0
        confidence_target_mode = "hard"
        train_dataset = $TrainDataset
    }
    $ConfigPath = Join-Path $ProfileDir "run-config.json"
    $ConfigJson = ConvertTo-Json -InputObject $Config -Depth 10
    if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
        $Existing = Get-Content -LiteralPath $ConfigPath -Raw
        if ($Existing.Trim() -ne $ConfigJson.Trim()) {
            throw "Refusing incompatible resume for $Profile; run-config.json differs"
        }
    }
    else {
        [System.IO.File]::WriteAllText($ConfigPath, $ConfigJson + "`n", $Utf8NoBom)
    }

    $Log = Join-Path $ProfileDir "training.log"
    "===== START $Profile $(Get-Date -Format o) =====" |
        Tee-Object -FilePath $Log -Append
    $Arguments = @(
        "run", "--rm", "--gpus", "all",
        "--shm-size", $DockerShmSize,
        "--mount", "type=bind,source=$TrainDataset,target=/data,readonly",
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
        "--trackrad_mask_supervision_weight", "$Weight",
        "--trackrad_mask_blur_radius", "4",
        "--skip_evaluation",
        "--offline_model",
        "--restore_ckpt", "/checkpoints/scaled_offline.pth",
        "--teacher_types", "cotracker2v1", "online_cotracker_three", "offline_cotracker_three",
        "--teacher_cotracker2_ckpt", "/checkpoints/cotracker2v1.pth",
        "--teacher_online_ckpt", "/checkpoints/baseline_online.pth",
        "--teacher_offline_ckpt", "/checkpoints/baseline_offline.pth",
        "--auxiliary_teacher_weight", "0",
        "--seed", "$TrainingSeed",
        "--paired_step_seed", "$TrainingSeed",
        "--teacher_seed", "$TeacherSeed",
        "--teacher_log_every", "1",
        "--confidence_target_mode", "hard",
        "--lr", "0.00005"
    )

    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
    $RunExit = $LASTEXITCODE
    $ErrorActionPreference = $PreviousPreference
    if ($RunExit -ne 0) {
        $Failures += $Profile
        "===== FAILED $Profile exit=$RunExit $(Get-Date -Format o) =====" |
            Tee-Object -FilePath $Log -Append
        continue
    }
    "===== DONE $Profile $(Get-Date -Format o) =====" |
        Tee-Object -FilePath $Log -Append
}

if ($Failures.Count -gt 0) {
    throw "Failed profiles: $($Failures -join ', '); rerun to resume"
}
Write-Host "All ground-truth-mask training profiles completed: $ResultsRoot"
