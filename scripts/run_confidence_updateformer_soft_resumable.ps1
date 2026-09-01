[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$HeadOnlyTrainingRoot = "C:\zhongliu\zhongliu-tuning\confidence-head-only-training-results",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\confidence-updateformer-training-results",
    [int]$NumSteps = 1000,
    [int]$SaveEverySteps = 25,
    [ValidateRange(0, 16)]
    [int]$DataLoaderWorkers = 0,
    [string]$DockerShmSize = "4g",
    [int]$TrainingSeed = 0,
    [int]$TeacherSeed = 20260902
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
if ($NumSteps -lt 1) { throw "NumSteps must be positive" }
if ($SaveEverySteps -lt 1) { throw "SaveEverySteps must be positive" }
foreach ($Directory in @($RepoRoot, $DatasetDir, $HeadOnlyTrainingRoot)) {
    if (-not (Test-Path -LiteralPath $Directory -PathType Container)) {
        throw "Directory not found: $Directory"
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
        throw "Missing training checkpoint: $Path"
    }
}

$HeadOnlyCheckpoint = Join-Path $HeadOnlyTrainingRoot `
    "confidence_head_soft_6_18\cotracker_three_final.pth"
$HeadOnlyAudit = Join-Path $HeadOnlyTrainingRoot `
    "confidence_head_soft_6_18\confidence-head-parameter-audit.json"
if (-not (Test-Path -LiteralPath $HeadOnlyCheckpoint -PathType Leaf)) {
    throw "Complete the confidence-head-only experiment first: $HeadOnlyCheckpoint"
}
if (-not (Test-Path -LiteralPath $HeadOnlyAudit -PathType Leaf)) {
    throw "Missing head-only frozen-parameter audit: $HeadOnlyAudit"
}
$HeadOnlyAuditValue = Get-Content -LiteralPath $HeadOnlyAudit -Raw | ConvertFrom-Json
if (-not $HeadOnlyAuditValue.passed) {
    throw "The head-only reference did not pass its parameter audit"
}
if ($null -ne $HeadOnlyAuditValue.scope -and
    $HeadOnlyAuditValue.scope -ne "head") {
    throw "The reference audit is not a head-only audit"
}
$HeadOnlyDesignPath = Join-Path $HeadOnlyTrainingRoot `
    "confidence-head-only-design.json"
if (Test-Path -LiteralPath $HeadOnlyDesignPath -PathType Leaf) {
    $HeadOnlyDesign = Get-Content -LiteralPath $HeadOnlyDesignPath -Raw |
        ConvertFrom-Json
    if ([int]$HeadOnlyDesign.training_seed -ne $TrainingSeed -or
        [int]$HeadOnlyDesign.teacher_seed -ne $TeacherSeed) {
        throw "Head-only reference seeds do not match this paired experiment"
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
catch { throw "Another confidence-UpdateFormer runner is using $ResultsRoot" }

function Write-AtomicText {
    param([string]$Path, [string]$Text)
    $Temporary = "$Path.tmp"
    [System.IO.File]::WriteAllText($Temporary, $Text, $Utf8NoBom)
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Copy-ReferenceCheckpoint {
    param([string]$Source, [string]$Destination)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) |
        Out-Null
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        $Temporary = "$Destination.tmp"
        Copy-Item -LiteralPath $Source -Destination $Temporary
        Move-Item -LiteralPath $Temporary -Destination $Destination -Force
    }
    $SourceHash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash
    $DestinationHash = (
        Get-FileHash -LiteralPath $Destination -Algorithm SHA256
    ).Hash
    if ($SourceHash -ne $DestinationHash) {
        throw "Reference checkpoint hash mismatch: $Destination"
    }
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
    finally { $ErrorActionPreference = $PreviousPreference }
    return $ExitCode
}

try {
    $BaseCheckpoint = Join-Path $CheckpointDir "scaled_offline.pth"
    $Design = [ordered]@{
        schema = 1
        objective = "Relearn shared UpdateFormer confidence representations with soft_6_18"
        dataset = (Resolve-Path -LiteralPath $DatasetDir).Path
        profile = "confidence_updateformer_soft_6_18"
        references = @("original_baseline", "confidence_head_soft_6_18")
        target_mode = "linear_soft"
        confidence_inner_radius = 6.0
        confidence_outer_radius = 18.0
        invariant = "Encoder, correlation modules, flow_head, and visibility row stay frozen"
        permitted_side_effect = "Coordinates may change because UpdateFormer representations are shared"
        excluded_losses = @("coordinate", "invisible-coordinate", "visibility")
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
        weight_decay = 0.00001
        base_checkpoint_sha256 = (
            Get-FileHash -LiteralPath $BaseCheckpoint -Algorithm SHA256
        ).Hash
        head_only_checkpoint_sha256 = (
            Get-FileHash -LiteralPath $HeadOnlyCheckpoint -Algorithm SHA256
        ).Hash
    }
    $DesignPath = Join-Path $ResultsRoot "confidence-updateformer-design.json"
    $DesignJson = ConvertTo-Json -InputObject $Design -Depth 20
    if (Test-Path -LiteralPath $DesignPath -PathType Leaf) {
        $Existing = Get-Content -LiteralPath $DesignPath -Raw |
            ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress
        $Requested = $DesignJson | ConvertFrom-Json |
            ConvertTo-Json -Depth 20 -Compress
        if ($Existing -ne $Requested) {
            throw "The saved design differs from this command. Use a new ResultsRoot."
        }
    }
    else { Write-AtomicText -Path $DesignPath -Text ($DesignJson + "`n") }

    Copy-ReferenceCheckpoint `
        -Source $BaseCheckpoint `
        -Destination (Join-Path $ResultsRoot `
            "original_baseline\cotracker_three_final.pth")
    Copy-ReferenceCheckpoint `
        -Source $HeadOnlyCheckpoint `
        -Destination (Join-Path $ResultsRoot `
            "confidence_head_soft_6_18\cotracker_three_final.pth")

    $AlgorithmDir = Join-Path $RepoRoot "cotracker-algorithm"
    $Image = "trackrad-cotracker-confidence-updateformer-training"
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

    $Profile = "confidence_updateformer_soft_6_18"
    $ProfileDir = Join-Path $ResultsRoot $Profile
    $FinalCheckpoint = Join-Path $ProfileDir "cotracker_three_final.pth"
    New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
    if (-not ((Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf) -and
        (Get-Item -LiteralPath $FinalCheckpoint).Length -gt 0)) {
        $Log = Join-Path $ProfileDir "training.log"
        "===== START $Profile $(Get-Date -Format o) =====" |
            Tee-Object -FilePath $Log -Append
        $RunStatus = Invoke-DockerLogged -LogPath $Log -Arguments @(
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
            "--auxiliary_teacher_seed", "20260903",
            "--teacher_log_every", "1",
            "--lr", "0.00005",
            "--wdecay", "0.00001",
            "--supervise_confidence",
            "--confidence_updateformer",
            "--confidence_loss_weight", "1.0",
            "--confidence_target_mode", "linear_soft",
            "--confidence_distance_threshold", "12",
            "--confidence_inner_radius", "6",
            "--confidence_outer_radius", "18"
        )
        if ($RunStatus -ne 0) {
            throw "Profile $Profile failed. Rerun the same command to resume."
        }
        "===== DONE $Profile $(Get-Date -Format o) =====" |
            Tee-Object -FilePath $Log -Append
    }
    else { Write-Host "SKIP completed profile: $Profile" }

    $AuditPath = Join-Path $ProfileDir "confidence-updateformer-parameter-audit.json"
    $AuditStatus = Invoke-DockerLogged `
        -LogPath (Join-Path $ProfileDir "parameter-audit.log") `
        -Arguments @(
            "run", "--rm",
            "--mount", "type=bind,source=$CheckpointDir,target=/checkpoints,readonly",
            "--mount", "type=bind,source=$ProfileDir,target=/results",
            "--entrypoint", "/opt/app/.pixi/envs/training/bin/python",
            $Image,
            "/opt/app/ext/co-tracker/audit_confidence_head_checkpoint.py",
            "--base", "/checkpoints/scaled_offline.pth",
            "--tuned", "/results/cotracker_three_final.pth",
            "--output", "/results/confidence-updateformer-parameter-audit.json",
            "--scope", "updateformer"
        )
    if ($AuditStatus -ne 0 -or
        -not (Test-Path -LiteralPath $AuditPath -PathType Leaf)) {
        throw "Confidence-UpdateFormer parameter audit failed"
    }
    Write-Host "Confidence-UpdateFormer soft-label training completed: $ResultsRoot"
}
finally {
    if ($null -ne $LockStream) { $LockStream.Dispose() }
}
