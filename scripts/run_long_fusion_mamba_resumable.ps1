[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$TrainDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40",
    [string]$ValidationDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10",
    [string]$TestDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\long-fusion-mamba",
    [int]$TrainingSteps = 500,
    [int]$SaveEverySteps = 25
)

$ErrorActionPreference = "Stop"
$AlgorithmDir = Join-Path $RepoRoot "cotracker-algorithm"
$Runner = Join-Path $RepoRoot "scripts\run_hierarchical_followup_resumable.ps1"
$Summarizer = Join-Path $RepoRoot "scripts\summarize_long_fusion_results.ps1"
$Image = "trackrad-algorithm-cotracker-algorithm"
$Log = Join-Path $ResultsRoot "queue.log"
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

$Expected = @{
    $TrainDataset = 40
    $ValidationDataset = 10
    $TestDataset = 38
}
foreach ($Entry in $Expected.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $Entry.Key -PathType Container)) {
        throw "Dataset is missing: $($Entry.Key)"
    }
    $Count = @(Get-ChildItem -LiteralPath $Entry.Key -Directory -Force).Count
    if ($Count -ne $Entry.Value) {
        throw "Expected $($Entry.Value) cases, found $Count in $($Entry.Key)"
    }
}

function Invoke-DockerLogged {
    param([string[]]$Arguments, [string]$StageLog)
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & docker @Arguments 2>&1 |
            Tee-Object -FilePath $StageLog -Append |
            ForEach-Object { Write-Host $_ }
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $Previous
    }
}

$TranscriptStarted = $false
try {
    Start-Transcript -Path $Log -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Long-fusion queue started"

    $BuildLog = Join-Path $ResultsRoot "docker-build.log"
    $Status = Invoke-DockerLogged -StageLog $BuildLog -Arguments @(
        "build", $AlgorithmDir,
        "--platform=linux/amd64",
        "--tag", $Image
    )
    if ($Status -ne 0) { throw "Algorithm image build failed: $Status" }

    $TrainCache = Join-Path $ResultsRoot "cache-train-40"
    $ValidationCache = Join-Path $ResultsRoot "cache-validation-10"
    foreach ($Cache in @(
        @{ Dataset = $TrainDataset; Output = $TrainCache; Count = 40 },
        @{ Dataset = $ValidationDataset; Output = $ValidationCache; Count = 10 }
    )) {
        New-Item -ItemType Directory -Force -Path $Cache.Output | Out-Null
        $CacheLog = Join-Path $Cache.Output "prepare.log"
        $Status = Invoke-DockerLogged -StageLog $CacheLog -Arguments @(
            "run", "--rm", "--gpus", "all", "--network", "none",
            "--mount", "type=bind,source=$($Cache.Dataset),target=/data,readonly",
            "--mount", "type=bind,source=$($Cache.Output),target=/cache",
            "--entrypoint", "/opt/app/.pixi/envs/cuda/bin/python",
            $Image,
            "/opt/app/experiments/prepare_long_fusion_cache.py",
            "--dataset-dir", "/data",
            "--output-dir", "/cache",
            "--expected-cases", "$($Cache.Count)"
        )
        if ($Status -ne 0) { throw "Feature cache preparation failed: $Status" }
    }

    $GateRoot = Join-Path $ResultsRoot "gates"
    $GateCheckpoints = @{}
    foreach ($Kind in @("mlp", "mamba")) {
        $GateDir = Join-Path $GateRoot $Kind
        New-Item -ItemType Directory -Force -Path $GateDir | Out-Null
        $GateLog = Join-Path $GateDir "training.log"
        $Status = Invoke-DockerLogged -StageLog $GateLog -Arguments @(
            "run", "--rm", "--gpus", "all", "--network", "none",
            "--mount", "type=bind,source=$TrainCache,target=/train-cache,readonly",
            "--mount", "type=bind,source=$ValidationCache,target=/validation-cache,readonly",
            "--mount", "type=bind,source=$GateDir,target=/results",
            "--entrypoint", "/opt/app/.pixi/envs/cuda/bin/python",
            $Image,
            "/opt/app/experiments/train_long_fusion_gate.py",
            "--kind", $Kind,
            "--train-cache", "/train-cache",
            "--validation-cache", "/validation-cache",
            "--output-dir", "/results",
            "--steps", "$TrainingSteps",
            "--save-every", "$SaveEverySteps",
            "--seed", "20260910"
        )
        if ($Status -ne 0) { throw "$Kind gate training failed: $Status" }
        $Final = Join-Path $GateDir "gate_final.pth"
        if (-not (Test-Path -LiteralPath $Final -PathType Leaf)) {
            throw "$Kind gate did not produce $Final"
        }
        $GateCheckpoints[$Kind] = $Final
    }

    $Profiles = @(
        "hierarchical_full_grid0_iterations2",
        "hierarchical_full_grid0_iterations2_mlp_gate",
        "hierarchical_full_grid0_iterations2_mamba_gate"
    )
    foreach ($Split in @(
        @{ Name = "validation-10"; Dataset = $ValidationDataset },
        @{ Name = "test-38"; Dataset = $TestDataset }
    )) {
        $SplitResults = Join-Path $ResultsRoot $Split.Name
        foreach ($Profile in $Profiles) {
            $Arguments = @{
                Repository = $RepoRoot
                Dataset = $Split.Dataset
                Results = $SplitResults
                Profiles = @($Profile)
                RequireDiagnostics = $true
            }
            if ($Profile -like "*_mlp_gate") {
                $Arguments.FusionGateCheckpoint = $GateCheckpoints.mlp
            }
            elseif ($Profile -like "*_mamba_gate") {
                $Arguments.FusionGateCheckpoint = $GateCheckpoints.mamba
            }
            & $Runner @Arguments
        }
        & $Summarizer -Results $SplitResults -Prefix "long-fusion-$($Split.Name)"
    }
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Long-fusion queue completed"
}
catch {
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] FAILED: $($_.Exception.Message)"
    throw
}
finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
