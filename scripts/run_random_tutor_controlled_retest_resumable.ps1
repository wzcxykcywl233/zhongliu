[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-controlled-retest-results",
    [int[]]$Seeds = @(0, 1, 2),
    [int]$NumSteps = 2000,
    [int]$SaveEverySteps = 25
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $RepoRoot "scripts\run_random_tutor_training_resumable.ps1"
$Profiles = @(
    "baseline_single_teacher",
    "random_tutor_w010",
    "random_tutor_w015",
    "random_tutor_w020",
    "random_tutor_w025",
    "same_teacher_control_w020"
)

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Training runner not found: $Runner"
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

$Design = [ordered]@{
    schema = 1
    purpose = "paired auxiliary-teacher weight retest"
    dataset = [System.IO.Path]::GetFullPath($DatasetDir)
    seeds = @($Seeds)
    num_steps = $NumSteps
    save_every_steps = $SaveEverySteps
    profiles = @(
        [ordered]@{ name = "baseline_single_teacher"; auxiliary_weight = 0.0; auxiliary_teacher = "none" },
        [ordered]@{ name = "random_tutor_w010"; auxiliary_weight = 0.10; auxiliary_teacher = "fixed independent sequence" },
        [ordered]@{ name = "random_tutor_w015"; auxiliary_weight = 0.15; auxiliary_teacher = "fixed independent sequence" },
        [ordered]@{ name = "random_tutor_w020"; auxiliary_weight = 0.20; auxiliary_teacher = "fixed independent sequence" },
        [ordered]@{ name = "random_tutor_w025"; auxiliary_weight = 0.25; auxiliary_teacher = "fixed independent sequence" },
        [ordered]@{ name = "same_teacher_control_w020"; auxiliary_weight = 0.20; auxiliary_teacher = "same as primary; exact no-op" }
    )
    invariants = @(
        "student initialization",
        "dataset order and augmentation RNG",
        "optimizer and schedule",
        "primary teacher sequence",
        "auxiliary teacher sequence for nonzero distinct-teacher profiles",
        "normalized total teacher-loss weight"
    )
}
$DesignPath = Join-Path $ResultsRoot "controlled-design.json"
$DesignJson = ConvertTo-Json -InputObject $Design -Depth 20
if (Test-Path -LiteralPath $DesignPath -PathType Leaf) {
    $ExistingDesign = Get-Content -LiteralPath $DesignPath -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 20
    $NormalizedDesign = $DesignJson | ConvertFrom-Json | ConvertTo-Json -Depth 20
    if ($ExistingDesign -ne $NormalizedDesign) {
        throw "Controlled design differs from the existing run: $DesignPath"
    }
}
else {
    $DesignTemporary = "$DesignPath.tmp"
    [System.IO.File]::WriteAllText(
        $DesignTemporary,
        $DesignJson + "`n",
        (New-Object System.Text.UTF8Encoding($false))
    )
    Move-Item -LiteralPath $DesignTemporary -Destination $DesignPath
}

foreach ($Seed in $Seeds) {
    $SeedRoot = Join-Path $ResultsRoot "seed_$Seed"
    $PrimaryTeacherSeed = 20260829 + (1000 * $Seed)
    $AuxiliaryTeacherSeed = 20260830 + (1000 * $Seed)
    Write-Host "===== CONTROLLED RETEST seed=$Seed ====="
    & $Runner `
        -RepoRoot $RepoRoot `
        -DatasetDir $DatasetDir `
        -ResultsRoot $SeedRoot `
        -NumSteps $NumSteps `
        -SaveEverySteps $SaveEverySteps `
        -DataLoaderWorkers 0 `
        -DockerShmSize "4g" `
        -TrainingSeed $Seed `
        -PrimaryTeacherSeed $PrimaryTeacherSeed `
        -AuxiliaryTeacherSeed $AuxiliaryTeacherSeed `
        -Profiles $Profiles
    if ($LASTEXITCODE -ne 0) {
        throw "Controlled retest seed $Seed failed with exit code $LASTEXITCODE"
    }
}

$AuditScript = Join-Path $RepoRoot "scripts\audit_random_tutor_controlled_retest.ps1"
& $AuditScript -TrainingRoot $ResultsRoot -Seeds $Seeds -ExpectedSteps $NumSteps
Write-Host "All controlled random-tutor training replicates completed: $ResultsRoot"
