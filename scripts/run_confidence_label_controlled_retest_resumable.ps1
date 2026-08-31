[CmdletBinding()]
param(
    [string]$RepoRoot = "",
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-controlled-retest-training",
    [int]$NumSteps = 1000,
    [int]$SaveEverySteps = 25,
    [ValidateRange(0, 16)]
    [int]$DataLoaderWorkers = 0,
    [string]$DockerShmSize = "4g"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}

$Runner = Join-Path $PSScriptRoot "run_confidence_label_experiments_resumable.ps1"
$Profiles = @("confidence_hard_12", "confidence_soft_6_18")
$Replicates = @(
    [ordered]@{ seed = 1; teacher_seed = 20260832 },
    [ordered]@{ seed = 2; teacher_seed = 20260833 }
)

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$Manifest = [ordered]@{
    schema = 1
    purpose = "Paired replication of hard_12 versus soft_6_18"
    existing_seed_zero_training = (Join-Path $RepoRoot "confidence-label-training-results")
    profiles = $Profiles
    num_steps = $NumSteps
    save_every_steps = $SaveEverySteps
    replicates = $Replicates
}
$ManifestPath = Join-Path $ResultsRoot "controlled-retest-design.json"
$ManifestJson = ConvertTo-Json -InputObject $Manifest -Depth 10
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
if (Test-Path -LiteralPath $ManifestPath -PathType Leaf) {
    $Existing = Get-Content -LiteralPath $ManifestPath -Raw |
        ConvertFrom-Json |
        ConvertTo-Json -Depth 10 -Compress
    $Requested = $ManifestJson | ConvertFrom-Json | ConvertTo-Json -Depth 10 -Compress
    if ($Existing -ne $Requested) {
        throw "The controlled retest design changed. Use a new ResultsRoot."
    }
}
else {
    $Temporary = "$ManifestPath.tmp"
    [System.IO.File]::WriteAllText($Temporary, $ManifestJson + "`n", $Utf8NoBom)
    Move-Item -LiteralPath $Temporary -Destination $ManifestPath -Force
}

foreach ($Replicate in $Replicates) {
    $Seed = [int]$Replicate.seed
    $SeedRoot = Join-Path $ResultsRoot "seed_$Seed"
    Write-Host "===== CONTROLLED CONFIDENCE RETEST seed=$Seed ====="
    & $Runner `
        -RepoRoot $RepoRoot `
        -DatasetDir $DatasetDir `
        -ResultsRoot $SeedRoot `
        -NumSteps $NumSteps `
        -SaveEverySteps $SaveEverySteps `
        -DataLoaderWorkers $DataLoaderWorkers `
        -DockerShmSize $DockerShmSize `
        -TrainingSeed $Seed `
        -TeacherSeed ([int]$Replicate.teacher_seed) `
        -Profiles $Profiles
}

Write-Host "All additional confidence-label replicates completed: $ResultsRoot"
