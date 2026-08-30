[CmdletBinding()]
param(
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-controlled-retest-results",
    [int[]]$Seeds = @(0, 1, 2),
    [int]$ExpectedSteps = 2000
)

$ErrorActionPreference = "Stop"
$Profiles = @(
    "baseline_single_teacher",
    "random_tutor_w010",
    "random_tutor_w015",
    "random_tutor_w020",
    "random_tutor_w025",
    "same_teacher_control_w020"
)
$TutorProfiles = @(
    "random_tutor_w010",
    "random_tutor_w015",
    "random_tutor_w020",
    "random_tutor_w025"
)

function Read-TeacherSequence {
    param([string]$LogPath)
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        throw "Missing training log: $LogPath"
    }
    $ByStep = @{}
    foreach ($Line in Get-Content -LiteralPath $LogPath) {
        if ($Line -match 'step=(\d+) primary=([^ ]+) auxiliary=([^ ]+) alpha=') {
            $Step = [int]$Matches[1]
            $ByStep[$Step] = [PSCustomObject]@{
                Step = $Step
                Primary = $Matches[2]
                Auxiliary = $Matches[3]
            }
        }
    }
    if ($ByStep.Count -ne $ExpectedSteps) {
        throw "$LogPath contains $($ByStep.Count) unique teacher steps; expected $ExpectedSteps"
    }
    return @(0..($ExpectedSteps - 1) | ForEach-Object {
        if (-not $ByStep.ContainsKey($_)) { throw "$LogPath is missing teacher step $_" }
        $ByStep[$_]
    })
}

$AuditSeeds = @()
foreach ($Seed in $Seeds) {
    $Sequences = @{}
    foreach ($Profile in $Profiles) {
        $Sequences[$Profile] = Read-TeacherSequence (
            Join-Path $TrainingRoot "seed_$Seed\$Profile\training.log"
        )
    }

    $Baseline = $Sequences["baseline_single_teacher"]
    foreach ($Profile in $Profiles) {
        $Candidate = $Sequences[$Profile]
        for ($Step = 0; $Step -lt $ExpectedSteps; $Step++) {
            if ($Candidate[$Step].Primary -ne $Baseline[$Step].Primary) {
                throw "Primary teacher mismatch: seed=$Seed profile=$Profile step=$Step"
            }
        }
    }

    $TutorReference = $Sequences[$TutorProfiles[0]]
    foreach ($Profile in $TutorProfiles) {
        $Candidate = $Sequences[$Profile]
        for ($Step = 0; $Step -lt $ExpectedSteps; $Step++) {
            if ($Candidate[$Step].Auxiliary -ne $TutorReference[$Step].Auxiliary) {
                throw "Auxiliary teacher mismatch: seed=$Seed profile=$Profile step=$Step"
            }
            if ($Candidate[$Step].Auxiliary -eq $Candidate[$Step].Primary) {
                throw "Auxiliary teacher equals primary: seed=$Seed profile=$Profile step=$Step"
            }
        }
    }

    for ($Step = 0; $Step -lt $ExpectedSteps; $Step++) {
        if ($Baseline[$Step].Auxiliary -ne "none") {
            throw "Baseline has an auxiliary teacher: seed=$Seed step=$Step"
        }
        $Control = $Sequences["same_teacher_control_w020"][$Step]
        if ($Control.Auxiliary -ne $Control.Primary) {
            throw "Same-teacher control mismatch: seed=$Seed step=$Step"
        }
    }

    $AuditSeeds += [ordered]@{
        seed = $Seed
        steps = $ExpectedSteps
        primary_sequence_identical = $true
        auxiliary_sequence_identical_across_weights = $true
        auxiliary_always_distinct = $true
        baseline_has_no_auxiliary = $true
        same_teacher_control_exact_identity = $true
    }
}

$Audit = [ordered]@{
    passed = $true
    generated_at = [DateTimeOffset]::Now.ToString("o")
    seeds = $AuditSeeds
}
$AuditPath = Join-Path $TrainingRoot "controlled-sequence-audit.json"
$Temporary = "$AuditPath.tmp"
[System.IO.File]::WriteAllText(
    $Temporary,
    (ConvertTo-Json -InputObject $Audit -Depth 20) + "`n",
    (New-Object System.Text.UTF8Encoding($false))
)
Move-Item -LiteralPath $Temporary -Destination $AuditPath -Force
Write-Host "Controlled sequence audit passed: $AuditPath"
