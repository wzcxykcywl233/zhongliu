[CmdletBinding()]
param(
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\gt-mask\train",
    [int]$ExpectedSteps = 1000
)

$ErrorActionPreference = "Stop"
$Profiles = @("control", "mask_w0025", "mask_w005", "mask_w010")
$ExpectedWeights = @{
    control = "0.0000"
    mask_w0025 = "0.0250"
    mask_w005 = "0.0500"
    mask_w010 = "0.1000"
}
$Maps = @{}
$Rows = @()

function ConvertTo-WrappedLiteralPattern([string]$Text) {
    return (($Text.ToCharArray() | ForEach-Object {
        [regex]::Escape([string]$_) + '\s*'
    }) -join '')
}

$Keys = @{}
foreach ($Name in @(
    "experiment", "step", "case", "query_sha256", "primary",
    "auxiliary", "alpha", "primary_loss", "auxiliary_loss",
    "mask_weight", "mask_loss"
)) {
    $Keys[$Name] = ConvertTo-WrappedLiteralPattern $Name
}
$RecordPattern = (
    '(?s)' +
    $Keys.experiment + '=(?<experiment>.*?)' +
    $Keys.step + '=(?<step>.*?)' +
    $Keys.case + '=(?<case>.*?)' +
    $Keys.query_sha256 + '=(?<query>.*?)' +
    $Keys.primary + '=(?<primary>.*?)' +
    $Keys.auxiliary + '=(?<auxiliary>.*?)' +
    $Keys.alpha + '=(?<alpha>.*?)' +
    $Keys.primary_loss + '=(?<primary_loss>.*?)' +
    $Keys.auxiliary_loss + '=(?<auxiliary_loss>.*?)' +
    $Keys.mask_weight + '=(?<weight>.*?)' +
    $Keys.mask_loss +
    '=\s*(?<mask_loss>None|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)

function Get-CompactGroupValue($Match, [string]$Name) {
    return [regex]::Replace($Match.Groups[$Name].Value, '\s+', '')
}

foreach ($Profile in $Profiles) {
    $Log = Join-Path $TrainingRoot "$Profile\training.log"
    $Checkpoint = Join-Path $TrainingRoot "$Profile\cotracker_three_final.pth"
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "$Profile is missing its final checkpoint"
    }
    $Map = @{}
    # Windows PowerShell can hard-wrap native Docker stderr in the middle of
    # field names and values before Tee-Object receives it. Match every key with
    # optional inter-character whitespace, then compact only captured values.
    $RawLog = Get-Content -LiteralPath $Log -Raw
    foreach ($Match in [regex]::Matches($RawLog, $RecordPattern)) {
        if ((Get-CompactGroupValue $Match "experiment") -ne $Profile) {
            continue
        }
        $Step = [int](Get-CompactGroupValue $Match "step")
        $Map[$Step] = [ordered]@{
            case = Get-CompactGroupValue $Match "case"
            query_sha256 = Get-CompactGroupValue $Match "query"
            teacher = Get-CompactGroupValue $Match "primary"
            auxiliary = Get-CompactGroupValue $Match "auxiliary"
            alpha = Get-CompactGroupValue $Match "alpha"
            weight = Get-CompactGroupValue $Match "weight"
            mask_loss = Get-CompactGroupValue $Match "mask_loss"
        }
    }
    $Missing = @(0..($ExpectedSteps - 1) | Where-Object { -not $Map.ContainsKey($_) })
    if ($Missing.Count -gt 0) {
        throw "$Profile is missing $($Missing.Count) audited steps"
    }
    foreach ($Step in 0..($ExpectedSteps - 1)) {
        $Record = $Map[$Step]
        if ($Record.auxiliary -ne "none" -or $Record.alpha -ne "0.0000") {
            throw "$Profile used an auxiliary teacher at step $Step"
        }
        if ($Record.weight -ne $ExpectedWeights[$Profile]) {
            throw "$Profile reported mask weight $($Record.weight) at step $Step"
        }
        if ($Profile -ne "control") {
            $ParsedLoss = 0.0
            if (-not [double]::TryParse(
                $Record.mask_loss,
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$ParsedLoss
            ) -or [double]::IsNaN($ParsedLoss) -or [double]::IsInfinity($ParsedLoss)) {
                throw "$Profile reported an invalid mask loss at step $Step"
            }
        }
    }
    $Maps[$Profile] = $Map
    $Rows += [PSCustomObject]@{
        Profile = $Profile
        Steps = $Map.Count
        FinalCheckpointBytes = (Get-Item -LiteralPath $Checkpoint).Length
    }
}

$Control = $Maps["control"]
foreach ($Profile in $Profiles | Where-Object { $_ -ne "control" }) {
    foreach ($Step in 0..($ExpectedSteps - 1)) {
        if ($Maps[$Profile][$Step].teacher -ne $Control[$Step].teacher) {
            throw "Teacher mismatch at step ${Step}: control vs $Profile"
        }
        if ($Maps[$Profile][$Step].case -ne $Control[$Step].case) {
            throw "Training-case mismatch at step ${Step}: control vs $Profile"
        }
        if ($Maps[$Profile][$Step].query_sha256 -ne $Control[$Step].query_sha256) {
            throw "Query-sampling mismatch at step ${Step}: control vs $Profile"
        }
        if ($Maps[$Profile][$Step].mask_loss -eq "None") {
            throw "$Profile did not report mask supervision at step $Step"
        }
    }
}
foreach ($Step in 0..($ExpectedSteps - 1)) {
    if ($Control[$Step].mask_loss -ne "None") {
        throw "Control unexpectedly used mask supervision at step $Step"
    }
}

$Audit = [ordered]@{
    schema_version = 1
    passed = $true
    expected_steps = $ExpectedSteps
    profiles = $Profiles
    teacher_sequence_mismatches = 0
    case_sequence_mismatches = 0
    query_sequence_mismatches = 0
    auxiliary_teacher_steps = 0
    configured_weights_verified = $true
    control_has_no_mask_loss = $true
    candidates_have_mask_loss = $true
}
$AuditPath = Join-Path $TrainingRoot "training-audit.json"
$Audit | ConvertTo-Json -Depth 10 |
    Set-Content -LiteralPath $AuditPath -Encoding UTF8
$Rows | Export-Csv (Join-Path $TrainingRoot "training-audit.csv") `
    -NoTypeInformation -Encoding UTF8
Write-Host "Ground-truth-mask training audit passed: $AuditPath"
