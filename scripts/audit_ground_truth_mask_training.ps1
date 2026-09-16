[CmdletBinding()]
param(
    [string]$TrainingRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\gt-mask\train",
    [int]$ExpectedSteps = 1000,
    [ValidateRange(0.5, 1.0)][double]$MinimumLogCoverage = 0.95
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
$ReferenceConfig = $null

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
function Get-CompactValue([string]$Value) {
    return [regex]::Replace($Value, '\s+', '')
}

function Get-RecordValue(
    [string]$Record,
    [string]$StartPattern,
    [string]$EndPattern
) {
    $Match = [regex]::Match(
        $Record,
        '(?s)' + $StartPattern + '=(?<value>.*?)' + $EndPattern + '='
    )
    if (-not $Match.Success) { return $null }
    return Get-CompactValue $Match.Groups["value"].Value
}

foreach ($Profile in $Profiles) {
    $Log = Join-Path $TrainingRoot "$Profile\training.log"
    $Checkpoint = Join-Path $TrainingRoot "$Profile\cotracker_three_final.pth"
    $ConfigPath = Join-Path $TrainingRoot "$Profile\run-config.json"
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "$Profile is missing its final checkpoint"
    }
    if ((Get-Item -LiteralPath $Checkpoint).Length -le 0) {
        throw "$Profile has an empty final checkpoint"
    }
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "$Profile is missing run-config.json"
    }
    $Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    if (
        $Config.profile -ne $Profile -or
        [double]$Config.mask_supervision_weight -ne
            [double]$ExpectedWeights[$Profile] -or
        [int]$Config.num_steps -ne $ExpectedSteps -or
        [double]$Config.auxiliary_teacher_weight -ne 0.0 -or
        $Config.confidence_target_mode -ne "hard"
    ) {
        throw "$Profile has an incompatible run-config.json"
    }
    $ComparableConfig = [ordered]@{
        num_steps = [int]$Config.num_steps
        training_seed = [int]$Config.training_seed
        teacher_seed = [int]$Config.teacher_seed
        teacher_types = @($Config.teacher_types) -join ","
        auxiliary_teacher_weight = [double]$Config.auxiliary_teacher_weight
        confidence_target_mode = [string]$Config.confidence_target_mode
        train_dataset = [string]$Config.train_dataset
    } | ConvertTo-Json -Compress
    if ($null -eq $ReferenceConfig) {
        $ReferenceConfig = $ComparableConfig
    }
    elseif ($ComparableConfig -ne $ReferenceConfig) {
        throw "$Profile differs from control in a frozen training variable"
    }
    $Map = @{}
    # Windows PowerShell can hard-wrap native Docker stderr in the middle of
    # field names and values before Tee-Object receives it. First isolate every
    # experiment record so a malformed line cannot consume the next one. Then
    # match each key with optional inter-character whitespace.
    $RawLog = Get-Content -LiteralPath $Log -Raw
    $RecordStarts = @([regex]::Matches($RawLog, $Keys.experiment + '='))
    for ($RecordIndex = 0; $RecordIndex -lt $RecordStarts.Count; $RecordIndex++) {
        $Start = $RecordStarts[$RecordIndex].Index
        $End = if ($RecordIndex + 1 -lt $RecordStarts.Count) {
            $RecordStarts[$RecordIndex + 1].Index
        }
        else {
            $RawLog.Length
        }
        $RecordText = $RawLog.Substring($Start, $End - $Start)
        $Experiment = Get-RecordValue $RecordText $Keys.experiment $Keys.step
        if ($Experiment -ne $Profile) {
            continue
        }
        $StepText = Get-RecordValue $RecordText $Keys.step $Keys.case
        $Case = Get-RecordValue $RecordText $Keys.case $Keys.query_sha256
        $Query = Get-RecordValue $RecordText $Keys.query_sha256 $Keys.primary
        $Primary = Get-RecordValue $RecordText $Keys.primary $Keys.auxiliary
        $Auxiliary = Get-RecordValue $RecordText $Keys.auxiliary $Keys.alpha
        $Alpha = Get-RecordValue $RecordText $Keys.alpha $Keys.primary_loss
        $Weight = Get-RecordValue $RecordText $Keys.mask_weight $Keys.mask_loss
        $MaskLossMatch = [regex]::Match(
            $RecordText,
            $Keys.mask_loss +
            '=\s*(?<value>None|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
        )
        if (
            $null -eq $StepText -or $null -eq $Case -or
            $null -eq $Query -or $null -eq $Primary -or
            $null -eq $Auxiliary -or $null -eq $Alpha -or
            $null -eq $Weight -or -not $MaskLossMatch.Success -or
            $StepText -notmatch '^\d+$' -or $Query -notmatch '^[0-9a-f]{64}$'
        ) {
            continue
        }
        $Step = [int]$StepText
        $Map[$Step] = [ordered]@{
            case = $Case
            query_sha256 = $Query
            teacher = $Primary
            auxiliary = $Auxiliary
            alpha = $Alpha
            weight = $Weight
            mask_loss = Get-CompactValue $MaskLossMatch.Groups["value"].Value
        }
    }
    $Missing = @(0..($ExpectedSteps - 1) | Where-Object { -not $Map.ContainsKey($_) })
    $Coverage = $Map.Count / [double]$ExpectedSteps
    if ($Coverage -lt $MinimumLogCoverage) {
        $Preview = ($Missing | Select-Object -First 30) -join ","
        throw "$Profile log coverage is $([math]::Round($Coverage, 4)); missing: $Preview"
    }
    foreach ($Step in @($Map.Keys)) {
        $Record = $Map[$Step]
        if ($Record.auxiliary -ne "none" -or $Record.alpha -ne "0.0000") {
            throw "$Profile used an auxiliary teacher at step $Step"
        }
        if ($Record.weight -ne $ExpectedWeights[$Profile]) {
            throw "$Profile reported mask weight $($Record.weight) at step $Step"
        }
        if ($Profile -eq "control") {
            if ($Record.mask_loss -ne "None") {
                throw "Control unexpectedly used mask supervision at step $Step"
            }
        }
        else {
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
        ExpectedSteps = $ExpectedSteps
        AuditedSteps = $Map.Count
        Coverage = $Coverage
        MissingSteps = $Missing -join ","
        FinalCheckpointBytes = (Get-Item -LiteralPath $Checkpoint).Length
    }
}

$PairedSteps = 0
$UnpairedSteps = @()
$MinimumProfilesObserved = $Profiles.Count
foreach ($Step in 0..($ExpectedSteps - 1)) {
    $Available = @($Profiles | Where-Object { $Maps[$_].ContainsKey($Step) })
    $MinimumProfilesObserved = [math]::Min(
        $MinimumProfilesObserved,
        $Available.Count
    )
    if ($Available.Count -lt 2) {
        $UnpairedSteps += $Step
        continue
    }
    $PairedSteps += 1
    $ReferenceProfile = $Available[0]
    $Reference = $Maps[$ReferenceProfile][$Step]
    foreach ($Profile in $Available | Select-Object -Skip 1) {
        $Record = $Maps[$Profile][$Step]
        if ($Record.teacher -ne $Reference.teacher) {
            throw "Teacher mismatch at step ${Step}: $ReferenceProfile vs $Profile"
        }
        if ($Record.case -ne $Reference.case) {
            throw "Training-case mismatch at step ${Step}: $ReferenceProfile vs $Profile"
        }
        if ($Record.query_sha256 -ne $Reference.query_sha256) {
            throw "Query-sampling mismatch at step ${Step}: $ReferenceProfile vs $Profile"
        }
    }
}
$PairedCoverage = $PairedSteps / [double]$ExpectedSteps
if ($PairedCoverage -lt $MinimumLogCoverage) {
    $Preview = ($UnpairedSteps | Select-Object -First 30) -join ","
    throw "Paired log coverage is $([math]::Round($PairedCoverage, 4)); unpaired: $Preview"
}

$Audit = [ordered]@{
    schema_version = 1
    passed = $true
    expected_steps = $ExpectedSteps
    profiles = $Profiles
    final_checkpoints_verified = $true
    frozen_run_configs_verified = $true
    minimum_required_log_coverage = $MinimumLogCoverage
    paired_steps = $PairedSteps
    paired_coverage = $PairedCoverage
    minimum_profiles_observed_at_any_step = $MinimumProfilesObserved
    unpaired_steps = $UnpairedSteps
    profile_coverage = $Rows
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
