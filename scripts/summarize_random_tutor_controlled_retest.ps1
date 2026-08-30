[CmdletBinding()]
param(
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\random-tutor-controlled-retest-evaluation"
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

$Rows = @(
    foreach ($SeedDir in Get-ChildItem -LiteralPath $EvaluationRoot -Directory -Filter "seed_*") {
        $Seed = [int]($SeedDir.Name -replace '^seed_', '')
        foreach ($Profile in $Profiles) {
            $MetricsPath = Join-Path $SeedDir.FullName "$Profile\evaluation\metrics.json"
            if (-not (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
                throw "Missing metrics: $MetricsPath"
            }
            $Aggregates = (Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json).aggregates
            [PSCustomObject]@{
                Seed = $Seed
                Profile = $Profile
                DSC = [double]$Aggregates.dice_similarity_coefficient
                HD95 = [double]$Aggregates.hausdorff_distance_95
                MASD = [double]$Aggregates.surface_distance_average
                CD = [double]$Aggregates.center_distance
                D98 = [double]$Aggregates.relative_d98_dose
            }
        }
    }
)

function Get-SampleStd([double[]]$Values) {
    if ($Values.Count -lt 2) { return 0.0 }
    $Mean = ($Values | Measure-Object -Average).Average
    $SumSquares = ($Values | ForEach-Object { [math]::Pow($_ - $Mean, 2) } | Measure-Object -Sum).Sum
    return [math]::Sqrt($SumSquares / ($Values.Count - 1))
}

$Summary = @(
    foreach ($Group in $Rows | Group-Object Profile) {
        $Values = $Group.Group
        [PSCustomObject]@{
            Profile = $Group.Name
            N = $Values.Count
            DSC_Mean = ($Values.DSC | Measure-Object -Average).Average
            DSC_SD = Get-SampleStd $Values.DSC
            HD95_Mean = ($Values.HD95 | Measure-Object -Average).Average
            HD95_SD = Get-SampleStd $Values.HD95
            MASD_Mean = ($Values.MASD | Measure-Object -Average).Average
            MASD_SD = Get-SampleStd $Values.MASD
            CD_Mean = ($Values.CD | Measure-Object -Average).Average
            CD_SD = Get-SampleStd $Values.CD
            D98_Mean = ($Values.D98 | Measure-Object -Average).Average
            D98_SD = Get-SampleStd $Values.D98
        }
    }
)

$PairedDeltas = @(
    foreach ($SeedGroup in $Rows | Group-Object Seed) {
        $Baseline = $SeedGroup.Group |
            Where-Object Profile -EQ "baseline_single_teacher" |
            Select-Object -First 1
        foreach ($Candidate in $SeedGroup.Group | Where-Object Profile -NE "baseline_single_teacher") {
            [PSCustomObject]@{
                Seed = $Candidate.Seed
                Profile = $Candidate.Profile
                Delta_DSC = $Candidate.DSC - $Baseline.DSC
                Delta_HD95 = $Candidate.HD95 - $Baseline.HD95
                Delta_MASD = $Candidate.MASD - $Baseline.MASD
                Delta_CD = $Candidate.CD - $Baseline.CD
                Delta_D98 = $Candidate.D98 - $Baseline.D98
            }
        }
    }
)

$Rows | Sort-Object Seed, Profile |
    Export-Csv (Join-Path $EvaluationRoot "controlled-retest-per-seed.csv") -NoTypeInformation -Encoding UTF8
$Summary | Sort-Object DSC_Mean -Descending |
    Export-Csv (Join-Path $EvaluationRoot "controlled-retest-summary.csv") -NoTypeInformation -Encoding UTF8
$PairedDeltas | Sort-Object Seed, Profile |
    Export-Csv (Join-Path $EvaluationRoot "controlled-retest-paired-deltas.csv") -NoTypeInformation -Encoding UTF8
$Summary | Sort-Object DSC_Mean -Descending | Format-Table -AutoSize
