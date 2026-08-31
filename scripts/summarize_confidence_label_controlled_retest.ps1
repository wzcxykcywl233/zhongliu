[CmdletBinding()]
param(
    [string]$SeedZeroEvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-evaluation-results",
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-controlled-retest-evaluation"
)

$ErrorActionPreference = "Stop"
$Profiles = @("confidence_hard_12", "confidence_soft_6_18")

function Get-SampleStd([double[]]$Values) {
    if ($Values.Count -lt 2) { return 0.0 }
    $Mean = ($Values | Measure-Object -Average).Average
    $SumSquares = (
        $Values |
            ForEach-Object { [math]::Pow($_ - $Mean, 2) } |
            Measure-Object -Sum
    ).Sum
    return [math]::Sqrt($SumSquares / ($Values.Count - 1))
}

$Rows = @(
    foreach ($Seed in 0, 1, 2) {
        $Root = if ($Seed -eq 0) {
            $SeedZeroEvaluationRoot
        }
        else {
            Join-Path $EvaluationRoot "seed_$Seed"
        }
        foreach ($Profile in $Profiles) {
            $MetricsPath = Join-Path $Root "$Profile\evaluation\metrics.json"
            if (-not (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
                throw "Missing metrics: $MetricsPath"
            }
            $A = (Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json).aggregates
            [PSCustomObject]@{
                Seed = $Seed
                Profile = $Profile
                DSC = [double]$A.dice_similarity_coefficient
                HD95 = [double]$A.hausdorff_distance_95
                MASD = [double]$A.surface_distance_average
                CD = [double]$A.center_distance
                D98 = [double]$A.relative_d98_dose
                TimeSec = [double]$A.total_time
            }
        }
    }
)

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
        $Hard = $SeedGroup.Group |
            Where-Object Profile -EQ "confidence_hard_12" |
            Select-Object -First 1
        $Soft = $SeedGroup.Group |
            Where-Object Profile -EQ "confidence_soft_6_18" |
            Select-Object -First 1
        [PSCustomObject]@{
            Seed = [int]$SeedGroup.Name
            Delta_DSC = $Soft.DSC - $Hard.DSC
            Delta_HD95 = $Soft.HD95 - $Hard.HD95
            Delta_MASD = $Soft.MASD - $Hard.MASD
            Delta_CD = $Soft.CD - $Hard.CD
            Delta_D98 = $Soft.D98 - $Hard.D98
        }
    }
)

$DeltaSummary = [PSCustomObject]@{
    N = $PairedDeltas.Count
    Delta_DSC_Mean = ($PairedDeltas.Delta_DSC | Measure-Object -Average).Average
    Delta_DSC_SD = Get-SampleStd $PairedDeltas.Delta_DSC
    Delta_HD95_Mean = ($PairedDeltas.Delta_HD95 | Measure-Object -Average).Average
    Delta_HD95_SD = Get-SampleStd $PairedDeltas.Delta_HD95
    Delta_MASD_Mean = ($PairedDeltas.Delta_MASD | Measure-Object -Average).Average
    Delta_MASD_SD = Get-SampleStd $PairedDeltas.Delta_MASD
    Delta_CD_Mean = ($PairedDeltas.Delta_CD | Measure-Object -Average).Average
    Delta_CD_SD = Get-SampleStd $PairedDeltas.Delta_CD
    Delta_D98_Mean = ($PairedDeltas.Delta_D98 | Measure-Object -Average).Average
    Delta_D98_SD = Get-SampleStd $PairedDeltas.Delta_D98
}

New-Item -ItemType Directory -Force -Path $EvaluationRoot | Out-Null
$Rows | Sort-Object Seed, Profile |
    Export-Csv (Join-Path $EvaluationRoot "confidence-retest-per-seed.csv") -NoTypeInformation -Encoding UTF8
$Summary | Sort-Object DSC_Mean -Descending |
    Export-Csv (Join-Path $EvaluationRoot "confidence-retest-summary.csv") -NoTypeInformation -Encoding UTF8
$PairedDeltas | Sort-Object Seed |
    Export-Csv (Join-Path $EvaluationRoot "confidence-retest-paired-deltas.csv") -NoTypeInformation -Encoding UTF8
@($DeltaSummary) |
    Export-Csv (Join-Path $EvaluationRoot "confidence-retest-paired-delta-summary.csv") -NoTypeInformation -Encoding UTF8

Write-Host "Metric directions: DSC and D98 higher are better; HD95, MASD, and CD lower are better."
$Rows | Sort-Object Seed, Profile | Format-Table -AutoSize
Write-Host "Mean and sample standard deviation across three paired seeds:"
$Summary | Sort-Object DSC_Mean -Descending | Format-Table -AutoSize
Write-Host "Per-seed soft_6_18 minus hard_12 deltas:"
$PairedDeltas | Format-Table -AutoSize
Write-Host "Paired-delta mean and sample standard deviation:"
$DeltaSummary | Format-List
