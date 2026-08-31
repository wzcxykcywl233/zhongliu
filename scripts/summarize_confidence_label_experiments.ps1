[CmdletBinding()]
param(
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-evaluation-results"
)

$ErrorActionPreference = "Stop"
$Profiles = @(
    "confidence_hard_12",
    "confidence_soft_8_16",
    "confidence_soft_6_18"
)

$Rows = @(
    foreach ($Profile in $Profiles) {
        $MetricsPath = Join-Path $EvaluationRoot "$Profile\evaluation\metrics.json"
        if (-not (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
            throw "Missing metrics: $MetricsPath"
        }
        $A = (Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json).aggregates
        [PSCustomObject]@{
            Profile = $Profile
            DSC = [double]$A.dice_similarity_coefficient
            HD95 = [double]$A.hausdorff_distance_95
            MASD = [double]$A.surface_distance_average
            CD = [double]$A.center_distance
            D98 = [double]$A.relative_d98_dose
            TimeSec = [double]$A.total_time
        }
    }
)

$Baseline = $Rows | Where-Object Profile -EQ "confidence_hard_12"
$Deltas = @(
    foreach ($Row in $Rows | Where-Object Profile -NE "confidence_hard_12") {
        [PSCustomObject]@{
            Profile = $Row.Profile
            Delta_DSC = $Row.DSC - $Baseline.DSC
            Delta_HD95 = $Row.HD95 - $Baseline.HD95
            Delta_MASD = $Row.MASD - $Baseline.MASD
            Delta_CD = $Row.CD - $Baseline.CD
            Delta_D98 = $Row.D98 - $Baseline.D98
        }
    }
)

$Rows | Export-Csv `
    (Join-Path $EvaluationRoot "confidence-label-results.csv") `
    -NoTypeInformation -Encoding UTF8
$Deltas | Export-Csv `
    (Join-Path $EvaluationRoot "confidence-label-deltas-vs-hard.csv") `
    -NoTypeInformation -Encoding UTF8

Write-Host "Metric directions: DSC and D98 higher are better; HD95, MASD, and CD lower are better."
$Rows | Sort-Object DSC -Descending | Format-Table -AutoSize
Write-Host "Deltas are candidate minus hard_12: positive DSC/D98 and negative HD95/MASD/CD are improvements."
$Deltas | Format-Table -AutoSize
