[CmdletBinding()]
param(
    [string]$EvaluationRoot = "C:\zhongliu\zhongliu-tuning\confidence-head-only-evaluation-results"
)

$ErrorActionPreference = "Stop"
$Profiles = @(
    "original_baseline",
    "confidence_head_hard_12",
    "confidence_head_soft_6_18"
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

function Get-DeltaRow {
    param($Candidate, $Reference, [string]$Comparison)
    [PSCustomObject]@{
        Comparison = $Comparison
        Delta_DSC = $Candidate.DSC - $Reference.DSC
        Delta_HD95 = $Candidate.HD95 - $Reference.HD95
        Delta_MASD = $Candidate.MASD - $Reference.MASD
        Delta_CD = $Candidate.CD - $Reference.CD
        Delta_D98 = $Candidate.D98 - $Reference.D98
    }
}

$Original = $Rows | Where-Object Profile -EQ "original_baseline"
$Hard = $Rows | Where-Object Profile -EQ "confidence_head_hard_12"
$Soft = $Rows | Where-Object Profile -EQ "confidence_head_soft_6_18"
$Deltas = @(
    Get-DeltaRow $Hard $Original "hard_12 minus original"
    Get-DeltaRow $Soft $Original "soft_6_18 minus original"
    Get-DeltaRow $Soft $Hard "soft_6_18 minus hard_12"
)

$Rows | Export-Csv (Join-Path $EvaluationRoot "confidence-head-only-results.csv") `
    -NoTypeInformation -Encoding UTF8
$Deltas | Export-Csv (Join-Path $EvaluationRoot "confidence-head-only-deltas.csv") `
    -NoTypeInformation -Encoding UTF8

Write-Host "Metric directions: DSC and D98 higher are better; HD95, MASD, and CD lower are better."
$Rows | Format-Table -AutoSize
Write-Host "For deltas, positive DSC/D98 and negative HD95/MASD/CD are improvements."
$Deltas | Format-Table -AutoSize
