[CmdletBinding()]
param(
    [string]$Results = "C:\zhongliu\zhongliu-tuning\public-test-38-results"
)

$ErrorActionPreference = "Stop"
$summaryPath = Join-Path $Results "summary.json"
if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
    throw "Summary does not exist: $summaryPath"
}

$summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
$rows = foreach ($property in $summary.PSObject.Properties) {
    $aggregates = $property.Value.aggregates
    [PSCustomObject]@{
        Profile = $property.Name
        DSC     = [double]$aggregates.dice_similarity_coefficient
        HD95    = [double]$aggregates.hausdorff_distance_95
        MASD    = [double]$aggregates.surface_distance_average
        CD      = [double]$aggregates.center_distance
        D98     = [double]$aggregates.relative_d98_dose
        TimeSec = [double]$aggregates.total_time
    }
}

$baseline = $rows | Where-Object Profile -eq "baseline" | Select-Object -First 1
if (-not $baseline) {
    throw "The public test summary does not contain baseline metrics"
}

$comparison = foreach ($row in $rows | Where-Object Profile -ne "baseline") {
    [PSCustomObject]@{
        Profile    = $row.Profile
        Delta_DSC  = $row.DSC - $baseline.DSC
        Delta_HD95 = $row.HD95 - $baseline.HD95
        Delta_MASD = $row.MASD - $baseline.MASD
        Delta_CD   = $row.CD - $baseline.CD
        Delta_D98  = $row.D98 - $baseline.D98
    }
}

$rowsPath = Join-Path $Results "public-test-38-results.csv"
$comparisonPath = Join-Path $Results "public-test-38-deltas-vs-baseline.csv"
$rows | Sort-Object DSC -Descending |
    Export-Csv -LiteralPath $rowsPath -NoTypeInformation -Encoding UTF8
$comparison | Sort-Object Delta_DSC -Descending |
    Export-Csv -LiteralPath $comparisonPath -NoTypeInformation -Encoding UTF8

Write-Host "Metric directions: DSC and D98 higher are better; HD95, MASD and CD lower are better."
$rows | Sort-Object DSC -Descending | Format-Table -AutoSize
Write-Host "Candidate minus baseline: positive DSC/D98 and negative HD95/MASD/CD indicate improvement."
$comparison | Sort-Object Delta_DSC -Descending | Format-Table -AutoSize
Write-Host "Results CSV: $rowsPath"
Write-Host "Delta CSV:   $comparisonPath"

