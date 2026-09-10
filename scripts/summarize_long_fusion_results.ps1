[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Results,
    [string]$Prefix = "long-fusion"
)

$ErrorActionPreference = "Stop"
$SummaryPath = Join-Path $Results "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Summary does not exist: $SummaryPath"
}
$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
$Rows = foreach ($Property in $Summary.PSObject.Properties) {
    $A = $Property.Value.aggregates
    [PSCustomObject]@{
        Profile = $Property.Name
        DSC = [double]$A.dice_similarity_coefficient
        HD95 = [double]$A.hausdorff_distance_95
        MASD = [double]$A.surface_distance_average
        CD = [double]$A.center_distance
        D98 = [double]$A.relative_d98_dose
        TimeSec = [double]$A.total_time
    }
}
$ReferenceName = "hierarchical_full_grid0_iterations2"
$Reference = $Rows | Where-Object Profile -eq $ReferenceName | Select-Object -First 1
if (-not $Reference) { throw "Missing reference profile: $ReferenceName" }
$Candidates = @(
    "hierarchical_full_grid0_iterations2_mlp_gate",
    "hierarchical_full_grid0_iterations2_mamba_gate"
)
$Deltas = foreach ($Name in $Candidates) {
    $Row = $Rows | Where-Object Profile -eq $Name | Select-Object -First 1
    if (-not $Row) { throw "Missing result: $Name" }
    [PSCustomObject]@{
        Profile = $Name
        Reference = $ReferenceName
        Delta_DSC = $Row.DSC - $Reference.DSC
        Delta_HD95 = $Row.HD95 - $Reference.HD95
        Delta_MASD = $Row.MASD - $Reference.MASD
        Delta_CD = $Row.CD - $Reference.CD
        Delta_D98 = $Row.D98 - $Reference.D98
        RuntimeRatio = $Row.TimeSec / $Reference.TimeSec
    }
}
$RowsPath = Join-Path $Results "$Prefix-results.csv"
$DeltaPath = Join-Path $Results "$Prefix-deltas-vs-fixed-control.csv"
$Rows | Sort-Object DSC -Descending |
    Export-Csv -LiteralPath $RowsPath -NoTypeInformation -Encoding UTF8
$Deltas | Sort-Object Delta_DSC -Descending |
    Export-Csv -LiteralPath $DeltaPath -NoTypeInformation -Encoding UTF8
Write-Host "Metric directions: DSC and D98 higher are better; HD95, MASD and CD lower are better."
$Rows | Sort-Object DSC -Descending | Format-Table -AutoSize
Write-Host "Candidate minus fixed control: positive DSC/D98 and negative HD95/MASD/CD indicate improvement."
$Deltas | Sort-Object Delta_DSC -Descending | Format-Table -AutoSize
Write-Host "Results CSV: $RowsPath"
Write-Host "Delta CSV:   $DeltaPath"
