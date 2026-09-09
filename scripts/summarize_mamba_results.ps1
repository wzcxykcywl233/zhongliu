[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Results,
    [string]$Prefix = "mamba"
)

$ErrorActionPreference = "Stop"
$SummaryPath = Join-Path $Results "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Summary does not exist: $SummaryPath"
}

$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
$Rows = foreach ($Property in $Summary.PSObject.Properties) {
    $Aggregates = $Property.Value.aggregates
    [PSCustomObject]@{
        Profile = $Property.Name
        DSC = [double]$Aggregates.dice_similarity_coefficient
        HD95 = [double]$Aggregates.hausdorff_distance_95
        MASD = [double]$Aggregates.surface_distance_average
        CD = [double]$Aggregates.center_distance
        D98 = [double]$Aggregates.relative_d98_dose
        TimeSec = [double]$Aggregates.total_time
    }
}

$DirectControl = $Rows |
    Where-Object Profile -eq "hierarchical_full_grid0" |
    Select-Object -First 1
if (-not $DirectControl) {
    throw "Missing direct control: hierarchical_full_grid0"
}

$MambaProfiles = @(
    "hierarchical_full_grid0_mamba",
    "hierarchical_full_grid0_mamba_replacement"
)
$ReportedProfiles = @("baseline", "hierarchical_full_grid0") + $MambaProfiles
$Deltas = foreach ($Profile in $MambaProfiles) {
    $Row = $Rows | Where-Object Profile -eq $Profile | Select-Object -First 1
    if (-not $Row) { throw "Missing Mamba result: $Profile" }
    [PSCustomObject]@{
        Profile = $Profile
        Reference = "hierarchical_full_grid0"
        Delta_DSC = $Row.DSC - $DirectControl.DSC
        Delta_HD95 = $Row.HD95 - $DirectControl.HD95
        Delta_MASD = $Row.MASD - $DirectControl.MASD
        Delta_CD = $Row.CD - $DirectControl.CD
        Delta_D98 = $Row.D98 - $DirectControl.D98
        RuntimeRatio = $Row.TimeSec / $DirectControl.TimeSec
    }
}

$RowsPath = Join-Path $Results "$Prefix-results.csv"
$DeltaPath = Join-Path $Results "$Prefix-deltas-vs-hierarchical-full-grid0.csv"
$Rows |
    Where-Object { $_.Profile -in $ReportedProfiles } |
    Sort-Object DSC -Descending |
    Export-Csv -LiteralPath $RowsPath -NoTypeInformation -Encoding UTF8
$Deltas |
    Sort-Object Delta_DSC -Descending |
    Export-Csv -LiteralPath $DeltaPath -NoTypeInformation -Encoding UTF8

Write-Host "Metric directions: DSC and D98 higher are better; HD95, MASD and CD lower are better."
$Rows |
    Where-Object { $_.Profile -in $ReportedProfiles } |
    Sort-Object DSC -Descending |
    Format-Table -AutoSize
Write-Host "Mamba minus hierarchical_full_grid0: positive DSC/D98 and negative HD95/MASD/CD indicate improvement."
$Deltas | Sort-Object Delta_DSC -Descending | Format-Table -AutoSize
Write-Host "Results CSV: $RowsPath"
Write-Host "Direct-control delta CSV: $DeltaPath"
