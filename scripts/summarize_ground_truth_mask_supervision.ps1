[CmdletBinding()]
param(
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\gt-mask",
    [ValidateSet("validation-10", "test-38")][string]$Split = "validation-10",
    [string[]]$Profiles = @("control", "mask_w0025", "mask_w005", "mask_w010"),
    [string]$InferenceProfile = "hierarchical_full_grid0_iterations2_memory_topk"
)

$ErrorActionPreference = "Stop"
$SplitDirectory = if ($Split -eq "validation-10") { "val" } else { "test" }
$SplitRoot = Join-Path $ResultsRoot $SplitDirectory
$Rows = @(
    foreach ($Profile in $Profiles) {
        $MetricsPath = Join-Path $SplitRoot "$Profile\$InferenceProfile\metrics.json"
        if (-not (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
            throw "Missing metrics: $MetricsPath"
        }
        $Metrics = Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json
        $A = $Metrics.aggregates
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

$Control = $Rows | Where-Object Profile -EQ "control" | Select-Object -First 1
if (-not $Control) { throw "Control metrics are required" }
$Deltas = @(
    foreach ($Candidate in $Rows | Where-Object Profile -NE "control") {
        [PSCustomObject]@{
            Profile = $Candidate.Profile
            Reference = "control"
            Delta_DSC = $Candidate.DSC - $Control.DSC
            Delta_HD95 = $Candidate.HD95 - $Control.HD95
            Delta_MASD = $Candidate.MASD - $Control.MASD
            Delta_CD = $Candidate.CD - $Control.CD
            Delta_D98 = $Candidate.D98 - $Control.D98
        }
    }
)

$Rows | Export-Csv (Join-Path $ResultsRoot "$Split-results.csv") `
    -NoTypeInformation -Encoding UTF8
$Deltas | Export-Csv (Join-Path $ResultsRoot "$Split-deltas-vs-control.csv") `
    -NoTypeInformation -Encoding UTF8

Write-Host "Metric directions: DSC and D98 higher are better; HD95, MASD and CD lower are better."
$Rows | Sort-Object @{ Expression = "DSC"; Descending = $true } | Format-Table -AutoSize

if ($Split -eq "validation-10") {
    $Winner = $Rows |
        Where-Object Profile -NE "control" |
        Sort-Object `
            @{ Expression = "DSC"; Descending = $true },
            @{ Expression = "HD95"; Ascending = $true },
            @{ Expression = "MASD"; Ascending = $true },
            @{ Expression = "CD"; Ascending = $true },
            @{ Expression = "D98"; Descending = $true } |
        Select-Object -First 1
    $Selection = [ordered]@{
        schema_version = 1
        selected_profile = $Winner.Profile
        selection_split = "validation-10"
        primary_metric = "DSC"
        tie_breakers = @("HD95", "MASD", "CD", "D98")
        candidate_dsc = $Winner.DSC
        control_dsc = $Control.DSC
        candidate_beats_control = $Winner.DSC -gt $Control.DSC
        test_profiles = @("control", $Winner.Profile)
    }
    $Selection | ConvertTo-Json -Depth 10 |
        Set-Content -LiteralPath (Join-Path $ResultsRoot "selection.json") -Encoding UTF8
    Write-Host "Validation-selected candidate: $($Winner.Profile)"
}
