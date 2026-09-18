[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [Parameter(Mandatory = $true)][string]$Results,
    [Parameter(Mandatory = $true)][int]$ExpectedCases,
    [Parameter(Mandatory = $true)][ValidateSet('validation-10', 'test-38')][string]$SplitName
)
$ErrorActionPreference = 'Stop'
$Manifest = Get-Content (Join-Path $RepoRoot 'cotracker-algorithm\experiments\memory-refinement-40-10-38.json') -Raw | ConvertFrom-Json
$Profiles = @($Manifest.profiles)
$MetricNames = [ordered]@{ DSC = 'dice_similarity_coefficient'; HD95 = 'hausdorff_distance_95'; MASD = 'surface_distance_average'; CD = 'center_distance'; D98 = 'relative_d98_dose' }
$ReferenceCases = @{}
$Rows = @()
$DiagnosticRows = @()
$Summary = [ordered]@{}
foreach ($Profile in $Profiles) {
    $Metrics = Get-Content (Join-Path $Results "$Profile\metrics.json") -Raw | ConvertFrom-Json
    if (@($Metrics.results).Count -ne $ExpectedCases) { throw "$Profile has incorrect case count" }
    $UniqueIds = @($Metrics.results | ForEach-Object { $_.case_id } | Sort-Object -Unique)
    if ($UniqueIds.Count -ne $ExpectedCases) { throw "$Profile has duplicate cases" }
    $Summary[$Profile] = $Metrics
    foreach ($Case in $Metrics.results) {
        $CaseId = [string]$Case.case_id
        if ($Profile -eq $Manifest.reference) { $ReferenceCases[$CaseId] = $Case }
        if (-not $ReferenceCases.ContainsKey($CaseId)) { throw "Unexpected case: $Profile/$CaseId" }
        $Row = [ordered]@{ Split = $SplitName; Profile = $Profile; Case = $CaseId }
        foreach ($Key in $MetricNames.Keys) {
            $Field = $MetricNames[$Key]
            if ($null -eq $Case.$Field -or $null -eq $ReferenceCases[$CaseId].$Field) { throw "Missing metric $Field" }
            $Row[$Key] = [double]$Case.$Field
            $Row["Delta_$Key"] = [double]$Case.$Field - [double]$ReferenceCases[$CaseId].$Field
        }
        $DiagnosticPath = Join-Path $Results "$Profile\checkpoint\jobs\$CaseId\diagnostics.json"
        $Diagnostic = Get-Content -LiteralPath $DiagnosticPath -Raw | ConvertFrom-Json
        if ($Diagnostic.profile -ne $Profile) { throw "Wrong diagnostic profile: $DiagnosticPath" }
        foreach ($Property in $Diagnostic.mechanism.PSObject.Properties) {
            if ($Property.Name -like 'memory_*' -or $Property.Name -like 'query_memory_*') {
                $DiagnosticRows += [PSCustomObject]@{ Split = $SplitName; Profile = $Profile; Case = $CaseId; Mechanism = $Property.Name; Value = $Property.Value }
            }
        }
        $Rows += [PSCustomObject]$Row
    }
}
$SummaryJson = ConvertTo-Json -InputObject $Summary -Depth 100
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $Results 'summary.json.tmp'), $SummaryJson, $Utf8)
Move-Item -LiteralPath (Join-Path $Results 'summary.json.tmp') -Destination (Join-Path $Results 'summary.json') -Force
$Rows | Export-Csv (Join-Path $Results "memory-refinement-$SplitName-cases.csv") -NoTypeInformation -Encoding UTF8
$DiagnosticRows | Export-Csv (Join-Path $Results "memory-refinement-$SplitName-diagnostics.csv") -NoTypeInformation -Encoding UTF8
$DiagnosticRows | Group-Object Profile,Mechanism | ForEach-Object {
    $GroupRows = @($_.Group)
    [PSCustomObject]@{
        Split = $SplitName; Profile = $GroupRows[0].Profile; Mechanism = $GroupRows[0].Mechanism
        Cases = $GroupRows.Count
        NonzeroCases = @($GroupRows | Where-Object { [double]$_.Value -ne 0 }).Count
        Total = ($GroupRows | Measure-Object Value -Sum).Sum
    }
} | Export-Csv (Join-Path $Results "memory-refinement-$SplitName-mechanisms.csv") -NoTypeInformation -Encoding UTF8
& (Join-Path $RepoRoot 'scripts\summarize_dynamic_query_memory.ps1') `
    -Results $Results -Prefix "memory-refinement-$SplitName" `
    -ReferenceName $Manifest.reference -Candidates @($Profiles | Where-Object { $_ -ne $Manifest.reference })
