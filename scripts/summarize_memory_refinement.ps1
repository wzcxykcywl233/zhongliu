[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [Parameter(Mandatory = $true)][string]$Results,
    [Parameter(Mandatory = $true)][int]$ExpectedCases,
    [Parameter(Mandatory = $true)][ValidateSet('validation-10', 'test-38')][string]$SplitName,
    [string]$ManifestRelative = 'cotracker-algorithm\experiments\memory-refinement-40-10-38.json',
    [string]$ResultPrefix = 'memory-refinement'
)
$ErrorActionPreference = 'Stop'
$ManifestPath = if ([System.IO.Path]::IsPathRooted($ManifestRelative)) { $ManifestRelative } else { Join-Path $RepoRoot $ManifestRelative }
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$Profiles = @($Manifest.profiles)
$MetricNames = [ordered]@{ DSC = 'dice_similarity_coefficient'; HD95 = 'hausdorff_distance_95'; MASD = 'surface_distance_average'; CD = 'center_distance'; D98 = 'relative_d98_dose' }
$ReferenceCases = @{}
$ReferenceHashes = @{}
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
        if ($Manifest.state_modes -or $Manifest.audit_output_hashes) {
            $OutputHash = [string]$Diagnostic.prediction.array_sha256
            if (-not $OutputHash) { throw "Missing output hash: $Profile/$CaseId" }
            if ($Profile -eq $Manifest.reference) { $ReferenceHashes[$CaseId] = $OutputHash }
            $Row['OutputMatchesControl'] = $ReferenceHashes[$CaseId] -eq $OutputHash
        }
        if ($Manifest.state_modes) {
            $ExpectedMode = [string]$Manifest.state_modes.$Profile
            if ($Diagnostic.config.query_state_inheritance -ne $ExpectedMode) { throw "Wrong inheritance mode: $Profile/$CaseId" }
            foreach ($Key in @('state_inherited_segments','state_v_values','state_c_values','state_v_abs_logit_sum','state_c_abs_logit_sum')) {
                if ($null -eq $Diagnostic.mechanism.$Key) { throw "Missing diagnostic $Key in $Profile/$CaseId" }
            }
            if ($ExpectedMode -in @('none','c') -and $Diagnostic.mechanism.state_v_values -ne 0) { throw 'Unexpected V inheritance' }
            if ($ExpectedMode -in @('none','v') -and $Diagnostic.mechanism.state_c_values -ne 0) { throw 'Unexpected C inheritance' }
        }
        foreach ($Property in $Diagnostic.mechanism.PSObject.Properties) {
            if ($Property.Name -like 'memory_*' -or $Property.Name -like 'query_memory_*' -or $Property.Name -like 'state_*' -or $Property.Name -like 'runtime_*' -or $Property.Name -like 'occlusion_*') {
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
if ($Manifest.state_modes) {
    foreach ($Profile in $Profiles) {
        $Mode = [string]$Manifest.state_modes.$Profile
        foreach ($Kind in @('v','c')) {
            if ($Mode -eq 'none' -or ($Mode -in @('v','c') -and $Mode -ne $Kind)) { continue }
            $Total = ($DiagnosticRows | Where-Object { $_.Profile -eq $Profile -and $_.Mechanism -eq "state_${Kind}_values" } | Measure-Object Value -Sum).Sum
            if ($Total -le 0) { throw "Inheritance never activated: $Profile/$Kind" }
        }
    }
}
$Rows | Export-Csv (Join-Path $Results "$ResultPrefix-$SplitName-cases.csv") -NoTypeInformation -Encoding UTF8
$DiagnosticRows | Export-Csv (Join-Path $Results "$ResultPrefix-$SplitName-diagnostics.csv") -NoTypeInformation -Encoding UTF8
$DiagnosticRows | Group-Object Profile,Mechanism | ForEach-Object {
    $GroupRows = @($_.Group)
    [PSCustomObject]@{
        Split = $SplitName; Profile = $GroupRows[0].Profile; Mechanism = $GroupRows[0].Mechanism
        Cases = $GroupRows.Count
        NonzeroCases = @($GroupRows | Where-Object { [double]$_.Value -ne 0 }).Count
        Total = ($GroupRows | Measure-Object Value -Sum).Sum
    }
} | Export-Csv (Join-Path $Results "$ResultPrefix-$SplitName-mechanisms.csv") -NoTypeInformation -Encoding UTF8
& (Join-Path $RepoRoot 'scripts\summarize_dynamic_query_memory.ps1') `
    -Results $Results -Prefix "$ResultPrefix-$SplitName" `
    -ReferenceName $Manifest.reference -Candidates @($Profiles | Where-Object { $_ -ne $Manifest.reference })
