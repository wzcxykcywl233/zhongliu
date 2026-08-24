param(
    [string]$Results = "C:\zhongliu\zhongliu-tuning\full-audit-results"
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$BaselineProfile = "baseline"
$SummaryPath = Join-Path $Results "audit-summary.json"
$CsvPath = Join-Path $Results "audit-results.csv"

function Write-AtomicJson {
    param([string]$Path, $Value)
    $Temporary = "$Path.tmp"
    $Json = ConvertTo-Json -InputObject $Value -Depth 100
    [System.IO.File]::WriteAllText($Temporary, $Json + "`n", $Utf8NoBom)
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Get-CaseDiagnostics {
    param([string]$Profile)
    $Jobs = Join-Path $Results "$Profile\checkpoint\jobs"
    if (-not (Test-Path -LiteralPath $Jobs -PathType Container)) {
        return @()
    }
    return @(
        Get-ChildItem -LiteralPath $Jobs -Directory |
            Sort-Object Name |
            ForEach-Object {
                $Path = Join-Path $_.FullName "diagnostics.json"
                if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                    throw "$Profile/$($_.Name) is missing diagnostics.json"
                }
                [PSCustomObject]@{
                    CaseId = $_.Name
                    Value = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
                }
            }
    )
}

function Get-MechanismSum {
    param([object[]]$Cases, [string]$Name)
    $Values = @(
        foreach ($Case in $Cases) {
            $Property = $Case.Value.mechanism.PSObject.Properties[$Name]
            if ($null -ne $Property) { [double]$Property.Value }
        }
    )
    if ($Values.Count -eq 0) { return $null }
    return ($Values | Measure-Object -Sum).Sum
}

$BaselineCases = Get-CaseDiagnostics -Profile $BaselineProfile
if ($BaselineCases.Count -eq 0) {
    throw "No baseline diagnostics found in $Results"
}
$BaselineArrayHashes = @{}
$BaselineFileHashes = @{}
foreach ($Case in $BaselineCases) {
    $BaselineArrayHashes[$Case.CaseId] = $Case.Value.prediction.array_sha256
    $BaselineFileHashes[$Case.CaseId] = $Case.Value.prediction.output_file_sha256
}

$Profiles = @(
    Get-ChildItem -LiteralPath $Results -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "metrics.json") } |
        Sort-Object Name
)
$Rows = New-Object System.Collections.Generic.List[object]
$Detail = [ordered]@{}

foreach ($ProfileDir in $Profiles) {
    $Profile = $ProfileDir.Name
    $Cases = Get-CaseDiagnostics -Profile $Profile
    $Metrics = Get-Content -LiteralPath (Join-Path $ProfileDir.FullName "metrics.json") -Raw |
        ConvertFrom-Json
    $Aggregates = $Metrics.aggregates
    $ExactMatches = 0
    $ExactFileMatches = 0
    $CaseComparison = [ordered]@{}
    foreach ($Case in $Cases) {
        $ArrayHash = $Case.Value.prediction.array_sha256
        $FileHash = $Case.Value.prediction.output_file_sha256
        $MatchesBaseline = (
            $BaselineArrayHashes.ContainsKey($Case.CaseId) -and
            $BaselineArrayHashes[$Case.CaseId] -eq $ArrayHash
        )
        $FileMatchesBaseline = (
            $BaselineFileHashes.ContainsKey($Case.CaseId) -and
            $BaselineFileHashes[$Case.CaseId] -eq $FileHash
        )
        if ($MatchesBaseline) { $ExactMatches++ }
        if ($FileMatchesBaseline) { $ExactFileMatches++ }
        $CaseComparison[$Case.CaseId] = [ordered]@{
            matches_baseline = $MatchesBaseline
            file_matches_baseline = $FileMatchesBaseline
            output_file_sha256 = $FileHash
            array_sha256 = $ArrayHash
        }
    }

    $ConfigHashes = @($Cases | ForEach-Object { $_.Value.config_sha256 } | Sort-Object -Unique)
    if ($ConfigHashes.Count -ne 1) {
        throw "$Profile contains $($ConfigHashes.Count) distinct configuration hashes"
    }
    $ChangedCases = $Cases.Count - $ExactMatches
    $Row = [PSCustomObject][ordered]@{
        Profile = $Profile
        Cases = $Cases.Count
        CasesChangedVsBaseline = $ChangedCases
        FilesChangedVsBaseline = $Cases.Count - $ExactFileMatches
        ExactOutputMatch = ($ChangedCases -eq 0)
        ConfigSHA256 = $ConfigHashes[0]
        DSC = [double]$Aggregates.dice_similarity_coefficient
        HD95 = [double]$Aggregates.hausdorff_distance_95
        MASD = [double]$Aggregates.surface_distance_average
        CD = [double]$Aggregates.center_distance
        D98 = [double]$Aggregates.relative_d98_dose
        TimeSec = [double]$Aggregates.total_time
        VisibilityBelowThreshold = Get-MechanismSum $Cases "visibility_below_threshold"
        ConfidenceBelowThreshold = Get-MechanismSum $Cases "confidence_below_threshold"
        ValidityRejectedPoints = Get-MechanismSum $Cases "validity_rejected_points"
        MedianChangedPointFrames = Get-MechanismSum $Cases "temporal_median_changed_point_frames"
        MorphChangedPixels = Get-MechanismSum $Cases "morph_close_changed_pixels"
        LargestComponentChangedPixels = Get-MechanismSum $Cases "largest_component_changed_pixels"
        LockFirstChangedPixels = Get-MechanismSum $Cases "lock_first_mask_changed_pixels"
        OcclusionMerges = Get-MechanismSum $Cases "occlusion_merges"
        FeatureGateCandidates = Get-MechanismSum $Cases "feature_gate_candidates"
        FeatureRevalidateCandidates = Get-MechanismSum $Cases "feature_revalidate_candidates"
        FeatureRevalidateCorrected = Get-MechanismSum $Cases "feature_revalidate_corrected"
    }
    $Rows.Add($Row)
    $Detail[$Profile] = [ordered]@{
        summary = $Row
        config = $Cases[0].Value.config
        cases = $CaseComparison
    }
}

$Audit = [ordered]@{
    generated_at = [DateTimeOffset]::Now.ToString("o")
    baseline_profile = $BaselineProfile
    baseline_repeat_deterministic = (
        $Detail.Contains("baseline_repeat") -and
        $Detail["baseline_repeat"].summary.ExactOutputMatch
    )
    profiles = $Detail
}
Write-AtomicJson -Path $SummaryPath -Value $Audit
$Rows | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding UTF8
$Rows | Format-Table Profile,CasesChangedVsBaseline,ExactOutputMatch,DSC,HD95,MASD,CD,D98 -AutoSize
Write-Host "Audit JSON: $SummaryPath"
Write-Host "Audit CSV:  $CsvPath"
