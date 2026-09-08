[CmdletBinding()]
param(
    [string]$Results = "C:\zhongliu\zhongliu-tuning\public-test-38-results",
    [int]$ExpectedCases = 38
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$BaselineJobs = Join-Path $Results "baseline\checkpoint\jobs"
$RepeatJobs = Join-Path $Results "baseline_repeat\checkpoint\jobs"
$CsvPath = Join-Path $Results "baseline-repeat-hash-audit.csv"
$JsonPath = Join-Path $Results "baseline-repeat-hash-audit.json"

foreach ($Path in @($BaselineJobs, $RepeatJobs)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Baseline audit input does not exist: $Path"
    }
}

$Rows = @(
    foreach ($CaseDir in Get-ChildItem -LiteralPath $BaselineJobs -Directory | Sort-Object Name) {
        $CaseId = $CaseDir.Name
        $BaselineOutput = Join-Path $CaseDir.FullName `
            "output\images\mri-linac-series-targets\output.mha"
        $RepeatOutput = Join-Path $RepeatJobs `
            "$CaseId\output\images\mri-linac-series-targets\output.mha"
        foreach ($Path in @($BaselineOutput, $RepeatOutput)) {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw "Baseline audit output does not exist: $Path"
            }
        }
        $BaselineHash = (Get-FileHash -LiteralPath $BaselineOutput -Algorithm SHA256).Hash.ToLowerInvariant()
        $RepeatHash = (Get-FileHash -LiteralPath $RepeatOutput -Algorithm SHA256).Hash.ToLowerInvariant()
        [PSCustomObject]@{
            CaseId = $CaseId
            ExactFileMatch = ($BaselineHash -eq $RepeatHash)
            BaselineSHA256 = $BaselineHash
            RepeatSHA256 = $RepeatHash
        }
    }
)

if ($Rows.Count -ne $ExpectedCases) {
    throw "Baseline hash audit found $($Rows.Count) cases; expected $ExpectedCases"
}
$ExactMatches = @($Rows | Where-Object ExactFileMatch).Count
$Summary = [ordered]@{
    generated_at = [DateTimeOffset]::Now.ToString("o")
    expected_cases = $ExpectedCases
    compared_cases = $Rows.Count
    exact_file_matches = $ExactMatches
    deterministic = ($ExactMatches -eq $ExpectedCases)
}

$Rows | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding UTF8
$Temporary = "$JsonPath.tmp"
[System.IO.File]::WriteAllText(
    $Temporary,
    (ConvertTo-Json -InputObject $Summary -Depth 10) + "`n",
    $Utf8NoBom
)
Move-Item -LiteralPath $Temporary -Destination $JsonPath -Force

$Summary | Format-List
Write-Host "Audit JSON: $JsonPath"
Write-Host "Audit CSV:  $CsvPath"
if (-not $Summary.deterministic) {
    Write-Warning "Baseline repeat is not byte-identical for every case"
}
