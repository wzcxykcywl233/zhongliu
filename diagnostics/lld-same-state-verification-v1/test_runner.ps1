$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
$TempBase=if ($env:COTRACKER_TEST_TMP) {$env:COTRACKER_TEST_TMP} else {$env:TEMP}
$Fixture=Join-Path $TempBase ('ss'+[guid]::NewGuid().ToString('N').Substring(0,6))
$Repo=Join-Path $Fixture 'repo'; $Old=Join-Path $Fixture 'old'; $Prior=Join-Path $Fixture 'prior'
$Data=Join-Path $Fixture 'lld'; $Audit=Join-Path $Fixture 'audit'; $Baseline=Join-Path $Fixture 'base'
$Output=Join-Path $Fixture 'out'; $Train=Join-Path $Fixture 'train-40'
foreach ($p in @($Repo,$Old,$Prior,$Data,$Audit,$Baseline,$Train,(Join-Path $Repo 'scripts'),(Join-Path $Repo 'training-checkpoints'))) {
    New-Item -ItemType Directory -Path $p -Force | Out-Null
}
Set-Content -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Value 'function Get-LocalImageId([string]$Image) { return $Image }' -Encoding UTF8
@{images=@{Training=('sha256:'+('a'*64))}} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Old 'manifest.json') -Encoding UTF8
@{Dataset=@(@{Split='train-40';Case='A_001';File=(Join-Path $Train 'A_001\images\frames.mha')})} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Prior 'frozen-study.json') -Encoding UTF8
Set-Content -LiteralPath (Join-Path $Audit 'data-manifest.json') -Value '{}' -Encoding UTF8
$global:SameStateCalls=[Collections.Generic.List[object]]::new()
$global:SameStateFail=$false
function global:docker {
    $a=@($args); $global:SameStateCalls.Add($a); $global:LASTEXITCODE=0
    if ($global:SameStateFail) { $global:LASTEXITCODE=1; 'Mock failure'; return }
    'PROBE COMPLETE mock'
}
function Assert([bool]$Value,[string]$Message) { if (-not $Value) { throw $Message } }
function Run([string]$Stage) {
    & (Join-Path $Here 'run_verify.ps1') -Stage $Stage -RepoRoot $Repo -PreviousRoot $Old -PriorStudy $Prior -DataRoot $Data -AuditRoot $Audit -BaselineRoot $Baseline -OutputRoot $Output
}
try {
    Run 'all'
    $Calls=@($global:SameStateCalls.ToArray())
    Assert ($Calls.Count -eq 21) '20 GPU jobs plus one CPU report'
    Assert (@($Calls | Where-Object {$_ -contains '--gpus'}).Count -eq 20) 'All GPU jobs bounded'
    foreach ($c in $Calls) {
        Assert (($c -contains 'never') -and ($c -contains 'none')) 'No pull/network'
        Assert (-not ($c -match 'validation-10|test-38|worker.py|target=/study')) 'No evaluation/production'
        foreach ($target in @('checkout','previous','baseline','weights')) {
            Assert (@($c | Where-Object {$_ -like "*target=/$target,readonly"}).Count -eq 1) 'Inputs must be read-only'
        }
        if ($c -contains '--gpus') {
            Assert ($c -contains '3600s') 'GPU timeout'
            Assert (@($c | Where-Object {$_ -like '*train-40*target=/data,readonly'}).Count -eq 1) 'Train-40 only'
        } else {
            Assert ($c -contains '1800s') 'CPU timeout'
            Assert (-not ($c -match 'target=/data|target=/lld,')) 'CPU report has no raw data'
        }
    }
    $global:SameStateCalls.Clear(); Run 'F0'
    Assert ($global:SameStateCalls.Count -eq 6) 'F0 only five jobs plus report'
    Assert (@($global:SameStateCalls.ToArray() | Where-Object {$_ -contains 'F001'}).Count -eq 0) 'No unrequested arms'
    $global:SameStateCalls.Clear(); Run 'report'
    Assert ($global:SameStateCalls.Count -eq 1) 'Report only CPU'
    Assert (-not ($global:SameStateCalls[0] -contains '--gpus')) 'CPU report cannot initialize GPU'
    $global:SameStateCalls.Clear(); $global:SameStateFail=$true
    $Failed=$false
    try { Run 'all' } catch { $Failed=$true }
    Assert $Failed 'Failure surfaced'
    Assert ($global:SameStateCalls.Count -eq 1) 'No subsequent GPU jobs after error'
    $Failed=$false
    try {
        & (Join-Path $Here 'run_verify.ps1') -Stage F0 -RepoRoot $Repo -PreviousRoot $Old -PriorStudy $Prior -DataRoot $Data -AuditRoot $Audit -BaselineRoot $Baseline -OutputRoot $Baseline
    } catch { $Failed=$true }
    Assert $Failed 'Output/input overlap refused'
    Write-Host 'RUNNER TESTS PASSED (mock Docker, no data/GPU access)'
} finally {
    Remove-Item Function:\docker
    $Resolved=[IO.Path]::GetFullPath($Fixture)
    $Parent=[IO.Path]::GetFullPath($TempBase).TrimEnd('\')+'\'
    if ($Resolved.StartsWith($Parent,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path $Resolved -Leaf) -match '^ss[0-9a-f]{6}$') {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    } else { throw 'Unsafe test fixture cleanup refused' }
}
