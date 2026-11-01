$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
$TempBase=if ($env:COTRACKER_TEST_TMP) {$env:COTRACKER_TEST_TMP} else {$env:TEMP}
$Fixture=Join-Path $TempBase ('st'+[guid]::NewGuid().ToString('N').Substring(0,6))
$Repo=Join-Path $Fixture 'repo'; $Old=Join-Path $Fixture 'old'; $Prior=Join-Path $Fixture 'prior'
$Data=Join-Path $Fixture 'lld'; $Audit=Join-Path $Fixture 'audit'; $Baseline=Join-Path $Fixture 'base'
$Evidence=Join-Path $Fixture 'evidence'; $Output=Join-Path $Fixture 'out'; $Train=Join-Path $Fixture 'train-40'
foreach ($p in @($Repo,$Old,$Prior,$Data,$Audit,$Baseline,$Evidence,$Train,(Join-Path $Repo 'scripts'),(Join-Path $Repo 'training-checkpoints'))) {
    New-Item -ItemType Directory -Path $p -Force | Out-Null
}
Set-Content -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Value 'function Get-LocalImageId([string]$Image) { return $Image }' -Encoding UTF8
@{images=@{Training=('sha256:'+('a'*64))}} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Old 'manifest.json') -Encoding UTF8
@{Dataset=@(@{Split='train-40';Case='A_001';File=(Join-Path $Train 'A_001\images\frames.mha')})} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Prior 'frozen-study.json') -Encoding UTF8
Set-Content -LiteralPath (Join-Path $Audit 'data-manifest.json') -Value '{}' -Encoding UTF8
$global:StabilityCalls=[Collections.Generic.List[object]]::new()
$global:StabilityFailAt=0
function global:docker {
    $a=@($args); $global:StabilityCalls.Add($a); $global:LASTEXITCODE=0
    if ($global:StabilityFailAt -eq $global:StabilityCalls.Count) { $global:LASTEXITCODE=1; 'Mock failure'; return }
    'TRAIN COMPLETE mock'
}
function Assert([bool]$Value,[string]$Message) { if (-not $Value) { throw $Message } }
function Run([string]$Stage) {
    & (Join-Path $Here 'run_verify.ps1') -Stage $Stage -RepoRoot $Repo -PreviousRoot $Old -PriorStudy $Prior -DataRoot $Data -AuditRoot $Audit -BaselineRoot $Baseline -EvidenceRoot $Evidence -OutputRoot $Output
}
try {
    Run 'all'
    $Calls=@($global:StabilityCalls.ToArray())
    Assert ($Calls.Count -eq 7) 'Graph+four training jobs+two CPU jobs'
    Assert (@($Calls | Where-Object {$_ -contains '--gpus'}).Count -eq 5) 'Exactly five GPU jobs'
    Assert ($Calls[0] -contains 'plan') 'Validate old evidence before GPU jobs'
    Assert ($Calls[1] -contains 'graph') 'Graph isolation before any updates'
    Assert ($Calls[6] -contains 'report') 'Finish with CPU report'
    foreach ($c in $Calls) {
        Assert (($c -contains 'never') -and ($c -contains 'none')) 'No pull/network'
        Assert (-not ($c -match 'validation-10|test-38|worker.py|/verify.py')) 'No evaluation/old GPU reruns'
        foreach ($target in @('checkout','previous','baseline','weights','evidence')) {
            Assert (@($c | Where-Object {$_ -like "*target=/$target,readonly"}).Count -eq 1) 'Inputs must be read-only'
        }
        if ($c -contains '--gpus') {
            if ($c -contains 'graph') { Assert ($c -contains '1200s') 'Graph timeout' }
            else { Assert ($c -contains '7200s') 'Bounded 100-update training timeout' }
            Assert (@($c | Where-Object {$_ -like '*train-40*target=/data,readonly'}).Count -eq 1) 'Train-40 only'
        } else {
            Assert ($c -contains '3600s') 'CPU timeout'
            Assert (-not ($c -match 'target=/data|target=/lld,')) 'CPU plan/report no raw data'
        }
    }
    foreach ($arm in @('F0','Fprobe','F001','F010')) {
        Assert (@($Calls | Where-Object {$_ -contains $arm}).Count -eq 1) 'Each arm once'
    }
    $global:StabilityCalls.Clear(); Run 'report'
    Assert ($global:StabilityCalls.Count -eq 1) 'Report only CPU'
    Assert (-not ($global:StabilityCalls[0] -contains '--gpus')) 'Report cannot initialize GPU'
    $global:StabilityCalls.Clear(); Run 'graph'
    Assert ($global:StabilityCalls.Count -eq 2) 'Graph-only stops before training'
    $global:StabilityCalls.Clear(); Run 'plan'
    Assert ($global:StabilityCalls.Count -eq 1) 'Plan-only stops before GPU'
    $global:StabilityCalls.Clear(); $global:StabilityFailAt=2
    $Failed=$false
    try { Run 'all' } catch { $Failed=$true }
    Assert $Failed 'Graph failure surfaced'
    Assert ($global:StabilityCalls.Count -eq 2) 'Graph failure blocks every training job'
    $global:StabilityCalls.Clear(); $global:StabilityFailAt=3
    $Failed=$false
    try { Run 'all' } catch { $Failed=$true }
    Assert $Failed 'Training failure surfaced'
    Assert ($global:StabilityCalls.Count -eq 3) 'Training failure blocks later arms'
    $Failed=$false
    try {
        & (Join-Path $Here 'run_verify.ps1') -Stage plan -RepoRoot $Repo -PreviousRoot $Old -PriorStudy $Prior -AuditRoot $Audit -BaselineRoot $Baseline -EvidenceRoot $Evidence -OutputRoot $Evidence
    } catch { $Failed=$true }
    Assert $Failed 'Output/evidence overlap refused'
    Write-Host 'RUNNER TESTS PASSED (mock Docker, no data/GPU access)'
} finally {
    Remove-Item Function:\docker
    $Resolved=[IO.Path]::GetFullPath($Fixture)
    $Parent=[IO.Path]::GetFullPath($TempBase).TrimEnd('\')+'\'
    if ($Resolved.StartsWith($Parent,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path $Resolved -Leaf) -match '^st[0-9a-f]{6}$') {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    } else { throw 'Unsafe test fixture cleanup refused' }
}
