$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
$TempBase=if ($env:COTRACKER_TEST_TMP) {$env:COTRACKER_TEST_TMP} else {$env:TEMP}
$Fixture=Join-Path $TempBase ('df'+[guid]::NewGuid().ToString('N').Substring(0,5))
$Repo=Join-Path $Fixture 'repo'; $Old=Join-Path $Fixture 'old'; $Prior=Join-Path $Fixture 'prior'
$Data=Join-Path $Fixture 'lld'; $Audit=Join-Path $Fixture 'audit'; $Baseline=Join-Path $Fixture 'base'
$Evidence=Join-Path $Fixture 'evidence'; $Stability=Join-Path $Fixture 'stable'
foreach ($p in @($Repo,$Old,$Prior,$Data,$Audit,$Baseline,$Evidence,$Stability,(Join-Path $Repo 'scripts'),(Join-Path $Repo 'training-checkpoints'))) {
    New-Item -ItemType Directory -Path $p -Force | Out-Null
}
Set-Content -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Value 'function Get-LocalImageId([string]$Image) { return $Image }' -Encoding UTF8
$Image='sha256:'+('a'*64)
@{images=@{Training=$Image;Evaluation=$Image}} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Old 'manifest.json') -Encoding UTF8
$Rows=@()
foreach ($Split in @('train-40','validation-10','test-38')) {
    $Dir=Join-Path $Fixture $Split; New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    $Rows+=@{Split=$Split;Case='A_001';File=(Join-Path $Dir 'A_001\images\A_001_frames.mha')}
}
@{Dataset=$Rows} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Prior 'frozen-study.json') -Encoding UTF8
Set-Content -LiteralPath (Join-Path $Audit 'data-manifest.json') -Value '{}' -Encoding UTF8
$global:FullCalls=[Collections.Generic.List[object]]::new(); $global:FullFail=''
function global:docker {
    $a=@($args); $global:FullCalls.Add($a); $global:LASTEXITCODE=0
    if ($global:FullFail -and ($a -contains $global:FullFail)) {
        $global:LASTEXITCODE=1; 'Mock failure'; return
    }
    if ($a -contains '/transport.py') { 'REGION_METRICS_BASE64=e30=' }
    else { 'CACHE VERIFIED mock' }
}
function Assert([bool]$Value,[string]$Message) { if (-not $Value) { throw $Message } }
function Run-Stage([string]$Stage,[string]$Name) {
    & (Join-Path $Here 'run_experiment.ps1') -Stage $Stage -RepoRoot $Repo -PreviousRoot $Old -PriorStudy $Prior -DataRoot $Data -AuditRoot $Audit -BaselineRoot $Baseline -EvidenceRoot $Evidence -StabilityRoot $Stability -OutputRoot (Join-Path $Fixture $Name)
}
try {
    Run-Stage 'all' 'out'
    $Calls=@($global:FullCalls.ToArray()); $Worker=@($Calls | Where-Object {$_ -contains '/checkout/diagnostics/lld-direct-full-v1/worker.py'})
    Assert ($Calls[0] -contains 'plan') 'Stability binding verified first'
    Assert (@($Worker | Where-Object {$_ -contains 'train'}).Count -eq 15) 'Exactly5arms x3seed full jobs'
    Assert (@($Worker | Where-Object {$_ -contains 'infer'}).Count -eq 16) '15 students +1 official reference'
    Assert (@($Calls | Where-Object {$_ -contains '/transport.py'}).Count -eq 16) '16 official evaluations'
    foreach ($c in $Calls) {
        Assert (-not ($c -match 'test-38|--phase')) 'all cannot access test or obsolete phase option'
        Assert (($c -contains 'none') -and ($c -contains 'never')) 'Offline/no pull'
        Assert (-not ($c -contains 'pilot')) 'No pilot rerun'
    }
    foreach ($c in $Worker) {
        foreach ($target in @('checkout','previous','baseline','evidence','stability','weights')) {
            Assert (@($c | Where-Object {$_ -like "*target=/$target,readonly"}).Count -eq 1) 'Immutable prior inputs'
        }
        if ($c -contains 'train') {
            Assert ($c -contains '43200s') 'Full jobs bounded by12h'
            Assert (@($c | Where-Object {$_ -like '*train-40*target=/data,readonly'}).Count -eq 1) 'Train gets train40 only'
            Assert (-not ($c -match 'validation-10|test-38')) 'Training cannot read heldout images'
        } elseif ($c -contains 'infer') {
            Assert (-not ($c -match 'target=/lld|train-40')) 'Inference has no training raw data'
        }
    }
    foreach ($Arm in @('M0','F0','Fprobe','F001','F010')) {
        Assert (@($Worker | Where-Object {($_ -contains 'train') -and ($_ -contains $Arm)}).Count -eq 3) 'Every arm all3seeds'
    }
    $global:FullCalls.Clear(); Run-Stage 'train' 'train'
    Assert (@($global:FullCalls.ToArray() | Where-Object {$_ -contains 'train'}).Count -eq 15) 'train only15'
    Assert (@($global:FullCalls.ToArray() | Where-Object {$_ -contains 'infer'}).Count -eq 0) 'train stage does not evaluate'
    $global:FullCalls.Clear(); Run-Stage 'validation' 'val'
    Assert (@($global:FullCalls.ToArray() | Where-Object {$_ -contains 'train'}).Count -eq 0) 'validation never trains'
    $global:FullCalls.Clear(); Run-Stage 'test' 'test'
    $Calls=@($global:FullCalls.ToArray())
    Assert (@($Calls | Where-Object {$_ -contains 'train'}).Count -eq 0) 'Test never trains'
    Assert (@($Calls | Where-Object {$_ -contains 'test-gate'}).Count -eq 1) 'Frozen test gate required'
    Assert (@($Calls | Where-Object {$_ -contains 'infer'}).Count -eq 16) 'All5 predefined arms tested'
    Assert (@($Calls | Where-Object {$_ -match 'train-40|validation-10|target=/lld'}).Count -eq 0) 'Test mounts only test raw data'
    $global:FullCalls.Clear(); $global:FullFail='plan'; $Failed=$false
    try { Run-Stage 'all' 'fail' } catch { $Failed=$true }
    Assert $Failed 'Stale evidence stops queue'; Assert ($global:FullCalls.Count -eq 1) 'No work after failed evidence binding'
    $global:FullCalls.Clear(); $global:FullFail='test-gate'; $Failed=$false
    try { Run-Stage 'test' 'failtest' } catch { $Failed=$true }
    Assert $Failed 'Test gate failure stops'; Assert (@($global:FullCalls.ToArray() | Where-Object {$_ -contains 'infer'}).Count -eq 0) 'No ungated test inference'
    Write-Host 'FULL RUNNER TESTS PASSED (mock Docker, no GPU or dataset access)'
} finally {
    Remove-Item Function:\docker
    $Resolved=[IO.Path]::GetFullPath($Fixture); $Parent=[IO.Path]::GetFullPath($TempBase).TrimEnd('\')+'\'
    if ($Resolved.StartsWith($Parent,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path $Resolved -Leaf) -match '^df[0-9a-f]{5}$') {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    } else { throw 'Unsafe test fixture cleanup refused' }
}
