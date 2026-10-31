$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
$TempBase=if ($env:COTRACKER_TEST_TMP) {$env:COTRACKER_TEST_TMP} else {$env:TEMP}
$Fixture=Join-Path $TempBase ('da'+[guid]::NewGuid().ToString('N').Substring(0,5))
$Repo=Join-Path $Fixture 'repo'; $Old=Join-Path $Fixture 'old'; $Prior=Join-Path $Fixture 'prior'
$Ac=Join-Path $Fixture 'ac'; $Audit=Join-Path $Fixture 'audit'
$Baseline=Join-Path $Fixture 'base'; $Diagnostic=Join-Path $Fixture 'diag'
foreach ($Path in @($Repo,$Old,$Prior,$Ac,$Audit,$Baseline,$Diagnostic,(Join-Path $Repo 'scripts'),(Join-Path $Repo 'training-checkpoints'))) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
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
$global:DomainCalls=[Collections.Generic.List[object]]::new()
$global:DomainGate=$true; $global:DomainFailSmoke=$false; $global:DiagnosticFail=$false
function global:docker {
    $a=@($args); $global:DomainCalls.Add($a); $global:LASTEXITCODE=0
    if (($a -contains 'audit') -and ($a -contains 'smoke') -and $global:DomainFailSmoke) {
        $global:LASTEXITCODE=1; 'Mock zero-control audit failed'; return
    }
    if (($a -contains '/checkout/diagnostics/lld-fnet-runtime-fix-v1/diagnose.py') -and ($a -contains 'report')) {
        if ($global:DiagnosticFail) { $global:LASTEXITCODE=1; 'RUNTIME GATE CLOSED'; return }
        Set-Content -LiteralPath (Join-Path $Diagnostic 'verified.json') -Value '{}' -Encoding UTF8
    }
    if ($a -contains 'roster') {
        $Mount=@($a | Where-Object {$_ -like '*target=/study'})[0]
        $Root=$Mount.Substring(17,$Mount.Length-17-14)
        $Arms=if ($global:DomainGate) {@('M0','F0','S001','F010')} else {@()}
        @{arms=@($Arms)} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'full-roster.json') -Encoding UTF8
    }
    if ($a -contains '/transport.py') { 'REGION_METRICS_BASE64=e30=' }
    else { 'CACHE VERIFIED mock' }
}
function Assert([bool]$Value,[string]$Message) { if (-not $Value) { throw $Message } }
function Run-Stage([string]$Mode,[string]$Name) {
    & (Join-Path $Here 'run_repair.ps1') -Stage $Mode -RepoRoot $Repo -PreviousRoot $Old -PriorStudy $Prior -DataRoot $Ac -AuditRoot $Audit -OutputRoot (Join-Path $Fixture $Name) -BaselineRoot $Baseline -DiagnosticRoot $Diagnostic
}
try {
    Run-Stage 'all' 'out'
    $Calls=@($global:DomainCalls.ToArray())
    Assert ($Calls[0] -contains 'probe') 'Runtime probe before training'
    $Probes=@($Calls | Where-Object {$_ -contains '/checkout/diagnostics/lld-fnet-runtime-fix-v1/diagnose.py'})
    Assert ($Probes.Count -eq 9) '8 bounded GPU jobs + report'
    foreach ($Call in $Probes) {
        Assert (-not ($Call -match 'validation-10|test-38|target=/study')) 'Diagnostic cannot access evaluation or production output'
        Assert ($Call -contains '900s') 'Each runtime job bounded'
    }
    $Worker=@($Calls | Where-Object {$_ -contains '/checkout/diagnostics/lld-fnet-runtime-fix-v1/worker.py'})
    Assert (@($Worker | Where-Object {$_ -contains 'train'}).Count -eq 28) '8 smoke + 8 pilot + 12 full jobs'
    Assert (@($Worker | Where-Object {$_ -contains 'infer'}).Count -eq 30) '8 smoke + 9 pilot + 13 full endpoints'
    foreach ($Call in $Calls) {
        Assert (-not ($Call -match 'test-38')) 'all cannot mount or run test'
        Assert (($Call -contains 'none') -and ($Call -contains 'never')) 'No network/pull'
    }
    foreach ($Call in $Worker) {
        Assert (@($Call | Where-Object {$_ -like '*target=/previous,readonly'}).Count -eq 1) 'Old results read-only'
        Assert (@($Call | Where-Object {$_ -like '*target=/baseline,readonly'}).Count -eq 1) 'Failed study read-only'
        Assert (@($Call | Where-Object {$_ -like '*target=/diagnostic,readonly'}).Count -eq 1) 'Production cannot change diagnostic'
        if ($Call -contains 'train') {
            Assert (@($Call | Where-Object {$_ -like '*train-40*target=/data,readonly'}).Count -eq 1) 'Training gets train40'
            Assert (-not ($Call -match 'validation-10|test-38')) 'Training cannot access evaluation split'
        }
    }
    $global:DomainCalls.Clear(); $global:DomainGate=$false
    Run-Stage 'all' 'closed'
    Assert (@($global:DomainCalls.ToArray() | Where-Object {$_ -contains 'train'}).Count -eq 16) 'Closed gate skips large jobs'
    $global:DomainCalls.Clear(); $global:DomainGate=$true; $global:DomainFailSmoke=$true
    $Failed=$false
    try { Run-Stage 'all' 'failed' } catch { $Failed=$true }
    Assert $Failed 'Smoke failure must stop'
    Assert (@($global:DomainCalls.ToArray() | Where-Object {($_ -contains 'pilot') -or ($_ -contains 'full')}).Count -eq 0) 'No later stages after audit failure'
    $global:DomainCalls.Clear(); $global:DomainFailSmoke=$false
    Run-Stage 'test' 'test'
    $Calls=@($global:DomainCalls.ToArray())
    Assert (@($Calls | Where-Object {$_ -contains 'train'}).Count -eq 0) 'Test never trains'
    Assert (@($Calls | Where-Object {$_ -contains 'test-gate'}).Count -eq 1) 'Test gate before inference'
    Assert (@($Calls | Where-Object {$_ -contains 'infer'}).Count -eq 13) 'Frozen test roster'
    Assert (@($Calls | Where-Object {$_ -match 'train-40|validation-10|target=/lld'}).Count -eq 0) 'Test mounts only test'
    $global:DomainCalls.Clear(); $global:DiagnosticFail=$true
    $Failed=$false
    try { Run-Stage 'all' 'badgate' } catch { $Failed=$true }
    Assert $Failed 'Failed runtime gate must stop'
    Assert (@($global:DomainCalls.ToArray() | Where-Object {$_ -contains '/checkout/diagnostics/lld-fnet-runtime-fix-v1/worker.py'}).Count -eq 0) 'No production job before runtime gate'
    $global:DomainCalls.Clear(); $global:DiagnosticFail=$false
    Run-Stage 'diagnose' 'only'
    Assert (@($global:DomainCalls.ToArray() | Where-Object {$_ -contains '/checkout/diagnostics/lld-fnet-runtime-fix-v1/worker.py'}).Count -eq 0) 'diagnose never trains production'
    $Failed=$false
    try {
        & (Join-Path $Here 'run_repair.ps1') -Stage diagnose -RepoRoot $Repo -PreviousRoot $Old -PriorStudy $Prior -DataRoot $Ac -AuditRoot $Audit -BaselineRoot $Baseline -OutputRoot (Join-Path $Fixture 'overlap') -DiagnosticRoot $Baseline
    } catch { $Failed=$true }
    Assert $Failed 'Overlapping diagnostic output forbidden'
    Write-Host 'RUNNER TESTS PASSED (mock Docker; no GPU/data accessed)'
} finally {
    Remove-Item Function:\docker
    $Resolved=[IO.Path]::GetFullPath($Fixture)
    $Parent=[IO.Path]::GetFullPath($TempBase).TrimEnd('\')+'\'
    if ($Resolved.StartsWith($Parent,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path $Resolved -Leaf) -match '^da[0-9a-f]{5}$') {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    } else { throw 'Unsafe test-fixture cleanup refused' }
}
