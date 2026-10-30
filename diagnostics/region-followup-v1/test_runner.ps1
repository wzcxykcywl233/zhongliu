$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
$TempBase=if ($env:COTRACKER_TEST_TMP) {$env:COTRACKER_TEST_TMP} else {$env:TEMP}
$Fixture=Join-Path $TempBase ('rf'+[guid]::NewGuid().ToString('N').Substring(0,5))
$Repo=Join-Path $Fixture 'repo'; $Previous=Join-Path $Fixture 'old'; $Prior=Join-Path $Fixture 'prior'
$Ac=Join-Path $Fixture 'ac'; $Audit=Join-Path $Fixture 'audit'
foreach ($Path in @($Repo,$Previous,$Prior,$Ac,$Audit,(Join-Path $Repo 'scripts'),(Join-Path $Repo 'training-checkpoints'))) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
}
$Runtime='function Get-LocalImageId([string]$Image) { return $Image }'
Set-Content -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Value $Runtime -Encoding UTF8
$Image='sha256:'+('a'*64)
@{images=@{Training=$Image;Evaluation=$Image}} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Previous 'manifest.json') -Encoding UTF8
$Rows=@()
foreach ($Split in @('train-40','validation-10','test-38')) {
    $Dir=Join-Path $Fixture $Split
    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    $Rows+=@{Split=$Split;Case='A_001';File=(Join-Path $Dir 'A_001\images\A_001_frames.mha')}
}
@{Dataset=$Rows} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Prior 'frozen-study.json') -Encoding UTF8
Set-Content -LiteralPath (Join-Path $Audit 'data-manifest.json') -Value '{}' -Encoding UTF8
$global:FollowupCalls=[Collections.Generic.List[object]]::new()
$global:FollowupFailDiagnosis=$false
function global:docker {
    $a=@($args); $global:FollowupCalls.Add($a)
    $global:LASTEXITCODE=0
    if ($a -contains 'diagnose' -and $global:FollowupFailDiagnosis) { $global:LASTEXITCODE=1; 'Mock diagnosis failure'; return }
    if ($a -contains '/transport.py') { 'REGION_METRICS_BASE64=e30=' }
    else { 'CACHE VERIFIED simulated worker' }
}
function Assert([bool]$Value,[string]$Message) { if (-not $Value) { throw $Message } }
function Run-Stage([string]$Mode,[string]$Out) {
    & (Join-Path $Here 'run_experiment.ps1') -Stage $Mode -RepoRoot $Repo -PreviousRoot $Previous -PriorStudy $Prior -DataRoot $Ac -AuditRoot $Audit -OutputRoot $Out
}
try {
    $Out=Join-Path $Fixture 'out'
    Run-Stage 'all' $Out
    $Calls=@($global:FollowupCalls.ToArray())
    Assert ($Calls[0] -contains 'plan') 'Plan must be first'
    $Worker=@($Calls | Where-Object { $_ -contains '/checkout/diagnostics/region-followup-v1/worker.py' })
    foreach ($Call in $Worker) {
        Assert (@($Call | Where-Object {$_ -like '*target=/previous,readonly'}).Count -eq 1) 'Old results must be read-only'
        Assert (($Call -contains 'none') -and ($Call -contains 'never')) 'No network/pull'
        if ($Call -contains 'train') {
            Assert (-not ($Call -match 'test-38')) 'Training cannot mount test split'
            Assert (@($Call | Where-Object {$_ -like '*train-40*target=/data,readonly'}).Count -eq 1) 'G1 gets train only'
        }
        if ($Call -contains 'diagnose') {
            Assert (-not ($Call -match 'test-38|target=/acdc|train-40')) 'Diagnosis only validation'
        }
    }
    Assert (@($Worker | Where-Object {$_ -contains 'train'}).Count -eq 3) 'Only three G1 training jobs'
    Assert (@($Worker | Where-Object {$_ -contains 'diagnose'}).Count -eq 3) 'Three diagnosis seeds'
    Assert (@($Worker | Where-Object {$_ -contains 'infer'}).Count -eq 13) 'Ten point evaluations and three G1 validation evaluations'
    foreach ($Call in @($Worker | Where-Object {($_ -contains 'infer') -and ($_ -contains 'G1')})) {
        Assert (-not ($Call -match 'test-38')) 'G1 never tests automatically'
    }
    $global:FollowupCalls.Clear(); $global:FollowupFailDiagnosis=$true
    $Failed=$false
    try { Run-Stage 'all' (Join-Path $Fixture 'fail') } catch { $Failed=$true }
    Assert $Failed 'Failed branch must leave nonzero final outcome'
    Assert (@($global:FollowupCalls.ToArray() | Where-Object {$_ -contains 'train'}).Count -eq 3) 'Diagnosis failure must not stop independent G1 branch'
    $global:FollowupCalls.Clear(); $global:FollowupFailDiagnosis=$false
    Run-Stage 'diagnose' (Join-Path $Fixture 'diag')
    Assert (@($global:FollowupCalls.ToArray() | Where-Object {$_ -match 'test-38|train-40|target=/acdc'}).Count -eq 0) 'Standalone diagnosis cannot access training/test datasets'
    Write-Host 'RUNNER TESTS PASSED (mock Docker; no GPU or data access)'
} finally {
    Remove-Item Function:\docker
    # Test fixtures only; avoid recursive cleanup of any unverified/broad path.
    $Resolved=[IO.Path]::GetFullPath($Fixture)
    $ExpectedParent=[IO.Path]::GetFullPath($TempBase).TrimEnd('\')+'\'
    if ($Resolved.StartsWith($ExpectedParent,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path $Resolved -Leaf) -match '^rf[0-9a-f]{5}$') {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    } else { throw 'Refusing unsafe test-fixture cleanup' }
}
