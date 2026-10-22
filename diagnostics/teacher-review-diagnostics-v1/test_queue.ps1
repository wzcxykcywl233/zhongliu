# Mock Docker orchestration. No network, GPU, images or real dataset required.
param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$Repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Fixture=Join-Path $TemporaryRoot ('tr'+[Guid]::NewGuid().ToString('N').Substring(0,6))
$Checkout=Join-Path $Fixture 'repo'
$Prior=Join-Path $Fixture 'prior'
$Collab=Join-Path $Fixture 'collab'
$Output=Join-Path $Fixture 'out'
$Utf8=New-Object System.Text.UTF8Encoding($false)
function Save-Fixture([string]$Path,[string]$Text) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Path -Parent) | Out-Null
    [IO.File]::WriteAllText($Path,$Text,$Utf8)
}
Save-Fixture (Join-Path $Checkout 'scripts\experiment_local_runtimes.ps1') (Get-Content -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Raw)
New-Item -ItemType Directory -Force -Path (Join-Path $Checkout 'training-checkpoints') | Out-Null
$Data=@()
foreach ($Pair in @(@{Split='train-40';Case='A'},@{Split='validation-10';Case='B'})) {
    $File=Join-Path $Fixture "$($Pair.Split)\$($Pair.Case)\file.mha"
    Save-Fixture $File 'fixture'
    $Data+=@{Split=$Pair.Split;Case=$Pair.Case;File=$File}
}
Save-Fixture (Join-Path $Prior 'frozen-study.json') (ConvertTo-Json -Depth 8 -InputObject @{Dataset=$Data})
Save-Fixture (Join-Path $Collab 'manifest.json') (ConvertTo-Json -InputObject @{images=@{Training=('sha256:'+('a'*64))}})
$global:TrCalls=New-Object System.Collections.Generic.List[string]
$global:TrFail=''
$global:TrSmoke=$false
$global:TrFull=$false
function global:docker {
    $Values=@($args)
    $global:LASTEXITCODE=0
    if ($Values[0] -eq 'image') { return 'sha256:'+('a'*64) }
    if ($Values[0] -ne 'run' -or $Values -notcontains 'never' -or $Values -notcontains 'none') { throw 'Network/build not allowed.' }
    if (@($Values | Where-Object {$_ -match 'target=/datasets/test'}).Count) { throw 'Test data mounted.' }
    if ($Values -contains '/package/train.py') { throw 'Training launched.' }
    foreach ($Mount in @($Values | Where-Object {$_ -like 'type=bind,*'})) {
        if ($Mount -notlike '*,target=/study' -and $Mount -notlike '*,readonly') { throw 'Input not read-only.' }
    }
    $Index=[array]::IndexOf($Values,'/package/diagnose.py')
    if ($Index -lt 0) { throw 'Wrong worker.' }
    $Command=$Values[$Index+1]
    $global:TrCalls.Add($Command)
    if ($Command -in @('smoke','full') -and $Values -notcontains '--gpus') { throw 'GPU missing.' }
    if ($Command -in @('plan','report') -and $Values -contains '--gpus') { throw 'CPU stage unexpectedly requested GPU.' }
    if ($global:TrFail -eq $Command) { $global:LASTEXITCODE=7; return 'Synthetic failure.' }
    if ($Command -eq 'smoke') { $global:TrSmoke=$true }
    if ($Command -eq 'full') {
        if (-not $global:TrSmoke) { $global:LASTEXITCODE=8; return 'Missing smoke.' }
        $global:TrFull=$true
    }
    if ($Command -eq 'report' -and -not $global:TrFull) { $global:LASTEXITCODE=9; return 'Incomplete full probes.' }
    return "PROBES COMPLETE $Command"
}
$Runner=Join-Path $PSScriptRoot 'run_diagnosis.ps1'
$Common=@{RepoRoot=$Checkout;PriorStudy=$Prior;CollaborationRoot=$Collab;OutputRoot=$Output}
function Expect-Failure([string]$Stage) {
    $Failed=$false
    try { & $Runner @Common -Stage $Stage } catch { $Failed=$true }
    if (-not $Failed) { throw 'Expected failure did not stop queue.' }
}
try {
    $global:TrFail='smoke'
    Expect-Failure 'all'
    if (($global:TrCalls -join ',') -ne 'smoke') { throw 'Smoke failure did not stop full probes.' }
    $global:TrCalls.Clear(); $global:TrFail='full'
    Expect-Failure 'all'
    if (($global:TrCalls -join ',') -ne 'smoke,full') { throw 'Full failure did not stop report.' }
    $global:TrCalls.Clear(); $global:TrFail=''
    & $Runner @Common -Stage all
    if (($global:TrCalls -join ',') -ne 'smoke,full,report') { throw 'Wrong automatic order.' }
    $global:TrCalls.Clear()
    & $Runner @Common -Stage full
    if (($global:TrCalls -join ',') -ne 'full,report') { throw 'Wrong continuation order.' }
    $global:TrCalls.Clear()
    & $Runner @Common -Stage report
    if (($global:TrCalls -join ',') -ne 'report') { throw 'Report triggered inference.' }
    Write-Host 'MOCK QUEUE PASSED: stop on failures; no training/test/build/network; read-only inputs.'
} finally { Remove-Item Function:\docker }
