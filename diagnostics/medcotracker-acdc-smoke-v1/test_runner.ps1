param([string]$TempRoot=[IO.Path]::GetTempPath())
$ErrorActionPreference='Stop'
$TestRoot=Join-Path $TempRoot ('acdc-smoke-test-'+[Guid]::NewGuid().ToString('N'))
$Repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
New-Item -ItemType Directory -Force -Path $TestRoot,(Join-Path $TestRoot 'data'),(Join-Path $TestRoot 'audit'),(Join-Path $TestRoot 'weights') | Out-Null
[IO.File]::WriteAllText((Join-Path $TestRoot 'audit\data-manifest.json'),'{}')
foreach ($Name in @('cotracker2v1.pth','baseline_online.pth','scaled_offline.pth')) {
    [IO.File]::WriteAllText((Join-Path $TestRoot ('weights\'+$Name)),'fixture only')
}
$global:SmokeCalls=New-Object System.Collections.Generic.List[object]
$global:SmokeImage='sha256:'+('a'*64)
$global:SmokeFail=$false
$global:SmokeOutput=Join-Path $TestRoot 'output'
function docker {
    $Call=@($args)
    $global:SmokeCalls.Add($Call)
    $global:LASTEXITCODE=0
    if ($Call[0] -eq 'info') { Write-Output 'linux'; return }
    if ($Call[0] -eq 'image' -and $Call[1] -eq 'inspect') { Write-Output $global:SmokeImage; return }
    if ($Call[0] -eq 'run') {
        if ($global:SmokeFail) { $global:LASTEXITCODE=1; Write-Output 'MOCK FAIL'; return }
        [IO.File]::WriteAllText((Join-Path $global:SmokeOutput 'summary.json'),'{"complete":true,"stage":"external-gpu-smoke","trackrad_training_started":false,"pilot_started":false}')
        Write-Output 'MOCK SMOKE PASSED'; return
    }
    throw 'Unexpected Docker action'
}
$Parameters=@{RepoRoot=$Repo; DataRoot=(Join-Path $TestRoot 'data'); AuditRoot=(Join-Path $TestRoot 'audit');
              WeightsRoot=(Join-Path $TestRoot 'weights'); OutputRoot=$global:SmokeOutput; RuntimeImage=$global:SmokeImage}
& (Join-Path $PSScriptRoot 'run_smoke.ps1') @Parameters
$Runs=@($global:SmokeCalls | Where-Object { $_[0] -eq 'run' })
if ($Runs.Count -ne 1) { throw 'Only one bounded smoke worker allowed' }
$Call=$Runs[0]
foreach ($Token in @('--pull','never','--network','none','--gpus','all','1800s')) {
    if ($Call -notcontains $Token) { throw "Missing safety/runtime token $Token" }
}
if (($Call -join ' ') -match 'trackrad2025|test-38|build|download') { throw 'Unexpected dataset/network/build scope' }
foreach ($Target in @('/checkout','/data','/audit','/weights')) {
    if (-not ($Call | Where-Object { $_ -like "*target=$Target,readonly" })) { throw "Missing readonly $Target" }
}
$global:SmokeCalls.Clear()
$global:SmokeFail=$true
$Failed=$false
try { & (Join-Path $PSScriptRoot 'run_smoke.ps1') @Parameters } catch { $Failed=$true }
if (-not $Failed -or @($global:SmokeCalls | Where-Object { $_[0] -eq 'run' }).Count -ne 1) {
    throw 'Failure must stop immediately; stale success marker must not mask failure'
}
Write-Host 'SMOKE POWERSHELL MOCK PASSED (no Docker/GPU/network calls)'
