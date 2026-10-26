param([string]$TempRoot=(Join-Path ([IO.Path]::GetTempPath()) 'acdc-qc-runner-tests'))
$ErrorActionPreference='Stop'
$Runner=Join-Path $PSScriptRoot 'run_audit.ps1'
$Tokens=$null; $Errors=$null
[void][System.Management.Automation.Language.Parser]::ParseFile($Runner,[ref]$Tokens,[ref]$Errors)
if ($Errors.Count) { throw ($Errors | Out-String) }
$Root=Join-Path $TempRoot ([Guid]::NewGuid().ToString('N').Substring(0,8))
foreach ($Part in @('data','audit','study','prior')) { New-Item -ItemType Directory -Force -Path (Join-Path $Root $Part) | Out-Null }
foreach ($Seed in 0,1,2) {
    $SeedRoot=Join-Path $Root "study\seed_$Seed"
    New-Item -ItemType Directory -Force -Path (Join-Path $SeedRoot 'full\labels'),(Join-Path $SeedRoot 'full\external\B1') | Out-Null
    foreach ($File in @('manifest.json','full\external\B1\steps.json','full\external\B1\verified.json')) {
        [IO.File]::WriteAllText((Join-Path $SeedRoot $File),'{}')
    }
}
$global:QcImage='sha256:'+('a'*64)
$global:QcCalls=New-Object System.Collections.Generic.List[object]
$global:QcFailSmoke=$false
function docker {
    $Call=@($args)
    $global:QcCalls.Add($Call)
    $global:LASTEXITCODE=0
    if ($Call[0] -eq 'info') { Write-Output 'linux'; return }
    if ($Call[0] -eq 'image') { Write-Output $global:QcImage; return }
    if ($Call[0] -ne 'run') { throw 'Forbidden Docker action' }
    if ($global:QcFailSmoke -and $Call -contains 'unittest') {
        $global:LASTEXITCODE=1
        Write-Output 'FAILED mock smoke'
        return
    }
    Write-Output 'COMPLETE mock worker'
}
$Params=@{DataRoot=(Join-Path $Root 'data'); AuditRoot=(Join-Path $Root 'audit');
    StudyRoot=(Join-Path $Root 'study'); OutputRoot=(Join-Path $Root 'out'); RuntimeImage=$global:QcImage}
& $Runner @Params -Stage all
$Runs=@($global:QcCalls | Where-Object { $_[0] -eq 'run' })
if ($Runs.Count -ne 3) { throw 'Expected dependency check, smoke then audit' }
if ($Runs[1] -notcontains 'unittest' -or $Runs[2] -notcontains '/package/audit.py') { throw 'Incorrect smoke gate ordering' }
foreach ($Call in $Runs) {
    foreach ($Token in @('--network','none','--pull','never','--read-only')) {
        if ($Call -notcontains $Token) { throw "Missing restriction $Token" }
    }
    if ($Call -contains '--gpus' -or ($Call -join ' ') -match 'test-38|training-checkpoints|build|train_transfer') { throw 'Unexpected GPU/training/test mount' }
    $Mounts=@($Call | Where-Object { $_ -like 'type=bind,*' })
    foreach ($Mount in $Mounts) {
        if ($Mount -notlike '*target=/output' -and $Mount -notlike '*,readonly') { throw 'Input not read-only' }
        if ($Mount -match 'target=/cache/seed_\d$|target=/study') { throw 'Broad study mount forbidden' }
    }
    if (@($Mounts | Where-Object { $_ -like '*target=/cache/*' }).Count -ne 12) { throw 'Expected only 4 external artifacts per seed' }
}
$global:QcCalls.Clear()
$global:QcFailSmoke=$true
$Failed=$false
try { & $Runner @Params -Stage all } catch { $Failed=$true }
if (-not $Failed -or @($global:QcCalls | Where-Object { $_ -contains '/package/audit.py' }).Count) { throw 'Failed smoke must stop audit' }
$global:QcFailSmoke=$false
$global:QcCalls.Clear()
$Params.StudyRoot=Join-Path $Root 'MUST-NOT-RESOLVE'
& $Runner @Params -Stage raw
if (@($global:QcCalls | Where-Object { ($_ -join ' ') -match 'target=/cache/' }).Count) { throw 'Raw stage should not mount labels' }
$global:QcCalls.Clear()
$Params.OutputRoot=Join-Path $Root 'data\nested-output'
$Failed=$false
try { & $Runner @Params -Stage raw } catch { $Failed=$true }
if (-not $Failed -or $global:QcCalls.Count) { throw 'Overlapping output must fail before Docker' }
Write-Host 'RUNNER TESTS PASSED: parse, readonly scope, default flow, failure gate, raw independence, overlap guard.'
Write-Host "Mock fixtures retained: $Root"
