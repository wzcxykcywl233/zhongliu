param([string]$TempRoot='D:\yijian\tmp')
$ErrorActionPreference='Stop'
$Runner=Join-Path $PSScriptRoot 'run_experiment.ps1'
$Tokens=$null; $Errors=$null
[System.Management.Automation.Language.Parser]::ParseFile($Runner,[ref]$Tokens,[ref]$Errors) | Out-Null
if ($Errors.Count) { throw 'PowerShell syntax error' }
$TestRoot=Join-Path $TempRoot ('r'+[Guid]::NewGuid().ToString('N').Substring(0,5))
foreach ($Part in @('repo\scripts','repo\training-checkpoints','prior','data','audit','train\A','val\B')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $TestRoot $Part) | Out-Null
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '..\..\scripts\experiment_local_runtimes.ps1') -Destination (Join-Path $TestRoot 'repo\scripts\experiment_local_runtimes.ps1')
$Inventory=@{Dataset=@(
    @{Split='train-40';Case='A';File=(Join-Path $TestRoot 'train\A\frames.mha')},
    @{Split='validation-10';Case='B';File=(Join-Path $TestRoot 'val\B\frames.mha')},
    @{Split='test-38';Case='X';File='Z:\must-not-resolve\X\frames.mha'})}
[IO.File]::WriteAllText((Join-Path $TestRoot 'prior\frozen-study.json'),($Inventory | ConvertTo-Json -Depth 5))
$global:RegionMockImage='sha256:'+('a'*64)
[IO.File]::WriteAllText((Join-Path $TestRoot 'prior\local-runtime-images.json'),(@{Training=$global:RegionMockImage;Evaluation=$global:RegionMockImage} | ConvertTo-Json))
$global:RegionCalls=New-Object System.Collections.Generic.List[object]
$global:RegionFail=''; $global:RegionGate=$false
function docker {
    $Call=@($args); $global:RegionCalls.Add($Call); $global:LASTEXITCODE=0
    if ($Call[0] -eq 'image') { Write-Output $global:RegionMockImage; return }
    if ($Call[0] -ne 'run') { throw 'Unexpected Docker mutation' }
    $Joined=$Call -join ' '
    if (($global:RegionFail -eq 'smoke' -and $Joined -match 'reports.py audit --phase smoke') -or
        ($global:RegionFail -eq 'test' -and $Call -contains 'test-gate')) {
        $global:LASTEXITCODE=1; Write-Output 'MOCK EXPECTED FAILURE'; return
    }
    if ($Joined -match 'reports.py report --phase pilot') {
        $Mount=@($Call | Where-Object { $_ -match 'target=/study$' })[0]
        $Out=($Mount -replace '^type=bind,source=','') -replace ',target=/study$',''
        $Folder=Join-Path $Out 'reports\pilot\validation-10'
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        [IO.File]::WriteAllText((Join-Path $Folder 'selection.json'),(@{region_gate=$global:RegionGate}|ConvertTo-Json))
    }
    if ($Joined -match '/transport.py') {
        Write-Output ('REGION_METRICS_BASE64='+[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('{"mock":true}')))
    } else { Write-Output 'AUDIT MOCK WORKER' }
}
$Params=@{RepoRoot=(Join-Path $TestRoot 'repo');PriorStudy=(Join-Path $TestRoot 'prior');DataRoot=(Join-Path $TestRoot 'data');
    AuditRoot=(Join-Path $TestRoot 'audit');OutputRoot=(Join-Path $TestRoot 'out')}
& $Runner @Params -Stage all
$Runs=@($global:RegionCalls | Where-Object { $_[0] -eq 'run' })
$Train=@($Runs | Where-Object { ($_ -join ' ') -match '/train.py' })
if ($Train.Count -ne 16) { throw "Expected 4 smoke + 3 pilot + 9 full, got $($Train.Count)" }
if (@($Train | Where-Object { ($_ -join ' ') -match '--phase full.*--arm G1' }).Count) { throw 'Closed gate ran G1' }
foreach ($Call in $Runs) {
    foreach ($Flag in @('--pull','never','--network','none')) { if ($Call -notcontains $Flag) { throw 'Missing offline restriction' } }
    $Joined=$Call -join ' '
    if ($Joined -match 'must-not-resolve|target=/test|type=bind,source=.*test-38') { throw 'Implicit test access' }
    if ($Joined -match '/train.py' -and $Joined -match 'target=/validation|target=/data,readonly.*\\val') { throw 'Training saw validation' }
    if ($Joined -match '/transport.py' -and ($Call -contains '--user' -or $Call -notcontains '/output:rw,mode=1777,size=512m')) { throw 'Evaluator permission/environment regression' }
}
$SmokeIndex=-1; $PilotIndex=-1
for ($i=0;$i -lt $Runs.Count;$i++) {
    $Joined=$Runs[$i] -join ' '
    if ($Joined -match 'reports.py audit --phase smoke') { $SmokeIndex=$i }
    if ($PilotIndex -lt 0 -and $Joined -match '/train.py --phase pilot') { $PilotIndex=$i }
}
if ($SmokeIndex -lt 0 -or $PilotIndex -le $SmokeIndex) { throw 'Pilot precedes smoke audit' }

$global:RegionCalls.Clear(); $global:RegionFail='smoke'
$Failed=$false
try { & $Runner @Params -Stage all } catch { $Failed=$true }
if (-not $Failed -or @($global:RegionCalls | Where-Object { ($_ -join ' ') -match '/train.py --phase (pilot|full)' }).Count) { throw 'Smoke failure did not stop queue' }

$global:RegionFail='test'; $global:RegionCalls.Clear(); $Failed=$false
try { & $Runner @Params -Stage test } catch { $Failed=$true }
if (-not $Failed -or @($global:RegionCalls | Where-Object { ($_ -join ' ') -match 'must-not-resolve|target=/test|/infer.py|/train.py' }).Count) { throw 'Test gate bypass' }

$global:RegionFail=''; $global:RegionGate=$true; $global:RegionCalls.Clear()
$Params.OutputRoot=Join-Path $TestRoot 'out2'
& $Runner @Params -Stage all
$Train=@($global:RegionCalls | Where-Object { ($_ -join ' ') -match '/train.py' })
if ($Train.Count -ne 19 -or @($Train | Where-Object { ($_ -join ' ') -match '--phase full.*--arm G1' }).Count -ne 3) { throw 'Open gate must add exactly three full G1 runs' }

$global:RegionCalls.Clear()
& $Runner @Params -Stage smoke
if (@($global:RegionCalls | Where-Object { ($_ -join ' ') -match '/train.py --phase (pilot|full)' }).Count) { throw 'smoke started large training' }
Write-Host "POWERSHELL MOCK QUEUE PASSED; no Docker/data/GPU work performed. Fixtures retained: $TestRoot"
