param([string]$TempRoot='D:\yijian\tmp')
$ErrorActionPreference='Stop'
$TestRoot=Join-Path $TempRoot ('f'+[Guid]::NewGuid().ToString('N').Substring(0,4))
$RealRepo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
foreach ($Part in @('repo\scripts','repo\training-checkpoints','prior','data','audit','external-smoke','pilot','train\T_001','val\V_001')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $TestRoot $Part) | Out-Null
}
Copy-Item -LiteralPath (Join-Path $RealRepo 'scripts\experiment_local_runtimes.ps1') -Destination (Join-Path $TestRoot 'repo\scripts\experiment_local_runtimes.ps1')
$Eval=@'
param($Repository,$Dataset,$Results,$Profiles,$ModelCheckpoint,[switch]$RequireDiagnostics,[switch]$FreezeImages,[switch]$UsePinnedImages,$ReferenceImagesPath,$SidecarRuntimeConfig)
if (-not $UsePinnedImages -or -not $RequireDiagnostics -or $Dataset -notlike '*\val') { throw 'Wrong evaluation scope' }
$global:TransferEvaluations.Add($ModelCheckpoint)
'@
[IO.File]::WriteAllText((Join-Path $TestRoot 'repo\scripts\run_hierarchical_followup_resumable.ps1'),$Eval)
$Inventory=@{Source=@();Dataset=@(
    @{Split='train-40';Case='T_001';File=(Join-Path $TestRoot 'train\T_001\image.mha')},
    @{Split='validation-10';Case='V_001';File=(Join-Path $TestRoot 'val\V_001\image.mha')},
    @{Split='test-38';Case='X_001';File='Z:\must-not-resolve\X_001\image.mha'})}
[IO.File]::WriteAllText((Join-Path $TestRoot 'prior\frozen-study.json'),($Inventory | ConvertTo-Json -Depth 6))
$global:TransferImage='sha256:'+('a'*64)
[IO.File]::WriteAllText((Join-Path $TestRoot 'prior\local-runtime-images.json'),(@{Training=$global:TransferImage;Inference=$global:TransferImage;Evaluation=$global:TransferImage} | ConvertTo-Json))
$global:TransferCalls=New-Object System.Collections.Generic.List[object]
$global:TransferEvaluations=New-Object System.Collections.Generic.List[string]
$global:TransferFail=$false
$global:TestGateFail=$false
function docker {
    $Call=@($args)
    $global:TransferCalls.Add($Call)
    $global:LASTEXITCODE=0
    if ($Call[0] -eq 'image') { Write-Output $global:TransferImage; return }
    if ($Call[0] -ne 'run') { throw 'Unexpected Docker action' }
    if ($global:TestGateFail -and $Call -contains 'test-gate') {
        $global:LASTEXITCODE=1; Write-Output 'MOCK TEST GATE FAIL'; return
    }
    if ($global:TransferFail -and $Call -contains 'inference-smoke' -and $Call -contains 'MCT_FULL_SEED=2') {
        $global:LASTEXITCODE=1; Write-Output 'MOCK INFERENCE FAIL'; return
    }
    Write-Output 'AUDIT MOCK WORKER'
}
$Parameters=@{RepoRoot=(Join-Path $TestRoot 'repo');PriorStudy=(Join-Path $TestRoot 'prior');
    DataRoot=(Join-Path $TestRoot 'data');AuditRoot=(Join-Path $TestRoot 'audit');ExternalSmoke=(Join-Path $TestRoot 'external-smoke');
    PilotRoot=(Join-Path $TestRoot 'pilot');OutputRoot=(Join-Path $TestRoot 'out')}
& (Join-Path $PSScriptRoot 'run_experiment.ps1') @Parameters -Stage all
$Runs=@($global:TransferCalls | Where-Object { $_[0] -eq 'run' })
$External=@($Runs | Where-Object { ($_ -join ' ') -match '/external.py' })
$Transfers=@($Runs | Where-Object { ($_ -join ' ') -match '/train_transfer.py' })
if ($External.Count -ne 3 -or $Transfers.Count -ne 12 -or $global:TransferEvaluations.Count -ne 7) { throw 'Wrong queue budgets/counts' }
$SmokeIndex=-1; $FirstExternal=-1
for ($i=0; $i -lt $Runs.Count; $i++) {
    $Call=$Runs[$i]
    if ($Call -contains 'inference-smoke') { $SmokeIndex=$i }
    if ($FirstExternal -lt 0 -and ($Call -join ' ') -match '/external.py') { $FirstExternal=$i }
    foreach ($Token in @('--pull','never','--network','none')) { if ($Call -notcontains $Token) { throw 'Missing offline restriction' } }
    if (($Call -join ' ') -match 'must-not-resolve|/datasets/test') { throw 'Test access' }
    if (($Call -join ' ') -match '/external.py' -and ($Call -join ' ') -match 'target=/data,|target=/datasets') { throw 'External phase has TrackRAD data' }
    if (($Call -join ' ') -match '/train_transfer.py' -and ($Call -join ' ') -match 'target=/acdc|target=/datasets/validation') { throw 'Transfer phase has external/validation input' }
}
if ($SmokeIndex -lt 0 -or $FirstExternal -le $SmokeIndex) { throw 'Pilot started before smoke' }
$global:TransferCalls.Clear(); $global:TransferEvaluations.Clear(); $global:TransferFail=$true
$Failed=$false
try { & (Join-Path $PSScriptRoot 'run_experiment.ps1') @Parameters -Stage all } catch { $Failed=$true }
if (-not $Failed -or @($global:TransferCalls | Where-Object { ($_ -join ' ') -match '/external.py' }).Count -gt 0 -or $global:TransferEvaluations.Count) {
    throw 'Smoke failure must prevent external pilot and evaluation'
}

$global:TransferCalls.Clear(); $global:TransferEvaluations.Clear(); $global:TransferFail=$false
& (Join-Path $PSScriptRoot 'run_experiment.ps1') @Parameters -Stage validation
if (@($global:TransferCalls | Where-Object { ($_ -join ' ') -match '/external.py|/train_transfer.py' }).Count -gt 0 -or $global:TransferEvaluations.Count -ne 7) {
    throw 'Validation-only stage must not train'
}
$global:TransferCalls.Clear(); $global:TransferEvaluations.Clear(); $global:TestGateFail=$true
$Message=''
try { & (Join-Path $PSScriptRoot 'run_experiment.ps1') @Parameters -Stage test } catch { $Message=$_.Exception.Message }
if ($Message -notmatch 'MedCoTracker failed' -or
    @($global:TransferCalls | Where-Object { ($_ -join ' ') -match 'target=/test-data|/external.py|/train_transfer.py' }).Count -gt 0 -or $global:TransferEvaluations.Count) {
    throw 'Missing validation gate must stop BEFORE test path resolution, test mount or inference'
}
$global:TestGateFail=$false
$EvalTest=$Eval.Replace("$" + "Dataset -notlike '*\val'", "$" + "Dataset -notlike '*\test'")
[IO.File]::WriteAllText((Join-Path $TestRoot 'repo\scripts\run_hierarchical_followup_resumable.ps1'),$EvalTest)
New-Item -ItemType Directory -Force -Path (Join-Path $TestRoot 'test\X_001') | Out-Null
$Inventory.Dataset[2].File=Join-Path $TestRoot 'test\X_001\image.mha'
[IO.File]::WriteAllText((Join-Path $TestRoot 'prior\frozen-study.json'),($Inventory | ConvertTo-Json -Depth 6))
$global:TransferCalls.Clear(); $global:TransferEvaluations.Clear()
& (Join-Path $PSScriptRoot 'run_experiment.ps1') @Parameters -Stage test
if ($global:TransferEvaluations.Count -ne 7 -or
    @($global:TransferCalls | Where-Object { ($_ -join ' ') -match '/external.py|/train_transfer.py' }).Count -gt 0) {
    throw 'Explicit test must evaluate fixed 7 arms without training'
}
$TestRuns=@($global:TransferCalls | Where-Object { $_[0] -eq 'run' })
$GateIndex=-1; $TestIndex=-1
for ($i=0; $i -lt $TestRuns.Count; $i++) {
    if ($TestRuns[$i] -contains 'test-gate') { $GateIndex=$i }
    if (($TestRuns[$i] -join ' ') -match 'target=/test-data') { $TestIndex=$i }
}
if ($GateIndex -lt 0 -or $TestIndex -le $GateIndex) { throw 'Test mounted before gate' }
Write-Host 'FULL POWERSHELL MOCK PASSED; no real Docker/GPU/network calls'
