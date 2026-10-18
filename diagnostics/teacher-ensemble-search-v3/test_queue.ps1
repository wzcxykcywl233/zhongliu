# Windows orchestration regression. No real Docker, training or dataset access.
param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$Repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Fixture=Join-Path $TemporaryRoot ('tp'+[Guid]::NewGuid().ToString('N').Substring(0,6))
$Checkout=Join-Path $Fixture 'repo'
$Prior=Join-Path $Fixture 'prior'
$Output=Join-Path $Fixture 'out'
if ($Output.Length -gt 32) { throw 'Pass a short writable test root.' }
$Utf8=New-Object System.Text.UTF8Encoding($false)
function Save-Fixture([string]$Path,[string]$Text) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Path -Parent) | Out-Null
    [IO.File]::WriteAllText($Path,$Text,$Utf8)
}
Save-Fixture (Join-Path $Checkout 'scripts\experiment_local_runtimes.ps1') (Get-Content -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Raw)
Save-Fixture (Join-Path $Checkout 'scripts\run_hierarchical_followup_resumable.ps1') @'
param($Repository,$Dataset,$Results,$Profiles,$ModelCheckpoint,[switch]$RequireDiagnostics,
      [switch]$FreezeImages,[switch]$UsePinnedImages,$ReferenceImagesPath,$SidecarRuntimeConfig)
if ($Profiles[0] -ne 'hierarchical_full_grid0_iterations2_memory_topk_diverse') { throw 'Student changed.' }
if (-not $RequireDiagnostics -or -not $FreezeImages -or -not $UsePinnedImages -or -not $SidecarRuntimeConfig) { throw 'Missing provenance guard.' }
if ($ModelCheckpoint.Contains('\smoke\')) { throw 'Smoke weights leaked into evaluation.' }
if ($Dataset -match 'test-38' -and -not $global:TpsSelected) { throw 'Test set used before final frozen selection.' }
if (-not (Test-Path -LiteralPath (Join-Path $Results 'evaluation-inputs.json'))) { throw 'Missing checkpoint binding.' }
$global:TpsEvaluations.Add($Results)
'@
$Data=@()
foreach ($Item in @(@{Split='train-40';Case='A_001'},@{Split='validation-10';Case='B_001'},@{Split='test-38';Case='C_001'})) {
    $File=Join-Path $Fixture "$($Item.Split)\$($Item.Case)\frames.mha"
    Save-Fixture $File 'fixture'
    $Data+=@{Split=$Item.Split;Case=$Item.Case;File=$File;SHA256=(Get-FileHash -LiteralPath $File).Hash}
}
$Source=@(foreach ($Name in @('scripts\experiment_local_runtimes.ps1','scripts\run_hierarchical_followup_resumable.ps1')) {
    @{File=$Name;SHA256=(Get-FileHash -LiteralPath (Join-Path $Checkout $Name)).Hash}
})
Save-Fixture (Join-Path $Prior 'frozen-study.json') (ConvertTo-Json -Depth 30 -InputObject @{Source=$Source;Dataset=$Data})
Save-Fixture (Join-Path $Prior 'local-runtime-images.json') (ConvertTo-Json -InputObject @{Training=('sha256:'+('a'*64));Inference=('sha256:'+('b'*64));Evaluation=('sha256:'+('c'*64))})
New-Item -ItemType Directory -Force -Path (Join-Path $Checkout 'training-checkpoints') | Out-Null
$global:TpsCalls=New-Object System.Collections.Generic.List[object]
$global:TpsEvaluations=New-Object System.Collections.Generic.List[object]
$global:TpsSmoke=$false
$global:TpsSelected=$false
$global:TpsFail=''
function Make-PlanFixture([string]$Name,[string]$Phase,[string[]]$Arms,[int[]]$Seeds) {
    $Pools=[ordered]@{}
    foreach ($Arm in $Arms) { $Pools[$Arm]=@() }
    $Steps=if ($Phase -eq 'smoke') {10} elseif ($Phase -eq 'pilot') {250} else {1000}
    Save-Fixture (Join-Path $Output "plans\$Name.json") (ConvertTo-Json -Depth 20 -InputObject @{stage=$Name;phase=$Phase;pools=$Pools;seeds=$Seeds;steps=$Steps})
    foreach ($Seed in $Seeds) { foreach ($Arm in $Arms) {
        Save-Fixture (Join-Path $Output "$Phase\train\seed_$Seed\$Arm\run-config.json") '{}'
    }}
}
function Get-ArgValue($Values,[string]$Key) {
    $i=[array]::IndexOf($Values,$Key)
    if ($i -ge 0) { return [string]$Values[$i+1] }
    return ''
}
function global:docker {
    $a=@($args)
    $global:LASTEXITCODE=0
    if ($a[0] -eq 'image') { return @($a | Where-Object {$_ -like 'sha256:*'})[0] }
    if ($a[0] -ne 'run' -or $a -notcontains 'never' -or $a -notcontains 'none') { throw 'Network/build command prohibited.' }
    $global:TpsCalls.Add($a)
    if ($a -contains '/package/study.py') {
        $Command=$a[([array]::IndexOf($a,'/package/study.py')+1)]
        $Name=Get-ArgValue $a '--stage'
        if ($Command -eq 'prepare') {
            foreach ($File in @('manifest.json','runtime.json','reference-images.json')) { Save-Fixture (Join-Path $Output $File) '{}' }
            Make-PlanFixture 'smoke' 'smoke' @('control','v2_xmean','on3_xguard','offold_xyguard','offscaled_xguard') @(0)
            $Arms=@('control','oldpool')
            foreach ($F in @('v2','on3','offold','offscaled')) { foreach ($V in @('xmean','xguard','xyguard')) { $Arms+="${F}_$V" } }
            Make-PlanFixture 'pilot' 'pilot' $Arms @(0)
        }
        if ($Command -eq 'smoke-gate' -and -not $global:TpsSmoke) { $global:LASTEXITCODE=2; return 'Missing smoke' }
        if ($Command -eq 'gpu-smoke') {
            if ($a -notcontains '--gpus' -or $a -notcontains '/opt/app/.pixi/envs/cuda/bin/python') { throw 'Wrong GPU environment.' }
            $global:TpsSmoke=$true
            Save-Fixture (Join-Path $Output 'smoke-gate.json') '{}'
        }
        if ($Command -eq 'plan-check' -and -not (Test-Path (Join-Path $Output "plans\$Name.json"))) { $global:LASTEXITCODE=3; return 'Missing plan' }
        if ($Command -eq 'select') {
            if ($Name -eq 'pilot') { Make-PlanFixture 'confirm' 'full' @('control','oldpool','v2_xmean','on3_xguard','offold_xyguard','offscaled_xguard') @(0,1,2) }
            if ($Name -eq 'confirm') {
                if ($global:TpsFail -eq 'selection') { $global:LASTEXITCODE=7; return 'Synthetic selection failure' }
                Make-PlanFixture 'combine' 'full' @('control','oldpool','v2_xmean','on3_xguard','offscaled_xguard','mix01','mix02','mix12','mix012') @(0,1,2)
            }
            if ($Name -eq 'combine') {
                Make-PlanFixture 'test' 'full' @('control','oldpool','v2_xmean','on3_xguard','offscaled_xguard','mix012') @(0,1,2)
                $global:TpsSelected=$true
            }
        }
        if ($Command -eq 'eval-bind') {
            $Plan=Get-Content -LiteralPath (Join-Path $Output "plans\$Name.json") -Raw | ConvertFrom-Json
            $Arm=Get-ArgValue $a '--arm'; $Seed=Get-ArgValue $a '--seed'; $Split=Get-ArgValue $a '--split'
            $Relative=if ($Arm -eq 'pretrained') {'pretrained'} else {"seed_$Seed\$Arm"}
            Save-Fixture (Join-Path $Output "$($Plan.phase)\$Split\$Relative\evaluation-inputs.json") '{}'
        }
    } elseif ($a -contains '/package/train.py') {
        if ($a -notcontains '--gpus' -or $a -notcontains '/opt/app/.pixi/envs/training/bin/python') { throw 'Wrong training environment.' }
        if ($global:TpsFail -eq 'smoke') { $global:LASTEXITCODE=7; return 'Synthetic training failure' }
        $Mount=@($a | Where-Object {$_ -like 'type=bind,source=*,target=/results'})[0]
        $Folder=$Mount.Substring('type=bind,source='.Length).Replace(',target=/results','')
        Save-Fixture (Join-Path $Folder 'cotracker_three_final.pth') 'fixture'
    } else { throw 'Unknown worker' }
}
$Runner=Join-Path $PSScriptRoot 'run_experiment.ps1'
try {
    $Blocked=$false
    try { & $Runner -Stage pilot -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output } catch { $Blocked=$true }
    if (-not $Blocked -or @($global:TpsCalls | Where-Object {$_ -contains '/package/train.py'}).Count) { throw 'Pilot bypassed smoke.' }
    $global:TpsFail='smoke'; $Blocked=$false
    try { & $Runner -Stage all -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output } catch { $Blocked=$true }
    if (-not $Blocked -or $global:TpsEvaluations.Count) { throw 'Failed smoke reached evaluation.' }
    $global:TpsCalls.Clear(); $global:TpsFail='selection'; $Blocked=$false
    try { & $Runner -Stage all -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output } catch { $Blocked=$true }
    if (-not $Blocked -or @($global:TpsEvaluations | Where-Object {$_ -match 'test-38'}).Count) { throw 'Failed selection reached test.' }
    $global:TpsFail=''
    & $Runner -Stage all -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output
    $Training=@($global:TpsCalls | Where-Object {$_ -contains '/package/train.py'})
    if ($Training.Count -ne 49) { throw "Expected 5 smoke + 14 pilot + 30 full; got $($Training.Count)." }
    $Unique=@($global:TpsEvaluations | Sort-Object -Unique)
    if ($Unique.Count -ne 65) { throw "Expected 65 unique evaluation folders, got $($Unique.Count)." }
    & $Runner -Stage test -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output
    if (@($global:TpsCalls | Where-Object {$_ -contains '/package/train.py'}).Count -ne $Training.Count) { throw 'Test or resume retrained completed groups.' }
    Write-Host "PASS: failure gates, automatic selection/test ordering, worst-case budget, cached full-arm reuse. Fixtures: $Fixture"
} finally { Remove-Item Function:\docker }
