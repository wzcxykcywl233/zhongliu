# Windows PowerShell orchestration test. Docker and evaluation are fixtures.
param([Parameter(Mandatory=$true)][string]$TemporaryRoot)
$ErrorActionPreference='Stop'
$Repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Fixture=Join-Path $TemporaryRoot ('ft'+[Guid]::NewGuid().ToString('N').Substring(0,6))
$Checkout=Join-Path $Fixture 'repo'
$Prior=Join-Path $Fixture 'prior'
$Output=Join-Path $Fixture 'out'
if ($Output.Length -gt 40) { throw 'Pass a short writable TemporaryRoot for the path-limit regression.' }
$Utf8=New-Object System.Text.UTF8Encoding($false)
function Save-Fixture([string]$Path,[string]$Text) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Path -Parent) | Out-Null
    [IO.File]::WriteAllText($Path,$Text,$Utf8)
}
$Helper=Join-Path $Checkout 'scripts\experiment_local_runtimes.ps1'
Save-Fixture $Helper (Get-Content -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Raw)
Save-Fixture (Join-Path $Checkout 'scripts\run_hierarchical_followup_resumable.ps1') @'
param($Repository,$Dataset,$Results,$Profiles,$ModelCheckpoint,[switch]$RequireDiagnostics,
      [switch]$FreezeImages,[switch]$UsePinnedImages,$ReferenceImagesPath,$SidecarRuntimeConfig)
if ($Profiles[0] -ne 'hierarchical_full_grid0_iterations2_memory_topk_diverse') { throw 'Wrong fixed student.' }
if (-not $FreezeImages -or -not $UsePinnedImages -or -not $SidecarRuntimeConfig -or -not $RequireDiagnostics) { throw 'Runtime guard missing.' }
if ($ModelCheckpoint -match 'smoke') { throw 'Smoke weights used for full evaluation.' }
$global:FtEvaluations.Add($Results)
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
$global:FtCalls=New-Object System.Collections.Generic.List[object]
$global:FtEvaluations=New-Object System.Collections.Generic.List[object]
$global:FtSmoke=$false
$global:FtFail=$false
function global:docker {
    $a=@($args)
    $global:LASTEXITCODE=0
    if ($a[0] -eq 'image') { return @($a | Where-Object {$_ -like 'sha256:*'})[0] }
    if ($a[0] -ne 'run' -or $a -notcontains 'never' -or $a -notcontains 'none') { throw 'Unexpected network/build command.' }
    $global:FtCalls.Add($a)
    if ($a -contains '/package/study.py') {
        $Command=$a[([array]::IndexOf($a,'/package/study.py')+1)]
        if ($Command -eq 'prepare') {
            Save-Fixture (Join-Path $Output 'manifest.json') '{}'
            Save-Fixture (Join-Path $Output 'runtime.json') '{}'
            Save-Fixture (Join-Path $Output 'reference-images.json') '{}'
            foreach ($Phase in @('smoke','full')) { foreach ($Seed in @(0,1,2)) { foreach ($Policy in @('old','scaled','grid0','iter6')) {
                Save-Fixture (Join-Path $Output "$Phase\train\seed_$Seed\$Policy\run-config.json") '{}'
            }}}
        }
        if ($Command -eq 'smoke-gate' -and -not $global:FtSmoke) { $global:LASTEXITCODE=1; return 'Missing smoke' }
        if ($Command -eq 'gpu-smoke') {
            if ($a -notcontains '--gpus' -or $a -notcontains '/opt/app/.pixi/envs/cuda/bin/python') { throw 'Wrong inference smoke environment.' }
            $global:FtSmoke=$true
        }
    } elseif ($a -contains '/package/train.py') {
        if ($a -notcontains '--gpus' -or $a -notcontains '/opt/app/.pixi/envs/training/bin/python') { throw 'Wrong training environment.' }
        if ($global:FtFail) { $global:LASTEXITCODE=7; return 'Synthetic failure' }
        $Mount=@($a | Where-Object {$_ -like 'type=bind,source=*,target=/results'})[0]
        $Folder=$Mount.Substring('type=bind,source='.Length).Replace(',target=/results','')
        Save-Fixture (Join-Path $Folder 'cotracker_three_final.pth') 'fixture'
    } else { throw 'Unknown worker' }
}
$Runner=Join-Path $PSScriptRoot 'run_experiment.ps1'
try {
    $Blocked=$false
    try { & $Runner -Stage train -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output } catch { $Blocked=$true }
    if (-not $Blocked -or @($global:FtCalls | Where-Object {$_ -contains '/package/train.py'}).Count) { throw 'Full training bypassed smoke.' }
    $global:FtFail=$true
    $Blocked=$false
    try { & $Runner -Stage smoke -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output } catch { $Blocked=$true }
    if (-not $Blocked -or $global:FtSmoke) { throw 'Failed smoke continued.' }
    $global:FtFail=$false
    $global:FtCalls.Clear()
    & $Runner -Stage smoke -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output
    & $Runner -Stage all -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output
    $Training=@($global:FtCalls | Where-Object {$_ -contains '/package/train.py'})
    if ($Training.Count -ne 16 -or $global:FtEvaluations.Count -ne 26) { throw "Wrong matrix: $($Training.Count) train, $($global:FtEvaluations.Count) eval." }
    if (@($Training | Where-Object {$_ -contains '--smoke'}).Count -ne 4) { throw 'Wrong smoke matrix.' }
    & $Runner -Stage train -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output
    if (@($global:FtCalls | Where-Object {$_ -contains '/package/train.py'}).Count -ne 16) { throw 'Completed arms trained again.' }
    Write-Host "PASS: smoke gating, failure stop, 4+12 arms, 26 evaluations, cached resume. Fixtures retained: $Fixture"
} finally {
    Remove-Item Function:\docker
}
