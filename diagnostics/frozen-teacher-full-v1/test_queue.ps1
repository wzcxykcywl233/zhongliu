# Windows PowerShell 5.1 orchestration regression. All Docker calls are mocks.
$ErrorActionPreference='Stop'
$Repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Fixture=Join-Path ([IO.Path]::GetTempPath()) ('tf-'+[Guid]::NewGuid().ToString('N').Substring(0,6))
$Checkout=Join-Path $Fixture 'checkout'
$Prior=Join-Path $Fixture 'prior'
$Output=Join-Path $Fixture 'out'
$Utf8=New-Object System.Text.UTF8Encoding($false)
function Save-Fixture([string]$Path,[string]$Text) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Path -Parent) | Out-Null
    [IO.File]::WriteAllText($Path,$Text,$Utf8)
}
New-Item -ItemType Directory -Force -Path $Checkout,$Prior | Out-Null
$Helper=Join-Path $Checkout 'scripts\experiment_local_runtimes.ps1'
New-Item -ItemType Directory -Force -Path (Split-Path $Helper -Parent) | Out-Null
Copy-Item -LiteralPath (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1') -Destination $Helper
Save-Fixture (Join-Path $Checkout 'scripts\run_hierarchical_followup_resumable.ps1') @'
param($Repository,$Dataset,$Results,$Profiles,$ModelCheckpoint,[switch]$RequireDiagnostics,
      [switch]$FreezeImages,[switch]$UsePinnedImages,$ReferenceImagesPath)
if ($Profiles[0] -ne 'rt_points_0') { throw 'Best-student profile was not used.' }
if (-not $FreezeImages -or -not $UsePinnedImages -or -not $ReferenceImagesPath) { throw 'Runtime was not pinned.' }
if ($ModelCheckpoint -like '*pilot*') { throw 'Pilot weights used for evaluation.' }
$global:FrozenEvalCalls.Add([ordered]@{Results=$Results;Checkpoint=$ModelCheckpoint;Dataset=$Dataset})
New-Item -ItemType Directory -Force -Path $Results | Out-Null
'@
$Data=@()
foreach ($Pair in @(@{Split='train-40';Alias='train';Case='A_001'},@{Split='validation-10';Alias='validation';Case='B_001'},@{Split='test-38';Alias='test';Case='C_001'})) {
    $Path=Join-Path $Fixture "$($Pair.Alias)\$($Pair.Case)\images\$($Pair.Case)_frames.mha"
    Save-Fixture $Path 'fixture'
    $Data += [ordered]@{Split=$Pair.Split;Case=$Pair.Case;File=$Path;SHA256=(Get-FileHash -LiteralPath $Path).Hash}
}
$Source=@(foreach ($Name in @('scripts\experiment_local_runtimes.ps1','scripts\run_hierarchical_followup_resumable.ps1')) {
    [ordered]@{File=$Name;SHA256=(Get-FileHash -LiteralPath (Join-Path $Checkout $Name)).Hash}
})
Save-Fixture (Join-Path $Prior 'frozen-study.json') (ConvertTo-Json -InputObject ([ordered]@{Source=$Source;Dataset=$Data}) -Depth 20)
$TrainingId='sha256:'+('a'*64); $InferenceId='sha256:'+('b'*64); $EvaluationId='sha256:'+('c'*64)
Save-Fixture (Join-Path $Prior 'local-runtime-images.json') (ConvertTo-Json -InputObject @{Training=$TrainingId;Inference=$InferenceId;Evaluation=$EvaluationId})
New-Item -ItemType Directory -Force -Path (Join-Path $Checkout 'training-checkpoints') | Out-Null
$global:FrozenDockerCalls=New-Object System.Collections.Generic.List[object]
$global:FrozenEvalCalls=New-Object System.Collections.Generic.List[object]
$global:FrozenMockWinner='iter6'
function global:docker {
    $a=@($args)
    $global:FrozenDockerCalls.Add($a)
    $global:LASTEXITCODE=0
    if ($a[0] -eq 'image' -and $a[1] -eq 'inspect') { return @($a | Where-Object {$_ -like 'sha256:*'})[0] }
    if ($a[0] -ne 'run') { throw 'Unexpected build/pull/tag/network operation.' }
    if ($a -notcontains '--pull' -or $a[([array]::IndexOf($a,'--pull')+1)] -ne 'never' -or
        $a[([array]::IndexOf($a,'--network')+1)] -ne 'none') { throw 'Network/pull guard missing.' }
    if (@($a | Where-Object {$_ -like '*,target=/prior' -or $_ -like '*,target=/checkout'}).Count) { throw 'Old inputs writable.' }
    $Mount=@($a | Where-Object {$_ -like 'type=bind,source=*,target=/results'})[0]
    $Folder=$Mount.Substring('type=bind,source='.Length).Replace(',target=/results','')
    if ($a -contains '/package/study.py') {
        $Command=$a[([array]::IndexOf($a,'/package/study.py')+1)]
        if ($Command -eq 'prepare') {
            Save-Fixture (Join-Path $Folder 'selection.json') (ConvertTo-Json -InputObject @{selected_policy=$global:FrozenMockWinner;student_inference_profile='rt_points_0';arm_timeout_seconds=7200})
            Save-Fixture (Join-Path $Folder 'reference-images.json') '{}'
            foreach ($Seed in @(0,1,2)) { foreach ($Policy in @('old','iter6')) {
                Save-Fixture (Join-Path $Folder "train\seed_$Seed\$Policy\run-config.json") '{}'
            }}
        } elseif ($Command -eq 'verify-arm') {
            $Seed=$a[([array]::IndexOf($a,'--seed')+1)];$Policy=$a[([array]::IndexOf($a,'--policy')+1)]
            if (-not (Test-Path -LiteralPath (Join-Path $Folder "train\seed_$Seed\$Policy\cotracker_three_final.pth"))) { throw 'Missing completed arm.' }
        } elseif ($Command -eq 'audit') {
            Save-Fixture (Join-Path $Folder 'training-audit.json') '{"complete":true}'
        } elseif ($Command -eq 'verify-training') {
            if (-not (Test-Path -LiteralPath (Join-Path $Folder 'training-audit.json'))) { throw 'Full training required before evaluation.' }
        } elseif ($Command -eq 'report') {
            $Split=$a[([array]::IndexOf($a,'--split')+1)]
            Save-Fixture (Join-Path $Folder "$Split\audit.json") '{"complete":true}'
        } elseif ($Command -eq 'test-gate') {
            if (-not (Test-Path -LiteralPath (Join-Path $Folder 'validation-10\audit.json'))) { throw 'Validation required before test.' }
        }
        return
    }
    if ($a -notcontains '/opt/app/ext/co-tracker/train_on_real_data.py') { throw 'Unknown Docker task.' }
    foreach ($Pair in @(@('--num_steps','1000'),@('--restore_ckpt','/checkpoints/scaled_offline.pth'),
        @('--train_iters','4'),@('--sequence_len','10'),@('--traj_per_sample','384'))) {
        if ($a[([array]::IndexOf($a,$Pair[0])+1)] -ne $Pair[1]) { throw "Wrong student training flag $($Pair[0])" }
    }
    $Policy=$a[([array]::IndexOf($a,'--offline_teacher_policy')+1)]
    if ($Policy -notin @('old','iter6')) { throw 'Unselected teacher was trained.' }
    $Offline=$a[([array]::IndexOf($a,'--teacher_offline_ckpt')+1)]
    $Expected=if ($Policy -eq 'old') {'/checkpoints/baseline_offline.pth'} else {'/checkpoints/scaled_offline.pth'}
    if ($Offline -ne $Expected -or $a -notcontains '/usr/bin/timeout' -or $a -contains '--mixed_precision') { throw 'Training recipe/timeout changed.' }
    if (@($a | Where-Object {$_ -like '*,target=/datasets/test*' -or $_ -like '*,target=/datasets/validation*'}).Count) { throw 'Held-out data mounted in training.' }
    Save-Fixture (Join-Path $Folder 'cotracker_three_final.pth') 'mock-finished'
}
function Invoke-Fixture([string]$Stage) {
    & (Join-Path $PSScriptRoot 'run_full.ps1') -Stage $Stage -RepoRoot $Checkout -PriorStudy $Prior -OutputRoot $Output
}
function Expect-Failure([scriptblock]$Action,[string]$Message) {
    $Failed=$false
    try { & $Action } catch { $Failed=$true; if ($_.Exception.Message -notlike "*$Message*") { throw } }
    if (-not $Failed) { throw "Expected failure: $Message" }
}
try {
    $Before=@(Get-ChildItem -LiteralPath $Prior -Recurse -File | Get-FileHash | Select-Object Path,Hash | ConvertTo-Json)
    Invoke-Fixture 'plan'
    if (@($global:FrozenDockerCalls | Where-Object {$_ -contains '--gpus'}).Count -or $global:FrozenEvalCalls.Count) { throw 'Plan unexpectedly trained/evaluated.' }
    Expect-Failure { Invoke-Fixture 'test' } 'Full training required'
    Invoke-Fixture 'all'
    $Training=@($global:FrozenDockerCalls | Where-Object {$_ -contains '--offline_teacher_policy'})
    if ($Training.Count -ne 6 -or $global:FrozenEvalCalls.Count -ne 14) { throw 'Expected six full training arms plus seven validation and seven test evaluations.' }
    $Policies=@($Training | ForEach-Object { $_[([array]::IndexOf($_,'--offline_teacher_policy')+1)] })
    if (($Policies -join ',') -ne 'old,iter6,old,iter6,old,iter6') { throw 'Teacher order changed.' }
    Invoke-Fixture 'all'
    if (@($global:FrozenDockerCalls | Where-Object {$_ -contains '--offline_teacher_policy'}).Count -ne 6) { throw 'Resume retrained completed arms.' }
    $After=@(Get-ChildItem -LiteralPath $Prior -Recurse -File | Get-FileHash | Select-Object Path,Hash | ConvertTo-Json)
    if (($Before -join '') -ne ($After -join '')) { throw 'Old study was modified.' }
    $global:FrozenMockWinner='scaled'
    Expect-Failure { Invoke-Fixture 'train' } 'Unexpected frozen teacher/student'
    [IO.File]::AppendAllText($Helper,"`n# modified fixture")
    Expect-Failure { Invoke-Fixture 'plan' } 'Original ts-v1 source changed'
    Write-Host "PASS PS5.1: no build/network, frozen old inputs, best profile, 6x1000 arms, validation-before-test, exact resume, source/selection guards. Docker/evaluation were mocked. Fixture: $Fixture"
} finally {
    Remove-Item Function:\docker -ErrorAction SilentlyContinue
}
