[CmdletBinding()]
param(
    [ValidateSet('plan','train','validation','test','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$OutputRoot='C:\zhongliu\ts-full-v1'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$PackageRoot=$PSScriptRoot
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -gt 65 -or [IO.Path]::GetPathRoot($OutputRoot) -eq $OutputRoot+'\') {
    throw 'Use a short, dedicated OutputRoot, e.g. C:\zhongliu\ts-full-v1.'
}
foreach ($Path in @($RepoRoot,$PriorStudy,$OutputRoot,$PackageRoot)) {
    if ($Path -match '[",\r\n]') { throw 'Paths with quotes, commas or newlines are unsupported.' }
}
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
# New experiment files live outside the old source inventory; the old run is
# neither rewritten nor made compatible by weakening its source guards.
foreach ($File in $Protocol.Source) {
    if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot $File.File)).Hash -ne $File.SHA256) {
        throw "Original ts-v1 source changed: $($File.File). Preserve the pilot and restore/use its exact checkout."
    }
}
$Weights=Join-Path $RepoRoot 'training-checkpoints'
$Datasets=@{}
foreach ($Split in @('train-40','validation-10','test-38')) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Original dataset inventory missing: $Split" }
    $First=$Rows[0]
    $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot resolve the original dataset root.' }
    $Datasets[$Split]=(Resolve-Path -LiteralPath $First.File.Substring(0,$Offset)).Path.TrimEnd('\')
}
foreach ($InputRoot in @($RepoRoot,$PriorStudy,$Weights,$Datasets['train-40'],$Datasets['validation-10'],$Datasets['test-38'])) {
    if ($InputRoot -match '[",\r\n]') { throw 'Input paths with quotes, commas or newlines are unsupported.' }
    if ($OutputRoot -eq $InputRoot -or $OutputRoot.StartsWith($InputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $InputRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must not overlap the original study, checkout, weights or datasets.'
    }
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Images=Get-Content -LiteralPath (Join-Path $PriorStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
foreach ($Kind in @('Training','Inference','Evaluation')) {
    $Id=[string]$Images.$Kind
    if ($Id -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Id) -ne $Id) {
        throw "Original local image missing: $Kind. Start Docker Desktop/check image ls. No build or pull attempted."
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Transcript=$false
function Invoke-FullWorker([string]$Command,[string[]]$Extra=@(),[ValidateSet('Training','Inference')][string]$Kind='Training') {
    $Python=if ($Kind -eq 'Training') {'/opt/app/.pixi/envs/training/bin/python'} else {'/opt/app/.pixi/envs/cuda/bin/python'}
    $Arguments=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
        '--env','COTRACKER_APP_DIR=/opt/app',
        '--env','PYTHONPATH=/opt/app/ext/co-tracker:/opt/app:/opt/app/experiments',
        '--mount',"type=bind,source=$PriorStudy,target=/prior,readonly",
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$Weights,target=/checkpoints,readonly",
        '--mount',"type=bind,source=$($Datasets['train-40']),target=/datasets/train,readonly",
        '--mount',"type=bind,source=$($Datasets['validation-10']),target=/datasets/validation,readonly",
        '--mount',"type=bind,source=$($Datasets['test-38']),target=/datasets/test,readonly",
        '--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/results",
        '--entrypoint','/usr/bin/timeout',$Images.$Kind,'--signal=TERM','--kill-after=30s','900s',
        $Python,'/package/study.py',$Command) + $Extra
    Invoke-LocalDocker $Arguments (Join-Path $OutputRoot 'audit.log')
}
function Invoke-FullEvaluation([ValidateSet('validation-10','test-38')][string]$Split) {
    Invoke-FullWorker 'prepare'
    Invoke-FullWorker 'verify-training'
    if ($Split -eq 'test-38') { Invoke-FullWorker 'test-gate' }
    $Dataset=$Datasets[$Split]
    $Runner=Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    $References=Join-Path $OutputRoot 'reference-images.json'
    $Profile='rt_points_0'
    & $Runner -Repository $RepoRoot -Dataset $Dataset -Results (Join-Path $OutputRoot "$Split\pretrained") `
        -Profiles @($Profile) -ModelCheckpoint (Join-Path $Weights 'scaled_offline.pth') `
        -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $References
    foreach ($Seed in @(0,1,2)) {
        foreach ($Policy in @('old','iter6')) {
            & $Runner -Repository $RepoRoot -Dataset $Dataset -Results (Join-Path $OutputRoot "$Split\seed_$Seed\$Policy") `
                -Profiles @($Profile) -ModelCheckpoint (Join-Path $OutputRoot "train\seed_$Seed\$Policy\cotracker_three_final.pth") `
                -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $References
        }
    }
    Invoke-FullWorker 'report' @('--split',$Split)
}
try {
    Start-Transcript -Path (Join-Path $OutputRoot 'queue.log') -Append | Out-Null; $Transcript=$true
    Write-Host "Frozen-teacher full experiment stage=$Stage. User-requested full run despite the degraded pilot."
    Write-Host 'No new teacher fine-tuning, no pilot-student warm start, no Mamba, no build/pull/install.'
    Invoke-FullWorker 'prepare'
    Invoke-FullWorker 'runtime' @('--kind','Training') -Kind Training
    Invoke-FullWorker 'runtime' @('--kind','Inference') -Kind Inference
    $Selection=Get-Content -LiteralPath (Join-Path $OutputRoot 'selection.json') -Raw | ConvertFrom-Json
    if ($Selection.selected_policy -ne 'iter6' -or $Selection.student_inference_profile -ne 'rt_points_0') {
        throw 'Unexpected frozen teacher/student; preserve results and inspect selection.json.'
    }
    if ($Stage -eq 'plan') {
        Write-Host "PLAN FROZEN: iter6; best base student=250-point memory-diverse; old/iter6 x3 seeds x1000 steps. $OutputRoot\selection.json"
        return
    }
    if ($Stage -in @('train','all')) {
        foreach ($Seed in @(0,1,2)) {
            foreach ($Policy in @('old','iter6')) {
                $Folder=Join-Path $OutputRoot "train\seed_$Seed\$Policy"
                $Final=Join-Path $Folder 'cotracker_three_final.pth'
                if (-not (Test-Path -LiteralPath $Final)) {
                    $Offline=if ($Policy -eq 'old') {'baseline_offline.pth'} else {'scaled_offline.pth'}
                    $Budget=[int]$Selection.arm_timeout_seconds
                    if ($Budget -lt 300 -or $Budget -gt 14400) { throw 'Invalid original training budget.' }
                    Write-Host "TRAIN seed_$Seed/${Policy}: fresh scaled initialization, 1000 steps, timeout=${Budget}s"
                    Invoke-LocalDocker @('run','--rm','--pull','never','--network','none','--gpus','device=0','--shm-size','4g',
                        '--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly",
                        '--mount',"type=bind,source=$Weights,target=/checkpoints,readonly",
                        '--mount',"type=bind,source=$Folder,target=/results",'--workdir','/opt/app/ext/co-tracker',
                        '--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=30s',"${Budget}s",
                        '/opt/app/.pixi/envs/training/bin/python','/opt/app/ext/co-tracker/train_on_real_data.py',
                        '--batch_size','1','--num_workers','0','--num_steps','1000','--ckpt_path','/results',
                        '--model_name','cotracker_three','--experiment_name',"frozen_seed_${Seed}_$Policy",
                        '--sequence_len','10','--traj_per_sample','384','--train_iters','4','--save_freq','100000',
                        '--save_every_n_steps','25','--keep_last_checkpoints','3','--save_every_n_epoch','1',
                        '--trackrad_data_dir','/data','--skip_evaluation','--offline_model',
                        '--restore_ckpt','/checkpoints/scaled_offline.pth',
                        '--teacher_types','cotracker2v1','online_cotracker_three','offline_cotracker_three',
                        '--teacher_cotracker2_ckpt','/checkpoints/cotracker2v1.pth',
                        '--teacher_online_ckpt','/checkpoints/baseline_online.pth','--teacher_offline_ckpt',"/checkpoints/$Offline",
                        '--auxiliary_teacher_weight','0','--seed',"$(20261017+$Seed)",'--paired_step_seed',"$(20261017+$Seed)",
                        '--teacher_seed',"$(20261017+$Seed)",'--teacher_log_every','1','--confidence_target_mode','hard','--lr','0.00005',
                        '--backbone_growth_study','--backbone_architecture','base','--teacher_student_study','--offline_teacher_policy',$Policy
                    ) (Join-Path $Folder 'training.log')
                }
                Invoke-FullWorker 'verify-arm' @('--seed',"$Seed",'--policy',$Policy)
            }
        }
        Invoke-FullWorker 'audit'
    }
    if ($Stage -in @('validation','all')) { Invoke-FullEvaluation 'validation-10' }
    if ($Stage -in @('test','all')) { Invoke-FullEvaluation 'test-38' }
    Write-Host "COMPLETED stage=${Stage}: $OutputRoot. Formal results belong to test-38, not validation-10."
} finally {
    if ($Transcript) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
