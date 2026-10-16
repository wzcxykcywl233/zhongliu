[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','train','validation','test','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$OutputRoot='C:\zhongliu\fst-v1'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$PackageRoot=$PSScriptRoot
if ($OutputRoot.Length -gt 40 -or [IO.Path]::GetPathRoot($OutputRoot) -eq $OutputRoot+'\') {
    throw 'Use a short dedicated OutputRoot, at most 40 characters, e.g. C:\zhongliu\fst-v1.'
}
foreach ($Path in @($RepoRoot,$PriorStudy,$OutputRoot,$PackageRoot)) {
    if ($Path -match '[",\r\n]') { throw 'Quotes, commas and newlines in paths are unsupported.' }
}
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
foreach ($File in $Protocol.Source) {
    if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot $File.File)).Hash -ne $File.SHA256) {
        throw "Frozen ts-v1 source changed: $($File.File). Use its matching checkout; preserve prior results."
    }
}
$Weights=Join-Path $RepoRoot 'training-checkpoints'
$Datasets=@{}
foreach ($Split in @('train-40','validation-10','test-38')) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Missing dataset inventory: $Split" }
    $First=$Rows[0]
    $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot locate frozen dataset root.' }
    $Datasets[$Split]=(Resolve-Path -LiteralPath $First.File.Substring(0,$Offset)).Path.TrimEnd('\')
}
foreach ($InputRoot in @($RepoRoot,$PriorStudy,$Weights,$Datasets['train-40'],$Datasets['validation-10'],$Datasets['test-38'])) {
    if ($InputRoot -match '[",\r\n]' -or $OutputRoot -eq $InputRoot -or
        $OutputRoot.StartsWith($InputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $InputRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must be separate from the checkout, prior study, weights and datasets.'
    }
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Images=Get-Content -LiteralPath (Join-Path $PriorStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
foreach ($Kind in @('Training','Inference','Evaluation')) {
    if ($Images.$Kind -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Images.$Kind) -ne $Images.$Kind) {
        throw "Pinned $Kind image unavailable. Start Docker Desktop/check local images."
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Transcript=$false
$Policies=@('old','scaled','grid0','iter6')

function Invoke-QuietDocker([string[]]$Arguments,[string]$Log) {
    $Previous=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line -match '^(PLAN |RUNTIME |VERIFIED |PAIRED |SMOKE |GPU |OFFICIAL |VALIDATION |FROZEN |FINISHED TRAINING|pretrained:|old:|scaled:|grid0:|iter6:)' -or
                ($Line -match 'experiment=.* step=(\d+) ' -and ([int]$Matches[1] % 25 -eq 0))) { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Previous }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "Experiment failed ($Status). Preserve results; log: $Log"
    }
}
function Invoke-Worker([string]$Command,[string[]]$Extra=@(),[string]$Kind='Training',[switch]$Gpu) {
    $Python=if ($Kind -eq 'Training') {'/opt/app/.pixi/envs/training/bin/python'} else {'/opt/app/.pixi/envs/cuda/bin/python'}
    $Arguments=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$PriorStudy,target=/prior,readonly",
        '--mount',"type=bind,source=$Weights,target=/checkpoints,readonly",
        '--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
        '--mount',"type=bind,source=$($Datasets['train-40']),target=/datasets/train,readonly",
        '--mount',"type=bind,source=$($Datasets['validation-10']),target=/datasets/validation,readonly",
        '--mount',"type=bind,source=$($Datasets['test-38']),target=/datasets/test,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/study")
    if ($Gpu) { $Arguments+=@('--gpus','device=0','--shm-size','4g') }
    $Arguments+=@('--entrypoint','/usr/bin/timeout',$Images.$Kind,'--signal=TERM','--kill-after=30s','1800s',
        $Python,'/package/study.py',$Command)+$Extra
    Invoke-QuietDocker $Arguments (Join-Path $OutputRoot 'audit.log')
}
function Invoke-Training([string]$Phase) {
    if ($Phase -eq 'full') { Invoke-Worker 'smoke-gate' }
    $Seeds=if ($Phase -eq 'smoke') {@(0)} else {@(0,1,2)}
    $Budget=if ($Phase -eq 'smoke') {600} else {14400}
    foreach ($Seed in $Seeds) { foreach ($Policy in $Policies) {
        $Folder=Join-Path $OutputRoot "$Phase\train\seed_$Seed\$Policy"
        if (-not (Test-Path -LiteralPath (Join-Path $Folder 'cotracker_three_final.pth'))) {
            Write-Host "TRAIN $Phase seed_$Seed/$Policy; frozen Mamba; log=$Folder\training.log"
            $Arguments=@('run','--rm','--pull','never','--network','none','--gpus','device=0','--shm-size','4g',
                '--env','PYTHONDONTWRITEBYTECODE=1',
                '--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly",
                '--mount',"type=bind,source=$Weights,target=/checkpoints,readonly",
                '--mount',"type=bind,source=$OutputRoot,target=/study,readonly",
                '--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
                '--mount',"type=bind,source=$Folder,target=/results",
                '--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=30s',"${Budget}s",
                '/opt/app/.pixi/envs/training/bin/python','/package/train.py','--seed',"$Seed",'--policy',$Policy)
            if ($Phase -eq 'smoke') { $Arguments+='--smoke' }
            Invoke-QuietDocker $Arguments (Join-Path $Folder 'training.log')
        }
        Invoke-Worker 'verify-arm' @('--phase',$Phase,'--seed',"$Seed",'--policy',$Policy)
    }}
    Invoke-Worker 'audit' @('--phase',$Phase)
    if ($Phase -eq 'smoke') { Invoke-Worker 'gpu-smoke' -Kind Inference -Gpu }
}
function Invoke-Evaluation([string]$Split) {
    Invoke-Worker 'verify-training' @('--phase','full')
    if ($Split -eq 'test-38') { Invoke-Worker 'test-gate' }
    $Runner=Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    $Entries=@(@{Name='pretrained';Checkpoint=(Join-Path $OutputRoot 'pretrained-frozen.pth')})
    foreach ($Seed in @(0,1,2)) { foreach ($Policy in $Policies) {
        $Entries+=@{Name="seed_$Seed\$Policy";Checkpoint=(Join-Path $OutputRoot "full\train\seed_$Seed\$Policy\cotracker_three_final.pth")}
    }}
    foreach ($Entry in $Entries) {
        $Destination=Join-Path $OutputRoot "full\$Split\$($Entry.Name)"
        Write-Host "EVALUATE $Split $($Entry.Name); progress log=$Destination\runner.log"
        # Suppress verbose case diagnostics only on the console; the original
        # runner keeps its complete runner.log and per-job logs.
        & $Runner -Repository $RepoRoot -Dataset $Datasets[$Split] -Results $Destination `
            -Profiles @('hierarchical_full_grid0_iterations2_memory_topk_diverse') `
            -ModelCheckpoint $Entry.Checkpoint -RequireDiagnostics -FreezeImages -UsePinnedImages `
            -ReferenceImagesPath (Join-Path $OutputRoot 'reference-images.json') `
            -SidecarRuntimeConfig (Join-Path $OutputRoot 'runtime.json') 6>$null
    }
    Invoke-Worker 'report' @('--split',$Split)
}
try {
    Start-Transcript -Path (Join-Path $OutputRoot 'queue.log') -Append | Out-Null
    $Transcript=$true
    Write-Host "Frozen-backbone teacher comparison stage=$Stage; old/scaled/grid0/iter6; compact Mamba."
    Invoke-Worker 'prepare'
    Invoke-Worker 'runtime' @('--kind','Training') -Kind Training
    Invoke-Worker 'runtime' @('--kind','Inference') -Kind Inference
    if ($Stage -eq 'plan') { return }
    if ($Stage -eq 'smoke') { Invoke-Training 'smoke' }
    if ($Stage -in @('train','all')) { Invoke-Training 'full' }
    if ($Stage -in @('validation','all')) { Invoke-Evaluation 'validation-10' }
    if ($Stage -in @('test','all')) { Invoke-Evaluation 'test-38' }
    Write-Host "COMPLETED stage=${Stage}: $OutputRoot. Formal results: full\test-38; N=3 training seeds."
} finally {
    if ($Transcript) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
