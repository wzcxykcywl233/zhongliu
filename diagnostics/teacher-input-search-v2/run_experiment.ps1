[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','pilot','confirm','combine','test','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$OutputRoot='C:\zhongliu\tis-v2'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$PackageRoot=$PSScriptRoot
if ($OutputRoot.Length -gt 32 -or [IO.Path]::GetPathRoot($OutputRoot) -eq $OutputRoot+'\') {
    throw 'Use a short dedicated OutputRoot (at most 32 characters), e.g. C:\zhongliu\tis-v2.'
}
foreach ($Path in @($RepoRoot,$PriorStudy,$OutputRoot,$PackageRoot)) {
    if ($Path -match '[",\r\n]') { throw 'Quotes, commas or newlines in paths are unsupported.' }
}
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
foreach ($File in $Protocol.Source) {
    if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot $File.File)).Hash -ne $File.SHA256) {
        throw "Frozen source changed: $($File.File). Preserve old results and use the matching checkout."
    }
}
$Weights=Join-Path $RepoRoot 'training-checkpoints'
$Datasets=@{}
foreach ($Split in @('train-40','validation-10','test-38')) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Missing dataset inventory: $Split" }
    $First=$Rows[0]
    $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot resolve frozen dataset root.' }
    $Datasets[$Split]=(Resolve-Path -LiteralPath $First.File.Substring(0,$Offset)).Path.TrimEnd('\')
}
foreach ($InputRoot in @($RepoRoot,$PriorStudy,$Weights,$Datasets['train-40'],$Datasets['validation-10'],$Datasets['test-38'])) {
    if ($InputRoot -match '[",\r\n]' -or $OutputRoot -eq $InputRoot -or
        $OutputRoot.StartsWith($InputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $InputRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must not overlap any input directory.'
    }
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Images=Get-Content -LiteralPath (Join-Path $PriorStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
foreach ($Kind in @('Training','Inference','Evaluation')) {
    if ($Images.$Kind -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Images.$Kind) -ne $Images.$Kind) {
        throw "Pinned $Kind image unavailable. Start Docker Desktop; no build/pull attempted."
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Transcript=$false

function Invoke-QuietDocker([string[]]$Arguments,[string]$Log) {
    $Previous=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line -match '^(PLAN |RUNTIME |AUDIT |SELECTION |GPU |REPORT |RESULT |FROZEN |FINISHED TRAINING)' -or
                ($Line -match 'experiment=.* step=(\d+) ' -and ([int]$Matches[1] % 25 -eq 0))) { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Previous }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "Experiment failed ($Status). Preserve results and repeat unchanged command. Log: $Log"
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
    $Arguments+=@('--entrypoint','/usr/bin/timeout',$Images.$Kind,'--signal=TERM','--kill-after=30s','3600s',
        $Python,'/package/study.py',$Command)+$Extra
    Invoke-QuietDocker $Arguments (Join-Path $OutputRoot 'audit.log')
}
function Get-StagePlan([string]$Name) {
    Invoke-Worker 'plan-check' @('--stage',$Name)
    return Get-Content -LiteralPath (Join-Path $OutputRoot "plans\$Name.json") -Raw | ConvertFrom-Json
}
function Invoke-Training([string]$Name) {
    if ($Name -ne 'smoke') { Invoke-Worker 'smoke-gate' }
    $Plan=Get-StagePlan $Name
    $Phase=$Plan.phase
    $Budget=if ($Phase -eq 'smoke') {600} elseif ($Phase -eq 'pilot') {3600} else {14400}
    foreach ($Seed in $Plan.seeds) { foreach ($Property in $Plan.pools.PSObject.Properties) {
        $Arm=$Property.Name
        $Folder=Join-Path $OutputRoot "$Phase\train\seed_$Seed\$Arm"
        if (-not (Test-Path -LiteralPath (Join-Path $Folder 'cotracker_three_final.pth'))) {
            Write-Host "TRAIN $Name seed_$Seed/$Arm steps=$($Plan.steps); log=$Folder\training.log"
            $Arguments=@('run','--rm','--pull','never','--network','none','--gpus','device=0','--shm-size','4g',
                '--env','PYTHONDONTWRITEBYTECODE=1',
                '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
                '--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly",
                '--mount',"type=bind,source=$Weights,target=/checkpoints,readonly",
                '--mount',"type=bind,source=$OutputRoot,target=/study,readonly",
                '--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
                '--mount',"type=bind,source=$Folder,target=/results",
                '--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=30s',"${Budget}s",
                '/opt/app/.pixi/envs/training/bin/python','/package/train.py','--phase',$Phase,'--seed',"$Seed",'--arm',$Arm)
            Invoke-QuietDocker $Arguments (Join-Path $Folder 'training.log')
        }
        Invoke-Worker 'verify-arm' @('--stage',$Name,'--seed',"$Seed",'--arm',$Arm)
    }}
    Invoke-Worker 'audit' @('--stage',$Name)
}
function Invoke-Evaluation([string]$Name) {
    $Plan=Get-StagePlan $Name
    $Split=if ($Name -eq 'test') {'test-38'} else {'validation-10'}
    Invoke-Worker 'verify-training' @('--stage',$Name)
    $Runner=Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    $Entries=@(@{Name='pretrained';Arm='pretrained';Seed=-1;Checkpoint=(Join-Path $OutputRoot 'pretrained-frozen.pth')})
    foreach ($Seed in $Plan.seeds) { foreach ($Property in $Plan.pools.PSObject.Properties) {
        $Arm=$Property.Name
        $Entries+=@{Name="seed_$Seed\$Arm";Arm=$Arm;Seed=$Seed;Checkpoint=(Join-Path $OutputRoot "$($Plan.phase)\train\seed_$Seed\$Arm\cotracker_three_final.pth")}
    }}
    foreach ($Entry in $Entries) {
        Invoke-Worker 'eval-bind' @('--stage',$Name,'--split',$Split,'--seed',"$($Entry.Seed)",'--arm',$Entry.Arm)
        $Destination=Join-Path $OutputRoot "$($Plan.phase)\$Split\$($Entry.Name)"
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        Write-Host "EVALUATE $Name/$Split $($Entry.Name); log=$Destination\launcher.log"
        & $Runner -Repository $RepoRoot -Dataset $Datasets[$Split] -Results $Destination `
            -Profiles @('hierarchical_full_grid0_iterations2_memory_topk_diverse') `
            -ModelCheckpoint $Entry.Checkpoint -RequireDiagnostics -FreezeImages -UsePinnedImages `
            -ReferenceImagesPath (Join-Path $OutputRoot 'reference-images.json') `
            -SidecarRuntimeConfig (Join-Path $OutputRoot 'runtime.json') *> (Join-Path $Destination 'launcher.log')
    }
    Invoke-Worker 'report' @('--stage',$Name,'--split',$Split)
}
function Invoke-Smoke {
    if (Test-Path -LiteralPath (Join-Path $OutputRoot 'smoke-gate.json')) { Invoke-Worker 'smoke-gate'; return }
    Invoke-Worker 'teacher-smoke' -Gpu
    Invoke-Training 'smoke'
    Invoke-Worker 'gpu-smoke' -Kind Inference -Gpu
}
try {
    Start-Transcript -Path (Join-Path $OutputRoot 'queue.log') -Append | Out-Null
    $Transcript=$true
    Write-Host "Teacher-input search stage=$Stage; fixed frozen Mamba. Validation selects; final test-38 reports only."
    Invoke-Worker 'prepare'
    Invoke-Worker 'runtime' @('--kind','Training') -Kind Training
    Invoke-Worker 'runtime' @('--kind','Inference') -Kind Inference
    if ($Stage -eq 'plan') { return }
    if ($Stage -in @('smoke','all')) { Invoke-Smoke }
    foreach ($Name in @('pilot','confirm','combine')) {
        if ($Stage -in @($Name,'all')) {
            Invoke-Training $Name
            Invoke-Evaluation $Name
            Invoke-Worker 'select' @('--stage',$Name)
        }
    }
    if ($Stage -in @('test','all')) {
        Invoke-Worker 'smoke-gate'
        # Test stage verifies existing full checkpoints; it never starts training.
        Invoke-Worker 'audit' @('--stage','test')
        Invoke-Evaluation 'test'
    }
    Write-Host "COMPLETED stage=${Stage}: $OutputRoot. Final report: reports\test\test-38\summary.csv"
} finally {
    if ($Transcript) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
