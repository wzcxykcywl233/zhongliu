[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','full','validation','test','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$AuditRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$ExternalSmoke='C:\zhongliu\acdc-ab-gpu-smoke-v1',
    [string]$PilotRoot='C:\zhongliu\mct-ab-v1',
    [string]$OutputRoot='C:\zhongliu\mct-b1-full-v1'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
$AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
$ExternalSmoke=(Resolve-Path -LiteralPath $ExternalSmoke).Path.TrimEnd('\')
$PilotRoot=(Resolve-Path -LiteralPath $PilotRoot).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$Weights=Join-Path $RepoRoot 'training-checkpoints'
if ($OutputRoot.Length -gt 25 -or $OutputRoot.Length -le 3) {
    throw 'Use a short dedicated OutputRoot, at most 25 characters (seed subfolders also need short paths).'
}
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
foreach ($File in $Protocol.Source) {
    if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot $File.File)).Hash -ne $File.SHA256) {
        throw "Frozen source changed: $($File.File). Do not mix experiment versions."
    }
}
# Only resolve the two authorized TrackRAD splits. Test IDs are checked by the Python plan, never opened.
$Datasets=@{}
foreach ($Split in @('train-40','validation-10')) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Missing dataset inventory: $Split" }
    $First=$Rows[0]
    $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot resolve frozen dataset root.' }
    $Datasets[$Split]=(Resolve-Path -LiteralPath $First.File.Substring(0,$Offset)).Path.TrimEnd('\')
}
foreach ($Path in @($RepoRoot,$PriorStudy,$DataRoot,$AuditRoot,$ExternalSmoke,$PilotRoot,$Weights,$Datasets['train-40'],$Datasets['validation-10'])) {
    if ($Path -match '[",\r\n]' -or $Path -eq $OutputRoot -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must not overlap any input folder; ordinary paths only.'
    }
}
if ($OutputRoot -match '[",\r\n]') { throw 'Invalid output path.' }
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
$Arms=@('R1','B1')
$Seeds=@(0,1,2)
$script:Seed=0

function Invoke-QuietDocker([string[]]$Arguments,[string]$Log) {
    $Previous=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line -match '^(PLAN |RUNTIME |AUDIT |HANDOFF |GPU |FULL |TEST |RESULT |FROZEN |CACHE |SMOKE |EXTERNAL |FINISHED TRAINING)' -or
                ($Line -match 'experiment=.* step=(\d+) ' -and ([int]$Matches[1] % 25 -eq 0))) { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Previous }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "MedCoTracker failed ($Status). Retain results; repeat unchanged command after diagnosis. Log: $Log"
    }
}

function Invoke-Worker([string]$Script,[string[]]$Extra=@(),[string]$Mode='audit',[string]$Kind='Training',[int]$Seconds=1800,[switch]$Gpu,[string]$Log='') {
    $SeedRoot=Join-Path $OutputRoot "seed_$script:Seed"
    New-Item -ItemType Directory -Force -Path $SeedRoot | Out-Null
    $Python=if ($Kind -eq 'Training') {'/opt/app/.pixi/envs/training/bin/python'} else {'/opt/app/.pixi/envs/cuda/bin/python'}
    $Arguments=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env',"MCT_FULL_SEED=$script:Seed",'--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$PriorStudy,target=/prior,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$AuditRoot,target=/audit,readonly",
        '--mount',"type=bind,source=$ExternalSmoke,target=/external-smoke,readonly",
        '--mount',"type=bind,source=$PilotRoot,target=/pilot-source,readonly",
        '--mount',"type=bind,source=$SeedRoot,target=/study")
    if ($Script -eq 'reports.py') {
        $Arguments+=@('--mount',"type=bind,source=$OutputRoot,target=/study-all")
    }
    if ($Mode -eq 'plan') {
        $Arguments+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/datasets/train,readonly",
                      '--mount',"type=bind,source=$($Datasets['validation-10']),target=/datasets/validation,readonly")
    } elseif ($Mode -eq 'external') {
        $Arguments+=@('--mount',"type=bind,source=$DataRoot,target=/acdc,readonly")
    } elseif ($Mode -eq 'transfer') {
        $Arguments+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly")
    }
    if ($Mode -eq 'test') {
        if ($Stage -ne 'test' -or -not $Datasets.ContainsKey('test-38')) { throw 'Explicit gated test stage required.' }
        $Arguments+=@('--mount',"type=bind,source=$($Datasets['test-38']),target=/test-data,readonly")
    }
    if ($Gpu) { $Arguments+=@('--gpus','device=0','--shm-size','4g') }
    $Arguments+=@('--entrypoint','/usr/bin/timeout',$Images.$Kind,'--signal=TERM','--kill-after=30s',"${Seconds}s",
                  $Python,"/checkout/diagnostics/medcotracker-acdc-full-v1/$Script")+$Extra
    if (-not $Log) { $Log=Join-Path $OutputRoot 'audit.log' }
    Invoke-QuietDocker $Arguments $Log
}

function Invoke-Transfer([string]$Phase) {
    foreach ($Arm in $Arms) {
        Invoke-Worker 'study.py' @('handoff','--phase',$Phase,'--arm',$Arm)
        $Folder=Join-Path $OutputRoot "seed_$script:Seed\$Phase\train\$Arm"
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        $Budget=if ($Phase -eq 'smoke') {900} else {10800}
        Write-Host "TRANSFER seed_$script:Seed $Phase/$Arm; log=$Folder\training.log"
        Invoke-Worker 'train_transfer.py' @('--phase',$Phase,'--arm',$Arm) -Mode transfer -Gpu -Seconds $Budget -Log (Join-Path $Folder 'training.log')
    }
    Invoke-Worker 'study.py' @('audit','--phase',$Phase)
}

function Invoke-Smoke {
    if (Test-Path -LiteralPath (Join-Path $OutputRoot "seed_$script:Seed\smoke\verified.json")) {
        Invoke-Worker 'study.py' @('gate')
        return
    }
    Invoke-Transfer 'smoke'
    Invoke-Worker 'study.py' @('inference-smoke') -Kind Inference -Gpu
}


function Invoke-Full {
    foreach ($s in $Seeds) {
        $script:Seed=$s
        Invoke-Worker 'study.py' @('gate')
    }
    foreach ($s in $Seeds) {
        $script:Seed=$s
        $Folder=Join-Path $OutputRoot "seed_$s\full\external\B1"
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        Write-Host "EXTERNAL seed_$s B1 3000 steps; log=$Folder\training.log"
        Invoke-Worker 'external.py' @('--route','B1') -Mode external -Gpu -Seconds 28800 -Log (Join-Path $Folder 'training.log')
        Invoke-Transfer 'full'
    }
}

function Invoke-Evaluation([string]$Split) {
    foreach ($s in $Seeds) {
        $script:Seed=$s
        Invoke-Worker 'study.py' @('audit','--phase','full')
    }
    $Runner=Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    foreach ($s in $Seeds) {
        $script:Seed=$s
        $SeedRoot=Join-Path $OutputRoot "seed_$s"
        $EvaluationArms=if ($s -eq 0) { @('pretrained')+$Arms } else { $Arms }
        foreach ($Arm in $EvaluationArms) {
            Invoke-Worker 'reports.py' @('eval-bind','--split',$Split,'--seed',"$s",'--arm',$Arm)
            $Checkpoint=if ($Arm -eq 'pretrained') {Join-Path $SeedRoot 'pretrained-frozen.pth'} else {Join-Path $SeedRoot "full\train\$Arm\cotracker_three_final.pth"}
            $Destination=Join-Path $SeedRoot "full\$Split\$Arm"
            New-Item -ItemType Directory -Force -Path $Destination | Out-Null
            Write-Host "EVALUATE $Split seed_$s/$Arm; log=$Destination\launcher.log"
            $EvalArgs=@{Repository=$RepoRoot;Dataset=$Datasets[$Split];Results=$Destination;
                Profiles=@('hierarchical_full_grid0_iterations2_memory_topk_diverse');
                ModelCheckpoint=$Checkpoint;RequireDiagnostics=$true;FreezeImages=$true;UsePinnedImages=$true;
                ReferenceImagesPath=(Join-Path $SeedRoot 'reference-images.json');
                SidecarRuntimeConfig=(Join-Path $SeedRoot 'runtime.json')}
            & $Runner @EvalArgs *> (Join-Path $Destination 'launcher.log')
        }
    }
    $script:Seed=0
    Invoke-Worker 'reports.py' @('report','--split',$Split) -Seconds 3600
}

try {
    Start-Transcript -Path (Join-Path $OutputRoot 'queue.log') -Append | Out-Null
    $Transcript=$true
    Write-Host "MedCoTracker FULL stage=$Stage; B1/R1; seeds0/1/2; external3000 + transfer1000. No build, pull or network."
    foreach ($s in $Seeds) {
        $script:Seed=$s
        Invoke-Worker 'study.py' @('plan') -Mode plan
        Invoke-Worker 'study.py' @('runtime') -Kind Inference
    }
    $script:Seed=0
    Invoke-Worker 'reports.py' @('freeze-plan')
    if ($Stage -in @('smoke','all')) {
        foreach ($s in $Seeds) { $script:Seed=$s; Invoke-Smoke }
    }
    if ($Stage -in @('full','all')) { Invoke-Full }
    if ($Stage -in @('full','validation','all')) { Invoke-Evaluation 'validation-10' }
    if ($Stage -eq 'test') {
        $script:Seed=0
        # Gate BEFORE resolving/mounting any test directory.
        Invoke-Worker 'reports.py' @('test-gate') -Seconds 3600
        $Rows=@($Protocol.Dataset | Where-Object Split -eq 'test-38')
        if (-not $Rows.Count) { throw 'Missing test inventory.' }
        $First=$Rows[0]
        $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
        if ($Offset -lt 0) { throw 'Cannot resolve frozen test root.' }
        $TestRoot=(Resolve-Path -LiteralPath $First.File.Substring(0,$Offset)).Path.TrimEnd('\')
        if ($TestRoot -match '[",\r\n]' -or $TestRoot -eq $OutputRoot -or
            $TestRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
            $OutputRoot.StartsWith($TestRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid test path.' }
        $Datasets['test-38']=$TestRoot
        Invoke-Worker 'reports.py' @('test-data') -Mode test -Seconds 3600
        Invoke-Evaluation 'test-38'
    }
    Write-Host "COMPLETED stage=$Stage : $OutputRoot. all/full stop after validation-10; test-38 requires -Stage test explicitly."
} finally {
    if ($Transcript) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
