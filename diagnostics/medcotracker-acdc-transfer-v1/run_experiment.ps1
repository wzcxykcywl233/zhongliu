[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','pilot','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$AuditRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$ExternalSmoke='C:\zhongliu\acdc-ab-gpu-smoke-v1',
    [string]$OutputRoot='C:\zhongliu\mct-ab-v1'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
$AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
$ExternalSmoke=(Resolve-Path -LiteralPath $ExternalSmoke).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$Weights=Join-Path $RepoRoot 'training-checkpoints'
if ($OutputRoot.Length -gt 32 -or $OutputRoot.Length -le 3) {
    throw 'Use a short dedicated OutputRoot, at most 32 characters.'
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
foreach ($Path in @($RepoRoot,$PriorStudy,$DataRoot,$AuditRoot,$ExternalSmoke,$Weights,$Datasets['train-40'],$Datasets['validation-10'])) {
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
$Arms=@('R1','A1','B1','B2','B3')

function Invoke-QuietDocker([string[]]$Arguments,[string]$Log) {
    $Previous=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line -match '^(PLAN |RUNTIME |AUDIT |HANDOFF |GPU |PILOT |RESULT |FROZEN |CACHE |SMOKE |EXTERNAL |FINISHED TRAINING)' -or
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
    $Python=if ($Kind -eq 'Training') {'/opt/app/.pixi/envs/training/bin/python'} else {'/opt/app/.pixi/envs/cuda/bin/python'}
    $Arguments=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$PriorStudy,target=/prior,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$AuditRoot,target=/audit,readonly",
        '--mount',"type=bind,source=$ExternalSmoke,target=/external-smoke,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/study")
    if ($Mode -eq 'plan') {
        $Arguments+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/datasets/train,readonly",
                      '--mount',"type=bind,source=$($Datasets['validation-10']),target=/datasets/validation,readonly")
    } elseif ($Mode -eq 'external') {
        $Arguments+=@('--mount',"type=bind,source=$DataRoot,target=/acdc,readonly")
    } elseif ($Mode -eq 'transfer') {
        $Arguments+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly")
    }
    if ($Gpu) { $Arguments+=@('--gpus','device=0','--shm-size','4g') }
    $Arguments+=@('--entrypoint','/usr/bin/timeout',$Images.$Kind,'--signal=TERM','--kill-after=30s',"${Seconds}s",
                  $Python,"/checkout/diagnostics/medcotracker-acdc-transfer-v1/$Script")+$Extra
    if (-not $Log) { $Log=Join-Path $OutputRoot 'audit.log' }
    Invoke-QuietDocker $Arguments $Log
}

function Invoke-Transfer([string]$Phase) {
    foreach ($Arm in $Arms) {
        Invoke-Worker 'study.py' @('handoff','--phase',$Phase,'--arm',$Arm)
        $Folder=Join-Path $OutputRoot "$Phase\train\$Arm"
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        $Budget=if ($Phase -eq 'smoke') {900} else {3600}
        Write-Host "TRANSFER $Phase/$Arm; log=$Folder\training.log"
        Invoke-Worker 'train_transfer.py' @('--phase',$Phase,'--arm',$Arm) -Mode transfer -Gpu -Seconds $Budget -Log (Join-Path $Folder 'training.log')
    }
    Invoke-Worker 'study.py' @('audit','--phase',$Phase)
}

function Invoke-Smoke {
    if (Test-Path -LiteralPath (Join-Path $OutputRoot 'smoke\verified.json')) {
        Invoke-Worker 'study.py' @('gate')
        return
    }
    Invoke-Transfer 'smoke'
    Invoke-Worker 'study.py' @('inference-smoke') -Kind Inference -Gpu
}

function Invoke-Pilot {
    Invoke-Worker 'study.py' @('gate')
    foreach ($Arm in @('A1','B1','B2','B3')) {
        $Folder=Join-Path $OutputRoot "pilot\external\$Arm"
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        Write-Host "EXTERNAL pilot/$Arm 500 steps; log=$Folder\training.log"
        Invoke-Worker 'external.py' @('--route',$Arm) -Mode external -Gpu -Seconds 10800 -Log (Join-Path $Folder 'training.log')
    }
    Invoke-Transfer 'pilot'
    $Runner=Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    foreach ($Arm in @('pretrained')+$Arms) {
        Invoke-Worker 'study.py' @('eval-bind','--arm',$Arm)
        $Checkpoint=if ($Arm -eq 'pretrained') {Join-Path $OutputRoot 'pretrained-frozen.pth'} else {Join-Path $OutputRoot "pilot\train\$Arm\cotracker_three_final.pth"}
        $Destination=Join-Path $OutputRoot "pilot\validation-10\$Arm"
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        Write-Host "EVALUATE validation-10 $Arm; log=$Destination\launcher.log"
        & $Runner -Repository $RepoRoot -Dataset $Datasets['validation-10'] -Results $Destination `
            -Profiles @('hierarchical_full_grid0_iterations2_memory_topk_diverse') `
            -ModelCheckpoint $Checkpoint -RequireDiagnostics -FreezeImages -UsePinnedImages `
            -ReferenceImagesPath (Join-Path $OutputRoot 'reference-images.json') `
            -SidecarRuntimeConfig (Join-Path $OutputRoot 'runtime.json') *> (Join-Path $Destination 'launcher.log')
    }
    Invoke-Worker 'study.py' @('report')
}

try {
    Start-Transcript -Path (Join-Path $OutputRoot 'queue.log') -Append | Out-Null
    $Transcript=$true
    Write-Host "MedCoTracker stage=$Stage; transfer smoke -> seed0 pilot -> validation-10. No test-38, formal training, build, pull or network."
    Invoke-Worker 'study.py' @('plan') -Mode plan
    Invoke-Worker 'study.py' @('runtime') -Kind Inference
    if ($Stage -in @('smoke','all')) { Invoke-Smoke }
    if ($Stage -in @('pilot','all')) { Invoke-Pilot }
    Write-Host "COMPLETED stage=$Stage : $OutputRoot. No automatic three-seed or test stage."
} finally {
    if ($Transcript) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
