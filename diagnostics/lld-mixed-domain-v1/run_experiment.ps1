[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','pilot','full','all','test')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PreviousRoot='C:\zhongliu\rs-v1',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\lld-mmri-v1',
    [string]$AuditRoot='C:\zhongliu\lld-qc-v1',
    [string]$OutputRoot='C:\zhongliu\lld-da-v1'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PreviousRoot=(Resolve-Path -LiteralPath $PreviousRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$Weights=(Resolve-Path -LiteralPath (Join-Path $RepoRoot 'training-checkpoints')).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -gt 35 -or $OutputRoot.Length -le 3) { throw 'Use a short separate OutputRoot, e.g. C:\zhongliu\lld-da-v1' }
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
$AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
$OldManifest=Get-Content -LiteralPath (Join-Path $PreviousRoot 'manifest.json') -Raw | ConvertFrom-Json
$Images=$OldManifest.images
$Datasets=@{}
function Check-InputPath([string]$Path) {
    if ($Path -match '[",\r\n]' -or $OutputRoot -match '[",\r\n]' -or $Path -eq $OutputRoot -or
        $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must not overlap any input' }
}
function Resolve-Dataset([string]$Split) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Missing dataset inventory $Split" }
    $Row=$Rows[0]; $Index=$Row.File.IndexOf('\'+$Row.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Index -lt 0) { throw 'Invalid dataset path' }
    $Path=(Resolve-Path -LiteralPath $Row.File.Substring(0,$Index)).Path.TrimEnd('\')
    Check-InputPath $Path; $Datasets[$Split]=$Path
}
foreach ($Path in @($RepoRoot,$PreviousRoot,$PriorStudy,$Weights,$AuditRoot)) { Check-InputPath $Path }
if ($Stage -in @('all','smoke','pilot','full')) {
    Resolve-Dataset 'train-40'; Resolve-Dataset 'validation-10'
    $DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
    $AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
    Check-InputPath $DataRoot; Check-InputPath $AuditRoot
}
if ($Stage -eq 'test') { Resolve-Dataset 'test-38' }
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
foreach ($Kind in @('Training','Evaluation')) {
    if ($Images.$Kind -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Images.$Kind) -ne $Images.$Kind) {
        throw "Pinned $Kind image not available; no build or pull attempted"
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Package='/checkout/diagnostics/lld-mixed-domain-v1'
$PilotArms=@('M0','S0','S001','S010','F0','Fprobe','F001','F010')
$script:MetricPayload=$null

function Invoke-LoggedDocker([string[]]$Arguments,[string]$Log) {
    $Saved=$ErrorActionPreference; $ErrorActionPreference='Continue'; $script:MetricPayload=$null
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line.StartsWith('REGION_METRICS_BASE64=')) { $script:MetricPayload=$Line.Substring(22) }
            elseif ($Line -match '^(PLAN |TRAIN |CACHE |FINISHED |AUDIT |INFERENCE |RESULT |REPORT )') { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Saved }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "Domain study failed ($Status). Preserve results; log: $Log"
    }
}

function Invoke-Worker([string[]]$Extra,[string]$Mode='audit',[int]$Seconds=3600,[switch]$Gpu,[string]$Log='') {
    $ArgsList=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--env','COTRACKER_SOURCE_DIR=/checkout/cotracker-algorithm/ext/co-tracker',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$PreviousRoot,target=/previous,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/study",
        '--mount',"type=bind,source=$(Join-Path $AuditRoot 'data-manifest.json'),target=/audit/data-manifest.json,readonly")
    if ($Mode -eq 'train') {
        $ArgsList+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly",
            '--mount',"type=bind,source=$DataRoot,target=/lld,readonly")
    } elseif ($Mode -in @('validation-10','test-38')) {
        if (-not $Datasets.ContainsKey($Mode)) { throw 'This stage does not authorize the dataset' }
        $ArgsList+=@('--mount',"type=bind,source=$($Datasets[$Mode]),target=/data,readonly")
    }
    if ($Gpu) { $ArgsList+=@('--gpus','device=0','--shm-size','4g') }
    $ArgsList+=@('--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=30s',"${Seconds}s",
        '/opt/app/.pixi/envs/training/bin/python',"$Package/worker.py")+$Extra
    if (-not $Log) { $Log=Join-Path $OutputRoot 'audit.log' }
    Invoke-LoggedDocker $ArgsList $Log
}

function Evaluate-Output([string]$Phase,[string]$Split,[int]$Seed,[string]$Arm) {
    $Relative="$Phase/$Split/seed_$Seed/$Arm"; $Folder=Join-Path $OutputRoot $Relative
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
    Write-Host "EVALUATE $Relative; log=$Folder\inference.log"
    Invoke-Worker @('infer','--phase',$Phase,'--split',$Split,'--seed',"$Seed",'--arm',$Arm) -Mode $Split -Gpu -Seconds 14400 -Log (Join-Path $Folder 'inference.log')
    if ($Phase -eq 'smoke') { return }
    if (-not (Test-Path -LiteralPath (Join-Path $Folder 'metrics.json'))) {
        $ArgsList=@('run','--rm','--pull','never','--network','none','--cpus','4',
            '--mount',"type=bind,source=$Folder,target=/input,readonly",
            '--mount',"type=bind,source=$($Datasets[$Split]),target=/opt/app/ground_truth,readonly",
            '--mount',"type=bind,source=$(Join-Path $RepoRoot 'diagnostics\region-supervision-v1\eval_transport.py'),target=/transport.py,readonly",
            '--tmpfs','/output:rw,mode=1777,size=512m','--entrypoint','/usr/bin/timeout',
            $Images.Evaluation,'--signal=TERM','--kill-after=30s','7200s','python','/transport.py')
        Invoke-LoggedDocker $ArgsList (Join-Path $Folder 'evaluation.log')
        if (-not $script:MetricPayload) { throw 'No official metric payload' }
        $MetricPath=Join-Path $Folder 'metrics.json'
        [IO.File]::WriteAllBytes($MetricPath+'.tmp',[Convert]::FromBase64String($script:MetricPayload))
        Move-Item -LiteralPath ($MetricPath+'.tmp') -Destination $MetricPath
    }
    Invoke-Worker @('record','--folder',$Relative)
}

function Run-Phase([string]$Phase,[string[]]$Arms,[int[]]$Seeds) {
    foreach ($Seed in $Seeds) { foreach ($Arm in $Arms) {
        $Folder=Join-Path $OutputRoot "$Phase\train\seed_$Seed\$Arm"
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        Write-Host "TRAIN START $Phase seed=$Seed $Arm; log=$Folder\training.log"
        Invoke-Worker @('train','--phase',$Phase,'--arm',$Arm,'--seed',"$Seed") -Mode train -Gpu -Seconds 43200 -Log (Join-Path $Folder 'training.log')
    } }
    Invoke-Worker @('audit','--phase',$Phase)
    if ($Phase -ne 'smoke') { Evaluate-Output $Phase 'validation-10' 0 'pretrained' }
    foreach ($Seed in $Seeds) { foreach ($Arm in $Arms) { Evaluate-Output $Phase 'validation-10' $Seed $Arm } }
    if ($Phase -eq 'smoke') { Invoke-Worker @('audit','--phase','smoke','--require-inference') }
    else { Invoke-Worker @('report','--phase',$Phase,'--split','validation-10') }
}

try {
    Write-Host "LLD synthetic mixed study stage=$Stage. 8 pilot arms; frozen and partial-encoder branches. No build/pull/network; serial GPU."
    Invoke-Worker @('plan') -Seconds 7200
    if ($Stage -eq 'plan') { return }
    if ($Stage -in @('all','smoke','pilot')) { Run-Phase 'smoke' $PilotArms @(0) }
    if ($Stage -in @('all','pilot')) { Run-Phase 'pilot' $PilotArms @(0) }
    if ($Stage -in @('all','full','test')) {
        Invoke-Worker @('roster')
        $Roster=Get-Content -LiteralPath (Join-Path $OutputRoot 'full-roster.json') -Raw | ConvertFrom-Json
        $FullArms=@($Roster.arms)
        if (-not $FullArms.Count) {
            if ($Stage -eq 'test') { throw 'Pilot gate closed; no full test authorized' }
            Write-Host 'PILOT GATE CLOSED: no large training. Review validation report.'
            return
        }
        if ($Stage -eq 'test') {
            Invoke-Worker @('test-gate')
            Evaluate-Output 'full' 'test-38' 0 'pretrained'
            foreach ($Seed in @(0,1,2)) { foreach ($Arm in $FullArms) { Evaluate-Output 'full' 'test-38' $Seed $Arm } }
            Invoke-Worker @('report','--phase','full','--split','test-38')
        } else { Run-Phase 'full' $FullArms @(0,1,2) }
    }
    Write-Host "COMPLETED stage=$Stage : $OutputRoot. all stops after validation; test-38 requires explicit -Stage test."
} finally { $Lock.Dispose() }
