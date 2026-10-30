[CmdletBinding()]
param(
    [ValidateSet('plan','points','diagnose','g1','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PreviousRoot='C:\zhongliu\rs-v1',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$AuditRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$OutputRoot='C:\zhongliu\rs-next-v1'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PreviousRoot=(Resolve-Path -LiteralPath $PreviousRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$Weights=(Resolve-Path -LiteralPath (Join-Path $RepoRoot 'training-checkpoints')).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -gt 35 -or $OutputRoot.Length -le 3) { throw 'Use a short dedicated OutputRoot, e.g. C:\zhongliu\rs-next-v1' }
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
$OldManifest=Get-Content -LiteralPath (Join-Path $PreviousRoot 'manifest.json') -Raw | ConvertFrom-Json
$Images=$OldManifest.images
$Datasets=@{}
function Check-InputPath([string]$Path) {
    if ($Path -match '[",\r\n]' -or $OutputRoot -match '[",\r\n]' -or $Path -eq $OutputRoot -or
        $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must be separate from all input directories' }
}
function Resolve-Dataset([string]$Split) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Missing dataset inventory $Split" }
    $Row=$Rows[0]; $Index=$Row.File.IndexOf('\'+$Row.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Index -lt 0) { throw 'Invalid dataset inventory path' }
    $Path=(Resolve-Path -LiteralPath $Row.File.Substring(0,$Index)).Path.TrimEnd('\')
    Check-InputPath $Path
    $Datasets[$Split]=$Path
}
foreach ($Path in @($RepoRoot,$PreviousRoot,$PriorStudy,$Weights)) { Check-InputPath $Path }
if ($Stage -in @('all','points')) { Resolve-Dataset 'test-38' }
if ($Stage -in @('all','diagnose','g1')) { Resolve-Dataset 'validation-10' }
if ($Stage -in @('all','g1')) {
    Resolve-Dataset 'train-40'
    $DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
    $AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
    Check-InputPath $DataRoot; Check-InputPath $AuditRoot
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
foreach ($Kind in @('Training','Evaluation')) {
    if ($Images.$Kind -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Images.$Kind) -ne $Images.$Kind) {
        throw "Pinned $Kind image unavailable. No build or pull attempted."
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Package='/checkout/diagnostics/region-followup-v1'
$script:MetricPayload=$null

function Invoke-LoggedDocker([string[]]$Arguments,[string]$Log) {
    $Saved=$ErrorActionPreference; $ErrorActionPreference='Continue'; $script:MetricPayload=$null
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line.StartsWith('REGION_METRICS_BASE64=')) { $script:MetricPayload=$Line.Substring(22) }
            elseif ($Line -match '^(PLAN |TRAIN |CACHE |FINISHED |AUDIT |INFERENCE |RESULT |REPORT |DIAGNOSE )') { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Saved }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "Follow-up failed ($Status). Preserve all results; log: $Log"
    }
}

function Invoke-Worker([string[]]$Extra,[string]$Mode='audit',[int]$Seconds=3600,[switch]$Gpu,[string]$Log='') {
    $ArgsList=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--env','COTRACKER_SOURCE_DIR=/checkout/cotracker-algorithm/ext/co-tracker',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$PreviousRoot,target=/previous,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/study")
    if ($Mode -eq 'train') {
        $ArgsList+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly",
            '--mount',"type=bind,source=$DataRoot,target=/acdc,readonly",
            '--mount',"type=bind,source=$(Join-Path $AuditRoot 'data-manifest.json'),target=/audit/data-manifest.json,readonly")
    } elseif ($Mode -in @('validation-10','test-38')) {
        if (-not $Datasets.ContainsKey($Mode)) { throw 'Split not authorized by this stage' }
        $ArgsList+=@('--mount',"type=bind,source=$($Datasets[$Mode]),target=/data,readonly")
    }
    if ($Gpu) { $ArgsList+=@('--gpus','device=0','--shm-size','4g') }
    $ArgsList+=@('--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=30s',"${Seconds}s",
        '/opt/app/.pixi/envs/training/bin/python',"$Package/worker.py")+$Extra
    if (-not $Log) { $Log=Join-Path $OutputRoot 'audit.log' }
    Invoke-LoggedDocker $ArgsList $Log
}

function Evaluate-Output([int]$Seed,[string]$Arm) {
    $Split=if ($Arm -eq 'G1') {'validation-10'} else {'test-38'}
    $Relative=if ($Arm -eq 'G1') {"g1/validation-10/seed_$Seed/G1/region"} else {"points/test-38/seed_$Seed/$Arm/points"}
    $Folder=Join-Path $OutputRoot $Relative
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
    Write-Host "EVALUATE $Split seed=$Seed $Arm; log=$Folder\inference.log"
    Invoke-Worker @('infer','--seed',"$Seed",'--arm',$Arm) -Mode $Split -Gpu -Seconds 14400 -Log (Join-Path $Folder 'inference.log')
    if (-not (Test-Path -LiteralPath (Join-Path $Folder 'metrics.json'))) {
        $ArgsList=@('run','--rm','--pull','never','--network','none','--cpus','4',
            '--mount',"type=bind,source=$Folder,target=/input,readonly",
            '--mount',"type=bind,source=$($Datasets[$Split]),target=/opt/app/ground_truth,readonly",
            '--mount',"type=bind,source=$(Join-Path $RepoRoot 'diagnostics\region-supervision-v1\eval_transport.py'),target=/transport.py,readonly",
            '--tmpfs','/output:rw,mode=1777,size=512m','--entrypoint','/usr/bin/timeout',
            $Images.Evaluation,'--signal=TERM','--kill-after=30s','7200s','python','/transport.py')
        Invoke-LoggedDocker $ArgsList (Join-Path $Folder 'evaluation.log')
        if (-not $script:MetricPayload) { throw 'Official evaluator produced no metric payload' }
        $MetricPath=Join-Path $Folder 'metrics.json'
        [IO.File]::WriteAllBytes($MetricPath+'.tmp',[Convert]::FromBase64String($script:MetricPayload))
        Move-Item -LiteralPath ($MetricPath+'.tmp') -Destination $MetricPath
    }
    Invoke-Worker @('record','--folder',$Relative)
}

try {
    Write-Host "Region follow-up stage=$Stage. Old results read-only; no build/pull/network; GPU jobs serial."
    # Freeze ALL branches before test access, never adapt G1 or diagnostic modes to test metrics.
    Invoke-Worker @('plan') -Seconds 7200
    if ($Stage -eq 'plan') { return }
    $Branches=if ($Stage -eq 'all') {@('points','diagnose','g1')} else {@($Stage)}
    $Failures=@()
    foreach ($Branch in $Branches) {
        try {
            if ($Branch -eq 'points') {
                Evaluate-Output 0 'pretrained'
                foreach ($Seed in @(0,1,2)) { foreach ($Arm in @('M0','W1','D1')) { Evaluate-Output $Seed $Arm } }
                Invoke-Worker @('report','--branch','points')
            } elseif ($Branch -eq 'diagnose') {
                foreach ($Seed in @(0,1,2)) {
                    Invoke-Worker @('diagnose','--seed',"$Seed") -Mode validation-10 -Gpu -Seconds 14400 -Log (Join-Path $OutputRoot "diagnosis-seed-$Seed.log")
                }
                Invoke-Worker @('diagnosis-report')
            } else {
                foreach ($Seed in @(0,1,2)) {
                    Write-Host "TRAIN START exploratory G1 seed=$Seed steps=2000; log=$OutputRoot\g1-seed-$Seed.log"
                    Invoke-Worker @('train','--seed',"$Seed") -Mode train -Gpu -Seconds 86400 -Log (Join-Path $OutputRoot "g1-seed-$Seed.log")
                    Evaluate-Output $Seed 'G1'
                }
                Invoke-Worker @('report','--branch','g1')
            }
        } catch {
            $Failures+=$Branch
            Write-Warning "BRANCH FAILED $Branch : $_. Other independent branches will continue."
        }
    }
    if ($Failures.Count) { throw "Incomplete branches: $($Failures -join ', '). Preserve results; rerun same command." }
    Write-Host "COMPLETED stage=$Stage : $OutputRoot. G1 validation only; no automatic G1 test."
} finally { $Lock.Dispose() }
