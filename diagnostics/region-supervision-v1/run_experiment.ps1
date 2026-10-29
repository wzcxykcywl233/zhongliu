[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','pilot','full','test','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$AuditRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$OutputRoot='C:\zhongliu\rs-v1'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
$AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
$Weights=(Resolve-Path -LiteralPath (Join-Path $RepoRoot 'training-checkpoints')).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -gt 40 -or $OutputRoot.Length -le 3) { throw 'Use a short dedicated OutputRoot, e.g. C:\zhongliu\rs-v1' }
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
$Images=Get-Content -LiteralPath (Join-Path $PriorStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
$Datasets=@{}
function Resolve-Dataset([string]$Split) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Missing inventory $Split" }
    $Row=$Rows[0]; $Index=$Row.File.IndexOf('\'+$Row.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Index -lt 0) { throw 'Invalid inventory path' }
    $Datasets[$Split]=(Resolve-Path -LiteralPath $Row.File.Substring(0,$Index)).Path.TrimEnd('\')
}
Resolve-Dataset 'train-40'
Resolve-Dataset 'validation-10'
foreach ($Path in @($RepoRoot,$PriorStudy,$DataRoot,$AuditRoot,$Weights,$Datasets['train-40'],$Datasets['validation-10'])) {
    if ($Path -match '[",\r\n]' -or $OutputRoot -match '[",\r\n]' -or $Path -eq $OutputRoot -or
        $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must not overlap inputs; ordinary paths only' }
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
foreach ($Kind in @('Training','Evaluation')) {
    if ($Images.$Kind -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Images.$Kind) -ne $Images.$Kind) {
        throw "Pinned $Kind image unavailable. No image build or pull attempted."
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Package='/checkout/diagnostics/region-supervision-v1'
$script:MetricPayload=$null

function Invoke-LoggedDocker([string[]]$Arguments,[string]$Log) {
    $Saved=$ErrorActionPreference; $ErrorActionPreference='Continue'; $script:MetricPayload=$null
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line.StartsWith('REGION_METRICS_BASE64=')) { $script:MetricPayload=$Line.Substring(22) }
            elseif ($Line -match '^(PLAN |SOURCE |TRAIN |CACHE |FINISHED |AUDIT |INFER |INFERENCE |RESULT |REPORT |TEST )') { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Saved }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 30 | ForEach-Object { Write-Host $_ }
        throw "Region study failed ($Status); preserve results. Repeat identical command after diagnosis. Log: $Log"
    }
}

function Invoke-Worker([string]$Script,[string[]]$Extra=@(),[string]$Mode='audit',[int]$Seconds=1800,[switch]$Gpu,[string]$Log='') {
    $ArgsList=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--env','COTRACKER_SOURCE_DIR=/checkout/cotracker-algorithm/ext/co-tracker',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/study")
    if ($Mode -eq 'plan') {
        foreach ($Leaf in @('frozen-study.json','local-runtime-images.json')) {
            $ArgsList+=@('--mount',"type=bind,source=$(Join-Path $PriorStudy $Leaf),target=/prior/$Leaf,readonly")
        }
        $ArgsList+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/train,readonly",
            '--mount',"type=bind,source=$($Datasets['validation-10']),target=/validation,readonly")
    }
    if ($Mode -in @('plan','train')) {
        $ArgsList+=@('--mount',"type=bind,source=$DataRoot,target=/acdc,readonly",
            '--mount',"type=bind,source=$(Join-Path $AuditRoot 'data-manifest.json'),target=/audit/data-manifest.json,readonly")
    }
    if ($Mode -eq 'train') { $ArgsList+=@('--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly") }
    if ($Mode -in @('validation-10','test-38')) {
        if ($Mode -eq 'test-38' -and ($Stage -ne 'test' -or -not $Datasets.ContainsKey('test-38'))) { throw 'Explicit test gate required' }
        $ArgsList+=@('--mount',"type=bind,source=$($Datasets[$Mode]),target=/data,readonly")
    }
    if ($Gpu) { $ArgsList+=@('--gpus','device=0','--shm-size','4g') }
    $ArgsList+=@('--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=30s',"${Seconds}s",
        '/opt/app/.pixi/envs/training/bin/python',"$Package/$Script")+$Extra
    if (-not $Log) { $Log=Join-Path $OutputRoot 'audit.log' }
    Invoke-LoggedDocker $ArgsList $Log
}

function Evaluate-Output([string]$Phase,[string]$Split,[int]$Seed,[string]$Arm,[string]$View,[switch]$SmokeCase) {
    $Folder=Join-Path $OutputRoot "$Phase\$Split\seed_$Seed\$Arm\$View"
    New-Item -ItemType Directory -Force -Path $Folder | Out-Null
    $Options=@('--phase',$Phase,'--split',$Split,'--seed',"$Seed",'--arm',$Arm,'--view',$View)
    if ($SmokeCase) { $Options+='--smoke-case' }
    Invoke-Worker 'infer.py' $Options -Mode $Split -Gpu -Seconds 10800 -Log (Join-Path $Folder 'inference.log')
    $Relative="$Phase/$Split/seed_$Seed/$Arm/$View"
    if (Test-Path -LiteralPath (Join-Path $Folder 'metrics.json')) {
        Invoke-Worker 'reports.py' @('record','--folder',$Relative)
        return
    }
    # Official image, original user, original Python and source. Only /output is writable tmpfs.
    $ArgsList=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--mount',"type=bind,source=$Folder,target=/input,readonly",
        '--mount',"type=bind,source=$($Datasets[$Split]),target=/opt/app/ground_truth,readonly",
        '--mount',"type=bind,source=$(Join-Path $RepoRoot 'diagnostics\region-supervision-v1\eval_transport.py'),target=/transport.py,readonly",
        '--tmpfs','/output:rw,mode=1777,size=512m','--entrypoint','/usr/bin/timeout',
        $Images.Evaluation,'--signal=TERM','--kill-after=30s','7200s','python','/transport.py')
    Invoke-LoggedDocker $ArgsList (Join-Path $Folder 'evaluation.log')
    if (-not $script:MetricPayload) { throw 'Official evaluator produced no metrics payload' }
    $Bytes=[Convert]::FromBase64String($script:MetricPayload)
    $MetricPath=Join-Path $Folder 'metrics.json'
    if (Test-Path -LiteralPath $MetricPath) {
        $Old=[IO.File]::ReadAllBytes($MetricPath)
        if ([Convert]::ToBase64String($Old) -ne $script:MetricPayload) { throw 'Official metrics changed on replay; retain existing file' }
    } else { [IO.File]::WriteAllBytes($MetricPath,$Bytes) }
    Invoke-Worker 'reports.py' @('record','--folder',$Relative)
}

function Train-Phase([string]$Phase,[string[]]$Arms,[int[]]$Seeds) {
    foreach ($Seed in $Seeds) { foreach ($Arm in $Arms) {
        $Folder=Join-Path $OutputRoot "$Phase\train\seed_$Seed\$Arm"
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        Write-Host "TRAIN START $Phase seed=$Seed $Arm; log=$Folder\training.log"
        $Seconds=if ($Phase -eq 'smoke') {1800} elseif ($Phase -eq 'pilot') {14400} else {86400}
        Invoke-Worker 'train.py' @('--phase',$Phase,'--seed',"$Seed",'--arm',$Arm) -Mode train -Gpu -Seconds $Seconds -Log (Join-Path $Folder 'training.log')
    } }
}

function Evaluate-Phase([string]$Phase,[string]$Split,[string[]]$Arms,[int[]]$Seeds) {
    Invoke-Worker 'reports.py' @('audit','--phase',$Phase)
    Evaluate-Output $Phase $Split 0 'pretrained' 'points'
    foreach ($Seed in $Seeds) { foreach ($Arm in $Arms) {
        $Views=if ($Arm -eq 'D1') {@('points','region')} elseif ($Arm -eq 'G1') {@('region')} else {@('points')}
        foreach ($View in $Views) { Evaluate-Output $Phase $Split $Seed $Arm $View }
    } }
    Invoke-Worker 'reports.py' @('report','--phase',$Phase,'--split',$Split)
}

try {
    Write-Host "Mixed region study stage=$Stage. Frozen backbone; no build/pull/network. all stops after validation."
    Invoke-Worker 'study.py' @('plan') -Mode plan -Seconds 7200
    if ($Stage -eq 'plan') { return }
    if ($Stage -in @('smoke','all')) {
        Train-Phase 'smoke' @('M0','W1','D1','G1') @(0)
        Evaluate-Output 'smoke' 'validation-10' 0 'M0' 'points' -SmokeCase
        Evaluate-Output 'smoke' 'validation-10' 0 'D1' 'region' -SmokeCase
        Invoke-Worker 'reports.py' @('audit','--phase','smoke')
        if ($Stage -eq 'smoke') { return }
    }
    if ($Stage -in @('pilot','all')) {
        Train-Phase 'pilot' @('M0','W1','D1') @(0)
        Evaluate-Phase 'pilot' 'validation-10' @('M0','W1','D1') @(0)
        if ($Stage -eq 'pilot') { return }
    }
    $Decision=Get-Content -LiteralPath (Join-Path $OutputRoot 'reports\pilot\validation-10\selection.json') -Raw | ConvertFrom-Json
    $Arms=@('M0','W1','D1')
    if ($Decision.region_gate) { $Arms+='G1' } else { Write-Host 'REGION GATE CLOSED: G1 large training skipped; M0/W1/D1 remain fixed comparisons.' }
    if ($Stage -in @('full','all')) {
        Train-Phase 'full' $Arms @(0,1,2)
        Evaluate-Phase 'full' 'validation-10' $Arms @(0,1,2)
    } elseif ($Stage -eq 'test') {
        Invoke-Worker 'reports.py' @('test-gate')
        Resolve-Dataset 'test-38'
        $TestRoot=$Datasets['test-38']
        if ($TestRoot -match '[",\r\n]' -or $TestRoot -eq $OutputRoot -or
            $TestRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
            $OutputRoot.StartsWith($TestRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Test path overlaps output' }
        Evaluate-Phase 'full' 'test-38' $Arms @(0,1,2)
    }
    Write-Host "COMPLETED stage=$Stage : $OutputRoot. test-38 requires explicit -Stage test."
} finally { $Lock.Dispose() }
