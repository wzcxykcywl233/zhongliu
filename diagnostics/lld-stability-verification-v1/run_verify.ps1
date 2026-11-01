[CmdletBinding()]
param(
    [ValidateSet('all','plan','graph','train','report')][string]$Stage='all',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PreviousRoot='C:\zhongliu\rs-v1',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\lld-mmri-v1',
    [string]$AuditRoot='C:\zhongliu\lld-qc-v1',
    [string]$BaselineRoot='C:\zhongliu\lld-da-v1',
    [string]$EvidenceRoot='C:\zhongliu\lld-sstate-v1',
    [string]$OutputRoot='C:\zhongliu\lld-stab-v1'
)
$ErrorActionPreference='Stop'
foreach ($Name in @('RepoRoot','PreviousRoot','PriorStudy','AuditRoot','BaselineRoot','EvidenceRoot')) {
    Set-Variable -Name $Name -Value (Resolve-Path -LiteralPath (Get-Variable -Name $Name -ValueOnly)).Path.TrimEnd('\')
}
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -gt 40 -or $OutputRoot.Length -le 3) { throw 'Use a short dedicated stability output directory' }
$Weights=(Resolve-Path -LiteralPath (Join-Path $RepoRoot 'training-checkpoints')).Path.TrimEnd('\')
$Old=Get-Content -LiteralPath (Join-Path $PreviousRoot 'manifest.json') -Raw | ConvertFrom-Json
$Image=$Old.images.Training
function Check-Path([string]$Path) {
    if ($Path -match '[",\r\n]' -or $OutputRoot -match '[",\r\n]' -or $Path -eq $OutputRoot -or
        $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Stability output must not overlap inputs; no files changed'
    }
}
foreach ($Path in @($RepoRoot,$PreviousRoot,$PriorStudy,$Weights,$AuditRoot,$BaselineRoot,$EvidenceRoot)) { Check-Path $Path }
if ($Stage -in @('all','graph','train')) {
    $DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\'); Check-Path $DataRoot
    $Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
    $Rows=@($Protocol.Dataset | Where-Object Split -eq 'train-40')
    if (-not $Rows.Count) { throw 'Missing train-40 inventory' }
    $Row=$Rows[0]; $Index=$Row.File.IndexOf('\'+$Row.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Index -lt 0) { throw 'Invalid train-40 path' }
    $TrainRoot=(Resolve-Path -LiteralPath $Row.File.Substring(0,$Index)).Path.TrimEnd('\'); Check-Path $TrainRoot
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
if ($Image -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Image) -ne $Image) {
    throw 'Pinned training image unavailable; no build or pull attempted'
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Package='/checkout/diagnostics/lld-stability-verification-v1/study.py'

function Invoke-Check([string[]]$Extra,[switch]$Gpu,[int]$Seconds=3600,[string]$Log) {
    $a=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--env','COTRACKER_SOURCE_DIR=/checkout/cotracker-algorithm/ext/co-tracker',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$PreviousRoot,target=/previous,readonly",
        '--mount',"type=bind,source=$BaselineRoot,target=/baseline,readonly",
        '--mount',"type=bind,source=$EvidenceRoot,target=/evidence,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/study",
        '--mount',"type=bind,source=$(Join-Path $AuditRoot 'data-manifest.json'),target=/audit/data-manifest.json,readonly")
    if ($Gpu) {
        $a+=@('--gpus','device=0','--shm-size','4g',
            '--mount',"type=bind,source=$TrainRoot,target=/data,readonly",
            '--mount',"type=bind,source=$DataRoot,target=/lld,readonly")
    }
    $a+=@('--entrypoint','/usr/bin/timeout',$Image,'--signal=TERM','--kill-after=30s',"${Seconds}s",
        '/opt/app/.pixi/envs/training/bin/python',$Package)+$Extra
    Write-Host "START $($Extra -join ' '); timeout=${Seconds}s; log=$Log"
    $Saved=$ErrorActionPreference; $ErrorActionPreference='Continue'
    try {
        & docker @a 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line -match '^(PLAN |GRAPH |TRAIN |PROBES |CACHE |RESUME |RESULT |REPORT )') { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Saved }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "Stability verification failed ($Status). Preserve results; no larger training follows. Log: $Log"
    }
}

try {
    Write-Host 'Independent 100-update stability check: four F arms; old strict gate unchanged. No validation/test, large queue, build, pull or network.'
    if ($Stage -ne 'report') {
        Invoke-Check @('plan') -Log (Join-Path $OutputRoot 'plan.log')
    }
    if ($Stage -in @('all','graph','train')) {
        Invoke-Check @('graph') -Gpu -Seconds 1200 -Log (Join-Path $OutputRoot 'graph.log')
    }
    if ($Stage -in @('all','train')) {
        foreach ($Arm in @('F0','Fprobe','F001','F010')) {
            Invoke-Check @('train','--arm',$Arm) -Gpu -Seconds 7200 -Log (Join-Path $OutputRoot ($Arm+'.log'))
        }
    }
    if ($Stage -in @('all','train','report')) {
        Invoke-Check @('report') -Log (Join-Path $OutputRoot 'report.log')
        Write-Host "STABILITY VERIFICATION COMPLETE: $OutputRoot\summary.json. Review before any next stage. No full-training authorization."
    }
} finally { $Lock.Dispose() }
