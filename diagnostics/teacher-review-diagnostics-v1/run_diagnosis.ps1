[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','full','report','all')][string]$Stage='plan',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$CollaborationRoot='C:\zhongliu\tc-v1',
    [string]$OutputRoot='C:\zhongliu\trd-rangefix-v2'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$PriorStudy=(Resolve-Path -LiteralPath $PriorStudy).Path.TrimEnd('\')
$CollaborationRoot=(Resolve-Path -LiteralPath $CollaborationRoot).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$PackageRoot=$PSScriptRoot
$Weights=Join-Path $RepoRoot 'training-checkpoints'
if ($OutputRoot.Length -gt 48 -or [IO.Path]::GetPathRoot($OutputRoot) -eq $OutputRoot+'\') {
    throw 'Use a short dedicated OutputRoot such as C:\zhongliu\trd-rangefix-v2.'
}
$Protocol=Get-Content -LiteralPath (Join-Path $PriorStudy 'frozen-study.json') -Raw | ConvertFrom-Json
$Study=Get-Content -LiteralPath (Join-Path $CollaborationRoot 'manifest.json') -Raw | ConvertFrom-Json
$Datasets=@{}
foreach ($Split in @('train-40','validation-10')) {
    $Rows=@($Protocol.Dataset | Where-Object Split -eq $Split)
    if (-not $Rows.Count) { throw "Missing inventory $Split" }
    $First=$Rows[0]
    $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot resolve dataset root.' }
    $Datasets[$Split]=(Resolve-Path -LiteralPath $First.File.Substring(0,$Offset)).Path.TrimEnd('\')
}
foreach ($InputRoot in @($RepoRoot,$PriorStudy,$CollaborationRoot,$PackageRoot,$Weights,$Datasets['train-40'],$Datasets['validation-10'])) {
    if ($InputRoot -match '[",\r\n]' -or $OutputRoot -match '[",\r\n]' -or
        $OutputRoot -eq $InputRoot -or $OutputRoot.StartsWith($InputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $InputRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Output directory must not overlap any input; quotes/commas/newlines unsupported.'
    }
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Image=$Study.images.Training
if ($Image -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Image) -ne $Image) {
    throw 'Start Docker Desktop and make the frozen local training image available. No build/pull attempted.'
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
function Invoke-Probe([string]$Command) {
    $Budget=if ($Command -eq 'smoke') {'1200s'} else {'7200s'}
    $Arguments=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
        '--env',"DIAGNOSTIC_IMAGE_ID=$Image",
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$PriorStudy,target=/prior,readonly",
        '--mount',"type=bind,source=$CollaborationRoot,target=/collaboration,readonly",
        '--mount',"type=bind,source=$Weights,target=/checkpoints,readonly",
        '--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
        '--mount',"type=bind,source=$($Datasets['train-40']),target=/datasets/train,readonly",
        '--mount',"type=bind,source=$($Datasets['validation-10']),target=/datasets/validation,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/study")
    if ($Command -in @('smoke','full')) { $Arguments+=@('--gpus','device=0','--shm-size','4g') }
    $Arguments+=@('--entrypoint','/usr/bin/timeout',$Image,'--signal=TERM','--kill-after=30s',$Budget,
        '/opt/app/.pixi/envs/training/bin/python','/package/diagnose.py',$Command)
    Write-Host "START $Command; log=$OutputRoot\$Command.log"
    $Previous=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $OutputRoot "$Command.log") -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line -match '^(PLAN |COMMITTED |CACHE |PROBES |REPORT )') { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Previous }
    if ($Status -ne 0) {
        Get-Content -LiteralPath (Join-Path $OutputRoot "$Command.log") -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "Diagnostic failed ($Status). Keep results and repeat unchanged command after fixing the cause."
    }
}
try {
    Write-Host 'Frozen teacher-label verification only: no student training, test-38 access, build, pull or container network.'
    if ($Stage -eq 'all') {
        Invoke-Probe 'smoke'
        Invoke-Probe 'full'
        Invoke-Probe 'report'
    } elseif ($Stage -eq 'full') {
        Invoke-Probe 'full'
        Invoke-Probe 'report'
    } else { Invoke-Probe $Stage }
    Write-Host "COMPLETED $Stage : $OutputRoot"
} finally { $Lock.Dispose() }
