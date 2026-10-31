[CmdletBinding()]
param(
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$EvidenceRoot='C:\zhongliu\lld-fdiag-v1',
    [string]$OutputRoot='C:\zhongliu\lld-numerics-v1',
    [string]$PreviousRoot='C:\zhongliu\rs-v1',
    [ValidateSet('F0','Fprobe','F001','F010')][string]$Arm='F0'
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$EvidenceRoot=(Resolve-Path -LiteralPath $EvidenceRoot).Path.TrimEnd('\')
$PreviousRoot=(Resolve-Path -LiteralPath $PreviousRoot).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -le 3) { throw 'Use a dedicated output directory' }
foreach ($Path in @($RepoRoot,$EvidenceRoot,$PreviousRoot)) {
    if ($Path -match '[",\r\n]' -or $OutputRoot -match '[",\r\n]' -or $Path -eq $OutputRoot -or
        $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must not overlap inputs' }
}
$Old=Get-Content -LiteralPath (Join-Path $PreviousRoot 'manifest.json') -Raw | ConvertFrom-Json
$Image=$Old.images.Training
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
if ($Image -notmatch '^sha256:[0-9a-f]{64}$' -or (Get-LocalImageId $Image) -ne $Image) { throw 'Pinned local training image missing; no build/pull attempted' }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
try {
    Write-Host 'Read-only CPU numerical audit of saved tensors. No GPU, training, dataset access, network, build, pull or tolerance changes.'
    $ArgsList=@('run','--rm','--pull','never','--network','none','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$EvidenceRoot,target=/evidence,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/output",
        '--entrypoint','/usr/bin/timeout',$Image,'--signal=TERM','--kill-after=30s','600s',
        '/opt/app/.pixi/envs/training/bin/python','/checkout/diagnostics/lld-numerical-audit-v1/analyze.py','--arm',$Arm)
    Invoke-LocalDocker $ArgsList (Join-Path $OutputRoot 'audit.log')
    Write-Host "COMPLETED: $OutputRoot\summary.json. This does not authorize restarting training."
} finally { $Lock.Dispose() }
