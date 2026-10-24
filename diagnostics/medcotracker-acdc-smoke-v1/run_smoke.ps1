[CmdletBinding()]
param(
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$AuditRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$WeightsRoot='',
    [string]$OutputRoot='C:\zhongliu\acdc-ab-gpu-smoke-v1',
    [string]$RuntimeImage=''
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
if (-not $WeightsRoot) { $WeightsRoot=Join-Path $RepoRoot 'training-checkpoints' }
$DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
$AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
$WeightsRoot=(Resolve-Path -LiteralPath $WeightsRoot).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
foreach ($Path in @($RepoRoot,$DataRoot,$AuditRoot,$WeightsRoot,$OutputRoot)) {
    if ($Path -match '[",\r\n]' -or $Path.Length -le 3) { throw 'Use dedicated ordinary absolute directories.' }
}
foreach ($Path in @($RepoRoot,$DataRoot,$AuditRoot,$WeightsRoot)) {
    if ($Path -eq $OutputRoot -or $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must be separate from source, data, weights and audit directories.'
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $AuditRoot 'data-manifest.json'))) { throw 'Run native-v2 preparation first.' }
foreach ($Name in @('cotracker2v1.pth','baseline_online.pth','scaled_offline.pth')) {
    if (-not (Test-Path -LiteralPath (Join-Path $WeightsRoot $Name) -PathType Leaf)) { throw "Missing fixed weight: $Name" }
}
if (-not $RuntimeImage) {
    $Images=Get-Content -LiteralPath (Join-Path $PriorStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
    $RuntimeImage=$Images.Training
}
if ($RuntimeImage -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Use an existing pinned image ID, not a mutable tag.' }
$OldPreference=$ErrorActionPreference
$ErrorActionPreference='Continue'
try {
    $Engine=& docker info --format '{{.OSType}}' 2>&1
    $DockerStatus=$LASTEXITCODE
    $Actual=& docker image inspect $RuntimeImage --format '{{.Id}}' 2>&1
    $InspectStatus=$LASTEXITCODE
} finally { $ErrorActionPreference=$OldPreference }
if ($DockerStatus -ne 0 -or [string]$Engine -ne 'linux') { throw 'Start Docker Desktop Linux engine. No image build/pull attempted.' }
if ($InspectStatus -ne 0 -or [string]$Actual -ne $RuntimeImage) { throw 'Pinned training runtime unavailable; no build/pull attempted.' }
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
try {
    Write-Host 'ACDC external GPU smoke: 6 teacher interfaces + 4 routes x 5 updates. No TrackRAD, pilot, test, network, build or pull.'
    $Arguments=@('run','--rm','--pull','never','--network','none','--gpus','all','--cpus','4',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--env',"ACDC_SMOKE_IMAGE_ID=$RuntimeImage",
        '--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly",
        '--mount',"type=bind,source=$DataRoot,target=/data,readonly",
        '--mount',"type=bind,source=$AuditRoot,target=/audit,readonly",
        '--mount',"type=bind,source=$WeightsRoot,target=/weights,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/output",
        '--entrypoint','/usr/bin/timeout',$RuntimeImage,'--signal=TERM','--kill-after=30s','1800s',
        '/opt/app/.pixi/envs/training/bin/python',
        '/checkout/diagnostics/medcotracker-acdc-smoke-v1/smoke.py')
    $Log=Join-Path $OutputRoot 'smoke.log'
    $OldPreference=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object { Write-Host ([string]$_) }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$OldPreference }
    if ($Status -ne 0) { throw "GPU smoke failed ($Status); retain all attempts. No larger queue starts. Log: $Log" }
    $Summary=Get-Content -LiteralPath (Join-Path $OutputRoot 'summary.json') -Raw | ConvertFrom-Json
    if (-not $Summary.complete -or $Summary.stage -ne 'external-gpu-smoke' -or
        $Summary.trackrad_training_started -or $Summary.pilot_started) { throw 'Missing/invalid external smoke completion proof.' }
    Write-Host "EXTERNAL GPU SMOKE COMPLETE: $OutputRoot\summary.json. No automatic next stage."
} finally { $Lock.Dispose() }
