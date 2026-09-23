[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$Results = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\rotation-backcheck\test-38',
    [string]$Output = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\rotation-selection-diagnostics'
)
$ErrorActionPreference='Stop'
$Identity=Get-Content -LiteralPath (Join-Path $Results 'frozen-images.json') -Raw | ConvertFrom-Json
$Image=[string]$Identity.'trackrad-algorithm-cotracker-algorithm'
if ($Image -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Missing recorded image identity' }
& docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) { throw 'Recorded inference image is unavailable; do not rebuild or silently substitute it' }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Lock=[System.IO.File]::Open((Join-Path $Output '.diagnostic.lock'),'OpenOrCreate','ReadWrite','None')
try {
    $Previous=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker run --rm --gpus all --network none --platform linux/amd64 `
            --mount "type=bind,source=$Results,target=/previous,readonly" `
            --mount "type=bind,source=$RepoRoot\cotracker-algorithm,target=/diagnostic-source,readonly" `
            --mount "type=bind,source=$Output,target=/diagnostic-output" `
            --entrypoint /opt/app/.pixi/envs/cuda/bin/python $Image `
            /diagnostic-source/experiments/diagnose_rotation_selection.py `
            --results /previous --output /diagnostic-output --image-id $Image 2>&1 |
            Tee-Object -FilePath (Join-Path $Output 'diagnostic.log') -Append | ForEach-Object { Write-Host $_ }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$Previous }
    if ($Status -ne 0) { throw "Rotation selection diagnostic failed ($Status); rerun same command to resume" }
} finally { $Lock.Dispose() }
