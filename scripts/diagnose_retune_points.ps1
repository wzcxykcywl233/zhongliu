[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\parameter-retune-gridfix-v2',
    [string]$OutputRoot = '',
    [string]$PythonImage = 'python:3.11-slim'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$ResultsRoot = (Resolve-Path -LiteralPath $ResultsRoot).Path
if (-not $OutputRoot) { $OutputRoot = Join-Path $ResultsRoot 'point-count-diagnostics' }
$OutputFull = [IO.Path]::GetFullPath($OutputRoot)
if ($OutputFull -eq $ResultsRoot) { throw 'OutputRoot must be distinct from ResultsRoot.' }
if (-not (Test-Path -LiteralPath (Join-Path $ResultsRoot 'final\test-38\rt_points_0\metrics.json'))) {
    throw 'The final 38-case rt_points_0 metrics are missing.'
}
New-Item -ItemType Directory -Force -Path $OutputFull | Out-Null
$PriorPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    & docker run --rm --pull never --network none --platform linux/amd64 `
        --mount "type=bind,source=$RepoRoot\cotracker-algorithm\experiments,target=/app,readonly" `
        --mount "type=bind,source=$ResultsRoot,target=/results,readonly" `
        --mount "type=bind,source=$OutputFull,target=/diagnostics" `
        --entrypoint python $PythonImage /app/diagnose_retune_points.py `
        --root /results --output /diagnostics 2>&1 | ForEach-Object { Write-Host $_ }
    $Status = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PriorPreference
}
if ($Status -ne 0) { throw "Point-count diagnosis failed with exit code $Status" }
Write-Host "Point-count diagnosis complete: $OutputFull"
