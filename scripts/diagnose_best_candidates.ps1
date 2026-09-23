[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [ValidateSet('test-38','validation-10')][string]$Split = 'test-38',
    [string]$Dataset = '',
    [string]$PreviousResults = '',
    [string]$OutputRoot = '',
    [ValidateRange(1,10000)][int]$SampleFrames = 12
)
$ErrorActionPreference = 'Stop'
if (-not $Dataset) {
    $Leaf = if ($Split -eq 'test-38') { 'trackrad2025_labeled_public_test_38' } else { 'trackrad2025_labeled_validation_10' }
    $Dataset = Join-Path 'C:\zhongliu\trackrad2025-main\dataset' $Leaf
}
if (-not $PreviousResults) { $PreviousResults = Join-Path $RepoRoot "protocol-40-10-38\query-state-inheritance\$Split" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'protocol-40-10-38\best-candidate-diagnostics' }
$Output = Join-Path $OutputRoot $Split
# Never build or silently replace the original runtime; analysis is CPU-only.
$Identity = Get-Content -LiteralPath (Join-Path $PreviousResults 'frozen-images.json') -Raw | ConvertFrom-Json
$Image = [string]$Identity.'trackrad-algorithm-cotracker-algorithm'
if ($Image -notmatch '^sha256:[0-9a-f]{64}$') { throw 'No frozen inference image found' }
& docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) { throw 'Original frozen image unavailable. Preserve results; do not substitute another image.' }
$Dataset = (Resolve-Path -LiteralPath $Dataset).Path
$PreviousResults = (Resolve-Path -LiteralPath $PreviousResults).Path
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Output = (Resolve-Path -LiteralPath $Output).Path
foreach ($InputFolder in @($Dataset,$PreviousResults)) {
    if ($Output.StartsWith($InputFolder.TrimEnd('\') + '\',[StringComparison]::OrdinalIgnoreCase) -or $Output -eq $InputFolder) {
        throw 'Output must be separate from source dataset and previous results'
    }
}
$Lock = [System.IO.File]::Open((Join-Path $Output '.diagnostic.lock'),'OpenOrCreate','ReadWrite','None')
$Utf8 = New-Object System.Text.UTF8Encoding($false)
try {
    Write-Host "Read-only CPU diagnosis: $Split; no training, inference or image build"
    $Old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker run --rm --network none --platform linux/amd64 `
            --mount "type=bind,source=$Dataset,target=/dataset,readonly" `
            --mount "type=bind,source=$PreviousResults,target=/previous,readonly" `
            --mount "type=bind,source=$RepoRoot\cotracker-algorithm\experiments,target=/diagnostic-source,readonly" `
            --mount "type=bind,source=$Output,target=/diagnostic-output" `
            --entrypoint /opt/app/.pixi/envs/cuda/bin/python $Image `
            /diagnostic-source/diagnose_best_candidates.py --dataset /dataset --previous /previous `
            --output /diagnostic-output --split $Split --sample-frames $SampleFrames --image-id $Image 2>&1 |
            ForEach-Object {
                $Line = $_.ToString()
                Write-Host $Line
                [System.IO.File]::AppendAllText((Join-Path $Output 'runner.log'),$Line + "`n",$Utf8)
            }
        $Status = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Old }
    if ($Status -ne 0) { throw "Diagnostic failed ($Status). Rerun the same command to resume; do not delete old results." }
    Write-Host "Completed: $Output\summary.json"
} finally { $Lock.Dispose() }
