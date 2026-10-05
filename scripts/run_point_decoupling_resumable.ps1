[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\point-decoupling-v1',
    [string]$TrainDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40',
    [string]$ValidationDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$TestDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38',
    [string]$PriorResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\parameter-retune-gridfix-v2',
    [ValidateSet('validation','test')][string]$Stage = 'validation'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
if ($PriorResultsRoot) { $PriorResultsRoot = (Resolve-Path -LiteralPath $PriorResultsRoot).Path }
$ReferenceImages = ''
$Split = 'validation-10'
if ($Stage -eq 'test') {
    $Split = 'test-38'
    $ReferenceImages = Join-Path $ResultsRoot 'validation-10\frozen-images.json'
    $AuditPath = Join-Path $ResultsRoot 'validation-10\point-decoupling-audit.json'
    if (-not (Test-Path -LiteralPath $AuditPath)) { throw 'Run and audit validation first.' }
    $Audit = Get-Content -LiteralPath $AuditPath -Raw | ConvertFrom-Json
    if (-not $Audit.complete -or -not $Audit.exact_dense_repeat -or -not $Audit.exact_replay_source -or -not $Audit.exact_matched_queries) {
        throw 'Validation pair audit did not pass.'
    }
    foreach ($Property in $Audit.source_metrics_sha256.PSObject.Properties) {
        $MetricsPath = Join-Path (Join-Path $ResultsRoot 'validation-10') $Property.Name
        if ((Get-FileHash -LiteralPath $MetricsPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Property.Value) {
            throw 'Validation metrics changed after the pair audit.'
        }
    }
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$OuterLock = [IO.File]::Open((Join-Path $ResultsRoot '.point-decoupling.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    & (Join-Path $RepoRoot 'scripts\run_memory_refinement_resumable.ps1') `
        -RepoRoot $RepoRoot -TrainDataset $TrainDataset -ValidationDataset $ValidationDataset `
        -TestDataset $TestDataset -ResultsRoot $ResultsRoot -Stage $Stage `
        -ManifestRelative 'cotracker-algorithm\experiments\point-decoupling-40-10-38.json' `
        -ResultPrefix 'point-decoupling' -UsePinnedImages -PointDecoupling -ReferenceImagesPath $ReferenceImages
    $SplitResults = Join-Path $ResultsRoot $Split
    $Images = Get-Content -LiteralPath (Join-Path $SplitResults 'frozen-images.json') -Raw | ConvertFrom-Json
    $Arguments = @('run', '--rm', '--pull', 'never', '--network', 'none', '--platform', 'linux/amd64',
        '--mount', "type=bind,source=$RepoRoot\cotracker-algorithm,target=/app,readonly",
        '--mount', "type=bind,source=$SplitResults,target=/results",
        '--entrypoint', '/opt/app/.pixi/envs/cuda/bin/python')
    $PriorArgs = @()
    if ($PriorResultsRoot) {
        $Arguments += @('--mount', "type=bind,source=$PriorResultsRoot,target=/prior,readonly")
        $PriorArgs = @('--prior', '/prior')
    }
    $Arguments += @($Images.'trackrad-algorithm-cotracker-algorithm',
        '/app/experiments/analyze_point_decoupling.py', '--results', '/results', '--split', $Split) + $PriorArgs
    $PriorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $SplitResults 'point-audit.log') -Append | ForEach-Object { Write-Host $_ }
        $Status = $LASTEXITCODE
    } finally { $ErrorActionPreference = $PriorPreference }
    if ($Status -ne 0) { throw "Point-decoupling audit failed ($Status). Keep all results for diagnosis." }
    Write-Host "Point-decoupling $Split complete: $SplitResults"
} finally { $OuterLock.Dispose() }
