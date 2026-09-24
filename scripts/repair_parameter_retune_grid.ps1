[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$OldResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\parameter-retune',
    [string]$NewResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\parameter-retune-gridfix',
    [ValidateSet('validation','test')][string]$Stage = 'validation'
)
$ErrorActionPreference = 'Stop'
& (Join-Path $RepoRoot 'scripts\run_parameter_retune_resumable.ps1') `
    -RepoRoot $RepoRoot -ResultsRoot $NewResultsRoot -ReuseResultsRoot $OldResultsRoot -Stage $Stage
