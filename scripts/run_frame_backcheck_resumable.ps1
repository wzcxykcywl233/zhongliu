[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$TrainDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40',
    [string]$ValidationDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$TestDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\frame-backcheck',
    [ValidateSet('all','validation','test')][string]$Stage = 'all'
)
$ErrorActionPreference = 'Stop'
& (Join-Path $RepoRoot 'scripts\run_memory_refinement_resumable.ps1') `
    -RepoRoot $RepoRoot -TrainDataset $TrainDataset -ValidationDataset $ValidationDataset `
    -TestDataset $TestDataset -ResultsRoot $ResultsRoot -Stage $Stage `
    -ManifestRelative 'cotracker-algorithm\experiments\frame-backcheck-40-10-38.json' `
    -ResultPrefix 'frame-backcheck'
