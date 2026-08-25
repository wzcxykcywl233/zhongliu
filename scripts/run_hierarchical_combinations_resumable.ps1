param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\hierarchical-combination-results"
)

$ErrorActionPreference = "Stop"

$Profiles = @(
    "hierarchical_feat05_grid0",
    "hierarchical_feat05_iterations2",
    "hierarchical_feat05_grid0_iterations2",
    "hierarchical_full_grid0",
    "hierarchical_full_iterations2",
    "hierarchical_full_grid0_iterations2"
)

& (Join-Path $PSScriptRoot "run_hierarchical_followup_resumable.ps1") `
    -Repository $Repository `
    -Dataset $Dataset `
    -Results $Results `
    -Profiles $Profiles `
    -RequireDiagnostics
