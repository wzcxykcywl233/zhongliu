param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\hierarchical-revalidation-results"
)

$Profiles = @(
    "hierarchical_full_feature_gate",
    "hierarchical_full_feature_revalidate_r4"
)

& (Join-Path $PSScriptRoot "run_hierarchical_followup_resumable.ps1") `
    -Repository $Repository `
    -Dataset $Dataset `
    -Results $Results `
    -Profiles $Profiles
