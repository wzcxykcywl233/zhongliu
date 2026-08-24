param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\full-audit-results"
)

$ErrorActionPreference = "Stop"

# Start with the duplicate baseline and historically unchanged/tiny-delta
# profiles, so the most important audit evidence is available first.  The same
# runner then covers every earlier combination and hierarchical experiment.
$Profiles = @(
    "baseline",
    "baseline_repeat",
    "visibility_0_5",
    "confidence_0_5",
    "morph_close_3",
    "largest_component",
    "lock_first_mask",
    "temporal_median_3",
    "points_500",
    "points_1500",
    "support_grid_0",
    "support_grid_15",
    "iterations_2",
    "iterations_6",
    "keyframe_stride_2",
    "grid0_iterations2",
    "grid0_stride2",
    "grid0_iterations2_stride2",
    "hierarchical_d10",
    "hierarchical_d10_original_feat_025",
    "hierarchical_d10_original_feat_05",
    "hierarchical_d10_original_feat_075",
    "hierarchical_d10_occlusion_merge",
    "hierarchical_d10_dual_anchor",
    "hierarchical_full_d5",
    "hierarchical_full",
    "hierarchical_full_d15",
    "hierarchical_full_feature_gate",
    "hierarchical_full_feature_revalidate_r4"
)

& (Join-Path $PSScriptRoot "run_hierarchical_followup_resumable.ps1") `
    -Repository $Repository `
    -Dataset $Dataset `
    -Results $Results `
    -Profiles $Profiles `
    -RequireDiagnostics

& (Join-Path $PSScriptRoot "summarize_full_audit.ps1") -Results $Results
