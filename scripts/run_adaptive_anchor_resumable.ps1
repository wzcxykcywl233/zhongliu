[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\adaptive-anchor-v1',
    [string]$TrainDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40',
    [string]$ValidationDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$TestDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38',
    [ValidateSet('validation','test')][string]$Stage = 'validation',
    [string]$InferenceRuntimeImage = 'trackrad-algorithm-cotracker-algorithm:latest'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Images = Initialize-LocalExperimentImages -RepoRoot $RepoRoot -ResultsRoot $ResultsRoot -InferenceRuntime $InferenceRuntimeImage
& (Join-Path $RepoRoot 'scripts\run_memory_refinement_resumable.ps1') -RepoRoot $RepoRoot `
    -TrainDataset $TrainDataset -ValidationDataset $ValidationDataset -TestDataset $TestDataset `
    -ResultsRoot $ResultsRoot -Stage $Stage -ManifestRelative 'cotracker-algorithm\experiments\adaptive-anchor-40-10-38.json' `
    -ResultPrefix 'adaptive-anchor' -ReferenceImagesPath (Join-Path $ResultsRoot 'reference-images.json') -UsePinnedImages
$Split = if ($Stage -eq 'test') {'test-38'} else {'validation-10'}
Invoke-LocalDocker -Log (Join-Path $ResultsRoot "$Split\anchor-audit.log") -Arguments @(
    'run','--rm','--pull','never','--network','none',
    '--mount',"type=bind,source=$ResultsRoot,target=/study", '--entrypoint','/opt/app/.pixi/envs/cuda/bin/python',
    $Images.Inference,'/opt/app/experiments/audit_adaptive_anchor.py','--root',"/study/$Split",'--split',$Split)
