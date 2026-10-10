[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$TrainingResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-modefix-v3',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-inferencefix-v4',
    [ValidateSet('validation','test')][string]$Stage = 'validation'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$TrainingResultsRoot = (Resolve-Path -LiteralPath $TrainingResultsRoot).Path.TrimEnd('\')
$ResultsRoot = [IO.Path]::GetFullPath($ResultsRoot).TrimEnd('\')
if ($ResultsRoot -eq $TrainingResultsRoot -or
    $ResultsRoot.StartsWith($TrainingResultsRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or
    $TrainingResultsRoot.StartsWith($ResultsRoot + '\',[StringComparison]::OrdinalIgnoreCase)) {
    throw 'Use a separate sibling results root. The completed training study stays read-only.'
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Old = Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'frozen-run.json') -Raw | ConvertFrom-Json
$OldImages = Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'local-runtime-images.json') -Raw | ConvertFrom-Json
$OldAudit = Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'train\training-audit.json') -Raw | ConvertFrom-Json
if (-not $OldAudit.complete -or $OldAudit.proofs.Count -ne 12 -or $OldAudit.steps -ne 1000) {
    throw 'All 12 training arms and the strict paired audit must already be complete. This entry never trains.'
}
$TrainingImageId = Get-LocalImageId $OldImages.Training
if ((Get-LocalImageId $OldImages.Evaluation) -ne $OldImages.Evaluation) { throw 'Original evaluator unavailable.' }
$DatasetPaths = @{}
$Seen = @{}
foreach ($Split in @(@{Name='train-40';Count=40},@{Name='validation-10';Count=10},@{Name='test-38';Count=38})) {
    $Rows = @($Old.Dataset | Where-Object Split -eq $Split.Name)
    $CaseIds = @($Rows.Case | Sort-Object -Unique)
    if ($CaseIds.Count -ne $Split.Count) { throw 'Original frozen split has wrong case count.' }
    foreach ($Case in $CaseIds) {
        if ($Seen.ContainsKey($Case)) { throw 'Original split contains overlapping case IDs.' }
        $Seen[$Case] = $Split.Name
    }
    $First = $Rows[0]
    $Boundary = '\' + $First.Case + '\'
    $Offset = $First.File.IndexOf($Boundary,[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot derive original dataset path.' }
    $Dataset = $First.File.Substring(0,$Offset)
    $ActualCases = @(Get-ChildItem -LiteralPath $Dataset -Directory | Sort-Object Name)
    if (($ActualCases.Name -join ',') -ne ($CaseIds -join ',')) { throw 'Dataset case IDs changed.' }
    $ActualFiles = @(Get-ChildItem -LiteralPath $Dataset -File -Recurse | Sort-Object FullName)
    if (($ActualFiles.FullName -join '|') -ne ((@($Rows.File | Sort-Object)) -join '|')) { throw 'Dataset file inventory changed.' }
    $DatasetPaths[$Split.Name] = $Dataset
    foreach ($Row in $Rows) {
        if ((Get-FileHash -LiteralPath $Row.File -Algorithm SHA256).Hash -ne $Row.SHA256) { throw "Dataset changed: $($Row.File)" }
    }
}
foreach ($Weight in $Old.Weights) {
    $Path = Join-Path $RepoRoot ('training-checkpoints\' + $Weight.Name)
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Weight.SHA256) { throw 'Original pretrained/teacher weights changed.' }
}
$Paths = @(& git -C $RepoRoot ls-files -- cotracker-algorithm evaluation scripts)
if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate current source.' }
$Source = @(
    foreach ($Relative in ($Paths | Sort-Object)) {
        if ($Relative -match '\.(py|ps1|sh|json|toml|lock|txt)$|Dockerfile') {
            [ordered]@{File=$Relative;SHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot $Relative) -Algorithm SHA256).Hash}
        }
    }
)
$Utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-ImmutableJson([string]$Path, $Value) {
    $Json = ConvertTo-Json -InputObject $Value -Depth 100
    if (Test-Path -LiteralPath $Path) {
        if ([IO.File]::ReadAllText($Path) -ne $Json) { throw 'Recovery inputs/code changed; preserve this root and use a new ResultsRoot.' }
    } else {
        [IO.File]::WriteAllText("$Path.tmp",$Json,$Utf8)
        Move-Item -LiteralPath "$Path.tmp" -Destination $Path
    }
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$Lock = [IO.File]::Open((Join-Path $ResultsRoot '.evaluation-recovery.lock'),'OpenOrCreate','ReadWrite','None')
$Started = $false
try {
    $Manifest = [ordered]@{Schema=1;TrainingResultsRoot=$TrainingResultsRoot;Source=$Source;
        OriginalProtocolSHA256=(Get-FileHash -LiteralPath (Join-Path $TrainingResultsRoot 'frozen-run.json') -Algorithm SHA256).Hash;
        OriginalTrainingAuditSHA256=(Get-FileHash -LiteralPath (Join-Path $TrainingResultsRoot 'train\training-audit.json') -Algorithm SHA256).Hash;
        OriginalTrainingImage=$TrainingImageId;OriginalInferenceBase=$OldImages.Inference;OriginalEvaluator=$OldImages.Evaluation}
    Write-ImmutableJson (Join-Path $ResultsRoot 'frozen-evaluation.json') $Manifest
    Write-ImmutableJson (Join-Path $ResultsRoot 'current-source.json') $Source
    Start-Transcript -Path (Join-Path $ResultsRoot 'queue.log') -Append | Out-Null
    $Started = $true
    Write-Host 'Evaluation-only repair: 12 completed checkpoints stay in the original study; NO TRAINING.'
    $AuditArguments = @('run','--rm','--pull','never','--network','none',
        '--mount',"type=bind,source=$RepoRoot,target=/repair,readonly",
        '--mount',"type=bind,source=$TrainingResultsRoot,target=/previous,readonly",
        '--mount',"type=bind,source=$ResultsRoot,target=/study",
        '--entrypoint','/opt/app/.pixi/envs/training/bin/python',$TrainingImageId,
        '/repair/cotracker-algorithm/experiments/verify_sidecar_evaluation_recovery.py',
        '--old','/previous','--new','/study','--repo','/repair','--source','/study/current-source.json')
    Invoke-LocalDocker $AuditArguments (Join-Path $ResultsRoot 'recovery-audit.log')
    # Reuse the original inference environment, copying only current code. No
    # dependency download and no overwriting the old image identity manifests.
    $Images = Initialize-LocalExperimentImages -RepoRoot $RepoRoot -ResultsRoot $ResultsRoot -InferenceRuntime $OldImages.Inference
    if ($Images.Evaluation -ne $OldImages.Evaluation) { throw 'Evaluation environment changed; do not mix evaluators.' }
    Invoke-LocalDocker @('run','--rm','--pull','never','--network','none','--gpus','device=0',
        '--mount',"type=bind,source=$TrainingResultsRoot\train,target=/training,readonly",
        '--entrypoint','/opt/app/.pixi/envs/cuda/bin/python',$Images.Inference,
        '/opt/app/experiments/preflight_sidecar_inference.py','--train','/training') (Join-Path $ResultsRoot 'inference-preflight.log')
    $Split = if ($Stage -eq 'test') {'test-38'} else {'validation-10'}
    if ($Stage -eq 'test') {
        $Validation = Get-Content -LiteralPath (Join-Path $ResultsRoot 'validation-10\feature-sidecar-audit.json') -Raw | ConvertFrom-Json
        if (-not $Validation.complete -or $Validation.training_steps -ne 1000 -or (@($Validation.seeds) -join ',') -ne '0,1,2') { throw 'Repaired validation must finish first.' }
        foreach ($Property in $Validation.metrics_sha256.PSObject.Properties) {
            $Path = Join-Path $ResultsRoot ('validation-10\' + $Property.Name)
            if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Property.Value) { throw 'Validation metrics changed.' }
        }
    }
    $Runner = Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    $Profile = $Old.InferenceProfile
    $ReferenceImages = Join-Path $ResultsRoot 'reference-images.json'
    & $Runner -Repository $RepoRoot -Dataset $DatasetPaths[$Split] -Results (Join-Path $ResultsRoot "$Split\pretrained") `
        -Profiles @($Profile) -ModelCheckpoint (Join-Path $TrainingResultsRoot 'pretrained-frozen.pth') `
        -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $ReferenceImages
    foreach ($Seed in $Old.Seeds) {
        foreach ($Kind in $Old.Architectures) {
            $Final = Join-Path $TrainingResultsRoot "train\seed_$Seed\$Kind\cotracker_three_final.pth"
            & $Runner -Repository $RepoRoot -Dataset $DatasetPaths[$Split] -Results (Join-Path $ResultsRoot "$Split\seed_$Seed\$Kind") `
                -Profiles @($Profile) -ModelCheckpoint $Final -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $ReferenceImages
        }
    }
    Invoke-LocalDocker @('run','--rm','--pull','never','--network','none',
        '--mount',"type=bind,source=$RepoRoot,target=/repair,readonly",
        '--mount',"type=bind,source=$ResultsRoot,target=/study",
        '--mount',"type=bind,source=$TrainingResultsRoot\train,target=/study/train,readonly",
        '--entrypoint','/opt/app/.pixi/envs/training/bin/python',$TrainingImageId,
        '/repair/cotracker-algorithm/experiments/summarize_feature_sidecar.py',
        '--root','/study','--seeds','0,1,2','--steps','1000','--split',$Split) (Join-Path $ResultsRoot 'summary.log')
    Write-Host "Feature-sidecar $Split recovered without training: $ResultsRoot\$Split"
} finally {
    if ($Started) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
