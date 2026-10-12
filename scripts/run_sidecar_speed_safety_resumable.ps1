[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$TrainingResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-modefix-v3',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\sidecar-speed-safety-loadfix-v2',
    [ValidateSet('profile','smoke','validation','test')][string]$Stage = 'profile',
    [ValidateRange(1.01,100)][double]$MinimumSpeedup = 2,
    [switch]$ReferenceDiagnosisOnly,
    [switch]$RepairOutputPaths
)
$ErrorActionPreference = 'Stop'
if ($ReferenceDiagnosisOnly -and $Stage -notin @('validation','test')) { throw 'Reference diagnosis is validation/test only; use the runtime diagnostic queue for timing.' }
if ($RepairOutputPaths -and (-not $ReferenceDiagnosisOnly -or $Stage -notin @('validation','test'))) { throw 'Path repair is independent sidecar validation/test only.' }
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$TrainingResultsRoot = (Resolve-Path -LiteralPath $TrainingResultsRoot).Path.TrimEnd('\')
$ResultsRoot = [IO.Path]::GetFullPath($ResultsRoot).TrimEnd('\')
if ($ResultsRoot -eq $TrainingResultsRoot -or $ResultsRoot.StartsWith($TrainingResultsRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $TrainingResultsRoot.StartsWith($ResultsRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Use a separate new results root; do not overwrite the old study.' }
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Old = Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'frozen-run.json') -Raw | ConvertFrom-Json
$OldImages = Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'local-runtime-images.json') -Raw | ConvertFrom-Json
$OldAudit = Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'train\training-audit.json') -Raw | ConvertFrom-Json
if (-not $OldAudit.complete -or $OldAudit.steps -ne 1000 -or $OldAudit.proofs.Count -ne 12) { throw 'Original 12 training arms must already be complete.' }
$Datasets = @{}
$Seen = @{}
foreach ($Split in @(@{Name='train-40';Count=40},@{Name='validation-10';Count=10},@{Name='test-38';Count=38})) {
    $Rows = @($Old.Dataset | Where-Object Split -eq $Split.Name)
    $Cases = @($Rows.Case | Sort-Object -Unique)
    if ($Cases.Count -ne $Split.Count) { throw 'Original split case count differs.' }
    foreach ($Case in $Cases) { if ($Seen.ContainsKey($Case)) { throw 'Overlapping dataset cases.' }; $Seen[$Case]=$Split.Name }
    $First=$Rows[0]
    $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot locate original dataset.' }
    $Dataset=$First.File.Substring(0,$Offset)
    if ((@(Get-ChildItem -LiteralPath $Dataset -Directory | Sort-Object Name).Name -join ',') -ne ($Cases -join ',')) { throw 'Dataset IDs changed.' }
    if ((@(Get-ChildItem -LiteralPath $Dataset -Recurse -File | Sort-Object FullName).FullName -join '|') -ne (@($Rows.File | Sort-Object) -join '|')) { throw 'Dataset inventory changed.' }
    foreach ($Row in $Rows) { if ((Get-FileHash -LiteralPath $Row.File).Hash -ne $Row.SHA256) { throw "Dataset file changed: $($Row.File)" } }
    $Datasets[$Split.Name]=$Dataset
}
foreach ($Weight in $Old.Weights) { if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot ('training-checkpoints\'+$Weight.Name))).Hash -ne $Weight.SHA256) { throw 'Pretrained/teacher weights changed.' } }
$Files = @(& git -C $RepoRoot ls-files -- cotracker-algorithm evaluation scripts)
if ($LASTEXITCODE -ne 0) { throw 'Cannot fingerprint current source.' }
$Source = @(foreach ($File in ($Files | Sort-Object)) { if ($File -match '\.(py|ps1|sh|json|toml|lock|txt)$|Dockerfile') { [ordered]@{File=$File;SHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot $File)).Hash} } })
$Utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-FrozenJson([string]$Path,$Value) {
    $Json = ConvertTo-Json -InputObject $Value -Depth 100
    if ($RepairOutputPaths -and [IO.Path]::GetFileName($Path) -eq 'frozen-study.json') {
        if (-not (Test-Path -LiteralPath $Path)) { throw 'Path repair requires an existing frozen study.' }
        . (Join-Path $RepoRoot 'scripts\sidecar_path_repair_provenance.ps1')
        Confirm-SidecarPathRepair -Path $Path -Value $Value -RepoRoot $RepoRoot
        return
    }
    if (Test-Path -LiteralPath $Path) { if ([IO.File]::ReadAllText($Path) -ne $Json) { throw 'Study source/settings changed; use a new results root.' } }
    else { [IO.File]::WriteAllText("$Path.tmp",$Json,$Utf8); Move-Item -LiteralPath "$Path.tmp" -Destination $Path }
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$Lock = [IO.File]::Open((Join-Path $ResultsRoot '.speed-safety.lock'),'OpenOrCreate','ReadWrite','None')
$Started=$false
try {
    Write-FrozenJson (Join-Path $ResultsRoot 'frozen-study.json') ([ordered]@{Schema=1;Source=$Source;TrainingResultsRoot=$TrainingResultsRoot;
        OldProtocolSHA256=(Get-FileHash -LiteralPath (Join-Path $TrainingResultsRoot 'frozen-run.json')).Hash;
        OldTrainingAuditSHA256=(Get-FileHash -LiteralPath (Join-Path $TrainingResultsRoot 'train\training-audit.json')).Hash;
        Images=$OldImages;MinimumSpeedup=$MinimumSpeedup;ReferenceDiagnosisOnly=[bool]$ReferenceDiagnosisOnly;Gains=@(1,.5,.25);CapInputPixels=.5;
        Selection='validation-10: DSC mean >= pretrained and D98 delta >= -0.001; DSC/HD95/CD/D98 ranking';
        ScanAgreementMeanDSC=0.00001;ScanAgreementMaxCaseDSC=0.0001;ScanAgreementMaxTrackPixels=0.001})
    Start-Transcript -Path (Join-Path $ResultsRoot 'queue.log') -Append | Out-Null
    $Started=$true
    $Images = Initialize-LocalExperimentImages -RepoRoot $RepoRoot -ResultsRoot $ResultsRoot -TrainingRuntime $OldImages.Training -InferenceRuntime $OldImages.Inference
    if ($Images.Evaluation -ne $OldImages.Evaluation) { throw 'Evaluator image changed.' }
    Invoke-LocalDocker @('run','--rm','--pull','never','--network','none',
        '--mount',"type=bind,source=$RepoRoot,target=/source,readonly",
        '--mount',"type=bind,source=$TrainingResultsRoot,target=/previous,readonly",
        '--mount',"type=bind,source=$ResultsRoot,target=/study",'--entrypoint','/opt/app/.pixi/envs/training/bin/python',
        $Images.Training,'/opt/app/experiments/audit_sidecar_speed_source.py','--old','/previous','--new','/study','--repo','/source') (Join-Path $ResultsRoot 'source-audit.log')
    $ProfilePath=Join-Path $ResultsRoot 'speed-profile.json'
    $LoaderSHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot 'cotracker-algorithm\resources\model.py')).Hash.ToLowerInvariant()
    if ($Stage -eq 'profile') {
        if (-not (Test-Path -LiteralPath $ProfilePath)) {
            Invoke-LocalDocker @('run','--rm','--pull','never','--network','none','--gpus','device=0',
                '--mount',"type=bind,source=$TrainingResultsRoot\train,target=/training,readonly",
                '--mount',"type=bind,source=$ResultsRoot,target=/study",'--entrypoint','/opt/app/.pixi/envs/training/bin/python',
                $Images.Training,'/opt/app/experiments/profile_sidecar_speed.py','--train','/training','--output','/study/speed-profile.json',
                '--minimum-speedup',([string]$MinimumSpeedup),'--loader-sha256',$LoaderSHA256) (Join-Path $ResultsRoot 'speed-profile.log')
        }
        Write-Host "Profile complete. Inspect $ProfilePath before any new large training."
        return
    }
    if ($ReferenceDiagnosisOnly) {
        $Runtime = [ordered]@{backend='reference';point_chunk_size=0;gain=1;max_correction_pixels=0;checkpoint_blocks=$false}
        Write-Host 'Independent degradation diagnosis: old weights, unchanged reference backend; no speed gate or new training.'
    } else {
    $Profile = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json
    if ($Profile.checkpoint_loader.sha256 -ne $LoaderSHA256) { throw 'Speed profile used a different checkpoint loader. Preserve results and use a new root.' }
    if (-not $Profile.complete -or -not $Profile.selected -or $Profile.selected.measured_speedup -lt $MinimumSpeedup) {
        throw 'No verified sufficiently faster backend. Do not start another large training/evaluation queue; share speed-profile.json.'
    }
    foreach ($Property in $Profile.source_checkpoints_sha256.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $TrainingResultsRoot ('train\'+$Property.Name))).Hash -ne $Property.Value) { throw 'Profile weights changed.' }
    }
    $Runtime = $Profile.selected.runtime
    $FastPath=Join-Path $ResultsRoot 'fast-training-runtime.json'
    Write-FrozenJson $FastPath $Runtime
    if ($Stage -eq 'smoke') {
        & (Join-Path $RepoRoot 'scripts\run_feature_sidecar_resumable.ps1') -RepoRoot $RepoRoot `
            -ResultsRoot (Join-Path $ResultsRoot 'accelerated-training') -Stage smoke -SidecarRuntimeConfig $FastPath `
            -TrainDataset $Datasets['train-40'] -ValidationDataset $Datasets['validation-10'] -TestDataset $Datasets['test-38'] `
            -TrainingRuntimeImage $Images.Training -InferenceRuntimeImage $Images.Inference
        Write-Host 'Only four 5-step real training arms ran. No new 1000-step training was launched.'
        return
    }
    $SmokeRoot=Join-Path $ResultsRoot 'accelerated-training-smoke'
    $SmokeAudit=Get-Content -LiteralPath (Join-Path $SmokeRoot 'train\training-audit.json') -Raw | ConvertFrom-Json
    $SmokeProtocol=Get-Content -LiteralPath (Join-Path $SmokeRoot 'frozen-run.json') -Raw | ConvertFrom-Json
    if (-not $SmokeAudit.complete -or $SmokeAudit.steps -ne 5 -or $SmokeAudit.proofs.Count -ne 4 -or
        $SmokeProtocol.SidecarRuntimeSHA256 -ne (Get-FileHash -LiteralPath $FastPath).Hash) { throw 'Run -Stage smoke first; the real 5-step training proof must match this fast runtime.' }
    foreach ($Property in $SmokeAudit.final_checkpoints_sha256.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath (Join-Path $SmokeRoot ('train\'+$Property.Name))).Hash -ne $Property.Value) { throw 'Accelerated smoke checkpoints changed.' }
    }
    }
    $Arms=@()
    foreach ($Kind in @('mlp','mamba')) {
        foreach ($Variant in @(@{Suffix='full';Gain=1;Cap=0},@{Suffix='gain050';Gain=.5;Cap=0},@{Suffix='gain025';Gain=.25;Cap=0},@{Suffix='gain050_cap05';Gain=.5;Cap=.5})) {
            $Setting=[ordered]@{backend=$Runtime.backend;point_chunk_size=$Runtime.point_chunk_size;gain=$Variant.Gain;max_correction_pixels=$Variant.Cap;checkpoint_blocks=$Runtime.checkpoint_blocks}
            $Name=$Kind+'_'+$Variant.Suffix
            $Config=Join-Path $ResultsRoot ($Name+'-runtime.json')
            Write-FrozenJson $Config $Setting
            $Arms += [ordered]@{name=$Name;kind=$Kind;runtime=$Setting}
        }
    }
    Write-FrozenJson (Join-Path $ResultsRoot 'study-spec.json') ([ordered]@{arms=$Arms;seeds=@(0,1,2);profile=$Old.InferenceProfile;requires_scan_agreement=(-not [bool]$ReferenceDiagnosisOnly)})
    $Split=if ($Stage -eq 'test') {'test-38'} else {'validation-10'}
    if ($Stage -eq 'test') {
        $Validation=Get-Content -LiteralPath (Join-Path $ResultsRoot 'validation-10\safety-audit.json') -Raw | ConvertFrom-Json
        if (-not $Validation.complete -or $Validation.cases -ne 10) { throw 'Validation must finish before test.' }
        if ((Get-FileHash -LiteralPath (Join-Path $ResultsRoot 'selection.json')).Hash -ne $Validation.selection_sha256) { throw 'Frozen validation selection changed.' }
        foreach ($Property in $Validation.metrics_sha256.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath (Join-Path $ResultsRoot ('validation-10\'+$Property.Name))).Hash -ne $Property.Value) { throw 'Validation metrics changed.' }
        }
        $Selection=Get-Content -LiteralPath (Join-Path $ResultsRoot 'selection.json') -Raw | ConvertFrom-Json
        if (-not $Selection.backend_agreement_passed -or @($Selection.selected_arms).Count -eq 0) { throw 'No validation-qualified correction setting / backend agreement failed. No test cases will run.' }
        $Included=@($Selection.selected_arms)+@('mlp_full','mamba_full')
        $Arms=@($Arms | Where-Object { $Included -contains $_.name })
    }
    $Runner=Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    $RepairFlags=@{}
    if ($RepairOutputPaths) { $RepairFlags['RecoverLongPathOutput']=$true }
    $Pinned=Join-Path $ResultsRoot 'reference-images.json'
    & $Runner -Repository $RepoRoot -Dataset $Datasets[$Split] -Results (Join-Path $ResultsRoot "$Split\pretrained") `
        -Profiles @($Old.InferenceProfile) -ModelCheckpoint (Join-Path $TrainingResultsRoot 'pretrained-frozen.pth') `
        -RequireDiagnostics -SidecarTrackCache -FreezeImages -UsePinnedImages -ReferenceImagesPath $Pinned @RepairFlags
    if ($Stage -eq 'validation' -and -not $ReferenceDiagnosisOnly) {
        $Shadow=Join-Path $ResultsRoot 'mamba_reference-runtime.json'
        Write-FrozenJson $Shadow ([ordered]@{backend='reference';point_chunk_size=0;gain=1;max_correction_pixels=0;checkpoint_blocks=$false})
        & $Runner -Repository $RepoRoot -Dataset $Datasets[$Split] -Results (Join-Path $ResultsRoot "$Split\seed_0\mamba_reference") `
            -Profiles @($Old.InferenceProfile) -ModelCheckpoint (Join-Path $TrainingResultsRoot 'train\seed_0\mamba\cotracker_three_final.pth') `
            -RequireDiagnostics -SidecarTrackCache -SidecarRuntimeConfig $Shadow -FreezeImages -UsePinnedImages -ReferenceImagesPath $Pinned @RepairFlags
    }
    foreach ($Seed in @(0,1,2)) {
        foreach ($Arm in $Arms) {
            & $Runner -Repository $RepoRoot -Dataset $Datasets[$Split] -Results (Join-Path $ResultsRoot "$Split\seed_$Seed\$($Arm.name)") `
                -Profiles @($Old.InferenceProfile) -ModelCheckpoint (Join-Path $TrainingResultsRoot "train\seed_$Seed\$($Arm.kind)\cotracker_three_final.pth") `
                -RequireDiagnostics -SidecarTrackCache -SidecarRuntimeConfig (Join-Path $ResultsRoot ($Arm.name+'-runtime.json')) `
                -FreezeImages -UsePinnedImages -ReferenceImagesPath $Pinned @RepairFlags
        }
    }
    Invoke-LocalDocker @('run','--rm','--pull','never','--network','none',
        '--mount',"type=bind,source=$ResultsRoot,target=/study",'--entrypoint','/opt/app/.pixi/envs/cuda/bin/python',
        $Images.Inference,'/opt/app/experiments/summarize_sidecar_speed_safety.py','--root','/study','--split',$Split) (Join-Path $ResultsRoot 'summary.log')
    Write-Host "Speed/safety $Split complete. No old training was changed."
} finally {
    if ($Started) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
