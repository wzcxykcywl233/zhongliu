[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\backbone-growth-v1',
    [string]$TrainDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40',
    [string]$ValidationDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$TestDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38',
    [ValidateSet('smoke','train','validation','test')][string]$Stage = 'validation',
    [ValidateRange(1,100000)][int]$NumSteps = 1000,
    [ValidateRange(1,10000)][int]$SaveEverySteps = 25,
    [string]$Seeds = '0,1,2',
    [string]$TrainingRuntimeImage = ''
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$Architectures = @('base','time6','space_time6')
$InferenceProfile = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
if ($Seeds -notmatch '^\d+(,\d+)*$') { throw 'Seeds must be comma-separated nonnegative integers.' }
$SeedValues = @($Seeds.Split(',') | ForEach-Object { [int]$_ })
if (@($SeedValues | Sort-Object -Unique).Count -ne $SeedValues.Count) { throw 'Duplicate seeds are not allowed.' }
if ($Stage -eq 'smoke') {
    $ResultsRoot = $ResultsRoot.TrimEnd('\') + '-smoke'
    $NumSteps = 5
    $SaveEverySteps = 1
    $Seeds = '0'
    $SeedValues = @(0)
}
$CheckpointDir = Join-Path $RepoRoot 'training-checkpoints'
$Weights = @()
foreach ($Name in @('scaled_offline.pth','baseline_online.pth','baseline_offline.pth','cotracker2v1.pth')) {
    $Path = Join-Path $CheckpointDir $Name
    $Weights += [ordered]@{ Name=$Name; SHA256=(Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash }
}
$Data = @()
$Seen = @{}
foreach ($Split in @(
    @{Name='train-40'; Path=$TrainDataset; Count=40},
    @{Name='validation-10'; Path=$ValidationDataset; Count=10},
    @{Name='test-38'; Path=$TestDataset; Count=38}
)) {
    $Cases = @(Get-ChildItem -LiteralPath $Split.Path -Directory | Sort-Object Name)
    if ($Cases.Count -ne $Split.Count) { throw "Wrong case count in $($Split.Path)" }
    foreach ($Case in $Cases) {
        if ($Seen.ContainsKey($Case.Name)) { throw "Overlapping case across splits: $($Case.Name)" }
        $Seen[$Case.Name] = $Split.Name
        foreach ($Required in @("images\$($Case.Name)_frames.mha", "targets\$($Case.Name)_first_label.mha", "targets\$($Case.Name)_labels.mha", 'frame-rate.json','b-field-strength.json','scanned-region.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $Case.FullName $Required) -PathType Leaf)) { throw "Missing dataset file: $Required" }
        }
        foreach ($File in @(Get-ChildItem -LiteralPath $Case.FullName -File -Recurse | Sort-Object FullName)) {
            $Data += [ordered]@{ Split=$Split.Name; Case=$Case.Name; File=$File.FullName; SHA256=(Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash }
        }
    }
}
$SourcePaths = @(& git -C $RepoRoot ls-files -- cotracker-algorithm evaluation scripts)
if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate source files.' }
$Source = @(
    foreach ($Relative in ($SourcePaths | Sort-Object)) {
        if ($Relative -match '\.(py|ps1|sh|json|toml|lock|txt)$|Dockerfile') {
            [ordered]@{ File=$Relative; SHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot $Relative) -Algorithm SHA256).Hash }
        }
    }
)
$Protocol = [ordered]@{
    Schema=2; Source=$Source; Weights=$Weights; Dataset=$Data; TrainingRuntimeImage=$TrainingRuntimeImage
    Architectures=$Architectures; Seeds=$SeedValues; Steps=$NumSteps; SaveEverySteps=$SaveEverySteps
    SequenceLength=10; Trajectories=384; TrainIterations=4; LearningRate=0.00005
    AuxiliaryTeacherWeight=0; MaskLossWeight=0; ConfidenceTarget='hard'; InferenceProfile=$InferenceProfile
}
$Utf8 = New-Object System.Text.UTF8Encoding($false)
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$Lock = [IO.File]::Open((Join-Path $ResultsRoot '.backbone.lock'),'OpenOrCreate','ReadWrite','None')
$TranscriptStarted = $false
function Write-AtomicJson([string]$Path, $Value) {
    [IO.File]::WriteAllText("$Path.tmp", (ConvertTo-Json -InputObject $Value -Depth 100), $Utf8)
    Move-Item -LiteralPath "$Path.tmp" -Destination $Path -Force
}
function Invoke-DockerLogged([string[]]$Arguments, [string]$Log) {
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object { Write-Host $_ }
        $Status = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Previous }
    if ($Status -ne 0) { throw "Docker task failed ($Status); repeat the same command to resume after fixing the cause." }
}
function Invoke-TrainingTool([string]$Name, [string[]]$Extra) {
    Invoke-DockerLogged -Log (Join-Path $ResultsRoot 'audit.log') -Arguments (@(
        'run','--rm','--pull','never','--network','none',
        '--mount',"type=bind,source=$ResultsRoot,target=/study",
        '--entrypoint','/opt/app/.pixi/envs/training/bin/python',
        $TrainingImageId,"/opt/app/experiments/$Name"
    ) + $Extra)
}
try {
    $Frozen = Join-Path $ResultsRoot 'frozen-run.json'
    $Json = ConvertTo-Json -InputObject $Protocol -Depth 100
    if (Test-Path -LiteralPath $Frozen) {
        if ([IO.File]::ReadAllText($Frozen) -ne $Json) { throw 'Code, weights, data or training protocol changed; use a new ResultsRoot.' }
    } else {
        if (@(Get-ChildItem -LiteralPath $ResultsRoot -Recurse -File -Filter '*.pth').Count -gt 0) { throw 'Existing unrecorded checkpoints: use a new ResultsRoot.' }
        Write-AtomicJson $Frozen $Protocol
    }
    Start-Transcript -Path (Join-Path $ResultsRoot 'queue.log') -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host "Backbone-growth stage=$Stage; seeds=$Seeds; steps=$NumSteps; architectures=$($Architectures -join ',')"
    $ImagePath = Join-Path $ResultsRoot 'training-image.json'
    if (Test-Path -LiteralPath $ImagePath) {
        $TrainingImageId = [string](Get-Content -LiteralPath $ImagePath -Raw | ConvertFrom-Json).Id
        if ($TrainingImageId -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Invalid pinned training image ID.' }
        & docker image inspect $TrainingImageId *> $null
        if ($LASTEXITCODE -ne 0) { throw 'Pinned training image unavailable; preserve results.' }
    } else {
        if ($Stage -eq 'test') { throw 'Run training and validation first.' }
        $Dockerfile = Join-Path $RepoRoot 'cotracker-algorithm\Dockerfile.training'
        $BuildExtra = @()
        $RuntimeId = ''
        if ($TrainingRuntimeImage) {
            # Inspect only local images; never pull or silently substitute a tag.
            $Previous = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try {
                $RuntimeId = [string](& docker image inspect --format '{{.Id}}' $TrainingRuntimeImage 2>$null)
                $InspectStatus = $LASTEXITCODE
            } finally { $ErrorActionPreference = $Previous }
            if ($InspectStatus -ne 0 -or $RuntimeId -notmatch '^sha256:[0-9a-f]{64}$') {
                throw "Local training runtime unavailable: $TrainingRuntimeImage. Run docker image ls; no download was attempted."
            }
            # A Dockerfile FROM needs a local named reference rather than a bare
            # image ID. Make a dedicated alias for the inspected immutable ID.
            $RuntimeAlias = 'trackrad-backbone-runtime-base:' + $RuntimeId.Substring(7,12)
            Invoke-DockerLogged -Log (Join-Path $ResultsRoot 'build.log') -Arguments @('tag',$RuntimeId,$RuntimeAlias)
            $Dockerfile = Join-Path $RepoRoot 'cotracker-algorithm\Dockerfile.training.reuse'
            $BuildExtra = @('--pull=false','--network=none','--build-arg',"TRAINING_BASE_IMAGE=$RuntimeAlias")
            Write-Host "Reusing local training environment: $RuntimeId; copying CURRENT training code; no Pixi install"
        }
        Invoke-DockerLogged -Log (Join-Path $ResultsRoot 'build.log') -Arguments (@(
            'build','--progress=plain','--platform=linux/amd64','--file', $Dockerfile,
            '--tag','trackrad-cotracker-backbone-growth') + $BuildExtra + @((Join-Path $RepoRoot 'cotracker-algorithm')))
        $TrainingImageId = [string](& docker image inspect --format '{{.Id}}' trackrad-cotracker-backbone-growth)
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect training image.' }
        Write-AtomicJson $ImagePath @{Id=$TrainingImageId; ReusedRuntimeId=$RuntimeId; RequestedRuntime=$TrainingRuntimeImage}
    }
    Invoke-TrainingTool 'check_backbone_runtime.py' @('--output','/study/runtime-check.json')
    $TrainingRoot = Join-Path $ResultsRoot 'train'
    if ($Stage -ne 'test') {
        foreach ($Seed in $SeedValues) {
            foreach ($Architecture in $Architectures) {
                $Profile = "seed_${Seed}_$Architecture"
                $Folder = Join-Path $TrainingRoot "seed_$Seed\$Architecture"
                New-Item -ItemType Directory -Force -Path $Folder | Out-Null
                $Config = [ordered]@{
                    profile=$Profile; architecture=$Architecture; steps=$NumSteps; save_every_steps=$SaveEverySteps
                    seed=$Seed; training_seed=(20261005+$Seed); paired_step_seed=(20261005+$Seed); teacher_seed=(20261005+$Seed)
                    dataset=$TrainDataset; source_protocol_sha256=(Get-FileHash -LiteralPath $Frozen -Algorithm SHA256).Hash
                    sequence_len=10; traj_per_sample=384; train_iters=4; lr=0.00005
                    auxiliary_teacher_weight=0; confidence_target_mode='hard'; mask_loss_weight=0
                }
                $ConfigPath = Join-Path $Folder 'run-config.json'
                if (Test-Path -LiteralPath $ConfigPath) {
                    if ([IO.File]::ReadAllText($ConfigPath) -ne (ConvertTo-Json -InputObject $Config -Depth 100)) { throw 'Training config differs from checkpoint.' }
                } else { Write-AtomicJson $ConfigPath $Config }
                if (Test-Path -LiteralPath (Join-Path $Folder 'cotracker_three_final.pth')) {
                    Write-Host "SKIP completed training: $Profile"
                    continue
                }
                Write-Host "TRAIN $Profile"
                Invoke-DockerLogged -Log (Join-Path $Folder 'training.log') -Arguments @(
                    'run','--rm','--pull','never','--network','none','--gpus','device=0','--shm-size','4g',
                    '--mount',"type=bind,source=$TrainDataset,target=/data,readonly",
                    '--mount',"type=bind,source=$CheckpointDir,target=/checkpoints,readonly",
                    '--mount',"type=bind,source=$Folder,target=/results", '--workdir','/opt/app/ext/co-tracker',
                    $TrainingImageId,'--batch_size','1','--num_workers','0','--num_steps',"$NumSteps",'--ckpt_path','/results',
                    '--model_name','cotracker_three','--experiment_name',$Profile,'--sequence_len','10','--traj_per_sample','384',
                    '--train_iters','4','--save_freq','100000','--save_every_n_steps',"$SaveEverySteps",'--keep_last_checkpoints','3',
                    '--save_every_n_epoch','1','--trackrad_data_dir','/data','--skip_evaluation','--offline_model',
                    '--restore_ckpt','/checkpoints/scaled_offline.pth','--teacher_types','cotracker2v1','online_cotracker_three','offline_cotracker_three',
                    '--teacher_cotracker2_ckpt','/checkpoints/cotracker2v1.pth','--teacher_online_ckpt','/checkpoints/baseline_online.pth',
                    '--teacher_offline_ckpt','/checkpoints/baseline_offline.pth','--auxiliary_teacher_weight','0',
                    '--seed',"$(20261005+$Seed)",'--paired_step_seed',"$(20261005+$Seed)",'--teacher_seed',"$(20261005+$Seed)",
                    '--teacher_log_every','1','--confidence_target_mode','hard','--lr','0.00005',
                    '--backbone_growth_study','--backbone_architecture',$Architecture)
            }
        }
    }
    Invoke-TrainingTool 'audit_backbone_growth.py' @('--root','/study/train','--seeds',$Seeds,'--steps',"$NumSteps")
    if ($Stage -in @('smoke','train')) {
        Write-Host "Training and pair audit complete: $TrainingRoot"
    } else {
        $Audit = Get-Content -LiteralPath (Join-Path $TrainingRoot 'training-audit.json') -Raw | ConvertFrom-Json
        $AuditPath = Join-Path $ResultsRoot 'frozen-trained-checkpoints.json'
        $TrainedJson = ConvertTo-Json -InputObject $Audit.final_checkpoints_sha256 -Depth 100
        if (Test-Path -LiteralPath $AuditPath) {
            if ([IO.File]::ReadAllText($AuditPath) -ne $TrainedJson) { throw 'Trained checkpoints changed since evaluation began.' }
        } else { Write-AtomicJson $AuditPath $Audit.final_checkpoints_sha256 }
        $Split = if ($Stage -eq 'test') {'test-38'} else {'validation-10'}
        $Dataset = if ($Stage -eq 'test') {$TestDataset} else {$ValidationDataset}
        $ReferenceImages = ''
        if ($Stage -eq 'test') {
            $ValidationAuditPath = Join-Path $ResultsRoot 'validation-10\backbone-growth-audit.json'
            if (-not (Test-Path -LiteralPath $ValidationAuditPath)) { throw 'Validation evaluation must finish first.' }
            $ValidationAudit = Get-Content -LiteralPath $ValidationAuditPath -Raw | ConvertFrom-Json
            if (-not $ValidationAudit.complete -or $ValidationAudit.training_steps -ne $NumSteps -or
                (@($ValidationAudit.seeds) -join ',') -ne ($SeedValues -join ',')) { throw 'Validation audit does not match this protocol.' }
            foreach ($Property in $ValidationAudit.metrics_sha256.PSObject.Properties) {
                $Path = Join-Path (Join-Path $ResultsRoot 'validation-10') $Property.Name
                if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Property.Value) {
                    throw 'Validation metrics changed after the audit.'
                }
            }
            $ReferenceImages = Join-Path $ResultsRoot 'validation-10\pretrained\frozen-images.json'
        }
        $Runner = Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
        & $Runner -Repository $RepoRoot -Dataset $Dataset -Results (Join-Path $ResultsRoot "$Split\pretrained") `
            -Profiles @($InferenceProfile) -ModelCheckpoint (Join-Path $CheckpointDir 'scaled_offline.pth') `
            -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $ReferenceImages
        $ReferenceImages = Join-Path $ResultsRoot "$Split\pretrained\frozen-images.json"
        foreach ($Seed in $SeedValues) {
            foreach ($Architecture in $Architectures) {
                & $Runner -Repository $RepoRoot -Dataset $Dataset -Results (Join-Path $ResultsRoot "$Split\seed_$Seed\$Architecture") `
                    -Profiles @($InferenceProfile) -ModelCheckpoint (Join-Path $TrainingRoot "seed_$Seed\$Architecture\cotracker_three_final.pth") `
                    -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $ReferenceImages
            }
        }
        Invoke-TrainingTool 'summarize_backbone_growth.py' @('--root','/study','--seeds',$Seeds,'--steps',"$NumSteps",'--split',$Split)
        Write-Host "Backbone-growth $Split completed: $ResultsRoot\$Split"
    }
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
