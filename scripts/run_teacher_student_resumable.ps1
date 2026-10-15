[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$ResultsRoot = 'C:\zhongliu\ts-v1',
    [string]$TrainDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40',
    [string]$ValidationDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$TestDataset = 'C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38',
    [ValidateSet('plan','smoke','pilot','train','validation','test')][string]$Stage = 'plan',
    [string]$TrainingRuntimeImage = 'trackrad-cotracker-gt-mask-training:latest',
    [string]$InferenceRuntimeImage = 'trackrad-algorithm-cotracker-algorithm:latest',
    [ValidateRange(5,120)][double]$MaxMedianStepSeconds = 15,
    [ValidateRange(300,14400)][int]$ArmTimeoutSeconds = 7200,
    [switch]$AllowDegradedPilot
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$ResultsRoot = [IO.Path]::GetFullPath($ResultsRoot).TrimEnd('\')
if ($ResultsRoot.Length -gt 70 -or $ResultsRoot -match ',') { throw 'Use a short new results root, e.g. C:\zhongliu\ts-v1; no commas.' }
if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot '.git'))) { throw 'RepoRoot must be the repository checkout.' }
$Policies = @('old','scaled','grid0','iter6')
$Profile = 'hierarchical_full_grid0_iterations2_memory_topk_diverse'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-StudyJson([string]$Path, $Value, [switch]$Frozen) {
    $Json = ConvertTo-Json -InputObject $Value -Depth 100
    if ($Frozen -and (Test-Path -LiteralPath $Path)) {
        if ([IO.File]::ReadAllText($Path) -ne $Json) { throw "Frozen protocol changed: $Path. Preserve this study and use a NEW ResultsRoot." }
        return
    }
    [IO.File]::WriteAllText("$Path.tmp",$Json,$Utf8)
    Move-Item -LiteralPath "$Path.tmp" -Destination $Path -Force
}
function Read-CheckedJson([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required completed stage missing: $Path" }
    return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')

# Fingerprint both existing tracked code and new source (before the first commit).
$Files = @(& git -C $RepoRoot ls-files --cached --others --exclude-standard -- cotracker-algorithm evaluation scripts)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inventory source.' }
$Source = @(foreach ($File in ($Files | Sort-Object -Unique)) {
    if ($File -match '\.(py|ps1|sh|json|toml|lock|txt)$|Dockerfile') {
        [ordered]@{File=$File; SHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot $File)).Hash}
    }
})
$Weights = @(foreach ($Name in @('scaled_offline.pth','baseline_offline.pth','baseline_online.pth','cotracker2v1.pth')) {
    [ordered]@{Name=$Name; SHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot "training-checkpoints\$Name")).Hash}
})
$Seen = @{}; $Data = @()
foreach ($Split in @(@{Name='train-40'; Path=$TrainDataset; Count=40},
    @{Name='validation-10'; Path=$ValidationDataset; Count=10}, @{Name='test-38'; Path=$TestDataset; Count=38})) {
    $Cases = @(Get-ChildItem -LiteralPath $Split.Path -Directory | Sort-Object Name)
    if ($Cases.Count -ne $Split.Count) { throw "Expected $($Split.Count) cases: $($Split.Path)" }
    foreach ($Case in $Cases) {
        if ($Seen.ContainsKey($Case.Name)) { throw "Overlapping split ID: $($Case.Name)" }
        $Seen[$Case.Name]=$Split.Name
        foreach ($Required in @("images\$($Case.Name)_frames.mha", "targets\$($Case.Name)_first_label.mha",
            "targets\$($Case.Name)_labels.mha", 'frame-rate.json','b-field-strength.json','scanned-region.json')) {
            if (-not (Test-Path -LiteralPath (Join-Path $Case.FullName $Required) -PathType Leaf)) { throw "Missing $($Case.FullName)\$Required" }
        }
        foreach ($Item in @(Get-ChildItem -LiteralPath $Case.FullName -Recurse -File | Sort-Object FullName)) {
            $Data += [ordered]@{Split=$Split.Name; Case=$Case.Name; File=$Item.FullName; SHA256=(Get-FileHash -LiteralPath $Item.FullName).Hash}
        }
    }
}
$Common = [ordered]@{Schema=1; Source=$Source; Weights=$Weights; Dataset=$Data;
    TrainingRuntime=$TrainingRuntimeImage; InferenceRuntime=$InferenceRuntimeImage;
    Policies=$Policies; Student='base-scaled-init'; TeacherPool=@('cotracker2v1','online_cotracker_three','offline_cotracker_three');
    SequenceLength=10; QueryPoints=384; TrainIterations=4; LearningRate=0.00005; BaseSeed=20261017;
    ConfidenceTarget='hard'; AuxiliaryWeight=0; MaskWeight=0; InferenceProfile=$Profile;
    MaxMedianStepSeconds=$MaxMedianStepSeconds; ArmTimeoutSeconds=$ArmTimeoutSeconds;
    SmokeSteps=5; PilotSteps=125; FullSteps=1000; FullSeeds=@(0,1,2)}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$Lock = [IO.File]::Open((Join-Path $ResultsRoot '.teacher-student.lock'),'OpenOrCreate','ReadWrite','None')
$Started = $false
try {
    $Frozen = Join-Path $ResultsRoot 'frozen-study.json'
    if (-not (Test-Path -LiteralPath $Frozen) -and @(Get-ChildItem -LiteralPath $ResultsRoot -Recurse -File -Filter '*.pth').Count) {
        throw 'Unrecorded checkpoints in ResultsRoot; use a new directory.'
    }
    Write-StudyJson $Frozen $Common -Frozen
    $ProtocolSHA = (Get-FileHash -LiteralPath $Frozen).Hash
    Write-Host "Teacher -> student stage=$Stage. 40/10/38 disjoint. Only offline slot changes; no Mamba/soft labels/mask supervision."
    if ($Stage -eq 'plan') {
        Write-Host 'PLAN VERIFIED. smoke=4x5 steps; pilot=4x125 steps + validation; full=3x4x1000 steps. No Docker/build/training in plan.'
        return
    }
    Start-Transcript -Path (Join-Path $ResultsRoot 'queue.log') -Append | Out-Null; $Started=$true
    $Phase = if ($Stage -eq 'smoke') {'smoke'} elseif ($Stage -eq 'pilot') {'pilot'} else {'full'}
    $PhaseRoot = Join-Path $ResultsRoot $Phase
    New-Item -ItemType Directory -Force -Path $PhaseRoot | Out-Null
    $Steps = if ($Phase -eq 'smoke') {5} elseif ($Phase -eq 'pilot') {125} else {1000}
    $Seeds = if ($Phase -eq 'full') {'0,1,2'} else {'0'}
    $SeedValues = @($Seeds.Split(',') | ForEach-Object { [int]$_ })
    if ($Stage -eq 'pilot') {
        $Smoke = Read-CheckedJson (Join-Path $ResultsRoot 'smoke\training-audit.json')
        $GpuProof = Read-CheckedJson (Join-Path $ResultsRoot 'smoke\gpu-teacher-proof.json')
        $StudentProof = Read-CheckedJson (Join-Path $ResultsRoot 'smoke\gpu-student-inference-proof.json')
        if (-not $Smoke.complete -or $Smoke.steps -ne 5 -or -not $GpuProof.complete -or -not $StudentProof.complete) { throw 'Complete smoke first.' }
    }
    if ($Stage -eq 'train') {
        $Pilot = Read-CheckedJson (Join-Path $ResultsRoot 'pilot\validation-10\teacher-student-audit.json')
        if (-not $Pilot.complete -or $Pilot.steps -ne 125 -or $Pilot.split -ne 'validation-10' -or
            ($Pilot.seeds -join ',') -ne '0' -or ($Pilot.policies -join ',') -ne ($Policies -join ',')) { throw 'Complete pilot and inspect runtime/label diagnostics first.' }
        $PilotValidationRoot=Join-Path $ResultsRoot 'pilot\validation-10'
        foreach ($Property in $Pilot.metrics_sha256.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath (Join-Path $PilotValidationRoot $Property.Name)).Hash.ToLowerInvariant() -ne $Property.Value) { throw 'Pilot official metrics changed.' }
        }
        # Gate on frozen official JSON, never a manually edited summary table.
        $PretrainedMetrics=Read-CheckedJson (Join-Path $PilotValidationRoot "pretrained\$Profile\metrics.json")
        $Pretrained=[double]$PretrainedMetrics.aggregates.dice_similarity_coefficient
        $Candidates=@(foreach ($Policy in @('scaled','grid0','iter6')) {
            $Metrics=Read-CheckedJson (Join-Path $PilotValidationRoot "seed_0\$Policy\$Profile\metrics.json")
            if ([double]$Metrics.aggregates.dice_similarity_coefficient -gt $Pretrained) { $Policy }
        })
        if (-not $Candidates.Count -and -not $AllowDegradedPilot) { throw 'No pilot candidate beats pretrained DSC. Inspect degradation first; full training requires explicit -AllowDegradedPilot.' }
        Write-StudyJson (Join-Path $PhaseRoot 'launch-decision.json') ([ordered]@{PilotAuditSHA256=(Get-FileHash -LiteralPath (Join-Path $ResultsRoot 'pilot\validation-10\teacher-student-audit.json')).Hash; AllowDegradedPilot=[bool]$AllowDegradedPilot}) -Frozen
    }
    $Images = Initialize-LocalExperimentImages -RepoRoot $RepoRoot -ResultsRoot $ResultsRoot `
        -TrainingRuntime $TrainingRuntimeImage -InferenceRuntime $InferenceRuntimeImage
    $ReferenceImages = Join-Path $ResultsRoot 'reference-images.json'
    function Invoke-StudyTool([string]$Command, [string[]]$Extra=@(), [switch]$Gpu, [switch]$Inference) {
        $Arguments = @('run','--rm','--pull','never','--network','none')
        if ($Gpu) { $Arguments += @('--gpus','device=0') }
        $Image = if ($Inference) {$Images.Inference} else {$Images.Training}
        $Python = if ($Inference) {'/opt/app/.pixi/envs/cuda/bin/python'} else {'/opt/app/.pixi/envs/training/bin/python'}
        $Arguments += @('--mount',"type=bind,source=$PhaseRoot,target=/study",
            '--mount',"type=bind,source=$RepoRoot\training-checkpoints,target=/checkpoints,readonly",
            '--entrypoint','/usr/bin/timeout',$Image,'--signal=TERM','--kill-after=30s','600s',
            $Python,'/opt/app/experiments/teacher_student_study.py',
            $Command,'--root','/study','--seeds',$Seeds,'--steps',"$Steps") + $Extra
        Invoke-LocalDocker $Arguments (Join-Path $PhaseRoot 'audit.log')
    }
    Invoke-StudyTool 'runtime-proof'
    $RuntimeProof = Read-CheckedJson (Join-Path $PhaseRoot 'runtime-proof.json')
    if (-not $RuntimeProof.complete -or $RuntimeProof.trainer_sha256 -ne
        (Get-FileHash -LiteralPath (Join-Path $RepoRoot 'cotracker-algorithm\ext\co-tracker\train_on_real_data.py')).Hash.ToLowerInvariant()) {
        throw 'Training image does not contain the frozen current trainer.'
    }
    if ($Stage -eq 'smoke') { Invoke-StudyTool 'gpu-proof' -Gpu }
    if ($Stage -in @('smoke','pilot','train')) {
        foreach ($Seed in $SeedValues) {
            foreach ($Policy in $Policies) {
                $Arm = Join-Path $PhaseRoot "train\seed_$Seed\$Policy"
                New-Item -ItemType Directory -Force -Path $Arm | Out-Null
                $OfflineWeight = if ($Policy -eq 'old') {'baseline_offline.pth'} else {'scaled_offline.pth'}
                Write-StudyJson (Join-Path $Arm 'run-config.json') ([ordered]@{policy=$Policy; offline_checkpoint=$OfflineWeight;
                    steps=$Steps; seed=$Seed; training_seed=(20261017+$Seed); source_protocol_sha256=$ProtocolSHA;
                    sequence_len=10; queries=384; train_iters=4; lr=0.00005; save_every_steps=25; phase=$Phase}) -Frozen
                if (Test-Path -LiteralPath (Join-Path $Arm 'cotracker_three_final.pth')) { Write-Host "VERIFY completed arm later: seed_$Seed/$Policy"; continue }
                Write-Host "TRAIN seed_$Seed/$Policy steps=$Steps; timeout=$ArmTimeoutSeconds seconds"
                Invoke-LocalDocker @('run','--rm','--pull','never','--network','none','--gpus','device=0','--shm-size','4g',
                    '--mount',"type=bind,source=$TrainDataset,target=/data,readonly",
                    '--mount',"type=bind,source=$RepoRoot\training-checkpoints,target=/checkpoints,readonly",
                    '--mount',"type=bind,source=$Arm,target=/results",'--workdir','/opt/app/ext/co-tracker',
                    '--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=30s',"${ArmTimeoutSeconds}s",
                    '/opt/app/.pixi/envs/training/bin/python','/opt/app/ext/co-tracker/train_on_real_data.py',
                    '--batch_size','1','--num_workers','0','--num_steps',"$Steps",'--ckpt_path','/results',
                    '--model_name','cotracker_three','--experiment_name',"ts_seed_${Seed}_$Policy",
                    '--sequence_len','10','--traj_per_sample','384','--train_iters','4','--save_freq','100000',
                    '--save_every_n_steps','25','--keep_last_checkpoints','3','--save_every_n_epoch','1',
                    '--trackrad_data_dir','/data','--skip_evaluation','--offline_model',
                    '--restore_ckpt','/checkpoints/scaled_offline.pth',
                    '--teacher_types','cotracker2v1','online_cotracker_three','offline_cotracker_three',
                    '--teacher_cotracker2_ckpt','/checkpoints/cotracker2v1.pth',
                    '--teacher_online_ckpt','/checkpoints/baseline_online.pth',
                    '--teacher_offline_ckpt',"/checkpoints/$OfflineWeight",'--auxiliary_teacher_weight','0',
                    '--seed',"$(20261017+$Seed)",'--paired_step_seed',"$(20261017+$Seed)",'--teacher_seed',"$(20261017+$Seed)",
                    '--teacher_log_every','1','--confidence_target_mode','hard','--lr','0.00005',
                    '--backbone_growth_study','--backbone_architecture','base',
                    '--teacher_student_study','--offline_teacher_policy',$Policy) (Join-Path $Arm 'training.log')
            }
        }
        Invoke-StudyTool 'audit' @('--max-median-seconds',"$MaxMedianStepSeconds")
    } else { Invoke-StudyTool 'verify-training' }
    if ($Stage -eq 'smoke') { Invoke-StudyTool 'student-smoke' -Gpu -Inference }
    if ($Stage -in @('smoke','train')) { Write-Host "Completed $Stage. No automatic next stage. Results: $PhaseRoot"; return }

    $Split = if ($Stage -eq 'test') {'test-38'} else {'validation-10'}
    $Dataset = if ($Split -eq 'test-38') {$TestDataset} else {$ValidationDataset}
    if ($Split -eq 'test-38') {
        $ValidationRoot = Join-Path $PhaseRoot 'validation-10'
        $Validation = Read-CheckedJson (Join-Path $ValidationRoot 'teacher-student-audit.json')
        if (-not $Validation.complete -or $Validation.steps -ne $Steps -or ($Validation.seeds -join ',') -ne $Seeds) { throw 'Complete full validation first.' }
        foreach ($Property in $Validation.metrics_sha256.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath (Join-Path $ValidationRoot $Property.Name)).Hash.ToLowerInvariant() -ne $Property.Value) { throw 'Validation metrics changed.' }
        }
        Write-StudyJson (Join-Path $PhaseRoot 'test-shortlist.json') ([ordered]@{SelectionSplit='validation-10';
            Policies=$Policies; Seeds=$SeedValues; Steps=$Steps; Rule='All four predeclared arms, no test-driven selection';
            ValidationAuditSHA256=(Get-FileHash -LiteralPath (Join-Path $ValidationRoot 'teacher-student-audit.json')).Hash}) -Frozen
    }
    $Runner = Join-Path $RepoRoot 'scripts\run_hierarchical_followup_resumable.ps1'
    & $Runner -Repository $RepoRoot -Dataset $Dataset -Results (Join-Path $PhaseRoot "$Split\pretrained") `
        -Profiles @($Profile) -ModelCheckpoint (Join-Path $RepoRoot 'training-checkpoints\scaled_offline.pth') `
        -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $ReferenceImages
    foreach ($Seed in $SeedValues) {
        foreach ($Policy in $Policies) {
            & $Runner -Repository $RepoRoot -Dataset $Dataset -Results (Join-Path $PhaseRoot "$Split\seed_$Seed\$Policy") `
                -Profiles @($Profile) -ModelCheckpoint (Join-Path $PhaseRoot "train\seed_$Seed\$Policy\cotracker_three_final.pth") `
                -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $ReferenceImages
        }
    }
    Invoke-StudyTool 'report' @('--split',$Split)
    Write-Host "Student evaluation complete: $PhaseRoot\$Split. Do NOT substitute teacher-generator scores."
} finally {
    if ($Started) { Stop-Transcript | Out-Null }
    $Lock.Dispose()
}
