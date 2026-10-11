# Offline orchestration proof only. Mock Docker and training; no GPU/network tasks.
$ErrorActionPreference = 'Stop'
$TaskRoot = Join-Path ([IO.Path]::GetTempPath()) ('sidecar-speed-test-' + [guid]::NewGuid())
$FixtureRepo = Join-Path $TaskRoot 'repo'
$OldRoot = Join-Path $TaskRoot 'old'
$NewRoot = Join-Path $TaskRoot 'new'
New-Item -ItemType Directory -Force -Path "$FixtureRepo\scripts","$FixtureRepo\training-checkpoints","$FixtureRepo\cotracker-algorithm\resources","$OldRoot\train" | Out-Null
$ActualRepo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Copy-Item -LiteralPath "$ActualRepo\scripts\experiment_local_runtimes.ps1" -Destination "$FixtureRepo\scripts\experiment_local_runtimes.ps1"
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$FixtureRepo\cotracker-algorithm\resources\model.py",'loader fixture',$Utf8)
function Save-Json($Path,$Value) { [IO.File]::WriteAllText($Path,(ConvertTo-Json -InputObject $Value -Depth 30),$Utf8) }
$Data = @()
foreach ($Split in @(@{Name='train-40';Count=40},@{Name='validation-10';Count=10},@{Name='test-38';Count=38})) {
    for ($Index=0; $Index -lt $Split.Count; $Index++) {
        $Case = $Split.Name + '-' + $Index.ToString('D2')
        $Folder = "$TaskRoot\data\$($Split.Name)\$Case"
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        $File = Join-Path $Folder 'sample.txt'
        [IO.File]::WriteAllText($File,'fixture',$Utf8)
        $Data += @{Split=$Split.Name;Case=$Case;File=$File;SHA256=(Get-FileHash -LiteralPath $File).Hash}
    }
}
$WeightFile = "$FixtureRepo\training-checkpoints\scaled_offline.pth"
[IO.File]::WriteAllText($WeightFile,'fixture',$Utf8)
Save-Json "$OldRoot\frozen-run.json" @{Dataset=$Data;Weights=@(@{Name='scaled_offline.pth';SHA256=(Get-FileHash -LiteralPath $WeightFile).Hash});InferenceProfile='hierarchical_full_grid0_iterations2_memory_topk_diverse'}
Save-Json "$OldRoot\train\training-audit.json" @{complete=$true;proofs=@(1..12 | ForEach-Object { @{complete=$true} });steps=1000}
$Image = 'sha256:' + ('a'*64)
Save-Json "$OldRoot\local-runtime-images.json" @{Training=$Image;Inference=$Image;Evaluation=$Image}
[IO.File]::WriteAllText("$OldRoot\pretrained-frozen.pth",'fixture',$Utf8)
$Runner = @'
param($Repository,$Dataset,$Results,$Profiles,$ModelCheckpoint,[switch]$RequireDiagnostics,[switch]$SidecarTrackCache,$SidecarRuntimeConfig,[switch]$FreezeImages,[switch]$UsePinnedImages,$ReferenceImagesPath)
if (-not $RequireDiagnostics -or -not $SidecarTrackCache -or -not $FreezeImages -or -not $UsePinnedImages) { throw 'Unpinned/undiagnosed evaluation' }
if ($SidecarRuntimeConfig -and -not (Test-Path -LiteralPath $SidecarRuntimeConfig)) { throw 'Missing intervention' }
$global:SpeedRunnerCalls += @{Results=$Results;Checkpoint=$ModelCheckpoint}
'@
[IO.File]::WriteAllText("$FixtureRepo\scripts\run_hierarchical_followup_resumable.ps1",$Runner,$Utf8)
$Trainer = @'
param($RepoRoot,$ResultsRoot,$Stage,$SidecarRuntimeConfig,$TrainDataset,$ValidationDataset,$TestDataset,$TrainingRuntimeImage,$InferenceRuntimeImage)
if ($Stage -ne 'smoke') { throw 'Large training forbidden in this queue' }
$global:SpeedTrainingCalls += $SidecarRuntimeConfig
$SmokeRoot=$ResultsRoot+'-smoke'
New-Item -ItemType Directory -Force -Path "$SmokeRoot\train" | Out-Null
$Encoding=New-Object System.Text.UTF8Encoding($false)
$Audit=@{complete=$true;steps=5;proofs=@(1..4 | ForEach-Object { @{complete=$true} });final_checkpoints_sha256=@{}}
[IO.File]::WriteAllText("$SmokeRoot\train\training-audit.json",(ConvertTo-Json $Audit -Depth 10),$Encoding)
[IO.File]::WriteAllText("$SmokeRoot\frozen-run.json",(ConvertTo-Json @{SidecarRuntimeSHA256=(Get-FileHash -LiteralPath $SidecarRuntimeConfig).Hash}),$Encoding)
'@
[IO.File]::WriteAllText("$FixtureRepo\scripts\run_feature_sidecar_resumable.ps1",$Trainer,$Utf8)
$global:SpeedRunnerCalls=@()
$global:SpeedTrainingCalls=@()
$global:SpeedDockerCalls=@()
$global:SpeedTestOutput=$NewRoot
function global:git { $global:LASTEXITCODE=0; return 'scripts/experiment_local_runtimes.ps1' }
function global:docker {
    $global:SpeedDockerCalls += ,@($args)
    $global:LASTEXITCODE=0
    if ($args[0] -eq 'image') { return ('sha256:' + ('a'*64)) }
    if ($args -contains '/opt/app/experiments/profile_sidecar_speed.py') {
        $LoaderIndex=[Array]::IndexOf($args,'--loader-sha256')
        if ($LoaderIndex -lt 0) { throw 'No real loader hash provided to GPU profile' }
        $Profile=@{complete=$true;checkpoint_loader=@{sha256=$args[$LoaderIndex+1]};selected=@{measured_speedup=4;runtime=@{backend='parallel';point_chunk_size=16;gain=1;max_correction_pixels=0;checkpoint_blocks=$true}};source_checkpoints_sha256=@{}}
        [IO.File]::WriteAllText("$global:SpeedTestOutput\speed-profile.json",(ConvertTo-Json $Profile -Depth 10),(New-Object System.Text.UTF8Encoding($false)))
    }
    if ($args -contains '/opt/app/experiments/profile_sidecar_isolated.py') {
        if (-not ($args -contains '/usr/bin/timeout') -or -not ($args -contains '180s')) { throw 'Unbounded synthetic probe' }
        $Folder=([string](@($args | Where-Object { $_ -like '*target=/probe' })[0])).Replace('type=bind,source=','').Replace(',target=/probe','')
        $Kind=$args[[Array]::IndexOf($args,'--kind')+1]
        $Backend=$args[[Array]::IndexOf($args,'--backend')+1]
        $Chunk=[int]$args[[Array]::IndexOf($args,'--chunk')+1]
        if ($args -contains '--checkpoint') { $global:LASTEXITCODE=124; return }
        Save-Json (Join-Path $Folder 'result.json') @{complete=$true;kind=$Kind;runtime=@{backend=$Backend;point_chunk_size=$Chunk;gain=1;max_correction_pixels=0;checkpoint_blocks=$false}}
    }
    if ($args -contains '/opt/app/experiments/profile_sidecar_training_phases.py') {
        if (-not ($args -contains '/usr/bin/timeout') -or -not ($args -contains '900s') -or -not ($args -contains '--steps') -or $args[[Array]::IndexOf($args,'--steps')+1] -ne '3') { throw 'Unbounded/wrong real diagnostic steps' }
        if (-not ($args -contains "type=bind,source=$global:DiagnosticTrainData,target=/data,readonly")) { throw 'Wrong data mount interpolation' }
        $Folder=([string](@($args | Where-Object { $_ -like '*target=/results' })[0])).Replace('type=bind,source=','').Replace(',target=/results','')
        Save-Json (Join-Path $Folder 'phase-profile.json') @{complete=$true;diagnostic_steps=3;scheduler_num_steps=1000}
    }
}
$Queue="$ActualRepo\scripts\run_sidecar_speed_safety_resumable.ps1"
$BeforeAudit=(Get-FileHash -LiteralPath "$OldRoot\train\training-audit.json").Hash
& $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage profile
if ($global:SpeedRunnerCalls.Count -or $global:SpeedTrainingCalls.Count) { throw 'Profile launched training/evaluation' }
$Refused=$false
try { & $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage validation } catch { $Refused=$true }
if (-not $Refused -or $global:SpeedRunnerCalls.Count) { throw 'Evaluation allowed before real smoke proof' }
& $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage smoke
if ($global:SpeedTrainingCalls.Count -ne 1 -or $global:SpeedRunnerCalls.Count) { throw 'Smoke stage not isolated' }
& $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage validation
if ($global:SpeedRunnerCalls.Count -ne 26) { throw 'Expected reference, reference-scan shadow and 8 x 3 arms' }
$Refused=$false
try { & $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage test } catch { $Refused=$true }
if (-not $Refused -or $global:SpeedRunnerCalls.Count -ne 26) { throw 'Test ran before complete validation' }
New-Item -ItemType Directory -Force -Path "$NewRoot\validation-10" | Out-Null
$MetricPath="$NewRoot\validation-10\fixture-metrics.json"
[IO.File]::WriteAllText($MetricPath,'verified metrics fixture',$Utf8)
$SelectionPath="$NewRoot\selection.json"
Save-Json $SelectionPath @{backend_agreement_passed=$true;selected_arms=@('mlp_gain050')}
Save-Json "$NewRoot\validation-10\safety-audit.json" @{complete=$true;cases=10;selection_sha256=(Get-FileHash -LiteralPath $SelectionPath).Hash;metrics_sha256=@{'fixture-metrics.json'=(Get-FileHash -LiteralPath $MetricPath).Hash}}
& $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage test
if ($global:SpeedRunnerCalls.Count -ne 36) { throw 'Expected reference plus full MLP/Mamba and one selected arm x 3' }
Save-Json $SelectionPath @{backend_agreement_passed=$true;selected_arms=@('mamba_gain025')}
$Refused=$false
try { & $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage test } catch { $Refused=$true }
if (-not $Refused -or $global:SpeedRunnerCalls.Count -ne 36) { throw 'Changed selection accepted' }
Save-Json $SelectionPath @{backend_agreement_passed=$true;selected_arms=@()}
Save-Json "$NewRoot\validation-10\safety-audit.json" @{complete=$true;cases=10;selection_sha256=(Get-FileHash -LiteralPath $SelectionPath).Hash;metrics_sha256=@{'fixture-metrics.json'=(Get-FileHash -LiteralPath $MetricPath).Hash}}
$Refused=$false
try { & $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage test } catch { $Refused=$true }
if (-not $Refused -or $global:SpeedRunnerCalls.Count -ne 36) { throw 'Unqualified study ran test cases' }
# New diagnosis must be usable with NO selected speed profile or training smoke.
$IndependentRoot=Join-Path $TaskRoot 'independent'
& $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $IndependentRoot -Stage validation -ReferenceDiagnosisOnly
if ($global:SpeedRunnerCalls.Count -ne 61 -or $global:SpeedTrainingCalls.Count -ne 1) { throw 'Independent diagnosis expected 25 evaluation groups and no training' }
if (Test-Path -LiteralPath "$IndependentRoot\speed-profile.json") { throw 'Independent diagnosis depends on speed profiling' }
$Spec=Get-Content -LiteralPath "$IndependentRoot\study-spec.json" -Raw | ConvertFrom-Json
if ($Spec.requires_scan_agreement -or @($Spec.arms | Where-Object { $_.runtime.backend -ne 'reference' -or $_.runtime.point_chunk_size -ne 0 -or $_.runtime.checkpoint_blocks }).Count) { throw 'Independent diagnosis changed scan execution' }
# Timing queue uses new roots and fresh Docker invocations, bounded per setting.
New-Item -ItemType Directory -Force -Path "$FixtureRepo\cotracker-algorithm\ext\co-tracker" | Out-Null
Copy-Item -LiteralPath "$ActualRepo\cotracker-algorithm\ext\co-tracker\train_on_real_data.py" -Destination "$FixtureRepo\cotracker-algorithm\ext\co-tracker\train_on_real_data.py"
$Original=Get-Content -LiteralPath "$OldRoot\frozen-run.json" -Raw | ConvertFrom-Json
$Original | Add-Member -NotePropertyName Source -NotePropertyValue @(@{File='cotracker-algorithm/ext/co-tracker/train_on_real_data.py';SHA256=(Get-FileHash -LiteralPath "$FixtureRepo\cotracker-algorithm\ext\co-tracker\train_on_real_data.py").Hash})
Save-Json "$OldRoot\frozen-run.json" $Original
$TimingRoot=Join-Path $TaskRoot 'timing'
$TimingQueue="$ActualRepo\scripts\run_sidecar_runtime_diagnostics.ps1"
$global:DiagnosticTrainData=Join-Path $TaskRoot 'data\train-40'
& $TimingQueue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $TimingRoot -Stage profile
$SyntheticCalls=@($global:SpeedDockerCalls | Where-Object { $_ -contains '/opt/app/experiments/profile_sidecar_isolated.py' })
if ($SyntheticCalls.Count -ne 11) { throw 'Expected 11 fresh-process synthetic configurations' }
$Timeouts=@(Get-ChildItem -LiteralPath "$TimingRoot\isolated" -Recurse -Filter result.json | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json } | Where-Object status -eq 'timeout')
if ($Timeouts.Count -ne 4 -or @($Timeouts | Where-Object complete).Count) { throw 'Timeouts incorrectly marked complete' }
& $TimingQueue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $TimingRoot -Stage profile
if (@($global:SpeedDockerCalls | Where-Object { $_ -contains '/opt/app/experiments/profile_sidecar_isolated.py' }).Count -ne 11) { throw 'Recorded profiles repeated unexpectedly' }
& $TimingQueue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $TimingRoot -Stage training
if (@($global:SpeedDockerCalls | Where-Object { $_ -contains '/opt/app/experiments/profile_sidecar_training_phases.py' }).Count -ne 2) { throw 'Expected only two real 3-step diagnostic arms' }
Save-Json "$TimingRoot\training\seed_0\mamba\phase-profile.json" @{complete=$false}
$Refused=$false
try { & $TimingQueue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $TimingRoot -Stage training } catch { $Refused=$true }
if (-not $Refused) { throw 'Mixed partial real training accepted' }
[IO.File]::WriteAllText($Data[0].File,'changed',$Utf8)
$Refused=$false
try { & $Queue -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage profile } catch { $Refused=$true }
if (-not $Refused) { throw 'Changed data accepted' }
if ((Get-FileHash -LiteralPath "$OldRoot\train\training-audit.json").Hash -ne $BeforeAudit) { throw 'Old audit modified' }
foreach ($Arguments in $global:SpeedDockerCalls) {
    if ($Arguments -contains '--num_steps' -or $Arguments -contains 'train_on_real_data.py') { throw 'Uncontrolled training' }
    if ($Arguments -contains '/opt/app/experiments/audit_sidecar_speed_source.py' -and -not ($Arguments -contains "type=bind,source=$OldRoot,target=/previous,readonly")) { throw 'Old study not read-only' }
}
Write-Host "PASS: separate stages, smoke/validation gates, immutable selection/data, no large training. Fixtures retained at $TaskRoot"
