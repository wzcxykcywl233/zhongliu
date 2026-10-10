# Offline orchestration test. Uses mock Docker/runner; never launches training.
$ErrorActionPreference = 'Stop'
$TaskRoot = Join-Path ([IO.Path]::GetTempPath()) ('sidecar-recovery-test-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $TaskRoot | Out-Null
$FixtureRepo = Join-Path $TaskRoot 'repo'
$OldRoot = Join-Path $TaskRoot 'old'
$NewRoot = Join-Path $TaskRoot 'new'
New-Item -ItemType Directory -Force -Path "$FixtureRepo\scripts","$FixtureRepo\training-checkpoints","$OldRoot\train" | Out-Null
$ActualRepo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Copy-Item -LiteralPath "$ActualRepo\scripts\experiment_local_runtimes.ps1" -Destination "$FixtureRepo\scripts\experiment_local_runtimes.ps1"
$Utf8 = New-Object System.Text.UTF8Encoding($false)
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
$Weights = @(@{Name='scaled_offline.pth';SHA256=(Get-FileHash -LiteralPath $WeightFile).Hash})
Save-Json "$OldRoot\frozen-run.json" @{Dataset=$Data;Weights=$Weights;Source=@();Seeds=@(0,1,2);Architectures=@('mlp','conv','mamba_short','mamba');InferenceProfile='hierarchical_full_grid0_iterations2_memory_topk_diverse'}
$Proofs = @(1..12 | ForEach-Object { @{complete=$true} })
Save-Json "$OldRoot\train\training-audit.json" @{complete=$true;proofs=$Proofs;steps=1000}
$Image = 'sha256:' + ('a'*64)
Save-Json "$OldRoot\local-runtime-images.json" @{Training=$Image;Inference=$Image;Evaluation=$Image}
[IO.File]::WriteAllText("$OldRoot\pretrained-frozen.pth",'fixture',$Utf8)
$Runner = @'
param($Repository,$Dataset,$Results,$Profiles,$ModelCheckpoint,[switch]$RequireDiagnostics,[switch]$FreezeImages,[switch]$UsePinnedImages,$ReferenceImagesPath)
if (-not $RequireDiagnostics -or -not $FreezeImages -or -not $UsePinnedImages) { throw 'Unpinned evaluation' }
$global:RecoveryRunnerCalls += $ModelCheckpoint
'@
[IO.File]::WriteAllText("$FixtureRepo\scripts\run_hierarchical_followup_resumable.ps1",$Runner,$Utf8)
$global:RecoveryRunnerCalls = @()
$global:RecoveryDockerCalls = @()
function global:git { $global:LASTEXITCODE=0; return 'scripts/experiment_local_runtimes.ps1' }
function global:docker {
    $global:RecoveryDockerCalls += ,@($args)
    $global:LASTEXITCODE=0
    if ($args[0] -eq 'image') { return ('sha256:' + ('a'*64)) }
}
$BeforeAudit = (Get-FileHash -LiteralPath "$OldRoot\train\training-audit.json").Hash
& "$ActualRepo\scripts\repair_feature_sidecar_inference.ps1" -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage validation
if ($global:RecoveryRunnerCalls.Count -ne 13) { throw 'Expected reference plus 12 trained evaluations' }
if (@($global:RecoveryRunnerCalls | Where-Object { -not $_.StartsWith($OldRoot + '\') }).Count -ne 0) { throw 'Wrong training checkpoint source' }
if ((Get-FileHash -LiteralPath "$OldRoot\train\training-audit.json").Hash -ne $BeforeAudit) { throw 'Old audit modified' }
foreach ($Arguments in $global:RecoveryDockerCalls) {
    if (@($Arguments | Where-Object { $_ -match 'train_on_real_data.py|--num_steps|--feature_sidecar' }).Count) { throw 'Training launched by recovery' }
}
$AuditCalls = @($global:RecoveryDockerCalls | Where-Object { $_ -contains '/repair/cotracker-algorithm/experiments/verify_sidecar_evaluation_recovery.py' })
if ($AuditCalls.Count -ne 1 -or -not ($AuditCalls[0] -contains "type=bind,source=$OldRoot,target=/previous,readonly")) { throw 'Old study not read-only' }
$MissingValidationRefused = $false
try { & "$ActualRepo\scripts\repair_feature_sidecar_inference.ps1" -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage test } catch { $MissingValidationRefused=$true }
if (-not $MissingValidationRefused) { throw 'Test ran before repaired validation' }
New-Item -ItemType Directory -Force -Path "$NewRoot\validation-10" | Out-Null
$MetricPath = "$NewRoot\validation-10\fixture-metrics.json"
[IO.File]::WriteAllText($MetricPath,'verified metrics fixture',$Utf8)
Save-Json "$NewRoot\validation-10\feature-sidecar-audit.json" @{complete=$true;training_steps=1000;seeds=@(0,1,2);metrics_sha256=@{'fixture-metrics.json'=(Get-FileHash -LiteralPath $MetricPath).Hash}}
& "$ActualRepo\scripts\repair_feature_sidecar_inference.ps1" -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage test
if ($global:RecoveryRunnerCalls.Count -ne 26) { throw 'Test must evaluate reference plus 12 trained arms' }
$ChangedMetricRefused = $false
[IO.File]::WriteAllText($MetricPath,'changed metrics',$Utf8)
try { & "$ActualRepo\scripts\repair_feature_sidecar_inference.ps1" -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot -Stage test } catch { $ChangedMetricRefused=$true }
if (-not $ChangedMetricRefused -or $global:RecoveryRunnerCalls.Count -ne 26) { throw 'Changed validation metrics accepted' }
$ChangedDataRefused = $false
[IO.File]::WriteAllText($Data[0].File,'changed',$Utf8)
try { & "$ActualRepo\scripts\repair_feature_sidecar_inference.ps1" -RepoRoot $FixtureRepo -TrainingResultsRoot $OldRoot -ResultsRoot $NewRoot } catch { $ChangedDataRefused=$true }
if (-not $ChangedDataRefused) { throw 'Changed data accepted' }
Write-Host "PASS: evaluation-only, 13 arms, read-only old training, validation/data guards. Fixtures retained at $TaskRoot"
