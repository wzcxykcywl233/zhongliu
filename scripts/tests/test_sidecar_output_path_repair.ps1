# Offline fixtures only: no actual Docker/GPU/network or training.
$ErrorActionPreference='Stop'
$ActualRepo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TaskRoot=Join-Path ([IO.Path]::GetTempPath()) ('pr-'+([guid]::NewGuid().ToString('N')).Substring(0,6))
$FixtureRepo=Join-Path $TaskRoot 'repo'
New-Item -ItemType Directory -Force -Path "$FixtureRepo\scripts","$FixtureRepo\cotracker-algorithm" | Out-Null
$Utf8=New-Object Text.UTF8Encoding($false)
function Save-Json($Path,$Value) { [IO.File]::WriteAllText($Path,(ConvertTo-Json $Value -Depth 100),$Utf8) }
foreach ($Name in @('scripts/run_hierarchical_followup_resumable.ps1','scripts/run_sidecar_speed_safety_resumable.ps1')) {
    Copy-Item -LiteralPath (Join-Path $ActualRepo $Name) -Destination (Join-Path $FixtureRepo $Name)
}
$Numerical='cotracker-algorithm/model.py'
[IO.File]::WriteAllText((Join-Path $FixtureRepo $Numerical),'unchanged numerical fixture',$Utf8)
$Before=@(
    @{File='scripts/run_hierarchical_followup_resumable.ps1';SHA256='0F2ADA56A71C127B5805015E5D8684E58361177EA8674D3899D766A9442242C9'},
    @{File='scripts/run_sidecar_speed_safety_resumable.ps1';SHA256='3234679A9AF278BD8B91945A5DC442479684693D79F12B6023C999349A14960C'},
    @{File=$Numerical;SHA256=(Get-FileHash (Join-Path $FixtureRepo $Numerical)).Hash})
$Current=@(foreach ($Row in $Before) { @{File=$Row.File;SHA256=(Get-FileHash (Join-Path $FixtureRepo $Row.File)).Hash} })
$Frozen=Join-Path $TaskRoot 'frozen-study.json'
$Value=[ordered]@{Schema=1;Source=$Before;ReferenceDiagnosisOnly=$true;Gains=@(1,.5,.25);Images=@{Inference='pinned'}}
Save-Json $Frozen $Value
$OriginalHash=(Get-FileHash $Frozen).Hash
$Value.Source=$Current
. "$ActualRepo\scripts\sidecar_path_repair_provenance.ps1"
Confirm-SidecarPathRepair -Path $Frozen -Value $Value -RepoRoot $FixtureRepo
Confirm-SidecarPathRepair -Path $Frozen -Value $Value -RepoRoot $FixtureRepo
if ((Get-FileHash $Frozen).Hash -ne $OriginalHash) { throw 'Original frozen study was rewritten.' }
$Value.Gains=@(1,.75)
$Refused=$false
try { Confirm-SidecarPathRepair -Path $Frozen -Value $Value -RepoRoot $FixtureRepo } catch { $Refused=$true }
if (-not $Refused) { throw 'Changed experimental gains allowed.' }
$Value.Gains=@(1,.5,.25)
[IO.File]::WriteAllText((Join-Path $FixtureRepo $Numerical),'changed model',$Utf8)
$Current[2].SHA256=(Get-FileHash (Join-Path $FixtureRepo $Numerical)).Hash
$Refused=$false
try { Confirm-SidecarPathRepair -Path $Frozen -Value $Value -RepoRoot $FixtureRepo } catch { $Refused=$true }
if (-not $Refused) { throw 'Numerical source changed through path repair.' }

# Actual runner with mocked recovery container. Legacy output is 261 chars;
# short output and committed job remain within the Windows path budget.
$Profile='hierarchical_full_grid0_iterations2_memory_topk_diverse'
$Suffix="\$Profile\checkpoint\.attempts\A_010-b2a86cdc\output\images\mri-linac-series-targets\output.mha"
$Length=261-$Suffix.Length
$Padding=$Length-$TaskRoot.Length-1
if ($Padding -lt 1) { throw 'Use a shorter TEMP for long-path test fixture.' }
$Results=Join-Path $TaskRoot ('x'*$Padding)
$Legacy=Join-Path $Results "$Profile\checkpoint\.attempts\A_010-b2a86cdc"
$LegacyOutput=Join-Path $Legacy 'output\images\mri-linac-series-targets\output.mha'
New-Item -ItemType Directory -Force -Path (Split-Path $LegacyOutput -Parent) | Out-Null
[IO.File]::WriteAllText($LegacyOutput,'known legacy mask bytes',$Utf8)
if ($LegacyOutput.Length -ne 261) { throw 'Long-path fixture is not 261 chars.' }
$Runtime=Join-Path $TaskRoot 'runtime.json'
Save-Json $Runtime @{backend='reference';point_chunk_size=0;gain=.5;max_correction_pixels=.5;checkpoint_blocks=$false}
$Checkpoint=Join-Path $TaskRoot 'train\seed_0\mamba\cotracker_three_final.pth'
New-Item -ItemType Directory -Force -Path (Split-Path $Checkpoint -Parent) | Out-Null
[IO.File]::WriteAllText($Checkpoint,'checkpoint fixture',$Utf8)
$Dataset=Join-Path $TaskRoot 'data'
New-Item -ItemType Directory -Force -Path "$Dataset\A_010","$Legacy\output" | Out-Null
$Diagnostic=@{profile=$Profile;prediction=@{};mechanism=@{query_memory_fusions=2;query_memory_writes_accepted=2}}
[IO.File]::WriteAllText("$Legacy\case.log",('TRACKRAD_DIAGNOSTICS_JSON='+(ConvertTo-Json $Diagnostic -Compress -Depth 20)+"`nRuntime algorithm: 112.11512 s`nRuntime writing: 0.14606 s`n"),$Utf8)
$FakeImageId='sha256:'+('a'*64)
$Pinned=Join-Path $TaskRoot 'images.json'
Save-Json $Pinned @{'trackrad-algorithm-cotracker-algorithm'=$FakeImageId;'trackrad-evaluation'=$FakeImageId}
$Finalizer=@'
param($Repository,$Dataset,$Results,$Profiles,$EvaluationImage)
foreach ($Profile in $Profiles) { [IO.File]::WriteAllText((Join-Path $Results "$Profile\metrics.json"),'{}') }
'@
[IO.File]::WriteAllText("$FixtureRepo\scripts\finalize_hierarchical_results.ps1",$Finalizer,$Utf8)
$global:RecoveryCalls=0
function global:docker {
    $global:LASTEXITCODE=0
    if ($args[0] -eq 'image') { return $FakeImageId }
    if ($args -notcontains '/repair/recover.py') { throw 'Model inference/build unexpectedly launched; recovery should reuse bytes.' }
    if ($args -contains '--gpus') { throw 'Recovery uses GPU unnecessarily.' }
    if ($args -notcontains "type=bind,source=$($Dataset)\A_010,target=/case,readonly") { throw 'Wrong case bind.' }
    $Destination=([string](@($args | Where-Object { $_ -like '*target=/recovered' })[0])).Replace('type=bind,source=','').Replace(',target=/recovered','')
    $NewOutput=Join-Path $Destination 'output\images\mri-linac-series-targets\output.mha'
    if ($NewOutput.Length -ge 256) { throw 'New transient path still too long.' }
    New-Item -ItemType Directory -Force -Path (Split-Path $NewOutput -Parent) | Out-Null
    [IO.File]::Copy($LegacyOutput,$NewOutput)
    Copy-Item "$Legacy\case.log" "$Destination\case.log"
    Save-Json "$Destination\recovery.json" @{recovered=$true;source_attempt='A_010-b2a86cdc';started_at='2026-10-11T15:43:22Z';completed_at='2026-10-11T15:45:18Z'}
    $global:RecoveryCalls++
}
& "$ActualRepo\scripts\run_hierarchical_followup_resumable.ps1" -Repository $FixtureRepo -Dataset $Dataset -Results $Results `
    -Profiles @($Profile) -ModelCheckpoint $Checkpoint -SidecarRuntimeConfig $Runtime -SidecarTrackCache `
    -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $Pinned -RecoverLongPathOutput
$Committed=Join-Path $Results "$Profile\checkpoint\jobs\A_010"
if (-not (Test-Path "$Committed\.complete") -or $global:RecoveryCalls -ne 1 -or -not [IO.File]::Exists($LegacyOutput)) { throw 'Recovery did not preserve/commit the old output.' }
& "$ActualRepo\scripts\run_hierarchical_followup_resumable.ps1" -Repository $FixtureRepo -Dataset $Dataset -Results $Results `
    -Profiles @($Profile) -ModelCheckpoint $Checkpoint -SidecarRuntimeConfig $Runtime -SidecarTrackCache `
    -RequireDiagnostics -FreezeImages -UsePinnedImages -ReferenceImagesPath $Pinned -RecoverLongPathOutput
if ($global:RecoveryCalls -ne 1) { throw 'Completed metrics were rerun.' }
Write-Host "PASS reviewed host-only migration, unchanged frozen source, 261-character recovery and no inference. Fixtures retained at $TaskRoot"
