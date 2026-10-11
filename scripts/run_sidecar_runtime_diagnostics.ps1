[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$TrainingResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-modefix-v3',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\sidecar-runtime-diagnostics-v3',
    [ValidateSet('profile','training','validation','test')][string]$Stage = 'training',
    [ValidateRange(1,5)][int]$ProbeSteps = 3,
    [ValidateRange(60,600)][int]$ProfileTimeoutSeconds = 180,
    [ValidateRange(120,1800)][int]$TrainingTimeoutSeconds = 900,
    [switch]$CompactScan
)
$ErrorActionPreference = 'Stop'
if ($CompactScan -and ($ProbeSteps -ne 3 -or $Stage -notin @('profile','training'))) { throw 'Compact scan allows GPU proof then three-step training diagnosis only.' }
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path
$TrainingResultsRoot=(Resolve-Path -LiteralPath $TrainingResultsRoot).Path.TrimEnd('\')
$ResultsRoot=[IO.Path]::GetFullPath($ResultsRoot).TrimEnd('\')
if ($ResultsRoot -eq $TrainingResultsRoot -or $ResultsRoot.StartsWith($TrainingResultsRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $TrainingResultsRoot.StartsWith($ResultsRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Use a separate new results root; preserve the old training study.' }
# Degradation is independent: deliberately no speed-profile / smoke prerequisite.
if ($Stage -in @('validation','test')) {
    & (Join-Path $RepoRoot 'scripts\run_sidecar_speed_safety_resumable.ps1') -RepoRoot $RepoRoot `
        -TrainingResultsRoot $TrainingResultsRoot -ResultsRoot (Join-Path $ResultsRoot 'degradation') `
        -Stage $Stage -ReferenceDiagnosisOnly
    return
}
. (Join-Path $RepoRoot 'scripts\experiment_local_runtimes.ps1')
$Old=Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'frozen-run.json') -Raw | ConvertFrom-Json
$OldImages=Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'local-runtime-images.json') -Raw | ConvertFrom-Json
$OldAudit=Get-Content -LiteralPath (Join-Path $TrainingResultsRoot 'train\training-audit.json') -Raw | ConvertFrom-Json
if (-not $OldAudit.complete -or $OldAudit.steps -ne 1000 -or $OldAudit.proofs.Count -ne 12) { throw 'Original twelve arms must be complete.' }
$Datasets=@{}; $Seen=@{}
foreach ($Split in @(@{Name='train-40';Count=40},@{Name='validation-10';Count=10},@{Name='test-38';Count=38})) {
    $Rows=@($Old.Dataset | Where-Object Split -eq $Split.Name)
    $Cases=@($Rows.Case | Sort-Object -Unique)
    if ($Cases.Count -ne $Split.Count) { throw 'Wrong original split count.' }
    foreach ($Case in $Cases) { if ($Seen.ContainsKey($Case)) { throw 'Dataset IDs overlap.' }; $Seen[$Case]=$Split.Name }
    $First=$Rows[0]; $Offset=$First.File.IndexOf('\'+$First.Case+'\',[StringComparison]::OrdinalIgnoreCase)
    if ($Offset -lt 0) { throw 'Cannot locate original dataset.' }
    $Dataset=$First.File.Substring(0,$Offset)
    if ((@(Get-ChildItem -LiteralPath $Dataset -Directory | Sort-Object Name).Name -join ',') -ne ($Cases -join ',')) { throw 'Dataset IDs changed.' }
    if ((@(Get-ChildItem -LiteralPath $Dataset -Recurse -File | Sort-Object FullName).FullName -join '|') -ne (@($Rows.File | Sort-Object) -join '|')) { throw 'Dataset inventory changed.' }
    foreach ($Row in $Rows) { if ((Get-FileHash -LiteralPath $Row.File).Hash -ne $Row.SHA256) { throw "Dataset changed: $($Row.File)" } }
    $Datasets[$Split.Name]=$Dataset
}
foreach ($Weight in $Old.Weights) { if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot ('training-checkpoints\'+$Weight.Name))).Hash -ne $Weight.SHA256) { throw 'Pretrained/teacher weights changed.' } }
$Trainer='cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
$TrainerSHA256=(@($Old.Source | Where-Object File -eq $Trainer)[0].SHA256).ToLowerInvariant()
if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot $Trainer)).Hash -ne $TrainerSHA256) { throw 'Original trainer changed; cannot instrument this study.' }
$Files=@(& git -C $RepoRoot ls-files -- cotracker-algorithm evaluation scripts)
if ($LASTEXITCODE -ne 0) { throw 'Cannot fingerprint repository.' }
$Source=@(foreach ($File in ($Files | Sort-Object)) { if ($File -match '\.(py|ps1|sh|json|toml|lock|txt)$|Dockerfile') { [ordered]@{File=$File;SHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot $File)).Hash} } })
$Utf8=New-Object System.Text.UTF8Encoding($false)
function Write-DiagnosticJson([string]$Path,$Value,[switch]$Frozen) {
    $Json=ConvertTo-Json -InputObject $Value -Depth 100
    if ($Frozen -and (Test-Path -LiteralPath $Path)) {
        if ([IO.File]::ReadAllText($Path) -ne $Json) { throw 'Diagnostic source/settings changed; use a new results root.' }
    } else { [IO.File]::WriteAllText("$Path.tmp",$Json,$Utf8); Move-Item -LiteralPath "$Path.tmp" -Destination $Path -Force }
}
function Invoke-BudgetDocker([string[]]$Arguments,[string]$Log) {
    $Previous=$ErrorActionPreference; $ErrorActionPreference='Continue'
    try { & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object { Write-Host $_ }; $Status=$LASTEXITCODE }
    finally { $ErrorActionPreference=$Previous }
    if ($Status -notin @(0,124,137)) { throw "Diagnostic Docker failed ($Status); preserve the log. No silent fallback." }
    return $Status
}
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$Lock=[IO.File]::Open((Join-Path $ResultsRoot '.runtime-diagnostics.lock'),'OpenOrCreate','ReadWrite','None')
$Started=$false
try {
    Write-DiagnosticJson (Join-Path $ResultsRoot 'frozen-study.json') ([ordered]@{Schema=1;Source=$Source;TrainingResultsRoot=$TrainingResultsRoot;
        OldProtocolSHA256=(Get-FileHash -LiteralPath (Join-Path $TrainingResultsRoot 'frozen-run.json')).Hash;
        OldAuditSHA256=(Get-FileHash -LiteralPath (Join-Path $TrainingResultsRoot 'train\training-audit.json')).Hash;
        Images=$OldImages;CompactScan=[bool]$CompactScan;ProbeSteps=$ProbeSteps;SchedulerNumSteps=1000;ProfileTimeoutSeconds=$ProfileTimeoutSeconds;
        TrainingTimeoutSeconds=$TrainingTimeoutSeconds;TimingKinds=@('mlp','mamba');SequenceLen=60;Points=128;Iterations=4}) -Frozen
    Start-Transcript -Path (Join-Path $ResultsRoot 'queue.log') -Append | Out-Null; $Started=$true
    $Images=Initialize-LocalExperimentImages -RepoRoot $RepoRoot -ResultsRoot $ResultsRoot -TrainingRuntime $OldImages.Training -InferenceRuntime $OldImages.Inference
    if ($Images.Evaluation -ne $OldImages.Evaluation) { throw 'Evaluator image changed.' }
    Invoke-LocalDocker @('run','--rm','--pull','never','--network','none',
        '--mount',"type=bind,source=$RepoRoot,target=/source,readonly",'--mount',"type=bind,source=$TrainingResultsRoot,target=/previous,readonly",
        '--mount',"type=bind,source=$ResultsRoot,target=/study",'--entrypoint','/opt/app/.pixi/envs/training/bin/python',
        $Images.Training,'/opt/app/experiments/audit_sidecar_speed_source.py','--old','/previous','--new','/study','--repo','/source') (Join-Path $ResultsRoot 'source-audit.log')
    if ($Stage -eq 'profile') {
        $LoaderSHA256=(Get-FileHash -LiteralPath (Join-Path $RepoRoot 'cotracker-algorithm\resources\model.py')).Hash.ToLowerInvariant()
        $Tasks=@()
        foreach ($Kind in @('mlp','mamba_short','mamba')) {
            $Variants=@(@{Backend='reference';Chunk=0;Checkpoint=$false})
            if ($CompactScan) { $Variants=@(@{Backend='compact';Chunk=0;Checkpoint=$false}) }
            elseif ($Kind -ne 'mlp') { foreach ($Checkpoint in @($false,$true)) { foreach ($Chunk in @(16,32)) { $Variants+=@{Backend='parallel';Chunk=$Chunk;Checkpoint=$Checkpoint} } } }
            foreach ($Variant in $Variants) {
                $Name=$Kind+'_'+$Variant.Backend+'_'+$Variant.Chunk+'_cp'+[int]$Variant.Checkpoint
                $Tasks+=[ordered]@{name=$Name;kind=$Kind;runtime=[ordered]@{backend=$Variant.Backend;point_chunk_size=$Variant.Chunk;gain=1;max_correction_pixels=0;checkpoint_blocks=$Variant.Checkpoint}}
            }
        }
        Write-DiagnosticJson (Join-Path $ResultsRoot 'probe-manifest.json') $Tasks -Frozen
        foreach ($Task in $Tasks) {
            $Folder=Join-Path $ResultsRoot ('isolated\'+$Task.name); New-Item -ItemType Directory -Force -Path $Folder | Out-Null
            $Output=Join-Path $Folder 'result.json'
            if (Test-Path -LiteralPath $Output) {
                $Previous=Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
                if ($Previous.complete -or $Previous.status -in @('timeout','killed')) { Write-Host "SKIP recorded isolated probe: $($Task.name)"; continue }
                throw 'Partial isolated probe found; use a new results root.'
            }
            Write-Host "FRESH PROCESS $($Task.name); budget=$ProfileTimeoutSeconds seconds"
            $Arguments=@('run','--rm','--pull','never','--network','none','--gpus','device=0',
                '--mount',"type=bind,source=$TrainingResultsRoot\train,target=/training,readonly",'--mount',"type=bind,source=$Folder,target=/probe",
                '--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=10s',"${ProfileTimeoutSeconds}s",
                '/opt/app/.pixi/envs/training/bin/python','/opt/app/experiments/profile_sidecar_isolated.py',
                '--train','/training','--output','/probe/result.json','--kind',$Task.kind,'--backend',$Task.runtime.backend,
                '--chunk',([string]$Task.runtime.point_chunk_size),'--loader-sha256',$LoaderSHA256)
            if ($Task.runtime.checkpoint_blocks) { $Arguments+='--checkpoint' }
            $Status=Invoke-BudgetDocker $Arguments (Join-Path $Folder 'profile.log')
            if ($Status -in @(124,137)) {
                $Failure=if ($Status -eq 124) {'timeout'} else {'killed'}
                Write-DiagnosticJson $Output ([ordered]@{complete=$false;status=$Failure;exit_code=$Status;budget_seconds=$ProfileTimeoutSeconds;kind=$Task.kind;runtime=$Task.runtime;
                    note='137 may be timeout kill escalation or another SIGKILL/OOM; not proof of the cause.'})
                Write-Host "BUDGET FAILURE recorded: $($Task.name) ($Failure)"
            } elseif (-not (Test-Path -LiteralPath $Output)) { throw 'Worker returned success without a result.' }
        }
    } else {
        $CompactRuntime=''
        if ($CompactScan) {
            # Re-verify GPU outputs, gradients, named weights and memory before
            # any real updates. Does not rerun the dangerous reference graph.
            Invoke-LocalDocker @('run','--rm','--pull','never','--network','none','--mount',"type=bind,source=$ResultsRoot,target=/study",
                '--entrypoint','/opt/app/.pixi/envs/training/bin/python',$Images.Training,'/opt/app/experiments/verify_compact_scan_proof.py','--root','/study') (Join-Path $ResultsRoot 'compact-proof.log')
            $Ready=Get-Content -LiteralPath (Join-Path $ResultsRoot 'compact-readiness.json') -Raw | ConvertFrom-Json
            if (-not $Ready.complete -or $Ready.rows.Count -ne 3) { throw 'GPU compact proof must pass first.' }
            foreach ($Property in $Ready.result_sha256.PSObject.Properties) {
                if ((Get-FileHash -LiteralPath (Join-Path $ResultsRoot $Property.Name)).Hash -ne $Property.Value) { throw 'Compact proof changed.' }
            }
            $CompactRuntime=Join-Path $ResultsRoot 'compact-runtime.json'
            Write-DiagnosticJson $CompactRuntime $Ready.runtime -Frozen
        }
        foreach ($Kind in @('mlp','mamba')) {
            $Folder=Join-Path $ResultsRoot "training\seed_0\$Kind"; New-Item -ItemType Directory -Force -Path $Folder | Out-Null
            $Output=Join-Path $Folder 'phase-profile.json'
            if (Test-Path -LiteralPath $Output) {
                $Previous=Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
                if ($Previous.complete) { Write-Host "SKIP completed phase probe: $Kind"; continue }
                throw 'Partial training probe found. Preserve it; use a new ResultsRoot. Do not mix resumed diagnostic updates.'
            }
            Write-Host "REAL PHASE PROBE $Kind; updates=$ProbeSteps; original schedule=1000; budget=$TrainingTimeoutSeconds seconds"
            $Arguments=@('run','--rm','--pull','never','--network','none','--gpus','device=0','--shm-size','4g')
            if ($CompactRuntime) { $Arguments+=@('--mount',"type=bind,source=$CompactRuntime,target=/opt/sidecar/runtime.json,readonly") }
            $Arguments+=@(
                '--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly",
                '--mount',"type=bind,source=$RepoRoot\training-checkpoints,target=/checkpoints,readonly",'--mount',"type=bind,source=$Folder,target=/results",
                '--entrypoint','/usr/bin/timeout',$Images.Training,'--signal=TERM','--kill-after=10s',"${TrainingTimeoutSeconds}s",
                '/opt/app/.pixi/envs/training/bin/python','/opt/app/experiments/profile_sidecar_training_phases.py',
                '--kind',$Kind,'--steps',([string]$ProbeSteps),'--dataset','/data','--checkpoints','/checkpoints',
                '--output','/results','--trainer-sha256',$TrainerSHA256)
            if ($CompactRuntime) { $Arguments+=@('--runtime-config','/opt/sidecar/runtime.json') }
            $Status=Invoke-BudgetDocker $Arguments (Join-Path $Folder 'training.log')
            if ($Status -in @(124,137)) {
                Write-DiagnosticJson (Join-Path $Folder 'timeout.json') ([ordered]@{complete=$false;status='timeout';exit_code=$Status;budget_seconds=$TrainingTimeoutSeconds})
                throw "Real $Kind probe timed out; retain partial phase-profile.json. No larger training will run."
            }
            if (-not (Test-Path -LiteralPath $Output)) { throw 'No phase profile returned.' }
        }
    }
    Invoke-LocalDocker @('run','--rm','--pull','never','--network','none','--mount',"type=bind,source=$ResultsRoot,target=/study",
        '--entrypoint','/opt/app/.pixi/envs/training/bin/python',$Images.Training,'/opt/app/experiments/summarize_sidecar_runtime.py',
        '--root','/study','--stage',$Stage,'--steps',([string]$ProbeSteps)) (Join-Path $ResultsRoot 'summary.log')
    if ($CompactScan -and $Stage -eq 'profile') {
        Invoke-LocalDocker @('run','--rm','--pull','never','--network','none','--mount',"type=bind,source=$ResultsRoot,target=/study",
            '--entrypoint','/opt/app/.pixi/envs/training/bin/python',$Images.Training,'/opt/app/experiments/verify_compact_scan_proof.py','--root','/study') (Join-Path $ResultsRoot 'compact-proof.log')
    }
    Write-Host "Diagnostics complete: $ResultsRoot. No new 1000-step training was launched."
} finally { if ($Started) { Stop-Transcript | Out-Null }; $Lock.Dispose() }
