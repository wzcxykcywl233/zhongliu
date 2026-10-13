[CmdletBinding()]
param(
    [ValidateSet('cached','probe','all')][string]$Stage='cached',
    [string]$StudyRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\sidecar-stability-v5',
    [string]$TrainingRoot='C:\zhongliu\zhongliu-tuning\protocol-40-10-38\feature-sidecar-modefix-v3',
    [string]$Dataset='C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$Weights='C:\zhongliu\zhongliu-tuning\training-checkpoints',
    [string]$OutputRoot='C:\zhongliu\sidecar-mechanism-v3-results',
    [string]$DiagnosticImage='',
    [string]$TrainingDiagnosticImage='',
    [ValidateRange(60,900)][int]$JobTimeoutSeconds=600,
    [ValidateRange(300,7200)][int]$TotalTimeoutSeconds=4500
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native_process.ps1')
$StudyRoot=(Resolve-Path -LiteralPath $StudyRoot).Path.TrimEnd('\')
$TrainingRoot=(Resolve-Path -LiteralPath $TrainingRoot).Path.TrimEnd('\')
$Dataset=(Resolve-Path -LiteralPath $Dataset).Path.TrimEnd('\')
$Weights=(Resolve-Path -LiteralPath $Weights).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -gt 150 -or [IO.Path]::GetPathRoot($OutputRoot) -eq $OutputRoot+'\') {throw 'Use a short dedicated output directory, not a drive root.'}
foreach ($InputRoot in @($StudyRoot,$TrainingRoot,$Dataset,$Weights,$PSScriptRoot)) {
    if ($OutputRoot -eq $InputRoot -or $OutputRoot.StartsWith($InputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $InputRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must not overlap any input or script directory.'
    }
}
foreach ($Path in @($StudyRoot,$TrainingRoot,$Dataset,$Weights,$OutputRoot,$PSScriptRoot)) {
    if ($Path -match '[",\r\n]') {throw 'Paths containing quotes, commas or newlines are unsupported.'}
}
$InferenceImages=Get-Content -LiteralPath (Join-Path $StudyRoot 'degradation\local-runtime-images.json') -Raw | ConvertFrom-Json
$TrainingImages=Get-Content -LiteralPath (Join-Path $TrainingRoot 'local-runtime-images.json') -Raw | ConvertFrom-Json
$InferenceImage=if ($DiagnosticImage) {$DiagnosticImage} else {[string]$InferenceImages.Inference}
$TrainingImage=if ($TrainingDiagnosticImage) {$TrainingDiagnosticImage} else {[string]$TrainingImages.Training}
foreach ($Image in @($InferenceImage,$TrainingImage)) {
    if ($Image -notmatch '^sha256:[0-9a-f]{64}$') {throw 'Installed immutable sha256 image IDs are required.'}
}
$PriorPreference=$ErrorActionPreference
$ErrorActionPreference='Continue'
try {
    & docker info *> $null
    $EngineStatus=$LASTEXITCODE
    if ($EngineStatus -eq 0) {
        foreach ($Image in @($InferenceImage,$TrainingImage)) {
            $Found=[string](& docker image inspect $Image --format '{{.Id}}' 2>$null)
            if ($LASTEXITCODE -ne 0 -or $Found -ne $Image) {throw "Pinned image unavailable: $Image. No pull/build was attempted."}
        }
    }
} finally {$ErrorActionPreference=$PriorPreference}
if ($EngineStatus -ne 0) {throw 'Start Docker Desktop first; no image build/pull was attempted.'}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
$Watch=[Diagnostics.Stopwatch]::StartNew()
function Invoke-ProbeWorker([string]$WorkerStage,[string]$Job,[string]$Surface='',[string]$Case='',[string]$Route='') {
    $GPU=$WorkerStage -in @('probe','smoke')
    if ($GPU -and -not (Test-Path -LiteralPath (Join-Path $OutputRoot 'smoke\verified.json')) -and $WorkerStage -ne 'smoke') {
        throw 'GPU smoke must pass before larger probes.'
    }
    $Remaining=$TotalTimeoutSeconds-$Watch.Elapsed.TotalSeconds
    if ($Remaining -lt 30) {throw 'Total diagnostic time limit reached. Preserve completed jobs; rerun unchanged inputs to resume.'}
    $Timeout=[Math]::Min($JobTimeoutSeconds,$Remaining)
    $Image=if ($Surface -eq 'training') {$TrainingImage} else {$InferenceImage}
    $Python=if ($Surface -eq 'training') {'/opt/app/.pixi/envs/training/bin/python'} else {'/opt/app/.pixi/envs/cuda/bin/python'}
    $Container='sidecar-probe-'+[Guid]::NewGuid().ToString('N').Substring(0,12)
    # Unique sibling launch logs do not interfere with initial manifest creation.
    $LogBase=$OutputRoot+'.launch-'+$WorkerStage+'-'+[Guid]::NewGuid().ToString('N').Substring(0,8)
    $Output='/results'
    if ($Job) {$Output='/results/'+$Job.Replace('\','/')}
    $Arguments=@('run','--rm','--pull','never','--network','none','--platform','linux/amd64',
        '--name',$Container,'--cpus','4','--shm-size','2g',
        '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','COTRACKER_SOURCE_DIR=/opt/app/ext/co-tracker',
        '--mount',"type=bind,source=$StudyRoot,target=/study,readonly",
        '--mount',"type=bind,source=$TrainingRoot,target=/training,readonly",
        '--mount',"type=bind,source=$Dataset,target=/dataset,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$PSScriptRoot,target=/probe-source,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/results")
    if ($GPU) {$Arguments += @('--gpus','device=0')}
    # The in-container timeout also holds if this PowerShell launcher exits.
    $Arguments += @('--entrypoint','/usr/bin/timeout',$Image,'--signal=TERM','--kill-after=10s',
        ([string]([int][Math]::Ceiling($Timeout))+'s'),$Python,'/probe-source/mechanism_worker.py',
        '--stage',$WorkerStage,'--training','/training','--study','/study/degradation',
        '--dataset','/dataset','--weights','/weights','--output',$Output,'--root','/results')
    if ($WorkerStage -eq 'plan') {$Arguments+=@('--inference-image',$InferenceImage,'--training-image',$TrainingImage)}
    if ($Surface) {$Arguments+=@('--surface',$Surface)}
    if ($Case) {$Arguments+=@('--case',$Case,'--route',$Route)}
    $Quoted=@($Arguments | ForEach-Object {'"'+$_+'"'})
    Write-Host "START $WorkerStage $Job $Surface (timeout $([int]$Timeout)s)"
    $Process=Start-DiagnosticProcess -FilePath 'docker.exe' -ArgumentList $Quoted `
        -Stdout ($LogBase+'.stdout.log') -Stderr ($LogBase+'.stderr.log')
    $JobWatch=[Diagnostics.Stopwatch]::StartNew()
    while (-not $Process.WaitForExit(10000)) {
        if ($JobWatch.Elapsed.TotalSeconds -gt $Timeout) {
            # Stop ONLY the exact uniquely named container created by this invocation.
            $Prior=$ErrorActionPreference;$ErrorActionPreference='Continue'
            try {& docker stop --time 5 $Container 2>&1 | ForEach-Object {Write-Host $_}} finally {$ErrorActionPreference=$Prior}
            throw "Diagnostic $WorkerStage $Job timed out. Partial files retained. No training or larger probe will start. Logs: $LogBase"
        }
        if ([int]$JobWatch.Elapsed.TotalSeconds % 60 -lt 11) {Write-Host "RUNNING $WorkerStage $Job elapsed=$([int]$JobWatch.Elapsed.TotalSeconds)s"}
    }
    try {$Status=Get-DiagnosticExitCode $Process}
    catch {
        Get-Content -LiteralPath ($LogBase+'.stderr.log') -Tail 60 -ErrorAction SilentlyContinue | ForEach-Object {Write-Host $_}
        throw "Could not read child exit status. Preserve inputs/results. Logs: $LogBase. $($_.Exception.Message)"
    }
    Get-Content -LiteralPath ($LogBase+'.stdout.log') -ErrorAction SilentlyContinue | ForEach-Object {Write-Host $_}
    if ($Status -ne 0) {
        Get-Content -LiteralPath ($LogBase+'.stderr.log') -Tail 60 -ErrorAction SilentlyContinue | ForEach-Object {Write-Host $_}
        throw "Diagnostic failed (exit code $Status); preserve inputs/results. Logs: $LogBase"
    }
    Write-Host "DONE $WorkerStage $Job $Surface"
}
try {
    Write-Host 'Validation-10 diagnosis only. No training, weight update, test-38 run, image build or network.'
    Invoke-ProbeWorker 'plan' ''
    Invoke-ProbeWorker 'inspect' '' 'training'
    Invoke-ProbeWorker 'inspect' '' 'inference'
    if ($Stage -in @('cached','all')) {Invoke-ProbeWorker 'cached' 'cached'}
    if ($Stage -in @('probe','all')) {
        Invoke-ProbeWorker 'smoke' 'smoke' '' 'B_006' 'deployment_raw'
        foreach ($Case in @('B_006','A_028','A_012','A_010')) {
            foreach ($Route in @('deployment_raw','deployment_normalized','training_exact')) {
                Invoke-ProbeWorker 'probe' ("probes\$Case-$Route") '' $Case $Route
            }
        }
    }
    Invoke-ProbeWorker 'summary' ''
    Write-Host "Completed requested stage=$Stage : $OutputRoot\summary.json"
} finally {$Lock.Dispose()}
