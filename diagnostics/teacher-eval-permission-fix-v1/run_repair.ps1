# Evaluation-only permission repair; original frozen package is mounted unchanged.
[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','validation','select','test')][string]$Stage='validation',
    [ValidateSet('weights','core','hierarchy','memory','state','span','context','anchor','refinement','all')][string]$Group='weights',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PackageRoot='',
    [string]$ProtocolRoot='',
    [string]$SourceStudy='',
    [string]$Weights='',
    [string]$TrainDataset='C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40',
    [string]$ValidationDataset='C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10',
    [string]$TestDataset='C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38',
    [string]$OutputRoot='C:\zhongliu\teacher-single-factor-v1-results',
    [string]$InferenceImage='',
    [string]$EvaluationImage='',
    [ValidateRange(300,14400)][int]$JobTimeoutSeconds=7200
)
$ErrorActionPreference='Stop'
if (-not $PackageRoot) {$PackageRoot=Join-Path $RepoRoot 'diagnostics\teacher-single-factor-v1'}
$PackageRoot=(Resolve-Path -LiteralPath $PackageRoot).Path.TrimEnd('\')
. (Join-Path $PackageRoot 'native_process.ps1')
if (-not $ProtocolRoot) {$ProtocolRoot=Join-Path $RepoRoot 'protocol-40-10-38\feature-sidecar-modefix-v3'}
if (-not $SourceStudy) {$SourceStudy=Join-Path $RepoRoot 'protocol-40-10-38\sidecar-stability-v5\degradation'}
if (-not $Weights) {$Weights=Join-Path $RepoRoot 'training-checkpoints'}
$ProtocolRoot=(Resolve-Path -LiteralPath $ProtocolRoot).Path.TrimEnd('\')
$SourceStudy=(Resolve-Path -LiteralPath $SourceStudy).Path.TrimEnd('\')
$Weights=(Resolve-Path -LiteralPath $Weights).Path.TrimEnd('\')
$TrainDataset=(Resolve-Path -LiteralPath $TrainDataset).Path.TrimEnd('\')
$ValidationDataset=(Resolve-Path -LiteralPath $ValidationDataset).Path.TrimEnd('\')
$TestDataset=(Resolve-Path -LiteralPath $TestDataset).Path.TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -gt 100 -or [IO.Path]::GetPathRoot($OutputRoot) -eq $OutputRoot+'\') {
    throw 'Use a short dedicated OutputRoot (at most 100 characters), not a drive root.'
}
foreach ($InputRoot in @($ProtocolRoot,$SourceStudy,$Weights,$TrainDataset,$ValidationDataset,$TestDataset,$PackageRoot,$PSScriptRoot)) {
    if ($OutputRoot -eq $InputRoot -or $OutputRoot.StartsWith($InputRoot+'\',[StringComparison]::OrdinalIgnoreCase) -or $InputRoot.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must not overlap any input or script directory.'
    }
}
foreach ($Path in @($ProtocolRoot,$SourceStudy,$Weights,$TrainDataset,$ValidationDataset,$TestDataset,$OutputRoot,$PackageRoot,$PSScriptRoot)) {
    if ($Path -match '[",\r\n]') {throw 'Paths containing quotes, commas or newlines are unsupported.'}
}
$Images=Get-Content -LiteralPath (Join-Path $SourceStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
if (-not $InferenceImage) {$InferenceImage=[string]$Images.Inference}
if (-not $EvaluationImage) {$EvaluationImage=[string]$Images.Evaluation}
foreach ($Image in @($InferenceImage,$EvaluationImage)) {
    if ($Image -notmatch '^sha256:[0-9a-f]{64}$') {throw 'Immutable installed image IDs are required. No build/pull is performed.'}
}
$Prior=$ErrorActionPreference;$ErrorActionPreference='Continue'
try {
    & docker info *> $null
    $EngineStatus=$LASTEXITCODE
    if ($EngineStatus -eq 0) {
        foreach ($Image in @($InferenceImage,$EvaluationImage)) {
            $Found=[string](& docker image inspect $Image --format '{{.Id}}' 2>$null)
            if ($LASTEXITCODE -ne 0 -or $Found -ne $Image) {throw "Pinned image missing: $Image. No build/pull attempted."}
        }
    }
} finally {$ErrorActionPreference=$Prior}
if ($EngineStatus -ne 0) {throw 'Start Docker Desktop first; no image build/pull was attempted.'}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
function Invoke-TeacherContainer([string[]]$Arguments,[string]$Label,[string]$Container,[switch]$ReturnLog) {
    $LogBase=$OutputRoot+'.launch-'+$Label+'-'+[Guid]::NewGuid().ToString('N').Substring(0,8)
    Write-Host "START $Label (timeout $JobTimeoutSeconds seconds)"
    $Process=Start-TeacherProcess -Arguments $Arguments -LogBase $LogBase
    $Watch=[Diagnostics.Stopwatch]::StartNew()
    while (-not $Process.WaitForExit(10000)) {
        if ($Watch.Elapsed.TotalSeconds -gt $JobTimeoutSeconds) {
            $Prior=$ErrorActionPreference;$ErrorActionPreference='Continue'
            try {& docker stop --time 5 $Container 2>&1 | ForEach-Object {Write-Host $_}} finally {$ErrorActionPreference=$Prior}
            throw "Timed out: $Label. Completed cases/partial files preserved. Logs: $LogBase"
        }
        if ([int]$Watch.Elapsed.TotalSeconds % 60 -lt 11) {
            Write-Host "RUNNING $Label elapsed=$([int]$Watch.Elapsed.TotalSeconds)s; log=$LogBase.stdout.log"
            Get-Content -LiteralPath ($LogBase+'.stdout.log') -Tail 2 -ErrorAction SilentlyContinue | Where-Object {$_ -notlike 'TEACHER_METRICS_JSON *'} | ForEach-Object {Write-Host $_}
        }
    }
    $Status=Get-TeacherExitCode $Process
    Get-Content -LiteralPath ($LogBase+'.stdout.log') -Tail 40 -ErrorAction SilentlyContinue | Where-Object {$_ -notlike 'TEACHER_METRICS_JSON *'} | ForEach-Object {Write-Host $_}
    if ($Status -ne 0) {
        Get-Content -LiteralPath ($LogBase+'.stderr.log') -Tail 60 -ErrorAction SilentlyContinue | ForEach-Object {Write-Host $_}
        throw "Teacher experiment failed (exit $Status): $Label. Preserve results; repeat unchanged command. Logs: $LogBase"
    }
    if ($ReturnLog) {return $LogBase}
}
function Invoke-TeacherWorker([string]$Action,[string]$Split='validation-10',[string]$Profile='old') {
    $Container='teacher-ofat-'+[Guid]::NewGuid().ToString('N').Substring(0,12)
    $Arguments=@('run','--rm','--pull','never','--network','none','--platform','linux/amd64','--name',$Container,
        '--cpus','4','--shm-size','2g','--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
        '--env','OPENBLAS_NUM_THREADS=1','--env','PYTHONUNBUFFERED=1','--env','PYTHONDONTWRITEBYTECODE=1',
        '--env','COTRACKER_SOURCE_DIR=/opt/app/ext/co-tracker',
        '--mount',"type=bind,source=$ProtocolRoot,target=/protocol,readonly",
        '--mount',"type=bind,source=$SourceStudy,target=/study,readonly",
        '--mount',"type=bind,source=$Weights,target=/weights,readonly",
        '--mount',"type=bind,source=$TrainDataset,target=/datasets/train,readonly",
        '--mount',"type=bind,source=$ValidationDataset,target=/datasets/validation,readonly",
        '--mount',"type=bind,source=$TestDataset,target=/datasets/test,readonly",
        '--mount',"type=bind,source=$PackageRoot,target=/teacher-source,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/results")
    if ($Action -in @('infer','smoke')) {$Arguments+=@('--gpus','device=0')}
    $Arguments+=@('--entrypoint','/usr/bin/timeout',$InferenceImage,'--signal=TERM','--kill-after=10s',
        ([string]$JobTimeoutSeconds+'s'),'/opt/app/.pixi/envs/cuda/bin/python','/teacher-source/worker.py',
        '--action',$Action,'--split',$Split,'--profile',$Profile,'--group',$Group,
        '--protocol','/protocol','--study','/study','--weights','/weights','--output','/results',
        '--train','/datasets/train','--validation','/datasets/validation','--test','/datasets/test',
        '--inference-image',$InferenceImage,'--evaluation-image',$EvaluationImage)
    Invoke-TeacherContainer $Arguments ($Action+'-'+$Profile) $Container
}
function Save-EvaluatorMessage([string]$LogBase,[string]$Tag,[string]$Path) {
    $Lines=@(Get-Content -LiteralPath ($LogBase+'.stdout.log') | Where-Object {$_.StartsWith($Tag,[StringComparison]::Ordinal)})
    if ($Lines.Count -ne 1) {throw 'Expected exactly one complete evaluator data message; retain launch logs.'}
    $Json=$Lines[0].Substring($Tag.Length)
    $null=$Json | ConvertFrom-Json
    $Parent=Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
    [IO.File]::WriteAllText($Path,$Json,(New-Object System.Text.UTF8Encoding($false)))
}
function Invoke-EvaluatorRuntimeAudit([string]$RuntimePath) {
    $Container='teacher-eval-runtime-'+[Guid]::NewGuid().ToString('N').Substring(0,12)
    $Arguments=@('run','--rm','--pull','never','--network','none','--platform','linux/amd64','--name',$Container,
        '--cap-drop','ALL','--security-opt','no-new-privileges','--read-only',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=256m,mode=1777','--cpus','2','--env','PYTHONDONTWRITEBYTECODE=1',
        '--mount',"type=bind,source=$PSScriptRoot,target=/repair-source,readonly",
        '--entrypoint','/usr/bin/timeout',$EvaluationImage,'--signal=TERM','--kill-after=10s',
        ([string]$JobTimeoutSeconds+'s'),'python','/repair-source/repair.py','--action','runtime')
    $AuditLog=Invoke-TeacherContainer $Arguments 'evaluator-runtime' $Container -ReturnLog
    Save-EvaluatorMessage $AuditLog 'TEACHER_EVALUATOR_RUNTIME ' $RuntimePath
}
function Invoke-EvaluatorPermissionPreflight([string]$Split,[string]$Profile,[string]$RuntimePath) {
    $Container='teacher-eval-preflight-'+[Guid]::NewGuid().ToString('N').Substring(0,12)
    $Arguments=@('run','--rm','--pull','never','--network','none','--platform','linux/amd64','--name',$Container,
        '--cpus','2','--env','PYTHONUNBUFFERED=1','--env','PYTHONDONTWRITEBYTECODE=1',
        '--mount',"type=bind,source=$PackageRoot,target=/teacher-source,readonly",
        '--mount',"type=bind,source=$PSScriptRoot,target=/repair-source,readonly",
        '--mount',"type=bind,source=$SourceStudy,target=/study,readonly",
        '--mount',"type=bind,source=$RuntimePath,target=/evaluator-runtime.json,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/results",
        '--entrypoint','/usr/bin/timeout',$InferenceImage,'--signal=TERM','--kill-after=10s',
        ([string]$JobTimeoutSeconds+'s'),'/opt/app/.pixi/envs/cuda/bin/python','/repair-source/repair.py',
        '--action','prepare','--split',$Split,'--profile',$Profile)
    Invoke-TeacherContainer $Arguments ('permission-'+$Profile) $Container
}
function Invoke-TeacherEvaluation([string]$Split,[string]$Profile) {
    $ProfileRoot=Join-Path $OutputRoot ($Split+'\'+$Profile)
    $Ready=Get-Content -LiteralPath (Join-Path $ProfileRoot 'evaluation-ready.json') -Raw | ConvertFrom-Json
    if (-not $Ready.needs_evaluation) {Write-Host "METRICS VERIFIED CACHE $Split/$Profile";return}
    $Transport=Join-Path $OutputRoot ('permission-repair\transport\'+[Guid]::NewGuid().ToString('N').Substring(0,8))
    $RuntimePath=Join-Path $Transport 'runtime.json'
    Invoke-EvaluatorRuntimeAudit $RuntimePath
    Invoke-EvaluatorPermissionPreflight $Split $Profile $RuntimePath
    $Pointer=Get-Content -LiteralPath (Join-Path $OutputRoot ('permission-repair\'+$Split+'\'+$Profile+'\latest.json')) -Raw | ConvertFrom-Json
    $ReceiptPath=[IO.Path]::GetFullPath((Join-Path $OutputRoot ([string]$Pointer.receipt).Replace('/','\')))
    if (-not $ReceiptPath.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Unsafe repair receipt path'}
    if ((Get-FileHash -LiteralPath $ReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$Pointer.sha256) {throw 'Repair receipt changed'}
    $Receipt=Get-Content -LiteralPath $ReceiptPath -Raw | ConvertFrom-Json
    if ($Receipt.manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $OutputRoot 'manifest.json')).Hash.ToLowerInvariant()) {throw 'Original manifest changed after preflight'}
    if ([string]$Receipt.owner_uid -notmatch '^\d+$' -or [string]$Receipt.owner_gid -notmatch '^\d+$') {throw 'Unsafe evaluator identity'}
    $WriterIdentity=([string]$Receipt.owner_uid)+':'+([string]$Receipt.owner_gid)
    $Runtime=$Receipt.original_evaluator_runtime
    if ([string]$Runtime.identity.uid -notmatch '^\d+$' -or [string]$Runtime.identity.gid -notmatch '^\d+$') {throw 'Original evaluator identity unavailable'}
    $ScratchOptions='/output:rw,nosuid,nodev,size=128m,mode=0700,uid='+([string]$Runtime.identity.uid)+',gid='+([string]$Runtime.identity.gid)
    $Dataset=if ($Split -eq 'test-38') {$TestDataset} else {$ValidationDataset}
    $Container='teacher-eval-'+[Guid]::NewGuid().ToString('N').Substring(0,12)
    $Arguments=@('run','--rm','--pull','never','--network','none','--platform','linux/amd64','--name',$Container,
        '--cap-drop','ALL','--security-opt','no-new-privileges','--read-only',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=256m,mode=1777','--cpus','4',
        '--tmpfs',$ScratchOptions,'--env','GRAND_CHALLENGE_MAX_WORKERS=4','--env','PYTHONDONTWRITEBYTECODE=1',
        '--mount',"type=bind,source=$ProfileRoot,target=/input,readonly",
        '--mount',"type=bind,source=$Dataset,target=/opt/app/ground_truth,readonly",
        '--mount',"type=bind,source=$PSScriptRoot,target=/repair-source,readonly",
        '--mount',"type=bind,source=$ReceiptPath,target=/repair-receipt.json,readonly",
        '--mount',"type=bind,source=$OutputRoot\manifest.json,target=/original-manifest.json,readonly",
        '--entrypoint','/usr/bin/timeout',$EvaluationImage,'--signal=TERM','--kill-after=10s',
        ([string]$JobTimeoutSeconds+'s'),'python','/repair-source/repair.py','--action','evaluate')
    Write-Host "EVALUATOR ORIGINAL IDENTITY $($Runtime.identity.uid):$($Runtime.identity.gid); metrics publisher=$WriterIdentity"
    $EvaluationLog=Invoke-TeacherContainer $Arguments ('evaluate-'+$Profile) $Container -ReturnLog
    $ResponsePath=Join-Path $Transport 'metrics-response.json'
    Save-EvaluatorMessage $EvaluationLog 'TEACHER_METRICS_JSON ' $ResponsePath
    $Publisher='teacher-metrics-publish-'+[Guid]::NewGuid().ToString('N').Substring(0,12)
    $PublishArguments=@('run','--rm','--pull','never','--network','none','--platform','linux/amd64','--name',$Publisher,
        '--user',$WriterIdentity,'--cap-drop','ALL','--security-opt','no-new-privileges','--read-only',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=64m,mode=1777','--cpus','2','--env','PYTHONDONTWRITEBYTECODE=1',
        '--mount',"type=bind,source=$PSScriptRoot,target=/repair-source,readonly",
        '--mount',"type=bind,source=$ReceiptPath,target=/repair-receipt.json,readonly",
        '--mount',"type=bind,source=$ResponsePath,target=/metrics-response.json,readonly",
        '--mount',"type=bind,source=$OutputRoot,target=/results",
        '--entrypoint','/usr/bin/timeout',$InferenceImage,'--signal=TERM','--kill-after=10s',
        ([string]$JobTimeoutSeconds+'s'),'/opt/app/.pixi/envs/cuda/bin/python','/repair-source/repair.py','--action','publish')
    Invoke-TeacherContainer $PublishArguments ('publish-'+$Profile) $Publisher
    Invoke-TeacherWorker 'record' $Split $Profile
}
try {
    Write-Host "Teacher evaluation permission repair stage=$Stage group=$Group; original package remains frozen. No student training, Mamba, build, pull or network."
    Invoke-TeacherWorker 'plan'
    if ($Stage -eq 'plan') {
        (Get-Content -LiteralPath (Join-Path $OutputRoot 'plan.json') -Raw | ConvertFrom-Json).profiles |
            Select-Object Profile,Group,Parent,Factor,Change | Format-Table -AutoSize
    } elseif ($Stage -eq 'smoke') {
        Invoke-TeacherWorker 'smoke'
    } elseif ($Stage -eq 'select') {
        Invoke-TeacherWorker 'select'
    } else {
        if ($Stage -eq 'validation' -and (Test-Path -LiteralPath (Join-Path $OutputRoot 'selection.json'))) {
            throw 'Selection is already frozen. Do not add validation arms after test selection; use a new study.'
        }
        $Split=if ($Stage -eq 'test') {'test-38'} else {'validation-10'}
        if ($Stage -eq 'test') {
            $Names=@((Get-Content -LiteralPath (Join-Path $OutputRoot 'selection.json') -Raw | ConvertFrom-Json).test_profiles)
        } else {
            $Plan=(Get-Content -LiteralPath (Join-Path $OutputRoot 'plan.json') -Raw | ConvertFrom-Json).profiles
            $Needed=@{}
            foreach ($Row in $Plan) {
                if ($Group -eq 'all' -or $Row.Group -eq $Group) {
                    $Name=$Row.Profile
                    while ($Name) {
                        $Needed[$Name]=$true
                        $Parent=@($Plan | Where-Object Profile -EQ $Name)[0].Parent
                        $Name=$Parent
                    }
                }
            }
            $Names=@($Plan | Where-Object {$Needed.ContainsKey($_.Profile)} | ForEach-Object {$_.Profile})
        }
        foreach ($Name in $Names) {
            Invoke-TeacherWorker 'infer' $Split $Name
            Invoke-TeacherEvaluation $Split $Name
        }
        Invoke-TeacherWorker 'report' $Split
    }
    Write-Host "COMPLETED stage=$Stage group=$Group : $OutputRoot"
} finally {$Lock.Dispose()}
