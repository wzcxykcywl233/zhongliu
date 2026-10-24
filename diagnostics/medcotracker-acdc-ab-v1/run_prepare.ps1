[CmdletBinding()]
param(
    [ValidateSet('download','prepare','check-model','offline','all')][string]$Stage='prepare',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$OutputRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$RuntimeImage='',
    [switch]$AcceptResearchTerms
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$DataRoot=[IO.Path]::GetFullPath($DataRoot).TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$PackageRoot=$PSScriptRoot
foreach ($Path in @($RepoRoot,$DataRoot,$OutputRoot,$PackageRoot)) {
    if ($Path -match '[",\r\n]' -or $Path.Length -le 3) { throw 'Use normal dedicated absolute directories.' }
}
foreach ($Pair in @(@($DataRoot,$OutputRoot),@($DataRoot,$RepoRoot),@($OutputRoot,$RepoRoot))) {
    if ($Pair[0] -eq $Pair[1] -or $Pair[0].StartsWith($Pair[1]+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $Pair[1].StartsWith($Pair[0]+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'DataRoot, OutputRoot and RepoRoot must be separate non-nested directories.'
    }
}
if (($Stage -eq 'download' -or $Stage -eq 'all') -and -not $AcceptResearchTerms) {
    throw 'ACDC is strictly non-commercial scientific research, CC BY-NC-SA 4.0. Read README.txt; then pass -AcceptResearchTerms if you agree.'
}
if (-not $RuntimeImage) {
    $Images=Get-Content -LiteralPath (Join-Path $PriorStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
    $RuntimeImage=$Images.Training
}
if ($RuntimeImage -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Provide an existing pinned training image ID, not a mutable tag.' }
$OldPreference=$ErrorActionPreference
$ErrorActionPreference='Continue'
try {
    & docker info --format '{{.OSType}}' 2>&1 | Out-Null
    $DockerStatus=$LASTEXITCODE
    $Actual=& docker image inspect $RuntimeImage --format '{{.Id}}' 2>&1
    $InspectStatus=$LASTEXITCODE
} finally { $ErrorActionPreference=$OldPreference }
if ($DockerStatus -ne 0) { throw 'Start Docker Desktop Linux engine first. No build or pull attempted.' }
if ($InspectStatus -ne 0 -or [string]$Actual -ne $RuntimeImage) { throw 'Pinned local training image unavailable. No build/pull attempted.' }
New-Item -ItemType Directory -Force -Path $DataRoot,$OutputRoot | Out-Null
$DataLock=[IO.File]::Open($DataRoot+'.lock','OpenOrCreate','ReadWrite','None')
try {
    $OutputLock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
    try {
        $Stages=if ($Stage -eq 'all') {@('download','prepare','check-model')}
                elseif ($Stage -eq 'offline') {@('prepare','check-model')} else {@($Stage)}
        foreach ($Part in $Stages) {
            Write-Host "ACDC $Part; no model training, no TrackRAD/test access, no image build/pull."
            $DockerArgs=@('run','--rm','--pull','never','--cpus','4','--env','PYTHONDONTWRITEBYTECODE=1',
                '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
                '--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
                '--mount',"type=bind,source=$OutputRoot,target=/output")
            $Budget=if ($Part -eq 'download') {'14400s'} else {'3600s'}
            if ($Part -eq 'download') {
                $DockerArgs+=@('--mount',"type=bind,source=$DataRoot,target=/data")
            } else {
                $DockerArgs+=@('--network','none','--mount',"type=bind,source=$DataRoot,target=/data,readonly")
            }
            if ($Part -eq 'check-model') {
                $DockerArgs+=@('--mount',"type=bind,source=$RepoRoot,target=/checkout,readonly")
                $Worker=@('/package/ab_modes.py','--output','/output')
            } else {
                $Worker=@('/package/acdc_data.py',$Part,'--root','/data','--output','/output')
                if ($Part -eq 'download') { $Worker+='--accept-research-terms' }
            }
            $DockerArgs+=@('--entrypoint','/usr/bin/timeout',$RuntimeImage,'--signal=TERM','--kill-after=30s',$Budget,
                     '/opt/app/.pixi/envs/training/bin/python')+$Worker
            $Log=Join-Path $OutputRoot ($Part+'.log')
            $OldPreference=$ErrorActionPreference
            $ErrorActionPreference='Continue'
            try {
                & docker @DockerArgs 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object { Write-Host ([string]$_) }
                $Status=$LASTEXITCODE
            } finally { $ErrorActionPreference=$OldPreference }
            if ($Status -ne 0) { throw "ACDC $Part failed ($Status); keep data/results and inspect $Log. No training starts." }
        }
        Write-Host "PREPARATION COMPLETE: $OutputRoot. This is NOT a GPU training smoke. No training queue started."
    } finally {
        if ($OutputLock) { $OutputLock.Dispose() }
    }
} finally { $DataLock.Dispose() }
