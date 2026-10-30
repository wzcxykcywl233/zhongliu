[CmdletBinding()]
param(
    [ValidateSet('network','download','audit','all')][string]$Stage='audit',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PreviousRoot='C:\zhongliu\rs-v1',
    [string]$DataRoot='C:\zhongliu\lld-mmri-v1',
    [string]$OutputRoot='C:\zhongliu\lld-qc-v1',
    [string]$RuntimeImage='',
    [ValidateSet('https://huggingface.co','https://hf-mirror.com')][string]$DownloadEndpoint='https://huggingface.co',
    [ValidateSet('inherit','direct','explicit')][string]$ProxyMode='inherit',
    [string]$ProxyUrl='',
    [switch]$AcceptResearchTerms
)
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath $RepoRoot).Path.TrimEnd('\')
$DataRoot=[IO.Path]::GetFullPath($DataRoot).TrimEnd('\')
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
$PackageRoot=$PSScriptRoot
$OriginalRoot=(Resolve-Path -LiteralPath (Join-Path $RepoRoot 'diagnostics\lld-mixed-domain-v1')).Path
if (($ProxyMode -eq 'explicit') -ne [bool]$ProxyUrl) { throw 'Explicit mode requires ProxyUrl; other modes must omit it.' }
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
    throw 'LLD-MMRI is non-commercial research only; no external redistribution; same-organization single-site internal use only. Read README.txt; then pass -AcceptResearchTerms if you agree.'
}
if (-not $RuntimeImage) {
    $Images=Get-Content -LiteralPath (Join-Path $PreviousRoot 'manifest.json') -Raw | ConvertFrom-Json
    $RuntimeImage=$Images.images.Training
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
        $Stages=if ($Stage -eq 'all') {@('download','audit')} else {@($Stage)}
        foreach ($Part in $Stages) {
            Write-Host "LLD transport repair $Part; no model training, no TrackRAD/test access, no image build/pull."
            $DockerArgs=@('run','--rm','--pull','never','--cpus','4','--env','PYTHONDONTWRITEBYTECODE=1',
                '--env','OMP_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
                '--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
                '--mount',"type=bind,source=$OriginalRoot,target=/original,readonly",
                '--mount',"type=bind,source=$OutputRoot,target=/output")
            $Budget=if ($Part -eq 'download') {'43200s'} elseif ($Part -eq 'network') {'600s'} else {'14400s'}
            if ($Part -eq 'download') {
                $DockerArgs+=@('--mount',"type=bind,source=$DataRoot,target=/data")
            } elseif ($Part -eq 'audit') {
                $DockerArgs+=@('--network','none','--mount',"type=bind,source=$DataRoot,target=/data,readonly")
            }
            if ($Part -eq 'audit') {
                $Worker=@('/original/dataset.py','audit','--data','/data','--output','/output')
            } else {
                $Worker=@('/package/download.py',$Part,'--data','/data','--output','/output','--endpoint',$DownloadEndpoint,'--proxy-mode',$ProxyMode)
                if ($ProxyUrl) { $Worker+=@('--proxy-url',$ProxyUrl) }
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
            if ($Status -ne 0) { throw "LLD-MMRI $Part failed ($Status); keep data/results and inspect $Log. No training starts." }
        }
        if ($Stage -eq 'network') {
            Write-Host "NETWORK CHECK COMPLETE: $OutputRoot. This is NOT a data audit or GPU training smoke."
        } else {
            Write-Host "PREPARATION COMPLETE: $OutputRoot. This is NOT a GPU training smoke. No training queue started."
        }
    } finally {
        if ($OutputLock) { $OutputLock.Dispose() }
    }
} finally { $DataLock.Dispose() }
