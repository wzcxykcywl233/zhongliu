[CmdletBinding()]
param(
    [ValidateSet('smoke','raw','labels','all')][string]$Stage='all',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$AuditRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$StudyRoot='C:\zhongliu\mct-full-v1',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$OutputRoot='C:\zhongliu\acdc-qc-v1',
    [string]$RuntimeImage='',
    [ValidateRange(0,12)][int]$ReplaysPerSeed=12,
    [ValidateRange(60,43200)][int]$TimeoutSeconds=10800
)
$ErrorActionPreference='Stop'
$PackageRoot=(Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd('\')
$DataRoot=(Resolve-Path -LiteralPath $DataRoot).Path.TrimEnd('\')
$AuditRoot=(Resolve-Path -LiteralPath $AuditRoot).Path.TrimEnd('\')
if (-not [IO.Path]::IsPathRooted($OutputRoot)) { throw 'OutputRoot must be absolute.' }
$OutputRoot=[IO.Path]::GetFullPath($OutputRoot).TrimEnd('\')
if ($OutputRoot.Length -le 3 -or $OutputRoot.Length -gt 70) { throw 'Use a short dedicated OutputRoot, e.g. C:\zhongliu\acdc-qc-v1.' }
$NeedLabels=$Stage -eq 'labels' -or $Stage -eq 'all'
$Inputs=@($DataRoot,$AuditRoot,$PackageRoot)
if ($NeedLabels) {
    $StudyRoot=(Resolve-Path -LiteralPath $StudyRoot).Path.TrimEnd('\')
    $Inputs+=$StudyRoot
}
foreach ($Path in $Inputs) {
    if ($Path -match '[",\r\n]' -or $Path -eq $OutputRoot -or
        $OutputRoot.StartsWith($Path+'\',[StringComparison]::OrdinalIgnoreCase) -or
        $Path.StartsWith($OutputRoot+'\',[StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputRoot must be separate from every input, with ordinary path characters.'
    }
}
if ($OutputRoot -match '[",\r\n]') { throw 'Invalid output path.' }
function Assert-PhysicalPath([string]$Path) {
    $Probe=$Path
    while (-not (Test-Path -LiteralPath $Probe)) {
        $Parent=[IO.Path]::GetDirectoryName($Probe)
        if (-not $Parent -or $Parent -eq $Probe) { break }
        $Probe=$Parent
    }
    $Item=Get-Item -LiteralPath $Probe -ErrorAction SilentlyContinue
    while ($Item) {
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Junction/symlink paths are unsupported; use physical input/output directories.' }
        $Item=$Item.Parent
    }
}
foreach ($Path in $Inputs+@($OutputRoot)) { Assert-PhysicalPath $Path }
if (-not $RuntimeImage) {
    $Images=Get-Content -LiteralPath (Join-Path $PriorStudy 'local-runtime-images.json') -Raw | ConvertFrom-Json
    $RuntimeImage=$Images.Training
}
if ($RuntimeImage -notmatch '^sha256:[0-9a-f]{64}$') { throw 'Specify an existing pinned training image ID, not a mutable tag.' }
$OldPreference=$ErrorActionPreference
$ErrorActionPreference='Continue'
try {
    $Engine=& docker info --format '{{.OSType}}' 2>&1
    $EngineStatus=$LASTEXITCODE
    $Actual=& docker image inspect $RuntimeImage --format '{{.Id}}' 2>&1
    $ImageStatus=$LASTEXITCODE
} finally { $ErrorActionPreference=$OldPreference }
if ($EngineStatus -ne 0 -or [string]$Engine -ne 'linux') { throw 'Start Docker Desktop Linux engine first. No build/pull attempted.' }
if ($ImageStatus -ne 0 -or [string]$Actual -ne $RuntimeImage) { throw 'Pinned local training image unavailable. No build/pull attempted.' }
$Mounts=@('--mount',"type=bind,source=$PackageRoot,target=/package,readonly",
    '--mount',"type=bind,source=$DataRoot,target=/data,readonly",
    '--mount',"type=bind,source=$AuditRoot,target=/audit,readonly")
if ($NeedLabels) {
    foreach ($Seed in 0,1,2) {
        # Mount ONLY external label artifacts. No TrackRAD data, models or test reports are mounted.
        $SeedRoot=Join-Path $StudyRoot "seed_$Seed"
        $Pairs=@(
            @('manifest.json','manifest.json'),
            @('full\labels','labels'),
            @('full\external\B1\steps.json','steps.json'),
            @('full\external\B1\verified.json','verified.json')
        )
        foreach ($Pair in $Pairs) {
            $Source=(Resolve-Path -LiteralPath (Join-Path $SeedRoot $Pair[0])).Path
            Assert-PhysicalPath $Source
            $Mounts+=@('--mount',"type=bind,source=$Source,target=/cache/seed_$Seed/$($Pair[1]),readonly")
        }
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Lock=[IO.File]::Open($OutputRoot+'.lock','OpenOrCreate','ReadWrite','None')
function Invoke-QC([string[]]$Worker,[string]$Label) {
    $Log=Join-Path $OutputRoot ($Label+'.log')
    $Arguments=@('run','--rm','--pull','never','--network','none','--cpus','4','--memory','8g',
        '--user','0:0','--read-only','--tmpfs','/tmp:rw,size=1g',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','PYTHONUNBUFFERED=1',
        '--env','CUDA_VISIBLE_DEVICES=-1','--env','NVIDIA_VISIBLE_DEVICES=void',
        '--env','OMP_NUM_THREADS=2','--env','MKL_NUM_THREADS=2',
        '--mount',"type=bind,source=$OutputRoot,target=/output")+$Mounts+@(
        '--entrypoint','/usr/bin/timeout',$RuntimeImage,'--signal=TERM','--kill-after=30s',"${TimeoutSeconds}s",
        '/opt/app/.pixi/envs/training/bin/python')+$Worker
    $OldPreference=$ErrorActionPreference
    $ErrorActionPreference='Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object {
            $Line=[string]$_
            if ($Line -match '^(AUDIT |LABEL |COMPLETE |OK|Ran |FAILED|ERROR|DEPENDENCIES )') { Write-Host $Line }
        }
        $Status=$LASTEXITCODE
    } finally { $ErrorActionPreference=$OldPreference }
    if ($Status -ne 0) {
        Get-Content -LiteralPath $Log -Tail 25 | ForEach-Object { Write-Host $_ }
        throw "Quality audit failed ($Status). Preserve inputs/results. Log: $Log"
    }
}
try {
    Write-Host "ACDC quality audit stage=${Stage}: CPU only; no training/inference/network/build/pull/deletion."
    Invoke-QC @('-c',"import numpy, PIL, SimpleITK, torch; print('DEPENDENCIES OK')") 'dependencies'
    Invoke-QC @('-m','unittest','discover','-s','/package','-p','test_qa.py','-v') 'smoke'
    if ($Stage -ne 'smoke') {
        Invoke-QC @('/package/audit.py',$Stage,'--runtime-id',$RuntimeImage,'--replays-per-seed',"$ReplaysPerSeed") 'audit'
        Write-Host "Completed: $OutputRoot\summary.json"
        Write-Host "Review locally: $OutputRoot\index.html (keep the whole results folder). No automatic filtering."
    }
} finally { $Lock.Dispose() }
