# Shared LOCAL-only environment reuse. No download, Pixi install or policy bypass.
function Invoke-LocalDocker([string[]]$Arguments, [string]$Log) {
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker @Arguments 2>&1 | Tee-Object -FilePath $Log -Append | ForEach-Object { Write-Host $_ }
        $Status = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Previous }
    if ($Status -ne 0) { throw "Docker failed ($Status). Preserve results and repeat after fixing the cause." }
}
function Get-LocalImageId([string]$Name) {
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $Id = [string](& docker image inspect --format '{{.Id}}' $Name 2>$null)
        $Status = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Previous }
    if ($Status -ne 0 -or $Id -notmatch '^sha256:[0-9a-f]{64}$') {
        throw "Local image unavailable: $Name. Start Docker Desktop and check docker image ls. No pull was attempted."
    }
    return $Id
}
function Initialize-LocalExperimentImages([string]$RepoRoot, [string]$ResultsRoot,
    [string]$TrainingRuntime = '', [string]$InferenceRuntime = 'trackrad-algorithm-cotracker-algorithm:latest') {
    $RuntimeLock = [IO.File]::Open((Join-Path $ResultsRoot '.runtime.lock'),'OpenOrCreate','ReadWrite','None')
    try {
    $Path = Join-Path $ResultsRoot 'local-runtime-images.json'
    $Utf8 = New-Object System.Text.UTF8Encoding($false)
    if (Test-Path -LiteralPath $Path) {
        $Info = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        foreach ($Id in @($Info.Inference, $Info.Evaluation, $Info.Training) | Where-Object { $_ }) {
            if ((Get-LocalImageId $Id) -ne $Id) { throw 'Pinned runtime mismatch.' }
        }
    } else {
        $Log = Join-Path $ResultsRoot 'build.log'
        $Base = Get-LocalImageId $InferenceRuntime
        $Alias = 'trackrad-experiment-inference-base:' + $Base.Substring(7,12)
        Invoke-LocalDocker @('tag',$Base,$Alias) $Log
        Invoke-LocalDocker @('build','--pull=false','--network=none','--progress=plain',
            '--file',(Join-Path $RepoRoot 'cotracker-algorithm\Dockerfile.inference.reuse'),
            '--build-arg',"INFERENCE_BASE_IMAGE=$Alias",'--tag','trackrad-experiment-inference',
            (Join-Path $RepoRoot 'cotracker-algorithm')) $Log
        $Info = [ordered]@{Inference=(Get-LocalImageId 'trackrad-experiment-inference');
            Evaluation=(Get-LocalImageId 'trackrad-evaluation:latest'); Training='';
            InferenceBase=$Base; TrainingBase=''}
        if ($TrainingRuntime) {
            $Base = Get-LocalImageId $TrainingRuntime
            $Alias = 'trackrad-experiment-training-base:' + $Base.Substring(7,12)
            Invoke-LocalDocker @('tag',$Base,$Alias) $Log
            Invoke-LocalDocker @('build','--pull=false','--network=none','--progress=plain',
                '--file',(Join-Path $RepoRoot 'cotracker-algorithm\Dockerfile.training.reuse'),
                '--build-arg',"TRAINING_BASE_IMAGE=$Alias",'--tag','trackrad-experiment-training',
                (Join-Path $RepoRoot 'cotracker-algorithm')) $Log
            $Info.Training = Get-LocalImageId 'trackrad-experiment-training'
            $Info.TrainingBase = $Base
        }
        [IO.File]::WriteAllText("$Path.tmp",(ConvertTo-Json -InputObject $Info -Depth 10),$Utf8)
        Move-Item -LiteralPath "$Path.tmp" -Destination $Path
    }
    if ($TrainingRuntime -and -not $Info.Training) { throw 'Training runtime missing in this study.' }
    $FrozenPath = Join-Path $ResultsRoot 'reference-images.json'
    $Frozen = [ordered]@{'trackrad-algorithm-cotracker-algorithm'=$Info.Inference; 'trackrad-evaluation'=$Info.Evaluation}
    $Json = ConvertTo-Json -InputObject $Frozen -Depth 10
    if (Test-Path -LiteralPath $FrozenPath) {
        if ([IO.File]::ReadAllText($FrozenPath) -ne $Json) { throw 'Reference images changed.' }
    } else { [IO.File]::WriteAllText($FrozenPath,$Json,$Utf8) }
    return $Info
    } finally { $RuntimeLock.Dispose() }
}
