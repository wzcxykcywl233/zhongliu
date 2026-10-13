$ErrorActionPreference='Stop'
$Temporary=Join-Path ([IO.Path]::GetTempPath()) ('mechanism-probe-'+[guid]::NewGuid().ToString('N').Substring(0,8))
$Study=Join-Path $Temporary 'study'
$Training=Join-Path $Temporary 'training'
$Dataset=Join-Path $Temporary 'dataset'
$Weights=Join-Path $Temporary 'weights'
$Output=Join-Path $Temporary 'output'
foreach ($Directory in @($Study,$Training,$Dataset,$Weights,(Join-Path $Study 'degradation'))) {New-Item -ItemType Directory -Force -Path $Directory | Out-Null}
$Image='sha256:'+('1'*64)
[IO.File]::WriteAllText((Join-Path $Study 'degradation\local-runtime-images.json'),('{"Inference":"'+$Image+'"}'))
[IO.File]::WriteAllText((Join-Path $Training 'local-runtime-images.json'),('{"Training":"'+$Image+'"}'))
$Global:ProbeCalls=New-Object System.Collections.Generic.List[object]
function global:docker {
    $Global:LASTEXITCODE=0
    if ($args[0] -eq 'image') {'sha256:'+('1'*64)}
}
function global:Start-Process {
    param($FilePath,$ArgumentList,$WindowStyle,[switch]$PassThru,$RedirectStandardOutput,$RedirectStandardError)
    if ($FilePath -ne 'docker.exe' -or $WindowStyle -ne 'Hidden') {throw 'Unexpected visible/native launch'}
    $Arguments=@($ArgumentList | ForEach-Object {$_.Trim('"')})
    $Global:ProbeCalls.Add($Arguments)
    [IO.File]::WriteAllText($RedirectStandardOutput,'Mock diagnostic, NOT an actual model run.')
    [IO.File]::WriteAllText($RedirectStandardError,'')
    $Index=[Array]::IndexOf($Arguments,'--stage')
    if ($Arguments[$Index+1] -eq 'smoke') {
        New-Item -ItemType Directory -Force -Path (Join-Path $Global:ProbeOutput 'smoke') | Out-Null
        [IO.File]::WriteAllText((Join-Path $Global:ProbeOutput 'smoke\verified.json'),'{}')
    }
    $Result=New-Object PSObject -Property @{ExitCode=0;Handle=1234}
    $Result | Add-Member ScriptMethod WaitForExit {param($Milliseconds) return $true}
    $Result | Add-Member ScriptMethod Refresh {throw 'Do not discard cached exit state via Refresh'}
    return $Result
}
try {
    $Global:ProbeOutput=$Output
    & (Join-Path $PSScriptRoot 'run_mechanism_probes.ps1') -Stage all -StudyRoot $Study -TrainingRoot $Training -Dataset $Dataset -Weights $Weights -OutputRoot $Output
    if ($Global:ProbeCalls.Count -ne 18) {throw 'Expected plan, two inspectors, cached, smoke, twelve probes and summary'}
    $GPURuns=0;$SeenSmoke=$false
    foreach ($Arguments in $Global:ProbeCalls) {
        if ($Arguments -contains 'build' -or $Arguments -contains '--privileged' -or $Arguments -contains '--cap-add') {throw 'Unexpected mutation/elevation'}
        if (-not ($Arguments -contains 'none') -or -not ($Arguments -contains 'never')) {throw 'Network/pull guard missing'}
        if (@($Arguments | Where-Object {$_ -like 'type=bind,*readonly'}).Count -ne 5) {throw 'Input mounts not read-only'}
        $Index=[Array]::IndexOf($Arguments,'--stage');$Current=$Arguments[$Index+1]
        if ($Current -eq 'smoke') {$SeenSmoke=$true}
        if ($Arguments -contains '--gpus') {
            $GPURuns++
            if ($Current -notin @('smoke','probe') -or -not $SeenSmoke) {throw 'Unexpected GPU run or missing smoke ordering'}
        }
    }
    if ($GPURuns -ne 13) {throw 'Expected only thirteen bounded forward GPU jobs'}
    if ((Get-ChildItem -LiteralPath $Study -Recurse -File).Count -ne 1) {throw 'Study changed'}
    $Guarded=$false
    try {& (Join-Path $PSScriptRoot 'run_mechanism_probes.ps1') -StudyRoot $Study -TrainingRoot $Training -Dataset $Dataset -Weights $Weights -OutputRoot (Join-Path $Study 'unsafe')} catch {$Guarded=$_.Exception.Message -like '*overlap*'}
    if (-not $Guarded) {throw 'Output overlap guard failed'}
    Write-Host 'PASS mocked launcher: readonly/pinned/network/CPU/GPU-smoke/overlap guards. No Docker/GPU executed.'
} finally {
    Remove-Item Function:\docker
    Remove-Item Function:\Start-Process
    Write-Host "Fixture retained: $Temporary"
}
