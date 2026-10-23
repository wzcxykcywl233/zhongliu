param([string]$TempRoot=[IO.Path]::GetTempPath())
$ErrorActionPreference='Stop'
$TestRoot=Join-Path $TempRoot ('acdc-runner-'+[Guid]::NewGuid().ToString('N'))
$Repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$global:AcDcMockCalls=New-Object System.Collections.Generic.List[object]
$global:AcDcMockImage='sha256:'+('a'*64)
function docker {
    $Call=@($args)
    $global:AcDcMockCalls.Add($Call)
    $global:LASTEXITCODE=0
    if ($Call[0] -eq 'info') { Write-Output 'linux'; return }
    if ($Call[0] -eq 'image' -and $Call[1] -eq 'inspect') { Write-Output $global:AcDcMockImage; return }
    if ($Call[0] -eq 'run') { Write-Output 'MOCK WORKER PASSED'; return }
    throw 'Unexpected Docker operation in data preparation test'
}
& (Join-Path $PSScriptRoot 'run_prepare.ps1') -Stage all -AcceptResearchTerms -RepoRoot $Repo `
    -DataRoot (Join-Path $TestRoot 'data') -OutputRoot (Join-Path $TestRoot 'audit') -RuntimeImage $global:AcDcMockImage
$Runs=@($global:AcDcMockCalls | Where-Object { $_[0] -eq 'run' })
if ($Runs.Count -ne 3) { throw 'Expected exactly three preparation workers' }
foreach ($Call in $Runs) {
    if ($Call -contains '--gpus' -or $Call -contains 'build' -or $Call -contains 'pull') { throw 'Unexpected GPU/build/pull' }
    if ($Call -notcontains '--pull' -or $Call -notcontains 'never') { throw 'Missing no-pull guard' }
}
if ($Runs[0] -contains '--network' -or $Runs[0] -notcontains '--accept-research-terms') { throw 'Wrong download guard' }
foreach ($Call in @($Runs[1],$Runs[2])) {
    if ($Call -notcontains '--network' -or $Call -notcontains 'none') { throw 'Offline preparation lost network isolation' }
}
if ($Runs[1] -notcontains 'prepare' -or $Runs[2] -notcontains '/package/ab_modes.py') { throw 'Wrong stage order' }
Write-Host 'POWERSHELL LAUNCHER MOCK PASSED (no Docker invocation or network)'
