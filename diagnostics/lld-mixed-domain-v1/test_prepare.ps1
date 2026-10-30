$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
$TempBase=if ($env:COTRACKER_TEST_TMP) {$env:COTRACKER_TEST_TMP} else {$env:TEMP}
$Fixture=Join-Path $TempBase ('lp'+[guid]::NewGuid().ToString('N').Substring(0,6))
$Repo=Join-Path $Fixture 'repo'; $Old=Join-Path $Fixture 'old'
$Data=Join-Path $Fixture 'data'; $Out=Join-Path $Fixture 'out'
New-Item -ItemType Directory -Path $Repo,$Old -Force | Out-Null
$Image='sha256:'+('b'*64)
@{images=@{Training=$Image}} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Old 'manifest.json') -Encoding UTF8
$global:LldCalls=[Collections.Generic.List[object]]::new(); $global:LldFail=$false
function global:docker {
    $a=@($args); $global:LldCalls.Add($a); $global:LASTEXITCODE=0
    if ($a -contains 'info') {'linux'; return}
    if ($a -contains 'inspect') {$Image; return}
    if ($global:LldFail -and $a -contains 'download') {$global:LASTEXITCODE=1; 'Mock failed download'; return}
    'AUDIT mock'
}
function Assert([bool]$Value,[string]$Message) {if (-not $Value) {throw $Message}}
try {
    $Failed=$false
    try { & (Join-Path $Here 'run_prepare.ps1') -Stage all -RepoRoot $Repo -PreviousRoot $Old -DataRoot $Data -OutputRoot $Out } catch {$Failed=$true}
    Assert $Failed 'Terms must be explicit'; Assert ($global:LldCalls.Count -eq 0) 'No Docker/network before terms acceptance'
    & (Join-Path $Here 'run_prepare.ps1') -Stage all -RepoRoot $Repo -PreviousRoot $Old -DataRoot $Data -OutputRoot $Out -AcceptResearchTerms
    $Calls=@($global:LldCalls.ToArray() | Where-Object {$_ -contains 'run'})
    Assert ($Calls.Count -eq 2) 'Only download + audit'
    Assert ($Calls[0] -contains 'download') 'Download first'
    Assert ($Calls[0] -contains '--accept-research-terms') 'Forward explicit terms'
    Assert (-not ($Calls[0] -contains 'none')) 'Download network enabled'
    Assert (($Calls[1] -contains 'audit') -and ($Calls[1] -contains 'none')) 'Offline audit'
    Assert (@($Calls[1] | Where-Object {$_ -like '*target=/data,readonly'}).Count -eq 1) 'Audit data read-only'
    foreach($Call in $Calls) {
        Assert (($Call -contains 'never') -and -not ($Call -contains '--gpus')) 'No builds/pulls/GPU'
        Assert (-not ($Call -match 'train-40|validation-10|test-38|target=/checkout')) 'No model or TrackRAD access'
    }
    $global:LldCalls.Clear(); $global:LldFail=$true; $Failed=$false
    try { & (Join-Path $Here 'run_prepare.ps1') -Stage all -RepoRoot $Repo -PreviousRoot $Old -DataRoot $Data -OutputRoot $Out -AcceptResearchTerms } catch {$Failed=$true}
    Assert $Failed 'Download failure stops'
    Assert (@($global:LldCalls.ToArray() | Where-Object {$_ -contains 'audit'}).Count -eq 0) 'No audit after failure'
    Write-Host 'PREPARE TESTS PASSED (mock Docker, no download)'
} finally {
    Remove-Item Function:\docker
    $Resolved=[IO.Path]::GetFullPath($Fixture); $Parent=[IO.Path]::GetFullPath($TempBase).TrimEnd('\')+'\'
    if ($Resolved.StartsWith($Parent,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path $Resolved -Leaf) -match '^lp[0-9a-f]{6}$') {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    } else {throw 'Unsafe fixture cleanup refused'}
}
