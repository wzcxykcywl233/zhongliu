$ErrorActionPreference='Stop'
$Here=Split-Path -Parent $MyInvocation.MyCommand.Path
$TempBase=if ($env:COTRACKER_TEST_TMP) {$env:COTRACKER_TEST_TMP} else {$env:TEMP}
$Fixture=Join-Path $TempBase ('num'+[guid]::NewGuid().ToString('N').Substring(0,6))
$Repo=Join-Path $Fixture 'repo'; $Old=Join-Path $Fixture 'old'; $Evidence=Join-Path $Fixture 'evidence'; $Out=Join-Path $Fixture 'out'
foreach ($Path in @($Repo,$Old,$Evidence,(Join-Path $Repo 'scripts'))) { New-Item -ItemType Directory -Path $Path -Force | Out-Null }
$Image='sha256:'+('a'*64)
@{images=@{Training=$Image}} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Old 'manifest.json') -Encoding UTF8
# Use the real local-only helper; Docker itself is mocked below.
Copy-Item -LiteralPath (Join-Path (Split-Path (Split-Path $Here -Parent) -Parent) 'scripts\experiment_local_runtimes.ps1') -Destination (Join-Path $Repo 'scripts\experiment_local_runtimes.ps1')
$global:NumericalCalls=[Collections.Generic.List[object]]::new(); $global:NumericalFail=$false
function global:docker {
    $a=@($args); $global:NumericalCalls.Add($a); $global:LASTEXITCODE=0
    if ($a[0] -eq 'image') { return $Image }
    if ($global:NumericalFail) { $global:LASTEXITCODE=1; 'Mock evidence failed'; return }
    'NUMERICAL AUDIT COMPLETE mock'
}
function Assert([bool]$Condition,[string]$Message) { if (-not $Condition) { throw $Message } }
try {
    & (Join-Path $Here 'run_audit.ps1') -RepoRoot $Repo -PreviousRoot $Old -EvidenceRoot $Evidence -OutputRoot $Out
    $Calls=@($global:NumericalCalls.ToArray() | Where-Object {$_[0] -eq 'run'})
    Assert ($Calls.Count -eq 1) 'Only one CPU job'
    $Call=$Calls[0]
    Assert (-not ($Call -contains '--gpus')) 'No GPU allowed'
    Assert (($Call -contains 'none') -and ($Call -contains 'never') -and ($Call -contains '600s')) 'Offline bounded execution'
    Assert (@($Call | Where-Object {$_ -like '*target=/evidence,readonly'}).Count -eq 1) 'Evidence read-only'
    Assert (@($Call | Where-Object {$_ -like '*target=/checkout,readonly'}).Count -eq 1) 'Source read-only'
    Assert (-not ($Call -match 'target=/data|train-40|validation-10|test-38|target=/weights')) 'No raw data or weights'
    $global:NumericalFail=$true; $Failed=$false
    try { & (Join-Path $Here 'run_audit.ps1') -RepoRoot $Repo -PreviousRoot $Old -EvidenceRoot $Evidence -OutputRoot $Out } catch { $Failed=$true }
    Assert $Failed 'Docker failure must propagate'
    $Failed=$false
    try { & (Join-Path $Here 'run_audit.ps1') -RepoRoot $Repo -PreviousRoot $Old -EvidenceRoot $Evidence -OutputRoot $Evidence } catch { $Failed=$true }
    Assert $Failed 'Output/evidence overlap refused'
    Write-Host 'NUMERICAL RUNNER TESTS PASSED (mock Docker)'
} finally {
    Remove-Item Function:\docker
    $Resolved=[IO.Path]::GetFullPath($Fixture)
    $Parent=[IO.Path]::GetFullPath($TempBase).TrimEnd('\')+'\'
    if ($Resolved.StartsWith($Parent,[StringComparison]::OrdinalIgnoreCase) -and (Split-Path $Resolved -Leaf) -match '^num[0-9a-f]{6}$') {
        Remove-Item -LiteralPath $Resolved -Recurse -Force
    } else { throw 'Unsafe fixture cleanup refused' }
}
