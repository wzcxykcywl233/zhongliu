$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native_process.ps1')
$Temporary=Join-Path ([IO.Path]::GetTempPath()) ('probe-exit-'+[guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $Temporary | Out-Null
$Shell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
foreach ($Expected in @(0,7)) {
    $Stdout=Join-Path $Temporary ("exit-$Expected.stdout.log")
    $Stderr=Join-Path $Temporary ("exit-$Expected.stderr.log")
    $Arguments=@('-NoProfile','-Command',('"Write-Output child-ok; Start-Sleep -Milliseconds 80; exit '+$Expected+'"'))
    $Child=Start-DiagnosticProcess -FilePath $Shell -ArgumentList $Arguments -Stdout $Stdout -Stderr $Stderr
    $Budget=[Diagnostics.Stopwatch]::StartNew()
    while (-not $Child.WaitForExit(200)) {
        if ($Budget.Elapsed.TotalSeconds -gt 10) {throw 'Native regression child exceeded its test time limit.'}
    }
    $Actual=Get-DiagnosticExitCode $Child
    if ($null -eq $Actual -or $Actual -ne $Expected) {throw "Native child status mismatch: expected=$Expected actual=$Actual"}
    if ((Get-Content -LiteralPath $Stdout -Raw).Trim() -ne 'child-ok') {throw 'Redirected output was not drained before status check.'}
    Write-Host "PASS real Windows PowerShell child exit=$Actual with redirected logs"
}
$Unknown=New-Object PSObject -Property @{ExitCode=$null}
$Unknown | Add-Member ScriptMethod WaitForExit {}
$Rejected=$false
try {Get-DiagnosticExitCode $Unknown | Out-Null} catch {$Rejected=$_.Exception.Message -like '*unavailable*'}
if (-not $Rejected) {throw 'Null exit code was accepted as success/failure.'}
Write-Host 'PASS unknown exit code rejected explicitly'
Write-Host "Fixture retained: $Temporary"
