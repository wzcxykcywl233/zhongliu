[CmdletBinding()]
param(
    [ValidateSet('plan','smoke','pilot','full','test','all')][string]$Stage='smoke',
    [string]$RepoRoot='C:\zhongliu\zhongliu-tuning',
    [string]$PriorStudy='C:\zhongliu\ts-v1',
    [string]$DataRoot='C:\zhongliu\acdc-official-v1',
    [string]$AuditRoot='C:\zhongliu\acdc-ab-native-v2',
    [string]$OutputRoot='C:\zhongliu\rs-v1'
)
$ErrorActionPreference='Stop'
$Runner=Join-Path $RepoRoot 'diagnostics\region-supervision-v1\run_experiment.ps1'
$Text=[IO.File]::ReadAllText($Runner).Replace("`r`n","`n")
$Hasher=[Security.Cryptography.SHA256]::Create()
try { $Hash=([BitConverter]::ToString($Hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text)))).Replace('-','').ToLowerInvariant() }
finally { $Hasher.Dispose() }
if ($Hash -ne '43c8b1b724178b59c284a004e1a7f644cf9ce51208de1b8b74a087c99a8b1583') {
    throw 'Original region runner changed; preserve frozen inputs and results.'
}
$Before="function Invoke-Worker([string]`$Script,[string[]]`$Extra=@(),[string]`$Mode='audit',[int]`$Seconds=1800,[switch]`$Gpu,[string]`$Log='') {`n"
$Insert=@'
    if ($Script -in @('train.py','infer.py','reports.py')) {
        $Extra=@($Script)+$Extra
        $Script='../region-short-teacher-fix-v1/worker.py'
    }
'@
if ([regex]::Matches($Text,[regex]::Escape($Before)).Count -ne 1) { throw 'Unexpected worker dispatch count' }
$Patched=$Text.Replace($Before,$Before+$Insert.Replace("`r`n","`n")+"`n")
$RunParameters=@{Stage=$Stage;RepoRoot=$RepoRoot;PriorStudy=$PriorStudy;DataRoot=$DataRoot;AuditRoot=$AuditRoot;OutputRoot=$OutputRoot}
Write-Host 'Short teacher repair: existing labels/checkpoints preserved; online teacher <=8 native frames executes one window. Includes prior region shape fix.'
& ([scriptblock]::Create($Patched)) @RunParameters
