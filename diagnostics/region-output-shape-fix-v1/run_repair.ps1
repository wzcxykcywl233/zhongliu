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
    throw 'Original region runner changed; this repair is bound to commit 8239037. No source or results were modified.'
}
$Before="Invoke-Worker 'infer.py' "
$After="Invoke-Worker '../region-output-shape-fix-v1/infer_fixed.py' "
if ([regex]::Matches($Text,[regex]::Escape($Before)).Count -ne 1) { throw 'Unexpected inference dispatch count' }
$Patched=$Text.Replace($Before,$After)
$RunParameters=@{Stage=$Stage;RepoRoot=$RepoRoot;PriorStudy=$PriorStudy;DataRoot=$DataRoot;AuditRoot=$AuditRoot;OutputRoot=$OutputRoot}
Write-Host 'Region output shape repair: preserve frozen manifest, completed training and point outputs. Region output is B,T,H,W.'
& ([scriptblock]::Create($Patched)) @RunParameters
