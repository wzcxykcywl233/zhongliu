param([string]$TempRoot='D:\yijian\tmp')
$ErrorActionPreference='Stop'
$Runner=Join-Path $PSScriptRoot 'run_repair.ps1'
$Tokens=$null; $Errors=$null
$Ast=[System.Management.Automation.Language.Parser]::ParseFile($Runner,[ref]$Tokens,[ref]$Errors)
if ($Errors.Count) { throw 'Repair runner syntax error' }
$Repo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Original=Join-Path $Repo 'diagnostics\region-supervision-v1\run_experiment.ps1'
$BeforeHash=(Get-FileHash -LiteralPath $Original).Hash
$script:CapturedRegionRunner=$null
# Exercise real hash/dispatch guard and capture, but do not invoke the resulting Docker runner.
$Source=[IO.File]::ReadAllText($Runner)
$Anchor='& ([scriptblock]::Create($Patched)) @RunParameters'
if ([regex]::Matches($Source,[regex]::Escape($Anchor)).Count -ne 1) { throw 'Invocation anchor changed' }
$Probe=$Source.Replace($Anchor,'$script:CapturedRegionRunner=$Patched')
& ([scriptblock]::Create($Probe)) -RepoRoot $Repo -Stage all
if (-not $script:CapturedRegionRunner) { throw 'Runner not captured' }
$OriginalText=[IO.File]::ReadAllText($Original).Replace("`r`n","`n")
$Recovered=$script:CapturedRegionRunner.Replace("Invoke-Worker '../region-output-shape-fix-v1/infer_fixed.py' ","Invoke-Worker 'infer.py' ")
if ($Recovered -cne $OriginalText -or (Get-FileHash -LiteralPath $Original).Hash -ne $BeforeHash) { throw 'Repair changed more than inference dispatch or modified source' }
$Tokens=$null; $Errors=$null
[System.Management.Automation.Language.Parser]::ParseInput($script:CapturedRegionRunner,[ref]$Tokens,[ref]$Errors) | Out-Null
if ($Errors.Count) { throw 'Adapted runner syntax error' }
$TestRoot=Join-Path $TempRoot ('rsf'+[Guid]::NewGuid().ToString('N').Substring(0,6))
$Dir=Join-Path $TestRoot 'diagnostics\region-supervision-v1'
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
# A different saved file must fail the guard before execution; no result folders are used.
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.txt') -Destination (Join-Path $Dir 'run_experiment.ps1')
$Failed=$false
try { & ([scriptblock]::Create($Probe)) -RepoRoot $TestRoot -Stage all } catch { $Failed=$_.Exception.Message -match 'Original region runner changed' }
if (-not $Failed) { throw 'Unrecognized original runner accepted' }
Write-Host 'SHAPE REPAIR RUNNER TEST PASSED; dispatch-only, original files retained, no Docker execution.'
