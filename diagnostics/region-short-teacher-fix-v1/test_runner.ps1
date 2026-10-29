param([string]$TempRoot='D:\yijian\tmp')
$ErrorActionPreference='Stop'
$Runner=Join-Path $PSScriptRoot 'run_repair.ps1'
$Tokens=$null; $Errors=$null
$Ast=[System.Management.Automation.Language.Parser]::ParseFile($Runner,[ref]$Tokens,[ref]$Errors)
if ($Errors.Count) { throw 'Repair runner syntax error' }
$Repo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$Original=Join-Path $Repo 'diagnostics\region-supervision-v1\run_experiment.ps1'
$OriginalText=[IO.File]::ReadAllText($Original).Replace("`r`n","`n")
$BeforeHash=(Get-FileHash -LiteralPath $Original).Hash
$Source=[IO.File]::ReadAllText($Runner)
$Anchor='& ([scriptblock]::Create($Patched)) @RunParameters'
if ([regex]::Matches($Source,[regex]::Escape($Anchor)).Count -ne 1) { throw 'Invocation anchor changed' }
$Probe=$Source.Replace($Anchor,'$script:CapturedShortRunner=$Patched')
$script:CapturedShortRunner=$null
& ([scriptblock]::Create($Probe)) -RepoRoot $Repo -Stage all
if (-not $script:CapturedShortRunner) { throw 'No captured runner' }
$Tokens=$null; $Errors=$null
$PatchedAst=[System.Management.Automation.Language.Parser]::ParseInput($script:CapturedShortRunner,[ref]$Tokens,[ref]$Errors)
if ($Errors.Count) { throw 'Adapted runner syntax error' }
$Added=@'
    if ($Script -in @('train.py','infer.py','reports.py')) {
        $Extra=@($Script)+$Extra
        $Script='../region-short-teacher-fix-v1/worker.py'
    }
'@
$Recovered=$script:CapturedShortRunner.Replace($Added.Replace("`r`n","`n")+"`n",'')
if ($Recovered -cne $OriginalText -or (Get-FileHash -LiteralPath $Original).Hash -ne $BeforeHash) { throw 'Changed more than worker dispatch' }
# Exercise the actual worker argument assembly without starting Docker.
$Function=$PatchedAst.Find({param($Node) $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Invoke-Worker'},$true)
Invoke-Expression $Function.Extent.Text
$script:Calls=@()
function Invoke-LoggedDocker([string[]]$Arguments,[string]$Log) { $script:Calls+=,@($Arguments) }
$Package='/checkout/diagnostics/region-supervision-v1'; $Weights='C:\weights'; $OutputRoot='C:\results'
$Images=[pscustomobject]@{Training='sha256:test'}
foreach ($Name in @('train.py','infer.py','reports.py','study.py')) {
    Invoke-Worker $Name @('--phase','pilot')
    $Call=$script:Calls[-1]
    if ($Name -eq 'study.py') {
        if ($Call[-3] -ne "$Package/study.py") { throw 'Plan dispatch changed' }
    } else {
        if ($Call[-4] -ne "$Package/../region-short-teacher-fix-v1/worker.py" -or $Call[-3] -ne $Name) { throw 'Worker routing or arguments wrong' }
    }
    if ($Call[-2] -ne '--phase' -or $Call[-1] -ne 'pilot') { throw 'Original arguments changed' }
}
$TestRoot=Join-Path $TempRoot ('rsw'+[Guid]::NewGuid().ToString('N').Substring(0,6))
$Dir=Join-Path $TestRoot 'diagnostics\region-supervision-v1'
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.txt') -Destination (Join-Path $Dir 'run_experiment.ps1')
$Failed=$false
try { & ([scriptblock]::Create($Probe)) -RepoRoot $TestRoot -Stage all } catch { $Failed=$_.Exception.Message -match 'Original region runner changed' }
if (-not $Failed) { throw 'Unrecognized original runner accepted' }
Write-Host 'SHORT TEACHER RUNNER TEST PASSED; original stages unchanged, routed train/infer/reports only; no Docker execution.'
