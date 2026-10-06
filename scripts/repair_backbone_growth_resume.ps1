[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [string]$OriginalResults = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\backbone-growth-offline-v1',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\backbone-growth-resumefix-v1',
    [ValidateSet('recover','train','validation','test')][string]$Stage = 'validation'
)
$ErrorActionPreference = 'Stop'
$OriginalResults = (Resolve-Path -LiteralPath $OriginalResults).Path
$Original = Get-Content -LiteralPath (Join-Path $OriginalResults 'frozen-run.json') -Raw | ConvertFrom-Json
if ($Original.Schema -ne 2 -or -not $Original.TrainingRuntimeImage) {
    throw 'This repair supports the original pinned-runtime backbone study only.'
}
& (Join-Path $RepoRoot 'scripts\run_backbone_growth_resumable.ps1') `
    -RepoRoot $RepoRoot -ResultsRoot $ResultsRoot -RecoverFrom $OriginalResults -Stage $Stage `
    -TrainingRuntimeImage ([string]$Original.TrainingRuntimeImage) `
    -NumSteps ([int]$Original.Steps) -SaveEverySteps ([int]$Original.SaveEverySteps) `
    -Seeds (@($Original.Seeds) -join ',')
