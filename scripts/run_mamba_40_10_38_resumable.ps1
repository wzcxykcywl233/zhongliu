[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [int]$NumSteps = 1000,
    [int]$SaveEverySteps = 25
)

$ErrorActionPreference = "Stop"
$Root = Join-Path $RepoRoot "protocol-40-10-38\mamba-experiments"
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Log = Join-Path $Root "queue.log"
$TranscriptStarted = $false

try {
    Start-Transcript -Path $Log -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Mamba 40/10/38 queue started"
    & (Join-Path $RepoRoot "scripts\run_mamba_training_resumable.ps1") `
        -RepoRoot $RepoRoot `
        -NumSteps $NumSteps `
        -SaveEverySteps $SaveEverySteps
    & (Join-Path $RepoRoot "scripts\evaluate_mamba_40_10_38_resumable.ps1") `
        -RepoRoot $RepoRoot
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Mamba queue completed"
}
catch {
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] FAILED: $($_.Exception.Message)"
    throw
}
finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
