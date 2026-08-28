[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Mirror = "https://hf-mirror.com"
)

$ErrorActionPreference = "Stop"
$CheckpointDir = Join-Path $RepoRoot "training-checkpoints"
New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null

function Get-Checkpoint {
    param([string]$Name, [string]$Url)

    $Destination = Join-Path $CheckpointDir $Name
    $Partial = "$Destination.part"
    if ((Test-Path $Destination) -and (Get-Item $Destination).Length -gt 1MB) {
        Write-Host "Checkpoint exists: $Destination"
        return
    }
    $Arguments = @(
        "-L", "--fail", "--http1.1",
        "--retry", "10", "--retry-delay", "5", "--retry-all-errors"
    )
    if (Test-Path $Partial) {
        $Arguments += @("--continue-at", "-")
    }
    $Arguments += @("--output", $Partial, $Url)
    & curl.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Name"
    }
    if ((Get-Item $Partial).Length -le 1MB) {
        throw "Downloaded file is unexpectedly small: $Partial"
    }
    Move-Item -LiteralPath $Partial -Destination $Destination -Force
}

$ExistingScaled = Join-Path $RepoRoot "cotracker-algorithm\torch\hub\checkpoints\scaled_offline.pth"
$ScaledDestination = Join-Path $CheckpointDir "scaled_offline.pth"
if ((Test-Path $ExistingScaled) -and -not (Test-Path $ScaledDestination)) {
    Copy-Item -LiteralPath $ExistingScaled -Destination $ScaledDestination
}

Get-Checkpoint "scaled_offline.pth" "$Mirror/facebook/cotracker3/resolve/main/scaled_offline.pth?download=true"
Get-Checkpoint "baseline_online.pth" "$Mirror/facebook/cotracker3/resolve/main/baseline_online.pth?download=true"
Get-Checkpoint "baseline_offline.pth" "$Mirror/facebook/cotracker3/resolve/main/baseline_offline.pth?download=true"
Get-Checkpoint "cotracker2v1.pth" "$Mirror/facebook/cotracker/resolve/main/cotracker2v1.pth?download=true"

Get-ChildItem $CheckpointDir -Filter "*.pth" |
    Select-Object Name,Length,LastWriteTime |
    Format-Table -AutoSize
