[CmdletBinding()]
param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$DatasetRoot = "C:\zhongliu\trackrad2025-main\dataset",
    [string]$HfdScript = "C:\zhongliu\hfd.sh",
    [string]$Mirror = "https://hf-mirror.com",
    [string]$CombinedDataset = "",
    [string]$AuditRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoId = "LMUK-RADONC-PHYS-RES/TrackRAD2025"
$Subsets = [ordered]@{
    "trackrad2025_labeled_pre-testing_data" = 8
    "trackrad2025_labeled_testing_data"     = 30
}

if (-not (Test-Path -LiteralPath $Repository -PathType Container)) {
    throw "Repository does not exist: $Repository"
}
if (-not (Test-Path -LiteralPath $HfdScript -PathType Leaf)) {
    throw "hfd.sh does not exist: $HfdScript"
}

$HfdText = [System.IO.File]::ReadAllText($HfdScript)
if ($HfdText -match "https://huggingface\.co") {
    throw "hfd.sh still contains huggingface.co. Apply the previously used hf-mirror pagination patch first: $HfdScript"
}

New-Item -ItemType Directory -Force -Path $DatasetRoot | Out-Null
$DatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$Repository = (Resolve-Path -LiteralPath $Repository).Path
$HfdScript = (Resolve-Path -LiteralPath $HfdScript).Path

if ([string]::IsNullOrWhiteSpace($CombinedDataset)) {
    $CombinedDataset = Join-Path $DatasetRoot "trackrad2025_labeled_public_test_38"
}
if ([string]::IsNullOrWhiteSpace($AuditRoot)) {
    $AuditRoot = Join-Path $Repository "dataset-audit-public-test"
}

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)

    $fullPath = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($fullPath -notmatch "^([A-Za-z]):\\(.*)$") {
        throw "Only drive-letter Windows paths are supported: $WindowsPath"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace("\", "/")
    return "/mnt/$drive/$tail"
}

function Invoke-HfdDownload {
    param([Parameter(Mandatory = $true)][string]$Subset)

    $wslHfd = Convert-ToWslPath $HfdScript
    $wslDatasetRoot = Convert-ToWslPath $DatasetRoot
    Write-Host ""
    Write-Host "===== DOWNLOAD $Subset ====="
    & wsl.exe -d Ubuntu -- env "HF_ENDPOINT=$Mirror" `
        bash $wslHfd $RepoId `
        --dataset `
        --include "$Subset/*" `
        --tool wget `
        --local-dir $wslDatasetRoot
    if ($LASTEXITCODE -ne 0) {
        throw "$Subset download failed with exit code $LASTEXITCODE. Rerun this script to resume."
    }
}

function Add-HardLinkedTree {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $sourceRoot = (Resolve-Path -LiteralPath $Source).Path.TrimEnd("\")
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    foreach ($directory in Get-ChildItem -LiteralPath $sourceRoot -Recurse -Directory -Force) {
        $relative = $directory.FullName.Substring($sourceRoot.Length).TrimStart("\")
        New-Item -ItemType Directory -Force -Path (Join-Path $Destination $relative) | Out-Null
    }

    foreach ($file in Get-ChildItem -LiteralPath $sourceRoot -Recurse -File -Force) {
        $relative = $file.FullName.Substring($sourceRoot.Length).TrimStart("\")
        $target = Join-Path $Destination $relative
        $targetParent = Split-Path -Parent $target
        New-Item -ItemType Directory -Force -Path $targetParent | Out-Null

        if (Test-Path -LiteralPath $target -PathType Leaf) {
            if ((Get-Item -LiteralPath $target).Length -ne $file.Length) {
                throw "Existing combined file has a different size: $target"
            }
            continue
        }
        New-Item -ItemType HardLink -Path $target -Target $file.FullName | Out-Null
    }
}

foreach ($subset in $Subsets.Keys) {
    Invoke-HfdDownload $subset
}

$caseNames = @{}
foreach ($subset in $Subsets.Keys) {
    $source = Join-Path $DatasetRoot $subset
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Downloaded subset is missing: $source"
    }
    $actual = @(Get-ChildItem -LiteralPath $source -Directory -Force).Count
    $expected = $Subsets[$subset]
    if ($actual -ne $expected) {
        throw "$subset contains $actual cases; expected $expected"
    }

    foreach ($caseDirectory in Get-ChildItem -LiteralPath $source -Directory -Force) {
        if ($caseNames.ContainsKey($caseDirectory.Name)) {
            throw "Duplicate case across public test subsets: $($caseDirectory.Name)"
        }
        $caseNames[$caseDirectory.Name] = $subset
    }
}

if ($caseNames.Count -ne 38) {
    throw "The two public test subsets contain $($caseNames.Count) unique cases; expected 38"
}

New-Item -ItemType Directory -Force -Path $CombinedDataset | Out-Null
$CombinedDataset = (Resolve-Path -LiteralPath $CombinedDataset).Path
foreach ($subset in $Subsets.Keys) {
    Add-HardLinkedTree `
        -Source (Join-Path $DatasetRoot $subset) `
        -Destination $CombinedDataset
}

$auditScript = Join-Path $Repository "scripts\audit_trackrad_dataset.ps1"
if (-not (Test-Path -LiteralPath $auditScript -PathType Leaf)) {
    throw "Dataset audit script is missing: $auditScript"
}

New-Item -ItemType Directory -Force -Path $AuditRoot | Out-Null
foreach ($subset in $Subsets.Keys) {
    & $auditScript `
        -DatasetDir (Join-Path $DatasetRoot $subset) `
        -OutputDir (Join-Path $AuditRoot $subset) `
        -ExpectedCaseCount $Subsets[$subset]
}
& $auditScript `
    -DatasetDir $CombinedDataset `
    -OutputDir (Join-Path $AuditRoot "combined-38") `
    -ExpectedCaseCount 38

$combinedSummary = Join-Path $AuditRoot "combined-38\dataset-summary.json"
$audit = Get-Content -LiteralPath $combinedSummary -Raw | ConvertFrom-Json
if (-not $audit.StructureAuditPassed) {
    throw "The combined 38-case dataset failed its structural audit: $combinedSummary"
}

Write-Host ""
Write-Host "Public TrackRAD test data is ready."
Write-Host "Combined dataset: $CombinedDataset"
Write-Host "Cases: $($audit.CaseCount)"
Write-Host "Audit: $combinedSummary"

