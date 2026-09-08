[CmdletBinding()]
param(
    [string]$DatasetDir = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_training_data",
    [string]$OutputDir = "",
    [int]$ExpectedCaseCount = 50
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DatasetDir -PathType Container)) {
    throw "Dataset directory does not exist: $DatasetDir"
}

$DatasetDir = (Resolve-Path -LiteralPath $DatasetDir).Path
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $DatasetDir) "dataset-audit"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

function Read-ScalarJson {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }

    try {
        return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
    }
    catch {
        return "INVALID_JSON"
    }
}

function Get-MhaHeader {
    param([string]$Path)

    $buffer = New-Object byte[] 65536
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $count = $stream.Read($buffer, 0, $buffer.Length)
    }
    finally {
        $stream.Dispose()
    }

    $text = [System.Text.Encoding]::ASCII.GetString($buffer, 0, $count)
    $localMarker = $text.IndexOf("ElementDataFile", [System.StringComparison]::OrdinalIgnoreCase)
    if ($localMarker -ge 0) {
        $lineEnd = $text.IndexOf("`n", $localMarker)
        if ($lineEnd -ge 0) {
            $text = $text.Substring(0, $lineEnd + 1)
        }
    }

    function Get-HeaderValue {
        param([string]$Name)

        $pattern = "(?im)^\s*" + [regex]::Escape($Name) + "\s*=\s*([^\r\n]+)"
        $match = [regex]::Match($text, $pattern)
        if ($match.Success) {
            return $match.Groups[1].Value.Trim()
        }
        return $null
    }

    $dimSizeText = Get-HeaderValue "DimSize"
    $dimensions = @()
    if ($dimSizeText) {
        $dimensions = @(
            $dimSizeText -split "\s+" |
                Where-Object { $_ -match "^\d+$" } |
                ForEach-Object { [int64]$_ }
        )
    }

    $frames = $null
    if ($dimensions.Count -ge 3) {
        $frames = $dimensions[-1]
    }
    elseif ($dimensions.Count -eq 2) {
        $frames = 1
    }

    [PSCustomObject]@{
        Path           = $Path
        NDims          = Get-HeaderValue "NDims"
        DimSize        = $dimSizeText
        ElementSpacing = Get-HeaderValue "ElementSpacing"
        ElementType    = Get-HeaderValue "ElementType"
        CompressedData = Get-HeaderValue "CompressedData"
        Frames         = $frames
    }
}

function Add-Issue {
    param(
        [string]$Case,
        [string]$Severity,
        [string]$Code,
        [string]$Detail
    )

    $script:Issues.Add([PSCustomObject]@{
        Case     = $Case
        Severity = $Severity
        Code     = $Code
        Detail   = $Detail
    }) | Out-Null
}

$Issues = [System.Collections.Generic.List[object]]::new()
$CaseRows = [System.Collections.Generic.List[object]]::new()
$MhaRows = [System.Collections.Generic.List[object]]::new()

$CaseDirectories = @(
    Get-ChildItem -LiteralPath $DatasetDir -Directory -Force |
        Where-Object { $_.Name -notlike ".*" } |
        Sort-Object Name
)

foreach ($caseDirectory in $CaseDirectories) {
    $caseName = $caseDirectory.Name
    $cohort = ($caseName -split "_")[0]
    $imagesDirectory = Join-Path $caseDirectory.FullName "images"
    $targetsDirectory = Join-Path $caseDirectory.FullName "targets"

    $imageFiles = @(
        Get-ChildItem -LiteralPath $imagesDirectory -Filter "*.mha" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.BaseName -match "_frames\d*$" } |
            Sort-Object Name
    )
    $firstLabelFiles = @(
        Get-ChildItem -LiteralPath $targetsDirectory -Filter "*_first_label.mha" -File -ErrorAction SilentlyContinue |
            Sort-Object Name
    )
    $labelFiles = @(
        Get-ChildItem -LiteralPath $targetsDirectory -Filter "*.mha" -File -ErrorAction SilentlyContinue |
            Where-Object { $_.BaseName -match "_labels[^_]*$" } |
            Sort-Object Name
    )

    if ($imageFiles.Count -eq 0) {
        Add-Issue $caseName "ERROR" "MISSING_IMAGE" "No *_frames*.mha file"
    }
    if ($firstLabelFiles.Count -eq 0) {
        Add-Issue $caseName "ERROR" "MISSING_FIRST_LABEL" "No *_first_label.mha file"
    }
    if ($labelFiles.Count -eq 0) {
        Add-Issue $caseName "ERROR" "MISSING_LABELS" "No *_labels*.mha file"
    }

    $fieldPath = @(
        Join-Path $caseDirectory.FullName "b-field-strength.json"
        Join-Path $caseDirectory.FullName "field-strength.json"
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    $frameRatePath = Join-Path $caseDirectory.FullName "frame-rate.json"
    $regionPath = Join-Path $caseDirectory.FullName "scanned-region.json"

    $fieldStrength = if ($fieldPath) { Read-ScalarJson $fieldPath } else { $null }
    $frameRate = Read-ScalarJson $frameRatePath
    $scannedRegion = Read-ScalarJson $regionPath

    if (-not $fieldPath) {
        Add-Issue $caseName "ERROR" "MISSING_FIELD_STRENGTH" "No field-strength metadata"
    }
    elseif ($fieldStrength -eq "INVALID_JSON") {
        Add-Issue $caseName "ERROR" "INVALID_FIELD_STRENGTH" $fieldPath
    }
    if ($null -eq $frameRate) {
        Add-Issue $caseName "ERROR" "MISSING_FRAME_RATE" $frameRatePath
    }
    elseif ($frameRate -eq "INVALID_JSON") {
        Add-Issue $caseName "ERROR" "INVALID_FRAME_RATE" $frameRatePath
    }
    if ($null -eq $scannedRegion) {
        Add-Issue $caseName "ERROR" "MISSING_SCANNED_REGION" $regionPath
    }
    elseif ($scannedRegion -eq "INVALID_JSON") {
        Add-Issue $caseName "ERROR" "INVALID_SCANNED_REGION" $regionPath
    }

    $imageHeaders = @()
    foreach ($file in $imageFiles) {
        $header = Get-MhaHeader $file.FullName
        $imageHeaders += $header
        $MhaRows.Add([PSCustomObject]@{
            Case           = $caseName
            Role           = "image"
            File           = $file.Name
            SizeMiB        = [math]::Round($file.Length / 1MB, 3)
            NDims          = $header.NDims
            DimSize        = $header.DimSize
            ElementSpacing = $header.ElementSpacing
            ElementType    = $header.ElementType
            Frames         = $header.Frames
        }) | Out-Null
    }

    $primaryImageDim = if ($imageHeaders.Count -gt 0) { $imageHeaders[0].DimSize } else { $null }
    $primaryLabelDim = $null
    foreach ($file in @($firstLabelFiles + $labelFiles)) {
        $header = Get-MhaHeader $file.FullName
        $role = if ($file.Name -like "*_first_label.mha") { "first_label" } else { "labels" }
        if ($role -eq "labels" -and -not $primaryLabelDim) {
            $primaryLabelDim = $header.DimSize
        }
        $MhaRows.Add([PSCustomObject]@{
            Case           = $caseName
            Role           = $role
            File           = $file.Name
            SizeMiB        = [math]::Round($file.Length / 1MB, 3)
            NDims          = $header.NDims
            DimSize        = $header.DimSize
            ElementSpacing = $header.ElementSpacing
            ElementType    = $header.ElementType
            Frames         = $header.Frames
        }) | Out-Null
    }

    if ($primaryImageDim -and $primaryLabelDim -and $primaryImageDim -ne $primaryLabelDim) {
        Add-Issue $caseName "ERROR" "IMAGE_LABEL_DIMENSION_MISMATCH" `
            "image=$primaryImageDim; labels=$primaryLabelDim"
    }

    $caseBytes = (
        Get-ChildItem -LiteralPath $caseDirectory.FullName -Recurse -File -Force |
            Measure-Object Length -Sum
    ).Sum
    $caseFrames = ($imageHeaders | Measure-Object Frames -Sum).Sum

    $CaseRows.Add([PSCustomObject]@{
        Case                  = $caseName
        Cohort                = $cohort
        FieldStrengthT        = $fieldStrength
        FrameRateHz           = $frameRate
        ScannedRegion         = $scannedRegion
        ImageFiles            = $imageFiles.Count
        FirstLabelFiles       = $firstLabelFiles.Count
        LabelFiles            = $labelFiles.Count
        Frames                = $caseFrames
        ImageDimSize          = $primaryImageDim
        LabelDimSize          = $primaryLabelDim
        ImageLabelDimensionsOK = [bool](
            $primaryImageDim -and $primaryLabelDim -and $primaryImageDim -eq $primaryLabelDim
        )
        SizeMiB               = [math]::Round($caseBytes / 1MB, 3)
    }) | Out-Null
}

$allFiles = @(
    Get-ChildItem -LiteralPath $DatasetDir -Recurse -File -Force
)
$totalBytes = ($allFiles | Measure-Object Length -Sum).Sum
$cohortSummary = @(
    $CaseRows |
        Group-Object Cohort |
        Sort-Object Name |
        ForEach-Object {
            [PSCustomObject]@{
                Cohort = $_.Name
                Cases  = $_.Count
            }
        }
)
$fieldSummary = @(
    $CaseRows |
        Group-Object FieldStrengthT |
        Sort-Object Name |
        ForEach-Object {
            [PSCustomObject]@{
                FieldStrengthT = $_.Name
                Cases          = $_.Count
            }
        }
)
$regionSummary = @(
    $CaseRows |
        Group-Object ScannedRegion |
        Sort-Object Name |
        ForEach-Object {
            [PSCustomObject]@{
                ScannedRegion = $_.Name
                Cases         = $_.Count
            }
        }
)

if ($ExpectedCaseCount -gt 0 -and $CaseDirectories.Count -ne $ExpectedCaseCount) {
    Add-Issue "DATASET" "ERROR" "UNEXPECTED_CASE_COUNT" `
        "expected=$ExpectedCaseCount; actual=$($CaseDirectories.Count)"
}

$summary = [ordered]@{
    GeneratedAt                   = (Get-Date).ToString("o")
    DatasetDir                    = $DatasetDir
    ExpectedCaseCount             = $ExpectedCaseCount
    CaseCount                     = $CaseDirectories.Count
    FileCount                     = $allFiles.Count
    TotalSizeMiB                  = [math]::Round($totalBytes / 1MB, 2)
    ImageMhaCount                 = @($MhaRows | Where-Object Role -eq "image").Count
    FirstLabelMhaCount            = @($MhaRows | Where-Object Role -eq "first_label").Count
    LabelMhaCount                 = @($MhaRows | Where-Object Role -eq "labels").Count
    TotalImageFrames              = ($MhaRows | Where-Object Role -eq "image" | Measure-Object Frames -Sum).Sum
    ErrorCount                    = @($Issues | Where-Object Severity -eq "ERROR").Count
    WarningCount                  = @($Issues | Where-Object Severity -eq "WARNING").Count
    StructureAuditPassed          = (@($Issues | Where-Object Severity -eq "ERROR").Count -eq 0)
    Cohorts                       = $cohortSummary
    FieldStrengths                = $fieldSummary
    ScannedRegions                = $regionSummary
}

$caseCsv = Join-Path $OutputDir "dataset-cases.csv"
$mhaCsv = Join-Path $OutputDir "dataset-mha-files.csv"
$issueCsv = Join-Path $OutputDir "dataset-issues.csv"
$summaryJson = Join-Path $OutputDir "dataset-summary.json"

$CaseRows | Export-Csv -LiteralPath $caseCsv -NoTypeInformation -Encoding UTF8
$MhaRows | Export-Csv -LiteralPath $mhaCsv -NoTypeInformation -Encoding UTF8
$Issues | Export-Csv -LiteralPath $issueCsv -NoTypeInformation -Encoding UTF8
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryJson -Encoding UTF8

Write-Host ""
Write-Host "TrackRAD2025 dataset audit"
Write-Host "Dataset: $DatasetDir"
Write-Host "Output:  $OutputDir"
Write-Host ""
[PSCustomObject]@{
    Cases            = $summary.CaseCount
    Files            = $summary.FileCount
    SizeMiB          = $summary.TotalSizeMiB
    ImageMha         = $summary.ImageMhaCount
    FirstLabelMha    = $summary.FirstLabelMhaCount
    LabelMha         = $summary.LabelMhaCount
    TotalImageFrames = $summary.TotalImageFrames
    Errors           = $summary.ErrorCount
    AuditPassed      = $summary.StructureAuditPassed
} | Format-List

Write-Host "Cases by cohort:"
$cohortSummary | Format-Table -AutoSize
Write-Host "Cases by field strength:"
$fieldSummary | Format-Table -AutoSize
Write-Host "Cases by scanned region:"
$regionSummary | Format-Table -AutoSize

if ($Issues.Count -gt 0) {
    Write-Warning "Dataset issues were found. See: $issueCsv"
    $Issues | Select-Object -First 20 | Format-Table -AutoSize
}
else {
    Write-Host "No structural issues found."
}

Write-Host "Summary JSON: $summaryJson"
Write-Host "Case CSV:     $caseCsv"
Write-Host "MHA CSV:      $mhaCsv"
Write-Host "Issues CSV:   $issueCsv"

