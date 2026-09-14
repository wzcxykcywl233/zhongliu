[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$TrainDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40",
    [string]$ValidationDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10",
    [string]$TestDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\dynamic-query-memory"
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $RepoRoot "scripts\run_hierarchical_followup_resumable.ps1"
$Summarizer = Join-Path $RepoRoot "scripts\summarize_dynamic_query_memory.ps1"
$Log = Join-Path $ResultsRoot "queue.log"
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null

$Expected = @{
    $TrainDataset = 40
    $ValidationDataset = 10
    $TestDataset = 38
}
foreach ($Entry in $Expected.GetEnumerator()) {
    if (-not (Test-Path -LiteralPath $Entry.Key -PathType Container)) {
        throw "Dataset is missing: $($Entry.Key)"
    }
    $Count = @(Get-ChildItem -LiteralPath $Entry.Key -Directory -Force).Count
    if ($Count -ne $Entry.Value) {
        throw "Expected $($Entry.Value) cases, found $Count in $($Entry.Key)"
    }
}

$Profiles = @(
    "hierarchical_full_grid0_iterations2",
    "hierarchical_full_grid0_iterations2_memory_latest",
    "hierarchical_full_grid0_iterations2_memory_topk",
    "hierarchical_full_grid0_iterations2_memory_topk_diverse"
)

$TranscriptStarted = $false
try {
    Start-Transcript -Path $Log -Append | Out-Null
    $TranscriptStarted = $true
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Dynamic query-memory queue started"
    Write-Host "The 40-case split is protocol-audited but not read by this parameter-free inference study."

    foreach ($Split in @(
        @{ Name = "validation-10"; Dataset = $ValidationDataset },
        @{ Name = "test-38"; Dataset = $TestDataset }
    )) {
        $SplitResults = Join-Path $ResultsRoot $Split.Name
        & $Runner `
            -Repository $RepoRoot `
            -Dataset $Split.Dataset `
            -Results $SplitResults `
            -Profiles $Profiles `
            -RequireDiagnostics
        & $Summarizer `
            -Results $SplitResults `
            -Prefix "dynamic-query-memory-$($Split.Name)"
    }
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] Dynamic query-memory queue completed"
}
catch {
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] FAILED: $($_.Exception.Message)"
    throw
}
finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
}
