[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\zhongliu\zhongliu-tuning",
    [string]$TrainDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_train_40",
    [string]$ValidationDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_validation_10",
    [string]$TestDataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\protocol-40-10-38\memory-refinement",
    [ValidateSet("all", "validation", "test")][string]$Stage = "all",
    [string]$ManifestRelative = 'cotracker-algorithm\experiments\memory-refinement-40-10-38.json',
    [string]$ResultPrefix = 'memory-refinement',
    [string]$ReferenceImagesPath = ''
)
$ErrorActionPreference = "Stop"
$Utf8 = New-Object System.Text.UTF8Encoding($false)
New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
$QueueLock = [System.IO.File]::Open((Join-Path $ResultsRoot ".queue.lock"), 'OpenOrCreate', 'ReadWrite', 'None')
$TranscriptStarted = $false
try {
    Start-Transcript -Path (Join-Path $ResultsRoot "queue.log") -Append | Out-Null
    $TranscriptStarted = $true
    $RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
    $ManifestPath = if ([System.IO.Path]::IsPathRooted($ManifestRelative)) { $ManifestRelative } else { Join-Path $RepoRoot $ManifestRelative }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $Profiles = @($Manifest.profiles)
    $Checkpoint = Join-Path $RepoRoot "cotracker-algorithm\torch\hub\checkpoints\scaled_offline.pth"
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw "Missing original checkpoint: $Checkpoint" }
    $SplitSpecs = @(
        @{ Name = "train-40"; Path = $TrainDataset; Count = 40 },
        @{ Name = "validation-10"; Path = $ValidationDataset; Count = 10 },
        @{ Name = "test-38"; Path = $TestDataset; Count = 38 }
    )
    $SeenCases = @{}
    $DataFingerprint = @()
    foreach ($Split in $SplitSpecs) {
        $Cases = @(Get-ChildItem -LiteralPath $Split.Path -Directory | Sort-Object Name)
        if ($Cases.Count -ne $Split.Count) { throw "Expected $($Split.Count) cases in $($Split.Path), found $($Cases.Count)" }
        foreach ($Case in $Cases) {
            if ($SeenCases.ContainsKey($Case.Name)) { throw "Case overlap across splits: $($Case.Name)" }
            $SeenCases[$Case.Name] = $Split.Name
            foreach ($Relative in @("images\$($Case.Name)_frames.mha", "targets\$($Case.Name)_first_label.mha", "targets\$($Case.Name)_labels.mha", "frame-rate.json", "b-field-strength.json", "scanned-region.json")) {
                $Required = Join-Path $Case.FullName $Relative
                if (-not (Test-Path -LiteralPath $Required -PathType Leaf) -or (Get-Item -LiteralPath $Required).Length -eq 0) { throw "Missing/empty dataset file: $Required" }
            }
            foreach ($File in @(Get-ChildItem -LiteralPath $Case.FullName -Recurse -File | Sort-Object FullName)) {
                $DataFingerprint += [ordered]@{ Split = $Split.Name; Case = $Case.Name; File = $File.FullName; SHA256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash }
            }
        }
    }
    Write-Host "Dataset counts and disjoint case IDs verified: 40 / 10 / 38. No training in this queue."
    $SourcePaths = @(& git -C $RepoRoot ls-files -- cotracker-algorithm evaluation scripts)
    if ($LASTEXITCODE -ne 0) { throw "Cannot enumerate tracked source files" }
    $SourceHashes = foreach ($Relative in ($SourcePaths | Sort-Object)) {
        if ($Relative -match '\.(py|ps1|sh|json|toml|lock|txt)$|Dockerfile') {
            [ordered]@{ File = $Relative; SHA256 = (Get-FileHash -LiteralPath (Join-Path $RepoRoot $Relative) -Algorithm SHA256).Hash }
        }
    }
    $Frozen = [ordered]@{
        Schema = 1; Manifest = $Manifest; Source = @($SourceHashes)
        CheckpointSHA256 = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash
        Dataset = $DataFingerprint
    }
    $FrozenJson = ConvertTo-Json -InputObject $Frozen -Depth 30
    $FrozenPath = Join-Path $ResultsRoot "frozen-run.json"
    if (Test-Path -LiteralPath $FrozenPath) {
        if ([System.IO.File]::ReadAllText($FrozenPath) -ne $FrozenJson) {
            throw "Source, weights, protocol or dataset changed. Use a NEW ResultsRoot; do not mix cached results."
        }
    } else {
        $ExistingMetrics = @(Get-ChildItem -LiteralPath $ResultsRoot -Recurse -Filter metrics.json -File)
        if ($ExistingMetrics.Count -gt 0) { throw "Unfingerprinted results already exist; choose a new ResultsRoot" }
        [System.IO.File]::WriteAllText("$FrozenPath.tmp", $FrozenJson, $Utf8)
        Move-Item -LiteralPath "$FrozenPath.tmp" -Destination $FrozenPath
    }
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] $ResultPrefix queue started: $Stage"
    foreach ($Split in $SplitSpecs | Where-Object Name -ne 'train-40') {
        if ($Stage -eq 'validation' -and $Split.Name -ne 'validation-10') { continue }
        if ($Stage -eq 'test' -and $Split.Name -ne 'test-38') { continue }
        if ($Split.Name -eq 'test-38') {
            foreach ($Profile in $Profiles) {
                $Prior = Join-Path $ResultsRoot "validation-10\$Profile\metrics.json"
                if (-not (Test-Path -LiteralPath $Prior)) { throw "Validation incomplete: $Profile. Run validation first." }
            }
        }
        $SplitResults = Join-Path $ResultsRoot $Split.Name
        & (Join-Path $RepoRoot "scripts\run_hierarchical_followup_resumable.ps1") `
            -Repository $RepoRoot -Dataset $Split.Path -Results $SplitResults `
            -Profiles $Profiles -RequireDiagnostics -FreezeImages -ModelCheckpoint $Checkpoint `
            -ReferenceImagesPath $ReferenceImagesPath
        & (Join-Path $RepoRoot "scripts\summarize_memory_refinement.ps1") `
            -RepoRoot $RepoRoot -Results $SplitResults -ExpectedCases $Split.Count -SplitName $Split.Name `
            -ManifestRelative $ManifestRelative -ResultPrefix $ResultPrefix
        if ($Manifest.backcheck_analysis) {
            $AnalysisMode = 'iterations'
            if ($Manifest.backcheck_analysis -eq 'fourway') { $AnalysisMode = 'fourway' }
            if ($Manifest.backcheck_analysis -eq 'rotation') { $AnalysisMode = 'rotation' }
            $AnalysisLog = Join-Path $SplitResults 'backcheck-analysis.log'
            $PreviousPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try {
                & docker run --rm --network none --platform linux/amd64 `
                    --mount "type=bind,source=$($Split.Path),target=/dataset,readonly" `
                    --mount "type=bind,source=$SplitResults,target=/results" `
                    --entrypoint /opt/app/.pixi/envs/cuda/bin/python `
                    trackrad-algorithm-cotracker-algorithm `
                    /opt/app/experiments/analyze_frame_backcheck.py `
                    --dataset /dataset --results /results --split $Split.Name --mode $AnalysisMode 2>&1 |
                    Tee-Object -FilePath $AnalysisLog -Append | ForEach-Object { Write-Host $_ }
                $AnalysisExit = $LASTEXITCODE
            } finally { $ErrorActionPreference = $PreviousPreference }
            if ($AnalysisExit -ne 0) { throw "Backcheck analysis failed: exit=$AnalysisExit; rerun same command to resume" }
        }
    }
    Write-Host "[$([DateTimeOffset]::Now.ToString('o'))] $ResultPrefix queue completed"
} finally {
    if ($TranscriptStarted) { Stop-Transcript | Out-Null }
    $QueueLock.Dispose()
}
