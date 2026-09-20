[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\zhongliu\zhongliu-tuning',
    [ValidateSet('test-38','validation-10')][string]$Split = 'test-38',
    [string]$Dataset = '',
    [string]$ResultsRoot = 'C:\zhongliu\zhongliu-tuning\protocol-40-10-38\memory-chain-diagnostics',
    [string]$PreviousResults = '',
    [ValidateRange(1,1000)][int]$SamplePoints = 64
)
$ErrorActionPreference = 'Stop'
if (-not $Dataset) {
    $Leaf = if ($Split -eq 'test-38') { 'trackrad2025_labeled_public_test_38' } else { 'trackrad2025_labeled_validation_10' }
    $Dataset = Join-Path 'C:\zhongliu\trackrad2025-main\dataset' $Leaf
}
if (-not $PreviousResults) { $PreviousResults = Join-Path $RepoRoot "protocol-40-10-38\memory-refinement\$Split" }
$Output = Join-Path $ResultsRoot $Split
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Lock = [System.IO.File]::Open((Join-Path $Output '.queue.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
$Utf8 = New-Object System.Text.UTF8Encoding($false)
function Invoke-LoggedDocker([string[]]$DockerArgs) {
    $Old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & docker @DockerArgs 2>&1 | ForEach-Object {
            $Line = $_.ToString()
            Write-Host $Line
            [System.IO.File]::AppendAllText((Join-Path $Output 'runner.log'), $Line + "`n", $Utf8)
        }
        $Code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $Old }
    if ($Code -ne 0) { throw "Docker exit=$Code; rerun the same command to resume completed cases/profiles" }
}
try {
    $Checkpoint = Join-Path $RepoRoot 'cotracker-algorithm\torch\hub\checkpoints\scaled_offline.pth'
    $Cases = @(Get-ChildItem -LiteralPath $Dataset -Directory | Sort-Object Name)
    $Expected = if ($Split -eq 'test-38') { 38 } else { 10 }
    if ($Cases.Count -ne $Expected) { throw "Expected $Expected cases; found $($Cases.Count)" }
    $DatasetParent = Split-Path -Parent $Dataset
    $SplitFolders = @(
        @{ Name = 'train-40'; Leaf = 'trackrad2025_labeled_train_40'; Count = 40 },
        @{ Name = 'validation-10'; Leaf = 'trackrad2025_labeled_validation_10'; Count = 10 },
        @{ Name = 'test-38'; Leaf = 'trackrad2025_labeled_public_test_38'; Count = 38 }
    )
    $Seen = @{}
    $SplitIds = @(foreach ($Spec in $SplitFolders) {
        $Folder = if ($Spec.Name -eq $Split) { $Dataset } else { Join-Path $DatasetParent $Spec.Leaf }
        $Ids = @(Get-ChildItem -LiteralPath $Folder -Directory | Sort-Object Name | Select-Object -ExpandProperty Name)
        if ($Ids.Count -ne $Spec.Count) { throw "Unexpected count for $($Spec.Name): $($Ids.Count)" }
        foreach ($Id in $Ids) {
            if ($Seen.ContainsKey($Id)) { throw "Case overlap across splits: $Id" }
            $Seen[$Id] = $Spec.Name
        }
        [ordered]@{ Split = $Spec.Name; Cases = $Ids }
    })
    Write-Host 'Verified disjoint 40/10/38 case IDs; no training or later-frame label access.'
    $DataHashes = foreach ($Case in $Cases) {
        foreach ($Relative in @("images\$($Case.Name)_frames.mha", "targets\$($Case.Name)_first_label.mha", 'frame-rate.json','b-field-strength.json','scanned-region.json')) {
            $File = Join-Path $Case.FullName $Relative
            [ordered]@{ Case = $Case.Name; File = $Relative; SHA256 = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash }
        }
    }
    $Image = 'trackrad-memory-chain-diagnostics'
    Invoke-LoggedDocker @('build', '--progress=plain', '--platform=linux/amd64', '-t', $Image, (Join-Path $RepoRoot 'cotracker-algorithm'))
    $ImageId = & docker image inspect --format '{{.Id}}' $Image
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect diagnostic image' }
    $PreviousAvailable = Test-Path -LiteralPath $PreviousResults -PathType Container
    $PreviousHashes = @()
    if ($PreviousAvailable) {
        $PreviousHashes = @(foreach ($Profile in @('memory_control','memory_pointwise_fusion','memory_current_retrieval')) {
            foreach ($Case in $Cases) {
                $Prior = Join-Path $PreviousResults "$Profile\checkpoint\jobs\$($Case.Name)\diagnostics.json"
                if (Test-Path -LiteralPath $Prior -PathType Leaf) {
                    [ordered]@{ Profile = $Profile; Case = $Case.Name; SHA256 = (Get-FileHash -LiteralPath $Prior -Algorithm SHA256).Hash }
                }
            }
        })
    }
    $Frozen = [ordered]@{ Schema = 1; Split = $Split; SplitIds = $SplitIds; SamplePoints = $SamplePoints; Image = [string]$ImageId; Dataset = @($DataHashes); Checkpoint = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash; PreviousAvailable = $PreviousAvailable; PreviousHashes = $PreviousHashes }
    $FrozenJson = ConvertTo-Json -InputObject $Frozen -Depth 15
    $FrozenPath = Join-Path $Output 'frozen-run.json'
    if (Test-Path -LiteralPath $FrozenPath) {
        if ([System.IO.File]::ReadAllText($FrozenPath) -ne $FrozenJson) { throw 'Image, data, weights or sampling changed; use a new ResultsRoot' }
    } else {
        if (Test-Path -LiteralPath (Join-Path $Output 'jobs')) { throw 'Unfingerprinted job data found; use a new ResultsRoot' }
        [System.IO.File]::WriteAllText("$FrozenPath.tmp", $FrozenJson, $Utf8)
        Move-Item -LiteralPath "$FrozenPath.tmp" -Destination $FrozenPath
    }
    $DockerArgs = @('run','--rm','--name','trackrad-memory-chain-diagnostics','--platform=linux/amd64','--network','none','--gpus','all',
        '--env','COTRACKER_CHECKPOINT=/weights/model.pth',
        '--mount',"type=bind,source=$Dataset,target=/dataset,readonly",
        '--mount',"type=bind,source=$Output,target=/results",
        '--mount',"type=bind,source=$Checkpoint,target=/weights/model.pth,readonly",
        '--entrypoint','/opt/app/.pixi/envs/cuda/bin/python')
    if ($PreviousAvailable) { $DockerArgs += @('--mount',"type=bind,source=$PreviousResults,target=/previous,readonly") }
    $DockerArgs += @($Image,'/opt/app/experiments/run_memory_chain_diagnostics.py','--dataset','/dataset','--output','/results','--split',$Split,'--sample-points',"$SamplePoints")
    if ($PreviousAvailable) { $DockerArgs += @('--previous-results','/previous') }
    Invoke-LoggedDocker $DockerArgs
    Write-Host "Completed: $Output\summary.json"
} finally { $Lock.Dispose() }
