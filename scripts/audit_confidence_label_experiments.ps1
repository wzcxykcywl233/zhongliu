[CmdletBinding()]
param(
    [string]$ResultsRoot = "C:\zhongliu\zhongliu-tuning\confidence-label-training-results",
    [int]$NumSteps = 1000,
    [string[]]$Profiles = @(
        "confidence_hard_12",
        "confidence_soft_8_16",
        "confidence_soft_6_18"
    )
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Reference = $null
$Rows = @()

foreach ($Profile in $Profiles) {
    $ProfileDir = Join-Path $ResultsRoot $Profile
    $FinalCheckpoint = Join-Path $ProfileDir "cotracker_three_final.pth"
    $LogPath = Join-Path $ProfileDir "training.log"
    if (-not (Test-Path -LiteralPath $FinalCheckpoint -PathType Leaf)) {
        throw "Missing completed checkpoint: $FinalCheckpoint"
    }
    if (-not (Test-Path -LiteralPath $LogPath -PathType Leaf)) {
        throw "Missing training log: $LogPath"
    }

    # Docker progress output can insert carriage returns that split a logical
    # logging record into several PowerShell lines. Parse the complete payload
    # with a bounded expression instead of relying on line boundaries.
    $RawLog = Get-Content -LiteralPath $LogPath -Raw
    $ByStep = @{}
    $LogMatches = [regex]::Matches(
        $RawLog,
        'step=(\d+)[\s\S]{0,500}?primary=([A-Za-z0-9_]+)'
    )
    foreach ($LogMatch in $LogMatches) {
        $ByStep[[int]$LogMatch.Groups[1].Value] = $LogMatch.Groups[2].Value
    }
    $Sequence = @(
        for ($Step = 0; $Step -lt $NumSteps; $Step++) {
            if (-not $ByStep.ContainsKey($Step)) {
                throw "$Profile is missing the teacher audit for step $Step"
            }
            $ByStep[$Step]
        }
    )
    $SequenceText = $Sequence -join "`n"
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($SequenceText)
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $SequenceHash = (
            [BitConverter]::ToString($Hasher.ComputeHash($Bytes))
        ).Replace("-", "")
    }
    finally { $Hasher.Dispose() }

    if ($null -eq $Reference) {
        $Reference = $Sequence
    }
    elseif (Compare-Object -ReferenceObject $Reference -DifferenceObject $Sequence) {
        throw "$Profile used a different primary-teacher sequence"
    }

    $Rows += [PSCustomObject]@{
        Profile = $Profile
        Steps = $NumSteps
        TeacherSequenceSHA256 = $SequenceHash
        FinalCheckpointSHA256 = (
            Get-FileHash -LiteralPath $FinalCheckpoint -Algorithm SHA256
        ).Hash
    }
}

$Audit = [ordered]@{
    completed_at = [DateTimeOffset]::Now.ToString("o")
    passed = $true
    invariant = "Identical primary-teacher sequence for every label profile"
    profiles = $Rows
}
$Path = Join-Path $ResultsRoot "confidence-label-audit.json"
$Temporary = "$Path.tmp"
$Json = ConvertTo-Json -InputObject $Audit -Depth 10
[System.IO.File]::WriteAllText($Temporary, $Json + "`n", $Utf8NoBom)
Move-Item -LiteralPath $Temporary -Destination $Path -Force
$Rows | Format-Table -AutoSize
Write-Host "Confidence-label audit passed: $Path"
