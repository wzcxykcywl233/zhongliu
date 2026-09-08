[CmdletBinding()]
param(
    [string]$Repository = "C:\zhongliu\zhongliu-tuning",
    [string]$Dataset = "C:\zhongliu\trackrad2025-main\dataset\trackrad2025_labeled_public_test_38",
    [string]$ReadyAudit = "C:\zhongliu\zhongliu-tuning\dataset-audit-public-test\combined-38\dataset-summary.json",
    [string]$Results = "C:\zhongliu\zhongliu-tuning\public-test-38-results",
    [int]$PollSeconds = 30,
    [int]$WaitTimeoutHours = 24
)

$ErrorActionPreference = "Stop"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
New-Item -ItemType Directory -Force -Path $Results | Out-Null
$queueLog = Join-Path $Results "queue.log"
$lockPath = Join-Path $Results ".queue.lock"

function Write-QueueMessage {
    param([string]$Message)

    $line = "[$([DateTimeOffset]::Now.ToString('yyyy-MM-ddTHH:mm:ss.fffzzz'))] $Message"
    Write-Host $line
    [System.IO.File]::AppendAllText($queueLog, $line + "`n", $Utf8NoBom)
}

try {
    $lockStream = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw "Another public-test queue is already active: $Results"
}

try {
    if ($PollSeconds -lt 5) {
        throw "PollSeconds must be at least 5"
    }

    $deadline = if ($WaitTimeoutHours -gt 0) {
        [DateTimeOffset]::Now.AddHours($WaitTimeoutHours)
    }
    else {
        [DateTimeOffset]::MaxValue
    }
    $nextProgressMessage = [DateTimeOffset]::MinValue

    Write-QueueMessage "Queue started; waiting for the audited 38-case public test dataset"
    Write-QueueMessage "Ready audit: $ReadyAudit"
    Write-QueueMessage "Profiles are frozen in cotracker-algorithm/experiments/public-test-38-profiles.json"

    while ($true) {
        $ready = $false
        if ((Test-Path -LiteralPath $Dataset -PathType Container) -and
            (Test-Path -LiteralPath $ReadyAudit -PathType Leaf)) {
            try {
                $audit = Get-Content -LiteralPath $ReadyAudit -Raw | ConvertFrom-Json
                $caseCount = @(Get-ChildItem -LiteralPath $Dataset -Directory -Force).Count
                $ready =
                    [bool]$audit.StructureAuditPassed -and
                    ([int]$audit.CaseCount -eq 38) -and
                    ($caseCount -eq 38)
            }
            catch {
                $ready = $false
            }
        }

        if ($ready) {
            break
        }
        if ([DateTimeOffset]::Now -ge $deadline) {
            throw "Timed out waiting for the audited 38-case public test dataset"
        }
        if ([DateTimeOffset]::Now -ge $nextProgressMessage) {
            Write-QueueMessage "Dataset not ready yet; waiting"
            $nextProgressMessage = [DateTimeOffset]::Now.AddMinutes(5)
        }
        Start-Sleep -Seconds $PollSeconds
    }

    Write-QueueMessage "Dataset audit passed; starting the public-test evaluation queue"
    & (Join-Path $PSScriptRoot "run_public_test_38_resumable.ps1") `
        -Repository $Repository `
        -Dataset $Dataset `
        -Results $Results
    Write-QueueMessage "Public-test evaluation queue completed"
}
catch {
    Write-QueueMessage "FAILED: $($_.Exception.Message)"
    throw
}
finally {
    if ($null -ne $lockStream) {
        $lockStream.Dispose()
    }
}

