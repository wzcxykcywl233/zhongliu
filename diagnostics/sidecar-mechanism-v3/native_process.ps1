# Windows PowerShell 5.1 needs a retained process handle for reliable ExitCode.
# No native DLL, reflection, encoded command or execution-policy bypass.
function Start-DiagnosticProcess {
    param([string]$FilePath,[string[]]$ArgumentList,[string]$Stdout,[string]$Stderr)
    $Child=Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    # Acquire immediately, before the child can exit. Do not Refresh afterward:
    # Refresh discards cached process state on affected PowerShell/.NET versions.
    $null=$Child.Handle
    return $Child
}

function Get-DiagnosticExitCode {
    param($Process)
    # Unbounded form here is only called AFTER the bounded loop saw completion;
    # it drains redirected output and finalizes the process exit state.
    $null=$Process.WaitForExit()
    $Status=$Process.ExitCode
    if ($null -eq $Status) {
        throw 'Child process ended but its exit code is unavailable; do not infer success or failure. Inspect the retained stdout/stderr logs.'
    }
    return [int]$Status
}
