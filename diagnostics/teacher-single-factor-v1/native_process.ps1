# Retain the .NET process handle: Windows PowerShell 5.1 can otherwise lose ExitCode.
function Start-TeacherProcess {
    param([string[]]$Arguments,[string]$LogBase)
    $Quoted=@($Arguments | ForEach-Object {'"'+$_+'"'})
    $Child=Start-Process -FilePath 'docker.exe' -ArgumentList $Quoted -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput ($LogBase+'.stdout.log') -RedirectStandardError ($LogBase+'.stderr.log')
    $null=$Child.Handle
    return $Child
}
function Get-TeacherExitCode {
    param($Process)
    $null=$Process.WaitForExit()
    $Status=$Process.ExitCode
    if ($null -eq $Status) {throw 'Exit status unavailable. Preserve launch logs; do not infer success.'}
    return [int]$Status
}
