$ErrorActionPreference='Stop'
$Temporary=Join-Path ([IO.Path]::GetTempPath()) ('teacher-test-'+[guid]::NewGuid().ToString('N').Substring(0,8))
$Study=Join-Path $Temporary 'study'
$Protocol=Join-Path $Temporary 'protocol'
$Weights=Join-Path $Temporary 'weights'
$Train=Join-Path $Temporary 'train'
$Validation=Join-Path $Temporary 'validation'
$Test=Join-Path $Temporary 'test'
$Output=Join-Path $Temporary 'out'
foreach ($Directory in @($Study,$Protocol,$Weights,$Train,$Validation,$Test)) {New-Item -ItemType Directory -Force -Path $Directory | Out-Null}
$Image='sha256:'+('1'*64)
[IO.File]::WriteAllText((Join-Path $Study 'local-runtime-images.json'),('{"Inference":"'+$Image+'","Evaluation":"'+$Image+'"}'))
$Global:TeacherCalls=New-Object System.Collections.Generic.List[object]
$Global:TeacherOutput=$Output
$Global:FailTeacher=$false
function global:docker {
    $Global:LASTEXITCODE=0
    if ($args[0] -eq 'image') {'sha256:'+('1'*64)}
}
function global:Start-Process {
    param($FilePath,$ArgumentList,$WindowStyle,[switch]$PassThru,$RedirectStandardOutput,$RedirectStandardError)
    if ($FilePath -ne 'docker.exe' -or $WindowStyle -ne 'Hidden') {throw 'Unexpected launch'}
    $Arguments=@($ArgumentList | ForEach-Object {$_.Trim('"')})
    $Global:TeacherCalls.Add($Arguments)
    [IO.File]::WriteAllText($RedirectStandardOutput,'Mock teacher job: no model/GPU executed.')
    [IO.File]::WriteAllText($RedirectStandardError,'')
    $Index=[Array]::IndexOf($Arguments,'--action')
    if ($Index -ge 0) {
        $Action=$Arguments[$Index+1]
        if ($Action -eq 'plan') {
            $Plan='{"profiles":[{"Profile":"old","Group":"weights","Parent":null},{"Profile":"scaled","Group":"weights","Parent":"old"},{"Profile":"grid0","Group":"core","Parent":"scaled"}]}'
            [IO.File]::WriteAllText((Join-Path $Global:TeacherOutput 'plan.json'),$Plan)
        }
        if ($Action -eq 'infer') {
            $Profile=$Arguments[([Array]::IndexOf($Arguments,'--profile')+1)]
            $Split=$Arguments[([Array]::IndexOf($Arguments,'--split')+1)]
            $Folder=Join-Path $Global:TeacherOutput ($Split+'\'+$Profile)
            New-Item -ItemType Directory -Force -Path (Join-Path $Folder 'evaluation') | Out-Null
            [IO.File]::WriteAllText((Join-Path $Folder 'evaluation-ready.json'),'{"needs_evaluation":true}')
        }
    }
    $Exit=if ($Global:FailTeacher -and $Index -ge 0 -and $Arguments[$Index+1] -eq 'infer') {7} else {0}
    $Result=New-Object PSObject -Property @{ExitCode=$Exit;Handle=1234}
    $Result | Add-Member ScriptMethod WaitForExit {param($Milliseconds) return $true}
    $Result | Add-Member ScriptMethod Refresh {throw 'Do not discard cached exit state'}
    return $Result
}
try {
    $Parameters=@{ProtocolRoot=$Protocol;SourceStudy=$Study;Weights=$Weights;TrainDataset=$Train;ValidationDataset=$Validation;TestDataset=$Test;OutputRoot=$Output}
    & (Join-Path $PSScriptRoot 'run_teacher_experiments.ps1') @Parameters -Stage validation -Group core
    $Actions=@()
    foreach ($Arguments in $Global:TeacherCalls) {
        if ($Arguments -contains 'build' -or $Arguments -contains 'pull' -or $Arguments -contains '--privileged') {throw 'Unexpected broad action'}
        if (-not ($Arguments -contains 'none') -or -not ($Arguments -contains 'never')) {throw 'Network/pull guard missing'}
        if (-not ($Arguments -contains '/usr/bin/timeout')) {throw 'Internal timeout missing'}
        $Index=[Array]::IndexOf($Arguments,'--action')
        if ($Index -ge 0) {
            $Action=$Arguments[$Index+1]
            $Actions+= $Action
            if (@($Arguments | Where-Object {$_ -like 'type=bind,*readonly'}).Count -ne 7) {throw 'Input mount not read-only'}
            if (($Arguments -contains '--gpus') -ne ($Action -eq 'infer')) {throw 'Unexpected GPU job'}
        } else {
            if ($Arguments -contains '--gpus') {throw 'Evaluator must run on CPU'}
            if (@($Arguments | Where-Object {$_ -like 'type=bind,*readonly'}).Count -ne 2) {throw 'Evaluator input mount changed'}
        }
    }
    if (($Actions -join ',') -ne 'plan,infer,record,infer,record,infer,record,report') {throw 'Ancestor order/evaluation handoff failed'}
    if ($Global:TeacherCalls.Count -ne 11) {throw 'Unexpected job count'}
    [IO.File]::WriteAllText((Join-Path $Output 'selection.json'),'{"test_profiles":["old","scaled"]}')
    $Global:TeacherCalls.Clear()
    & (Join-Path $PSScriptRoot 'run_teacher_experiments.ps1') @Parameters -Stage test
    $GPU=@($Global:TeacherCalls | Where-Object {$_ -contains '--gpus'})
    if ($GPU.Count -ne 2) {throw 'Test must only run frozen shortlist'}
    $Rejected=$false
    try {& (Join-Path $PSScriptRoot 'run_teacher_experiments.ps1') @Parameters -Stage validation} catch {$Rejected=$_.Exception.Message -like '*already frozen*'}
    if (-not $Rejected) {throw 'Validation after selection must stop'}
    $Unsafe=@{} + $Parameters
    $Unsafe.OutputRoot=Join-Path $Study 'unsafe'
    $Rejected=$false
    try {& (Join-Path $PSScriptRoot 'run_teacher_experiments.ps1') @Unsafe} catch {$Rejected=$_.Exception.Message -like '*overlap*'}
    if (-not $Rejected) {throw 'Input/output overlap not rejected'}
    $Global:FailTeacher=$true
    $Global:TeacherCalls.Clear()
    $Rejected=$false
    try {& (Join-Path $PSScriptRoot 'run_teacher_experiments.ps1') @Parameters -Stage test} catch {$Rejected=$_.Exception.Message -like '*exit 7*'}
    if (-not $Rejected -or $Global:TeacherCalls.Count -ne 2) {throw 'Failure did not stop before evaluator/next teacher'}
    Write-Host 'PASS mocked teacher launcher: pinned/readonly/offline, ancestry, explicit shortlist, stop-on-failure. No Docker/GPU run.'
} finally {
    Remove-Item Function:\docker
    Remove-Item Function:\Start-Process
    Write-Host "Fixture retained: $Temporary"
}
. (Join-Path $PSScriptRoot 'native_process.ps1')
$Shell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
foreach ($Expected in @(0,7)) {
    $Child=Start-Process -FilePath $Shell -ArgumentList @('-NoProfile','-Command',('"Write-Output child-ok; exit '+$Expected+'"')) `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Temporary "exit-$Expected.out") `
        -RedirectStandardError (Join-Path $Temporary "exit-$Expected.err")
    $null=$Child.Handle
    $Watch=[Diagnostics.Stopwatch]::StartNew()
    while (-not $Child.WaitForExit(200)) {if ($Watch.Elapsed.TotalSeconds -gt 10) {throw 'Exit test exceeded time budget'}}
    $Actual=Get-TeacherExitCode $Child
    if ($Actual -ne $Expected) {throw "Exit code lost: expected=$Expected actual=$Actual"}
    Write-Host "PASS real Windows child exit=$Actual"
}
