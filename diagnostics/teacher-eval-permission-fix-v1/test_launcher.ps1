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
$OriginalPackage=Join-Path (Split-Path -Parent $PSScriptRoot) 'teacher-single-factor-v1'
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
            [IO.File]::WriteAllText((Join-Path $Global:TeacherOutput 'manifest.json'),'{"mock":1}')
        }
        if ($Action -eq 'infer') {
            $Profile=$Arguments[([Array]::IndexOf($Arguments,'--profile')+1)]
            $Split=$Arguments[([Array]::IndexOf($Arguments,'--split')+1)]
            $Folder=Join-Path $Global:TeacherOutput ($Split+'\'+$Profile)
            New-Item -ItemType Directory -Force -Path (Join-Path $Folder 'evaluation') | Out-Null
            [IO.File]::WriteAllText((Join-Path $Folder 'evaluation-ready.json'),'{"needs_evaluation":true}')
        }
    }
    if ($Index -ge 0 -and $Arguments[$Index+1] -eq 'prepare') {
        $Profile=$Arguments[([Array]::IndexOf($Arguments,'--profile')+1)]
        $Split=$Arguments[([Array]::IndexOf($Arguments,'--split')+1)]
        $Folder=Join-Path $Global:TeacherOutput ('permission-repair\'+$Split+'\'+$Profile)
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        $ReceiptPath=Join-Path $Folder 'preflight.json'
        $ManifestSHA=(Get-FileHash -LiteralPath (Join-Path $Global:TeacherOutput 'manifest.json')).Hash.ToLowerInvariant()
        [IO.File]::WriteAllText($ReceiptPath,('{"owner_uid":997,"owner_gid":998,"original_evaluator_runtime":{"identity":{"uid":999,"gid":999}},"manifest_sha256":"'+$ManifestSHA+'"}'))
        $ReceiptSHA=(Get-FileHash -LiteralPath $ReceiptPath).Hash.ToLowerInvariant()
        [IO.File]::WriteAllText((Join-Path $Folder 'latest.json'),('{"receipt":"permission-repair/'+$Split+'/'+$Profile+'/preflight.json","sha256":"'+$ReceiptSHA+'"}'))
    }
    if ($Index -ge 0 -and $Arguments[$Index+1] -eq 'runtime') {
        [IO.File]::WriteAllText($RedirectStandardOutput,'TEACHER_EVALUATOR_RUNTIME {"identity":{"uid":999,"gid":999},"modules":{"psutil":{}}}')
    }
    if ($Index -ge 0 -and $Arguments[$Index+1] -eq 'evaluate') {
        [IO.File]::WriteAllText($RedirectStandardOutput,'TEACHER_METRICS_JSON {"schema":2,"type":"teacher-official-metrics","metrics_json":"{\n\"results\":[]}"}')
    }
    $Exit=if ($Global:FailTeacher -and $Index -ge 0 -and $Arguments[$Index+1] -eq 'infer') {7} else {0}
    $Result=New-Object PSObject -Property @{ExitCode=$Exit;Handle=1234}
    $Result | Add-Member ScriptMethod WaitForExit {param($Milliseconds) return $true}
    $Result | Add-Member ScriptMethod Refresh {throw 'Do not discard cached exit state'}
    return $Result
}
try {
    $Parameters=@{ProtocolRoot=$Protocol;SourceStudy=$Study;Weights=$Weights;TrainDataset=$Train;ValidationDataset=$Validation;TestDataset=$Test;OutputRoot=$Output;PackageRoot=$OriginalPackage}
    & (Join-Path $PSScriptRoot 'run_repair.ps1') @Parameters -Stage validation -Group core
    $Actions=@()
    foreach ($Arguments in $Global:TeacherCalls) {
        if ($Arguments -contains 'build' -or $Arguments -contains 'pull' -or $Arguments -contains '--privileged') {throw 'Unexpected broad action'}
        if (-not ($Arguments -contains 'none') -or -not ($Arguments -contains 'never')) {throw 'Network/pull guard missing'}
        if (-not ($Arguments -contains '/usr/bin/timeout')) {throw 'Internal timeout missing'}
        $Index=[Array]::IndexOf($Arguments,'--action')
        if ($Index -lt 0) {throw 'All repair jobs must specify an explicit action'}
        $Action=$Arguments[$Index+1]
        $Actions+= $Action
        if (($Arguments -contains '--gpus') -ne ($Action -eq 'infer')) {throw 'Unexpected GPU job'}
        $ReadOnly=@($Arguments | Where-Object {$_ -like 'type=bind,*readonly'}).Count
        if ($Action -eq 'evaluate') {
            if ($ReadOnly -ne 5) {throw 'Evaluation input/source/receipt/manifest mounts must be read-only'}
            $UserIndex=[Array]::IndexOf($Arguments,'--user')
            if ($UserIndex -ge 0 -or $Arguments -contains 'HOME=/home/user' -or $Arguments -contains 'PYTHONUSERBASE=/home/user/.local') {throw 'Evaluator original identity/dependency environment changed'}
            if ($Arguments -notcontains '/output:rw,nosuid,nodev,size=128m,mode=0700,uid=999,gid=999') {throw 'Evaluator output must be private original-user scratch'}
            if (@($Arguments | Where-Object {$_ -like 'type=bind,*/output*'}).Count -ne 0) {throw 'Evaluator must not write host output directly'}
            foreach ($Guard in @('--read-only','--cap-drop','ALL','no-new-privileges')) {
                if ($Arguments -notcontains $Guard) {throw "Missing evaluator guard: $Guard"}
            }
            if ($Arguments -notcontains '/tmp:rw,nosuid,nodev,size=256m,mode=1777') {throw 'Restricted writable temp directory missing'}
        } elseif ($Action -eq 'runtime') {
            if ($ReadOnly -ne 1 -or $Arguments -contains '--user') {throw 'Original runtime audit identity changed'}
        } elseif ($Action -eq 'prepare') {
            if ($ReadOnly -ne 4) {throw 'Preflight input mounts must remain read-only'}
        } elseif ($Action -eq 'publish') {
            $UserIndex=[Array]::IndexOf($Arguments,'--user')
            if ($ReadOnly -ne 3 -or $UserIndex -lt 0 -or $Arguments[$UserIndex+1] -ne '997:998') {throw 'Only publisher adopts actual output owner'}
        } else {
            if ($ReadOnly -ne 7) {throw 'Original worker input mounts changed'}
            if ($Arguments -notcontains ('type=bind,source='+$OriginalPackage+',target=/teacher-source,readonly')) {throw 'Original frozen package was replaced by repair package'}
            if ($Arguments -contains '--user') {throw 'Inference/worker identity changed'}
        }
    }
    if (($Actions -join ',') -ne 'plan,infer,runtime,prepare,evaluate,publish,record,infer,runtime,prepare,evaluate,publish,record,infer,runtime,prepare,evaluate,publish,record,report') {throw 'Ancestor order/evaluation handoff failed'}
    if ($Global:TeacherCalls.Count -ne 20) {throw 'Unexpected job count'}
    [IO.File]::WriteAllText((Join-Path $Output 'selection.json'),'{"test_profiles":["old","scaled"]}')
    $Global:TeacherCalls.Clear()
    & (Join-Path $PSScriptRoot 'run_repair.ps1') @Parameters -Stage test
    $GPU=@($Global:TeacherCalls | Where-Object {$_ -contains '--gpus'})
    if ($GPU.Count -ne 2) {throw 'Test must only run frozen shortlist'}
    $Rejected=$false
    try {& (Join-Path $PSScriptRoot 'run_repair.ps1') @Parameters -Stage validation} catch {$Rejected=$_.Exception.Message -like '*already frozen*'}
    if (-not $Rejected) {throw 'Validation after selection must stop'}
    $Unsafe=@{} + $Parameters
    $Unsafe.OutputRoot=Join-Path $Study 'unsafe'
    $Rejected=$false
    try {& (Join-Path $PSScriptRoot 'run_repair.ps1') @Unsafe} catch {$Rejected=$_.Exception.Message -like '*overlap*'}
    if (-not $Rejected) {throw 'Input/output overlap not rejected'}
    $Global:FailTeacher=$true
    $Global:TeacherCalls.Clear()
    $Rejected=$false
    try {& (Join-Path $PSScriptRoot 'run_repair.ps1') @Parameters -Stage test} catch {$Rejected=$_.Exception.Message -like '*exit 7*'}
    if (-not $Rejected -or $Global:TeacherCalls.Count -ne 2) {throw 'Failure did not stop before evaluator/next teacher'}
    Write-Host 'PASS mocked permission repair: original-user evaluator, owner-matched publisher, tagged JSON data transport, frozen package and guards. No Docker/GPU run.'
} finally {
    Remove-Item Function:\docker
    Remove-Item Function:\Start-Process
    Write-Host "Fixture retained: $Temporary"
}
. (Join-Path $OriginalPackage 'native_process.ps1')
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
