# Offline launcher regression. Docker/Git are mocks; no GPU/build/network runs.
$ErrorActionPreference = 'Stop'
$Repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$Fixture = Join-Path ([IO.Path]::GetTempPath()) ('ts-' + [Guid]::NewGuid().ToString('N').Substring(0,6))
$Checkout = Join-Path $Fixture 'repo'
$StudyRoot = Join-Path $Fixture 'r'
$Utf8 = New-Object System.Text.UTF8Encoding($false)
function Save-Fixture([string]$Path, [string]$Text) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Path -Parent) | Out-Null
    [IO.File]::WriteAllText($Path,$Text,$Utf8)
}
New-Item -ItemType Directory -Force -Path (Join-Path $Checkout '.git') | Out-Null
foreach ($File in @('scripts\run_teacher_student_resumable.ps1','scripts\experiment_local_runtimes.ps1',
    'cotracker-algorithm\ext\co-tracker\train_on_real_data.py')) {
    $Destination = Join-Path $Checkout $File
    New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
    Copy-Item -LiteralPath (Join-Path $Repo $File) -Destination $Destination
}
foreach ($Name in @('scaled_offline.pth','baseline_offline.pth','baseline_online.pth','cotracker2v1.pth')) {
    Save-Fixture (Join-Path $Checkout "training-checkpoints\$Name") "fixture:$Name"
}
Save-Fixture (Join-Path $Checkout 'scripts\run_hierarchical_followup_resumable.ps1') @'
param($Repository,$Dataset,$Results,$Profiles,$ModelCheckpoint,[switch]$RequireDiagnostics,
      [switch]$FreezeImages,[switch]$UsePinnedImages,$ReferenceImagesPath)
$Folder=Join-Path $Results $Profiles[0]
New-Item -ItemType Directory -Force -Path $Folder | Out-Null
$Dsc=if ($Results -like '*pretrained') {0.88} else {0.89}
[IO.File]::WriteAllText((Join-Path $Folder 'metrics.json'),('{"aggregates":{"dice_similarity_coefficient":'+$Dsc+'}}'))
'@
$Datasets = @{}
foreach ($Split in @(@{Name='train';Count=40},@{Name='validation';Count=10},@{Name='test';Count=38})) {
    $Dataset = Join-Path $Fixture $Split.Name
    $Datasets[$Split.Name] = $Dataset
    for ($i=0; $i -lt $Split.Count; $i++) {
        $Case = '{0}_{1:d3}' -f $Split.Name,$i
        foreach ($File in @("images\${Case}_frames.mha","targets\${Case}_first_label.mha",
            "targets\${Case}_labels.mha",'frame-rate.json','b-field-strength.json','scanned-region.json')) {
            Save-Fixture (Join-Path $Dataset "$Case\$File") 'fixture-not-medical-data'
        }
    }
}
$global:TsCalls = New-Object System.Collections.Generic.List[object]
$global:TsFixtureCheckout = $Checkout
function global:git {
    $global:LASTEXITCODE=0
    'scripts/run_teacher_student_resumable.ps1'
    'scripts/experiment_local_runtimes.ps1'
    'cotracker-algorithm/ext/co-tracker/train_on_real_data.py'
}
function global:docker {
    $a = @($args)
    $global:TsCalls.Add($a)
    $global:LASTEXITCODE=0
    if ($a[0] -eq 'image' -and $a[1] -eq 'inspect') { return ('sha256:' + ('a'*64)) }
    if ($a[0] -ne 'run') { return }
    if ($a -contains '--network' -and $a[([array]::IndexOf($a,'--network')+1)] -ne 'none') { throw 'Network was enabled.' }
    $Mount = @($a | Where-Object { $_ -like 'type=bind,source=*,target=/study' })
    if ($Mount.Count) {
        $Output = $Mount[0].Substring('type=bind,source='.Length).Replace(',target=/study','')
        if ($a -contains 'runtime-proof') {
            $Sha = (Get-FileHash -LiteralPath (Join-Path $global:TsFixtureCheckout 'cotracker-algorithm\ext\co-tracker\train_on_real_data.py')).Hash.ToLowerInvariant()
            [IO.File]::WriteAllText((Join-Path $Output 'runtime-proof.json'), ('{"complete":true,"trainer_sha256":"'+$Sha+'"}'))
        } elseif ($a -contains 'gpu-proof') {
            [IO.File]::WriteAllText((Join-Path $Output 'gpu-teacher-proof.json'),' {"complete":true}')
        } elseif ($a -contains 'student-smoke') {
            if ($a -notcontains '/opt/app/.pixi/envs/cuda/bin/python') { throw 'Inference smoke did not use inference runtime.' }
            [IO.File]::WriteAllText((Join-Path $Output 'gpu-student-inference-proof.json'),' {"complete":true}')
        } elseif ($a -contains 'audit') {
            $Steps = $a[([array]::IndexOf($a,'--steps')+1)]
            [IO.File]::WriteAllText((Join-Path $Output 'training-audit.json'),('{"complete":true,"steps":'+$Steps+'}'))
        } elseif ($a -contains 'report') {
            $Split=$a[([array]::IndexOf($a,'--split')+1)]
            $Steps=[int]$a[([array]::IndexOf($a,'--steps')+1)]
            $Seeds=@($a[([array]::IndexOf($a,'--seeds')+1)].Split(',') | ForEach-Object { [int]$_ })
            $Folder=Join-Path $Output $Split
            $Hashes=[ordered]@{}
            foreach ($File in Get-ChildItem -LiteralPath $Folder -Recurse -Filter 'metrics.json' -File) {
                $Hashes[$File.FullName.Substring($Folder.Length+1)]=(Get-FileHash -LiteralPath $File.FullName).Hash.ToLowerInvariant()
            }
            $Proof=[ordered]@{complete=$true;split=$Split;steps=$Steps;seeds=$Seeds;policies=@('old','scaled','grid0','iter6');metrics_sha256=$Hashes}
            [IO.File]::WriteAllText((Join-Path $Folder 'teacher-student-audit.json'),(ConvertTo-Json -InputObject $Proof -Depth 20))
        }
        return
    }
    if ($a -contains '/opt/app/ext/co-tracker/train_on_real_data.py') {
        if ($a -notcontains '--teacher_student_study' -or $a -notcontains '/usr/bin/timeout' -or $a -contains 'mamba') { throw 'Wrong training command.' }
        $Policy=$a[([array]::IndexOf($a,'--offline_teacher_policy')+1)]
        $Weight=$a[([array]::IndexOf($a,'--teacher_offline_ckpt')+1)]
        $Expected=if ($Policy -eq 'old') {'/checkpoints/baseline_offline.pth'} else {'/checkpoints/scaled_offline.pth'}
        if ($Weight -ne $Expected) { throw 'Wrong offline teacher weight.' }
        $Mount=@($a | Where-Object { $_ -like 'type=bind,source=*,target=/results' })[0]
        $Output=$Mount.Substring('type=bind,source='.Length).Replace(',target=/results','')
        [IO.File]::WriteAllText((Join-Path $Output 'cotracker_three_final.pth'),'mock-complete')
    }
}
function Invoke-Fixture([string]$Stage) {
    & (Join-Path $Checkout 'scripts\run_teacher_student_resumable.ps1') -RepoRoot $Checkout -ResultsRoot $StudyRoot `
        -TrainDataset $Datasets.train -ValidationDataset $Datasets.validation -TestDataset $Datasets.test -Stage $Stage
}
function Expect-Failure([scriptblock]$Action, [string]$Message) {
    $Failed=$false
    try { & $Action } catch { $Failed=$true; if ($_.Exception.Message -notlike "*$Message*") { throw } }
    if (-not $Failed) { throw "Expected failure: $Message" }
}
try {
    Invoke-Fixture 'plan'
    if ($global:TsCalls.Count) { throw 'Plan unexpectedly called Docker.' }
    Expect-Failure { Invoke-Fixture 'pilot' } 'Required completed stage missing'
    if ($global:TsCalls.Count) { throw 'Missing smoke unexpectedly launched Docker.' }
    Invoke-Fixture 'smoke'
    $TrainingCalls=@($global:TsCalls | Where-Object { $_ -contains '--offline_teacher_policy' })
    if ($TrainingCalls.Count -ne 4) { throw "Wrong smoke training count: $($TrainingCalls.Count)" }
    foreach ($Call in $TrainingCalls) {
        if ($Call[([array]::IndexOf($Call,'--num_steps')+1)] -ne '5') { throw 'Smoke started a large run.' }
        if (($Call -join '|') -like '*target=/data*test*') { throw 'Test data was mounted for training.' }
    }
    # A second smoke may redo the GPU proof but must not retrain completed arms.
    Invoke-Fixture 'smoke'
    if (@($global:TsCalls | Where-Object { $_ -contains '--offline_teacher_policy' }).Count -ne 4) { throw 'Resume retrained completed arms.' }
    Expect-Failure { Invoke-Fixture 'train' } 'Required completed stage missing'
    Invoke-Fixture 'pilot'
    if (@($global:TsCalls | Where-Object { $_ -contains '--offline_teacher_policy' }).Count -ne 8) { throw 'Pilot did not run exactly four new arms.' }
    Invoke-Fixture 'train'
    $BeforeEvaluation=@($global:TsCalls | Where-Object { $_ -contains '--offline_teacher_policy' }).Count
    if ($BeforeEvaluation -ne 20) { throw 'Full stage did not run exactly twelve new arms.' }
    Invoke-Fixture 'validation'
    Invoke-Fixture 'test'
    if (@($global:TsCalls | Where-Object { $_ -contains '--offline_teacher_policy' }).Count -ne $BeforeEvaluation) { throw 'Evaluation unexpectedly started training.' }
    # Modified source must be rejected, rather than accepting cached checkpoints.
    [IO.File]::AppendAllText((Join-Path $Checkout 'cotracker-algorithm\ext\co-tracker\train_on_real_data.py'),"`n# changed fixture")
    Expect-Failure { Invoke-Fixture 'smoke' } 'Frozen protocol changed'
    Write-Host "PASS: PS5.1 plan/smoke/pilot/full/validation/test, stage gates, timeout/network guards, resume and source freeze. Docker/inference were mocks. Fixture retained: $Fixture"
} finally {
    Remove-Item Function:\docker -ErrorAction SilentlyContinue
    Remove-Item Function:\git -ErrorAction SilentlyContinue
}
