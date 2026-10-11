# Called only inside the frozen, source/data/weight verified compact queue.
function Invoke-SidecarStabilityTasks {
    if (-not $CompactSoak -or -not $CompactRuntime) { throw 'Missing explicit compact soak gate.' }
    foreach ($Kind in @('mlp','mamba')) {
        foreach ($Task in @(@{Group='training';Steps=100;Resume=$false},@{Group='resumed';Steps=50;Resume=$false},@{Group='resumed';Steps=100;Resume=$true})) {
            $Folder=Join-Path $ResultsRoot "$($Task.Group)\seed_0\$Kind"
            New-Item -ItemType Directory -Force -Path $Folder | Out-Null
            $Output=Join-Path $Folder 'phase-profile.json'
            if (Test-Path -LiteralPath $Output) {
                $Previous=Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
                if (-not $Previous.complete) { throw 'Partial soak retained. Use a new ResultsRoot; do not mix interrupted diagnostic updates.' }
                if ($Previous.diagnostic_steps -eq $Task.Steps -or ($Task.Steps -eq 50 -and $Previous.diagnostic_steps -eq 100)) {
                    Write-Host "SKIP completed $($Task.Group)/$Kind segment $($Task.Steps)"; continue
                }
                if (-not $Task.Resume -or $Previous.diagnostic_steps -ne 50) { throw 'Unexpected soak process boundary.' }
            } elseif ($Task.Resume) { throw 'Complete the 50-step segment first.' }
            Write-Host "BOUNDED SOAK $($Task.Group)/$Kind to $($Task.Steps); resume=$($Task.Resume); budget=$TrainingTimeoutSeconds seconds"
            $Arguments=@('run','--rm','--pull','never','--network','none','--gpus','device=0','--shm-size','4g',
                '--mount',"type=bind,source=$CompactRuntime,target=/opt/sidecar/runtime.json,readonly",
                '--mount',"type=bind,source=$($Datasets['train-40']),target=/data,readonly",
                '--mount',"type=bind,source=$RepoRoot\training-checkpoints,target=/checkpoints,readonly",
                '--mount',"type=bind,source=$Folder,target=/results",'--entrypoint','/usr/bin/timeout',$Images.Training,
                '--signal=TERM','--kill-after=10s',"${TrainingTimeoutSeconds}s",
                '/opt/app/.pixi/envs/training/bin/python','/opt/app/experiments/profile_sidecar_training_phases.py',
                '--kind',$Kind,'--steps',([string]$Task.Steps),'--soak','--dataset','/data','--checkpoints','/checkpoints',
                '--output','/results','--trainer-sha256',$TrainerSHA256,'--runtime-config','/opt/sidecar/runtime.json')
            if ($Task.Resume) { $Arguments+='--resume-50' }
            $Status=Invoke-BudgetDocker $Arguments (Join-Path $Folder 'training.log')
            if ($Status -ne 0) {
                Write-DiagnosticJson (Join-Path $Folder 'timeout.json') ([ordered]@{complete=$false;exit_code=$Status;budget_seconds=$TrainingTimeoutSeconds})
                throw 'Soak exceeded process budget or was killed. Preserve partial evidence; no large training will run.'
            }
            if (-not (Test-Path -LiteralPath $Output)) { throw 'Missing soak phase report.' }
            $Completed=Get-Content -LiteralPath $Output -Raw | ConvertFrom-Json
            if (-not $Completed.complete -or $Completed.diagnostic_steps -ne $Task.Steps) { throw 'Wrong soak update count.' }
        }
    }
}
