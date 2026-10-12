# A reviewed host-only migration. Never overwrite the original frozen study.
function Confirm-SidecarPathRepair([string]$Path,$Value,[string]$RepoRoot) {
    $Raw=[IO.File]::ReadAllText($Path)
    $Old=$Raw | ConvertFrom-Json
    if (-not $Old.ReferenceDiagnosisOnly) { throw 'Only the original reference-backend degradation study is supported.' }
    $Before=[ordered]@{}; $After=[ordered]@{}
    foreach ($P in $Old.PSObject.Properties) { if ($P.Name -ne 'Source') { $Before[$P.Name]=$P.Value } }
    foreach ($Key in $Value.Keys) { if ($Key -ne 'Source') { $After[$Key]=$Value[$Key] } }
    if ((ConvertTo-Json $Before -Depth 100 -Compress) -ne (ConvertTo-Json $After -Depth 100 -Compress)) { throw 'Path repair cannot change images, weights, study settings or training provenance.' }
    $Known=@{
        'scripts/run_hierarchical_followup_resumable.ps1'=@('0F2ADA56A71C127B5805015E5D8684E58361177EA8674D3899D766A9442242C9','1FC7544389968F0B1628DD32C5FC23847BFE51CDC0D2B5788D3290DE6720FA87');
        'scripts/run_sidecar_speed_safety_resumable.ps1'=@('3234679A9AF278BD8B91945A5DC442479684693D79F12B6023C999349A14960C','AA6DCB327C22B5F9FB8EDDBB3CCD94FB069772BC01B1FC75FC69EA39DC73FCCF')
    }
    $Reviewed=@{
        'scripts/run_hierarchical_followup_resumable.ps1'='29279838599172E362384DD8C606A0C52BC26ABE0A46607FEBD3C99295B9BAB9';
        'scripts/run_sidecar_speed_safety_resumable.ps1'='39883376EAD9F40F4CAC32F0C8F307C408CA62149D95AA2E6991677058B9CD86'
    }
    $Added=@('scripts/sidecar_path_repair_provenance.ps1','scripts/repair_sidecar_output_paths.ps1',
        'scripts/tests/test_sidecar_output_path_repair.ps1','cotracker-algorithm/experiments/recover_sidecar_output.py',
        'cotracker-algorithm/tests/test_sidecar_output_recovery.py')
    $OldMap=@{}; $NewMap=@{}
    foreach ($Row in $Old.Source) { if ($OldMap.ContainsKey($Row.File)) { throw 'Duplicate original source path.' }; $OldMap[$Row.File]=$Row.SHA256 }
    foreach ($Row in $Value.Source) { if ($NewMap.ContainsKey($Row.File)) { throw 'Duplicate current source path.' }; $NewMap[$Row.File]=$Row.SHA256 }
    foreach ($Name in $OldMap.Keys) {
        if (-not $NewMap.ContainsKey($Name)) { throw "Source deleted: $Name" }
        if ($NewMap[$Name] -eq $OldMap[$Name]) { continue }
        if (-not $Known.ContainsKey($Name) -or $OldMap[$Name] -notin $Known[$Name]) { throw "Not a reviewed host-only edit: $Name" }
        $Body=[IO.File]::ReadAllText((Join-Path $RepoRoot $Name)).Replace("`r`n","`n")
        $Hash=([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($Body)))).Replace('-','')
        if ($Hash -ne $Reviewed[$Name]) { throw "Additional edits in reviewed host file: $Name" }
    }
    foreach ($Name in $NewMap.Keys) {
        if (-not $OldMap.ContainsKey($Name) -and $Name -notin $Added) { throw "Unreviewed new source: $Name" }
        if ((Get-FileHash -LiteralPath (Join-Path $RepoRoot $Name)).Hash -ne $NewMap[$Name]) { throw "Source changed during repair: $Name" }
    }
    $Certificate=[ordered]@{Schema=1;OriginalFrozenSHA256=(Get-FileHash -LiteralPath $Path).Hash;CurrentSource=$Value.Source;
        NumericalSourceUnchanged=$true;ImagesUnchanged=$true;Mode='host-path-check-and-legacy-output-recovery-only'}
    $CertificatePath=Join-Path (Split-Path -Parent $Path) 'path-repair-provenance.json'
    $Json=ConvertTo-Json $Certificate -Depth 100
    if (Test-Path -LiteralPath $CertificatePath) {
        if ([IO.File]::ReadAllText($CertificatePath) -ne $Json) { throw 'Repair provenance changed; preserve results and stop.' }
    } else {
        [IO.File]::WriteAllText("$CertificatePath.tmp",$Json,(New-Object Text.UTF8Encoding($false)))
        Move-Item -LiteralPath "$CertificatePath.tmp" -Destination $CertificatePath
    }
    Write-Host 'Reviewed host path repair accepted. Original frozen-study.json remains unchanged; old runtime images stay pinned.'
}
