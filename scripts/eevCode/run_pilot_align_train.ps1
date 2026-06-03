# Wait for Modal NPZs, build aligned dataset, train SVR, LOVO eval.

$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
$env:PYTHONPATH = $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Log = Join-Path $Root "scout_data\eevCode\manifests\pilot_align_train.log"
$Pilot = Join-Path $Root "scout_data\eevCode\manifests\pilot_50.json"
$ModalOut = Join-Path $Root "scout_data\eevCode\intermediates_modal"
$DoneFlag = Join-Path $Root "scout_data\eevCode\manifests\pilot_modal.done"
$AlignedOut = Join-Path $Root "scout_data\eevCode\aligned\eev_aligned_pilot_modal_v1.npz"

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
Write-Log 'Pilot align+train waiting for Modal completion flag'

$manifest = Get-Content $Pilot -Raw | ConvertFrom-Json
$need = @($manifest.video_ids).Count
while (-not (Test-Path $DoneFlag)) {
    $npz = (Get-ChildItem $ModalOut -Filter '*_features.npz' -ErrorAction SilentlyContinue).Count
    Write-Log "Modal done flag missing; NPZs so far: $npz / $need"
    Start-Sleep -Seconds 60
}

$npz = (Get-ChildItem $ModalOut -Filter '*_features.npz' -ErrorAction SilentlyContinue).Count
if ($npz -eq 0) {
    Write-Log 'Aborting: pilot_modal.done set but intermediates_modal has zero NPZs'
    exit 1
}

Write-Log "Modal phase complete ($npz NPZs); building aligned dataset"
& $Py scripts/eevCode/build_aligned_dataset.py `
    --split train `
    --search-lag `
    --video-ids-file $Pilot `
    --intermediates-dir $ModalOut `
    --output $AlignedOut 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Align exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Log 'Training Multi-Output SVR on pilot Modal features'
& $Py scripts/eevCode/train_svr.py --aligned-npz $AlignedOut --model-name eev_svr_pilot_modal_v1 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Train exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Log 'LOVO evaluation'
& $Py scripts/eevCode/evaluate_svr.py --mode lovo --aligned-npz $AlignedOut --model-path (Join-Path $Root 'scout_data\eevCode\models\eev_svr_pilot_modal_v1.joblib') --report (Join-Path $Root 'scout_data\eevCode\models\pilot_modal_report.md') 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Eval exit=$LASTEXITCODE"
Write-Log 'Pilot align+train finished'
