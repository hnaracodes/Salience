# EEV pilot end-to-end: download -> Modal TRIBE -> align -> train -> LOVO eval
# Scoped to pilot_50.json only (does not touch full_split manifest).

$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
$env:PYTHONPATH = $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Log = Join-Path $Root "scout_data\eevCode\manifests\pilot_e2e.log"
$Pilot = Join-Path $Root "scout_data\eevCode\manifests\pilot_50.json"
$Intermediates = Join-Path $Root "scout_data\eevCode\intermediates"

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
Write-Log 'EEV pilot E2E start (50 videos max)'

if (-not (Test-Path $Pilot)) {
    Write-Log "ERROR: missing pilot manifest at $Pilot"
    exit 1
}

$manifest = Get-Content $Pilot -Raw | ConvertFrom-Json
foreach ($id in $manifest.video_ids) {
    $npz = Join-Path $Intermediates "${id}_features.npz"
    if (Test-Path $npz) {
        Remove-Item $npz -Force
        Write-Log "cleared stale feature NPZ: $id"
    }
}

Write-Log 'Step 1/5: download pilot videos (yt-dlp)'
& $Py scripts/eevCode/download_videos.py --download --video-ids-file $Pilot 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Step 1 exit code: $LASTEXITCODE"

Write-Log 'Step 2/5: Modal TRIBE features'
& $Py scripts/eevCode/generate_features.py --video-ids-file $Pilot --execute-tribe --skip-existing 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Step 2 exit code: $LASTEXITCODE"

Write-Log 'Step 3/5: align + lag search (train split, pilot IDs only)'
& $Py scripts/eevCode/build_aligned_dataset.py --split train --search-lag --video-ids-file $Pilot 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Step 3 exit code: $LASTEXITCODE"

Write-Log 'Step 4/5: train Multi-Output SVR'
& $Py scripts/eevCode/train_svr.py 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Step 4 exit code: $LASTEXITCODE"

Write-Log 'Step 5/5: LOVO evaluation + pilot_report.md'
& $Py scripts/eevCode/evaluate_svr.py --mode lovo 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Step 5 exit code: $LASTEXITCODE"

Write-Log 'EEV pilot E2E finished'
