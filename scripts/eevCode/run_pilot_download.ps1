# Re-download pilot videos with single-file mp4 format (no ffmpeg merge).

$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
$env:PYTHONPATH = $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Log = Join-Path $Root "scout_data\eevCode\manifests\pilot_download.log"
$Pilot = Join-Path $Root "scout_data\eevCode\manifests\pilot_50.json"

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null
Write-Log 'Pilot download start'
& $Py scripts/eevCode/download_videos.py --download --video-ids-file $Pilot 2>&1 | ForEach-Object { Write-Log $_ }
Write-Log "Pilot download finished exit=$LASTEXITCODE"
$statusPath = Join-Path $Root "scout_data\eevCode\manifests\download_status.json"
$ok = 0
if (Test-Path $statusPath) {
    $rows = Get-Content $statusPath -Raw | ConvertFrom-Json
    $ok = @($rows | Where-Object { $_.status -in @('ok','skipped') }).Count
}
Write-Log "Download summary ok/skipped=$ok"
$DoneFlag = Join-Path $Root "scout_data\eevCode\manifests\pilot_download.done"
Set-Content -Path $DoneFlag -Value (Get-Date -Format 'o')
Write-Log "Wrote done flag: $DoneFlag"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
