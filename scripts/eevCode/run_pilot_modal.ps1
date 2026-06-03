# Modal TRIBE inference for pilot 50 -> scout_data/eevCode/intermediates_modal/
# Safe to re-run: uses --skip-existing; PowerShell lock + Python file lock in generate_features.py.

$ErrorActionPreference = "Continue"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $Root
$env:PYTHONPATH = $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Log = Join-Path $Root "scout_data\eevCode\manifests\pilot_modal.log"
$Pilot = Join-Path $Root "scout_data\eevCode\manifests\pilot_50.json"
$Videos = Join-Path $Root "scout_data\eevCode\videos"
$ModalOut = Join-Path $Root "scout_data\eevCode\intermediates_modal"
$DoneFlag = Join-Path $Root "scout_data\eevCode\manifests\pilot_modal.done"
$Lock = Join-Path $Root "scout_data\eevCode\manifests\pilot_modal.lock"
$DownloadDone = Join-Path $Root "scout_data\eevCode\manifests\pilot_download.done"

function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

function Get-NpzCount {
    return (Get-ChildItem $ModalOut -Filter '*_features.npz' -ErrorAction SilentlyContinue).Count
}

New-Item -ItemType Directory -Force -Path $ModalOut | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Log -Parent) | Out-Null

if (Test-Path $Lock) {
    Write-Log 'Another Modal worker holds pilot_modal.lock; exiting to avoid duplicate charges'
    exit 0
}

if (Test-Path $DoneFlag) { Remove-Item $DoneFlag -Force }
Set-Content -Path $Lock -Value "$PID $(Get-Date -Format 'o')"

try {
    Write-Log 'Pilot Modal inference start'
    Write-Log 'Do not start a second generate_features worker while this runs'
    Write-Log "Output directory: $ModalOut"

    $manifest = Get-Content $Pilot -Raw | ConvertFrom-Json
    $need = @($manifest.video_ids).Count

    if (-not (Test-Path $DownloadDone)) {
        Write-Log "Waiting for download completion flag: $DownloadDone"
        while (-not (Test-Path $DownloadDone)) {
            $n = (Get-ChildItem $Videos -Filter '*.mp4' -ErrorAction SilentlyContinue).Count
            Write-Log "Download in progress; videos on disk: $n / $need"
            Start-Sleep -Seconds 30
        }
    }

    $n = (Get-ChildItem $Videos -Filter '*.mp4' -ErrorAction SilentlyContinue).Count
    Write-Log "Running Modal on $n available videos (pilot list has $need); existing NPZs: $(Get-NpzCount)"

    & $Py scripts/eevCode/generate_features.py `
        --video-ids-file $Pilot `
        --output-dir $ModalOut `
        --execute-tribe `
        --skip-existing
    $exitCode = $LASTEXITCODE

    $npz = Get-NpzCount
    Write-Log "Modal inference finished exit=$exitCode npz=$npz"

    if ($exitCode -ne 0 -or $npz -eq 0) {
        Write-Log 'Not writing pilot_modal.done (no successful feature NPZs)'
        exit 1
    }

    Set-Content -Path $DoneFlag -Value (Get-Date -Format 'o')
    Write-Log "Wrote done flag: $DoneFlag (npz=$npz)"
    exit 0
}
finally {
    if (Test-Path $Lock) { Remove-Item $Lock -Force -ErrorAction SilentlyContinue }
}
