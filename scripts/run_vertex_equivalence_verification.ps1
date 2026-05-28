# Run unit tests + Modal vertex-equivalence proof (tribe.py local entrypoint).
# Logs: scout_data/neuroemo/vertex_equivalence_run.log
#       scout_data/neuroemo/vertex_equivalence_modal.log  (Modal step only)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

$Log = Join-Path $Root "scout_data\neuroemo\vertex_equivalence_run.log"
$ModalLog = Join-Path $Root "scout_data\neuroemo\vertex_equivalence_modal.log"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Modal = Join-Path $Root ".venv\Scripts\modal.exe"
$BoldRel = "scout_data\neuroemo\raw\sub-01\func\sub-01_task-fe_bold.nii.gz"
$Bold = Join-Path $Root $BoldRel

function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line -Encoding utf8
}

New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
"=== vertex equivalence run started $(Get-Date -Format o) ===" | Set-Content -Path $Log -Encoding utf8
"" | Set-Content -Path $ModalLog -Encoding utf8

if (-not (Test-Path $Py)) { throw "Missing venv python: $Py" }
if (-not (Test-Path $Bold)) { throw "Missing BOLD file: $Bold" }
if (-not (Test-Path $Modal)) { throw "Missing modal CLI: $Modal" }

Write-Log "Step 1/2: pytest tests/test_vertex_equivalence.py"
& $Py -m pytest tests/test_vertex_equivalence.py -v --tb=short 2>&1 | ForEach-Object { Write-Log $_ }
if ($LASTEXITCODE -ne 0) { throw "pytest failed with exit code $LASTEXITCODE" }

Write-Log "Step 2/2: modal run tribe.py::verify_vertex_equivalence"
Write-Log "BOLD: $Bold"
Write-Log "Modal log: $ModalLog"

$modalArgs = @(
    "run", "tribe.py::verify_vertex_equivalence",
    "--bold-path", "scout_data/neuroemo/raw/sub-01/func/sub-01_task-fe_bold.nii.gz",
    "--output", "scout_data/neuroemo/vertex_equivalence_report.json",
    "--require-status", "projection_equivalence_verified"
)

Write-Log "Command: $Modal $($modalArgs -join ' ')"

# UTF-8 console: Modal CLI prints Unicode (e.g. checkmarks) that break Windows charmap pipes.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try { chcp 65001 | Out-Null } catch { }
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

# Modal writes DeprecationWarnings to stderr; with $ErrorActionPreference Stop that aborts the script.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Modal @modalArgs 2>&1 | Tee-Object -FilePath $ModalLog | ForEach-Object { Write-Log "[modal] $_" }
    $procExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $prevEap
}

if ($procExit -ne 0) {
    Write-Log "Modal exited with code $procExit. See $ModalLog"
    throw "Modal verifier failed with exit code $procExit"
}

Write-Log "Done. Report: scout_data/neuroemo/vertex_equivalence_report.json"
exit 0
