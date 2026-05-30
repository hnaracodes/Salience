# Run NeuroEmo experiment matrix phases 3-6 sequentially (see plan 2026-05-27).
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
$runner = Join-Path $Root "scripts\neuroEmoCode\run_neuroemo_experiment_matrix.py"
$log = Join-Path $Root "scout_data\neuroEmoCode\phases3-6_matrix_run.log"

function Write-Log([string]$Message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

New-Item -ItemType Directory -Path (Split-Path $log) -Force | Out-Null
"=== phases 3-6 matrix started $(Get-Date -Format o) ===" | Set-Content -Path $log -Encoding utf8

$phases = @(
    "timing-grid",
    "dynamic-features",
    "preprocessing-roi",
    "structured-decoding"
)

foreach ($phase in $phases) {
    Write-Log "START phase $phase --execute --skip-existing"
    & $py $runner --phase $phase --execute --skip-existing 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Log "FAIL phase $phase exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Log "DONE phase $phase; refreshing manifest"
    & $py $runner --phase all 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Log "FAIL manifest refresh after $phase exit=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Write-Log "=== phases 3-6 matrix completed $(Get-Date -Format o) ==="
