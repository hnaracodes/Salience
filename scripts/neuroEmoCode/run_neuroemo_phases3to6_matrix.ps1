$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root
$env:PYTHONWARNINGS = "ignore::RuntimeWarning"

$py = Join-Path $Root ".venv\Scripts\python.exe"
$runner = Join-Path $Root "scripts\run_neuroemo_experiment_matrix.py"
$log = Join-Path $Root "scout_data\neuroEmoCode\models\phases3to6_matrix_run.log"

New-Item -ItemType Directory -Path (Split-Path $log) -Force | Out-Null
"=== phases 3-6 matrix started $(Get-Date -Format o) ===" | Set-Content -Path $log -Encoding utf8

$phases = @("timing-grid", "dynamic-features", "preprocessing-roi", "structured-decoding")
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    foreach ($phase in $phases) {
        Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START phase=$phase" -Encoding utf8
        & $py $runner --phase $phase --execute --skip-existing 2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) {
            Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAIL phase=$phase exit=$LASTEXITCODE" -Encoding utf8
            exit $LASTEXITCODE
        }
        Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE phase=$phase" -Encoding utf8
    }
} finally {
    $ErrorActionPreference = $prevEap
}

Add-Content -Path $log -Value "=== phases 3-6 matrix completed $(Get-Date -Format o) ===" -Encoding utf8
