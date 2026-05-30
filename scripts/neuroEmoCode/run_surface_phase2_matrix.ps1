$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
$out = Join-Path $Root "scout_data\neuroEmoCode\models\2026-05-27_surface_annot_matrix_phase2"
$log = Join-Path $out "matrix_run.log"

New-Item -ItemType Directory -Path $out -Force | Out-Null
"=== matrix run started $(Get-Date -Format o) ===" | Set-Content -Path $log -Encoding utf8

$models = @("logistic_saga", "sgd_logistic", "linear_svc")
foreach ($m in $models) {
    Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $m" -Encoding utf8
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $py scripts/neuroEmoCode/train_neuroemo_emotion_model.py `
            --train-npz scout_data/neuroEmoCode/tribev2_surface/neuroemo_tribev2_train.npz `
            --model-type $m `
            --temporal-window-trs 10 `
            --temporal-contiguity contiguous `
            --roi-reducers mean,std,mean_abs `
            --exclude-labels neutral `
            --metrics-json "scout_data/neuroEmoCode/models/2026-05-27_surface_annot_matrix_phase2/${m}_5class_10tr_metrics.json" `
            --output-model "scout_data/neuroEmoCode/models/2026-05-27_surface_annot_matrix_phase2/${m}_5class_10tr.joblib" 2>&1 `
            | Tee-Object -FilePath $log -Append
    } finally {
        $ErrorActionPreference = $prevEap
    }

    if ($LASTEXITCODE -ne 0) {
        Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAIL $m exit=$LASTEXITCODE" -Encoding utf8
        throw "Matrix run failed for $m with exit code $LASTEXITCODE"
    }
    Add-Content -Path $log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE $m" -Encoding utf8
}

Add-Content -Path $log -Value "=== matrix run completed $(Get-Date -Format o) ===" -Encoding utf8
