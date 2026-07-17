# Resume Horikawa full-corpus Modal + ridge training from last checkpoint.
# Run from TribeV2/ with .venv311 active (or paths below work as-is).

Set-Location $PSScriptRoot\..\..
Write-Host "Consolidating completed Modal runs..."
.venv311\Scripts\python.exe scripts/horikawaCode/consolidate_horikawa_modal_checkpoint.py
Write-Host ""
Write-Host "To continue Modal feature extraction + train when ready:"
Write-Host "  .venv311\Scripts\python.exe scripts/horikawaCode/run_full_horikawa_training.py --skip-labels --resume"
