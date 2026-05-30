# E2E: dual-track (engagement + activation + emotion) -> heatmaps -> analyze + grounding
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$SessionId = "e6f04960ad5f40c4805f3e6c65b850e3"
$Log = Join-Path $Root "scout_data\e2e_dual_track_grounding.log"
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

"" | Set-Content $Log
Log "E2E start session=$SessionId"

Log "Step 1/4: dual_track (engagement + activation + emotion)"
& $Py scripts/run_dual_track.py --session-id $SessionId 2>&1 | Tee-Object -FilePath $Log -Append

Log "Step 2/4: uniform heatmaps + section refresh"
& $Py scripts/extract_section_heatmaps.py --session-id $SessionId --uniform-heatmap --refresh-sections 2>&1 | Tee-Object -FilePath $Log -Append

Log "Step 3/4: analyze + website sections + marketing scores + grounding"
& $Py scripts/analyze_session.py --session-id $SessionId --norm-id synthetic_bootstrap_v1 --website --ground --with-heatmaps 2>&1 | Tee-Object -FilePath $Log -Append

Log "Step 4/4: UX viewer export"
& $Py scripts/export_ux_viewer.py --session-id $SessionId 2>&1 | Tee-Object -FilePath $Log -Append

Log "E2E complete — validating bundle"
& $Py -c @"
import json
from pathlib import Path
p = Path('scout_data/sessions/$SessionId/analysis_bundle.json')
b = json.loads(p.read_text(encoding='utf-8'))
et = b.get('engagement_track') or {}
at = b.get('activation_track') or {}
emo = b.get('emotion_track')
triggers = b.get('grounding_triggers') or []
events = b.get('events') or []
eng_ok = sum(1 for s in (et.get('scores') or []) if s is not None)
print(f'schema_version={b.get(\"schema_version\")} dual_track_v={b.get(\"dual_track_schema_version\")}')
print(f'engagement: flag={et.get(\"baseline_flag\")} scored_trs={eng_ok}/{len(et.get(\"scores\") or [])}')
print(f'activation: mode={at.get(\"comparison_mode\")} flag={at.get(\"baseline_flag\")}')
print(f'emotion: templates={len((emo or {}).get(\"template_names\") or [])}')
print(f'grounding_triggers={len(triggers)} events={len(events)}')
ms = b.get('marketing_scores') or {}
print(f'marketing_scores overall={ms.get(\"overall_score\")}')
"@ 2>&1 | Tee-Object -FilePath $Log -Append

Log "Done. Log: $Log"
