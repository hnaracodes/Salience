---
id: FEATURE-002
session_id: CS-20260620-EMOTION-ACCURACY-ADR7
status: completed
owner: agent
files:
  - scout_data/sessions/CS-20260620-BOOTSTRAP-FULL/
  - scripts/run_website_session.py
  - scripts/run_dual_track.py
---

# Bootstrap full pipeline integration run

## Goal

Prove end-to-end website pipeline with **real Modal TRIBE** (cortical + subcortical) and **bootstrap Horikawa decoder** shadow track on a real walkthrough capture — without flipping production `emotion.mode`.

## Implementation Notes

- Session: `CS-20260620-BOOTSTRAP-FULL`
- Script: `configs/walkthrough_scripts/localhost_demo.yaml` (5 TR localhost fixture)
- Decoder: `scout_models/horikawa_ridge_v1/` (`status: bootstrap`, synthetic train data)
- Stages: capture (done) → tribe → dual_track → heatmaps (uniform) → analyze → export_viewer

## Expected artifacts

| File | Check |
|------|-------|
| `preds.npz` | T×20484 |
| `preds_subcortical.npz` | T×8802 |
| `analysis_bundle.json` | `emotion_track` + `emotion_decoder_track` |
| `ux_viewer/` | export_viewer output |

## Validation

```powershell
python scripts/inspect_session.py --session-id CS-20260620-BOOTSTRAP-FULL
# Confirm bundle: parallel_decoder_enabled, emotion_mode template
```

## Note

Bootstrap decoder probabilities are **not** scientifically valid — this run validates **plumbing only**.
