# Website session runbook

End-to-end flow: **scripted Playwright capture** (MP4 + DOM manifest) → **TRIBEv2 preds** → **dual-track** → **sparse ViT heatmaps** → **section_report** → **UX viewer** → **LLM narrative**.

Video remains the stimulus path to TRIBEv2; Playwright provides alignment and DOM credit only.

## Prerequisites

```bash
pip install -r requirements.txt
playwright install chromium
# Optional: ffmpeg on PATH for frame extraction
# Modal + HuggingFace secret for tribe inference and heatmaps
```

## 1. Capture (MP4 + session_manifest.json)

Serve local fixture (for `localhost_demo.yaml`):

```bash
python -m http.server 8765 --directory tests/fixtures/walkthrough_site
```

Record session (quick fixture, 5 TRs):

```bash
python scripts/record_website_session.py --script configs/walkthrough_scripts/localhost_demo.yaml
```

Cinematic fixture (recommended for emotion-rich capture, ~14 TRs):

```bash
python scripts/record_website_session.py --script configs/walkthrough_scripts/aurora_showcase.yaml
```

Preview fixture: `python -m http.server 8765 --directory tests/fixtures/walkthrough_site` → open `index.html` (Aurora showcase).

Note the printed `session_id`.

## 2. TRIBEv2 inference

```bash
modal run tribe.py::record_session --session-id <session_id>
```

Validates `preds.shape[0]` vs manifest snapshot count (±1 TR).

## 3. Dual-track (engagement + activation + emotion)

Generate the shared gray-background baseline once (Modal GPU):

```bash
modal run tribe.py::record_baseline
```

Writes `scout_data/baseline/preds_baseline.npz` from `gray background.mp4`. This file is used automatically for Track 1 (VAN/DMN Z vs baseline) and Track 3 (mean vertex activation Z).

Run dual-track (auto-loads baseline when present):

```bash
python scripts/run_dual_track.py --session-id <session_id>
```

Override baseline:

```bash
python scripts/run_dual_track.py --session-id <session_id> --baseline-session-id <baseline_id>
# or
python scripts/run_dual_track.py --session-id <session_id> --baseline-preds path/to/preds_baseline.npz
```

**Tracks in `analysis_bundle.json`:**

| Track | Key | Metric |
|-------|-----|--------|
| 1 | `engagement_track` | Z(VAN) − Z(DMN) vs gray baseline (Schaefer SalVentAttn / Default) |
| 2 | `emotion_track` | Kragel template cosine + session Z |
| 3 | `activation_track` | mean(\|preds\|) per TR; baseline Z or session Z |

Atlas: `configs/vertex_regions.csv` (Schaefer 400 @ fsaverage5, verified CBIG .annot alignment).

## 4. Sparse ViT heatmaps

Offline placeholder:

```bash
python scripts/extract_section_heatmaps.py --session-id <session_id> --uniform-heatmap --refresh-sections
```

Production (Modal DINOv2):

```bash
python scripts/extract_section_heatmaps.py --session-id <session_id> --modal --refresh-sections
```

## 5. Analyze + section report + marketing scores

```bash
python scripts/analyze_session.py --session-id <session_id> --norm-id <norm_id> --website --ground
```

This writes `section_report[]` and `marketing_scores` (schema v4): a session-relative **0–100 display curve**, five rubric metrics, drop moments, and per-section scores. Marketing scores are derived from `engagement_track` (VAN/DMN Z) plus preds novelty — they do **not** replace dual-track outputs or claim eye-tracking accuracy. Tune display weights in `configs/marketing_scores.yaml`.

## 6. UX viewer export

```bash
python scripts/export_ux_viewer.py --session-id <session_id>
```

Open `scout_data/sessions/<id>/ux_viewer/index.html?base=.`

## 7. LLM narrative (structured JSON only)

```bash
python scripts/generate_session_narrative.py --session-id <session_id> --provider template
```

Set `OPENAI_API_KEY` and `provider: openai` in `configs/llm_narrative.yaml` for live LLM.

## Orchestrator (all stages)

```bash
python scripts/run_website_session.py --stage all \
  --script configs/walkthrough_scripts/localhost_demo.yaml \
  --norm-id <norm_id> \
  --baseline-session-id <baseline_id> \
  --uniform-heatmap
```

## Alignment checklist

- [ ] `session_manifest.json` schema_version 2 with `video.path` and `tr_mapping.tr_duration_sec`
- [ ] Viewport 1920×1080 (or script viewport) matches heatmap dimensions
- [ ] `preds T` ≈ `len(dom_snapshots)` (see manifest `alignment.preds_validation`)
- [ ] `walkthrough.mp4` or `walkthrough.webm` present in session dir

## Artifact layout

```
scout_data/sessions/<id>/
  walkthrough.mp4
  session_manifest.json
  walkthrough_script.yaml
  preds.npz
  analysis_bundle.json   # schema v4 includes marketing_scores with --website
  heatmaps/t_<N>.npy
  ux_viewer/index.html
```

### marketing_scores vs engagement_track

| Field | Meaning |
|-------|---------|
| `engagement_track.scores` | Raw VAN−DMN Z vs gray baseline (scientific track) |
| `activation_track.raw_scores` | Mean \|preds\| per TR (ViralAnalyser attention proxy) |
| `activation_track.scores` | Baseline-relative or session-relative Z of mean activation |
| `marketing_scores.display_curve` | Min–max 0–100 compound of engagement Z + preds novelty |
| `marketing_scores.overall_score` | Average of five rubric metrics (opening, sustain, …) |
| `marketing_scores.sections[].score` | Per-section 0–100 from section mean engagement Z |

Disable with `analyze_session.py --no-marketing-scores`. Force on any session with `--marketing-scores`.
