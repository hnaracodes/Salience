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

Record session:

```bash
python scripts/record_website_session.py --script configs/walkthrough_scripts/localhost_demo.yaml
```

Note the printed `session_id`.

## 2. TRIBEv2 inference

```bash
modal run tribe.py::record_session --session-id <session_id>
```

Validates `preds.shape[0]` vs manifest snapshot count (±1 TR).

## 3. Dual-track (emotion Z + engagement)

Requires baseline preds for engagement track:

```bash
python scripts/run_dual_track.py --session-id <session_id> --baseline-session-id <baseline_id>
```

Or emotion-only:

```bash
python scripts/run_dual_track.py --session-id <session_id>
```

## 4. Sparse ViT heatmaps

Offline placeholder:

```bash
python scripts/extract_section_heatmaps.py --session-id <session_id> --uniform-heatmap --refresh-sections
```

Production (Modal DINOv2):

```bash
python scripts/extract_section_heatmaps.py --session-id <session_id> --modal --refresh-sections
```

## 5. Analyze + section report

```bash
python scripts/analyze_session.py --session-id <session_id> --norm-id <norm_id> --website --ground
```

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
  analysis_bundle.json
  heatmaps/t_<N>.npy
  ux_viewer/index.html
```
