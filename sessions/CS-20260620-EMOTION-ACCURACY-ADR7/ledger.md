# Ledger

## Step 1 - Context

- Files read: `sessions/INDEX.md`, ADR-6 report, emotion-accuracy pipeline plan, `dual_track.yaml`, `tribe.py`, `affect_features.py`, `mvpa_engine.py`, `tribev2_adapter.py`
- Prior sessions consulted: `CS-20260528-WEBSITE-CONSOLIDATED`, `CS-20260602-EEV-PIPELINE-PILOT`
- Open issues considered: domain gap (movies ≠ websites), bootstrap CV metrics near zero on synthetic data

## Step 2 - Implementation (prior turn)

- Added `scout_core/subcortical/` package, `preds_subcortical.npz` artifact path, Modal `predict_brain_both_npz`
- Added Horikawa training scripts + `scout_models/horikawa_ridge_v1/` export
- Wired decoder path in `dual_track.py` / `run_dual_track.py` behind `emotion.mode`
- Added UX validation scaffolding (`scripts/validation/`, `validation_study.yaml`)

## Step 3 - Bootstrap + shadow track (this turn)

- Files touched:
  - `configs/dual_track.yaml` — `parallel_decoder: true`, production `mode: template`
  - `scripts/run_dual_track.py` — Track 2b shadow decoder → `emotion_decoder_track` in bundle
  - `scripts/validation/aggregate_validation_metrics.py` — prefer `emotion_decoder_track` when present
  - `sessions/CS-20260620-EMOTION-ACCURACY-ADR7/` — session ledger + audit
- What changed: Decoder runs as separate bundle field without replacing Kragel grounding/SQLite traces
- Expected effect: Compare template vs decoder per session before product switch

## Step 4 - Bootstrap full run + handoff (2026-06-30)

- Retrained bootstrap decoder: `prepare_horikawa_tribev2.py --demo` → `train_horikawa_ridge_decoder.py` → `scout_models/horikawa_ridge_v1/model.joblib`
- Target session: `CS-20260620-BOOTSTRAP-FULL` (localhost demo walkthrough.webm, 5 TR)
- Chained pipeline started: Modal `record_session --no-visualize` → `run_dual_track` → heatmaps → analyze → export_viewer
- **Completed 2026-06-30:** Modal both-NPZ (14×20484 + 14×8802), dual_track + shadow decoder, heatmaps, analyze, ux_viewer export
- Re-ran `run_dual_track` after analyze to restore `emotion_decoder_track` (see BUG-004)
- Handoff for next agent: `TribeV2/AGENT-HANDOFF-EMOTION-PIPELINE.md`
- Ledger additions:
  - `bugs/BUG-002-playwright-browsers-missing.md`
  - `bugs/BUG-003-modal-network-interrupt.md`
  - `features/FEATURE-002-bootstrap-full-pipeline-run.md`
  - `features/FEATURE-003-real-horikawa-tribe-training.md`
  - `sprints/SPRINT-001-emotion-pipeline-completion.md`
- Real training plan: `.cursor/plans/horikawa_real_tribe_training_1327d1e4.plan.md`

## Step 5 - Real Horikawa TRIBE training pipeline (2026-06-30)

- Implemented FEATURE-003 phases 0–5 (code complete; Modal pilot run pending videos):
  - `scout_core/horikawaCode/constants.py` — HORIKAWA_34/14, CATEGORY_34_TO_PRODUCT_8
  - `scout_core/horikawaCode/labels.py` — figshare zip parser, load_horikawa_ratings, build_y_*
  - `scout_core/horikawaCode/cv.py`, `prepare.py` — LOVO CV + clip-level 808-dim framing
  - `scripts/horikawaCode/download_horikawa_labels.py`
  - `scripts/horikawaCode/build_clip_manifest.py`
  - `scripts/horikawaCode/generate_tribev2_features.py` (Modal + synthetic)
  - Extended prepare/train/evaluate; `configs/horikawa_decoding_dims.yaml`
  - CV scheme fixed: `leave_one_video_out` (one group per stimulus_id)
  - Tests expanded: `tests/horikawaCode/test_horikawa_training.py` (9 tests)
  - Post-audit fix: `cv.py` stacked OOF predictions for valid LOVO Pearson r; separate train/eval report filenames
- Bootstrap path unchanged (`--demo`); production gate ISSUE-001 still open until real pilot train + UX

## Step 6 - Readiness audit + video gate (2026-07-01)

- **Video folder:** `scout_data/horikawaCode/videos/` — place MP4s named `{stimulus_id}.mp4` or `{id:04d}.mp4` (see `pilot_150.json` ids: `1`, `10`, `1000`, …)
- **Labels:** cached — `labels/ratings_cache.npz` (n=2196), `labels/features.zip`
- **Pilot manifest:** `manifests/pilot_150.json` (118 clips stratified); **full manifest not built yet**
- **Modal intermediates:** none (`intermediates_modal/` empty — waiting on videos)
- **Orchestrator:** `scripts/horikawaCode/run_pilot_training.py` — $20 budget cap, 3-clip probe before scale-up
- **Tests:** `tests/horikawaCode/` 9/9 passed (2026-07-01)
- **Plan phases 0–5 code:** implemented; **phase 6 (full scale)** and **real Modal pilot** blocked on videos + Cowen approval
- **Open blockers:** BUG-004 (`emotion_decoder_track` stripped by analyze), ISSUE-001 prod gate, ML specialist review ×2 not run
- **Code audit verdict:** Approve with caveats — ready to run pilot once MP4s land; do not flip `emotion.mode: decoder` until real train + UX gate

### Pilot run (after videos)

```powershell
cd TribeV2
.venv311\Scripts\activate
modal deploy tribe.py
python scripts/horikawaCode/run_pilot_training.py --budget-usd 20 --probe-n 3 --pilot-n 150 --skip-labels
```

### Full corpus training (2026-07-02)

- **Subcortical fix:** `load_tribev2_checkpoint_model()` — HF subcortical ckpt missing `model_build_args`; proxy fallback disabled on Modal
- **Manifest:** `full_2181.json` n=2196, all with MP4
- **Orchestrator:** `scripts/horikawaCode/run_full_horikawa_training.py` — Modal ALL clips first, ridge train only after 100% intermediates
- **Monitor:** `scripts/horikawaCode/monitor_full_training.py` (5 min poll) → `pilot_run/full_training_monitor.log`
- **Tests:** 17/17 horikawa+affect+subcortical passed (2026-07-02)
- **Audit:** Approve with caveats — full Modal run in progress; ISSUE-001 prod gate still open


```powershell
python -m pytest tests/horikawaCode/ tests/test_affect_features.py tests/test_mvpa_engine.py tests/test_subcortical_io.py -v --tb=short
# 17 passed (2026-06-30)
```

## Validation

- **Bootstrap full run `CS-20260620-BOOTSTRAP-FULL` (2026-06-30):**
  - `preds.npz` 14×20484, `preds_subcortical.npz` 14×8802 (real Modal, `facebook/tribev2-subcortical`)
  - Track 2b peak probs written; `emotion_decoder_track` in bundle (re-run dual_track after analyze)
  - `ux_viewer/index.html` exported
  - WARN: preds T=14 vs manifest snapshots=5 (alignment delta); WARN: analyze may strip decoder fields (BUG-004)
- Prior gray-video session `CS-20260622-EMOTION-BOOTSTRAP`: Modal both-NPZ OK (45×20484 + 45×8802)
