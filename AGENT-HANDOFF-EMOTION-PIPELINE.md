# Agent handoff — emotion pipeline (read this first)

**Created:** 2026-06-30 (session continuation)  
**Parent session:** `CS-20260620-EMOTION-ACCURACY-ADR7`  
**Plan:** `.cursor/plans/horikawa_real_tribe_training_1327d1e4.plan.md`

Context is **not** saved in chat. Start here tomorrow.

---

## What Salience emotion pipeline is

1. **Modal TRIBE** → `preds.npz` (T×20484 cortical) + `preds_subcortical.npz` (T×8802)
2. **Track 2 production** → Kragel template cosines (`emotion_track`) — drives grounding/SQLite
3. **Track 2b shadow** → Horikawa ridge decoder (`emotion_decoder_track`) — bundle only until UX gate
4. **Production gate** → `emotion.mode` stays `template` until real Horikawa training + `transfer_r_valence >= 0.25`

Config: `configs/dual_track.yaml` (`parallel_decoder: true`, `mode: template`).

---

## State when this handoff was written

### Completed (ADR-7 code)

- `tribe.py::predict_brain_both_npz`, subcortical package, `affect_features.py`, `mvpa_engine.py`
- `scripts/horikawaCode/` prepare/train/evaluate (bootstrap/demo path only)
- Shadow track in `run_dual_track.py` → `emotion_decoder_track` in bundle
- UX validation scaffolding: `scripts/validation/`, `configs/validation_study.yaml`

### Bootstrap integration run

- **Session ID:** `CS-20260620-BOOTSTRAP-FULL` — **COMPLETE (2026-06-30)**
- **Artifacts:** `preds.npz` (14×20484), `preds_subcortical.npz` (14×8802), `analysis_bundle.json` with `emotion_decoder_track`, `ux_viewer/`
- **Decoder:** bootstrap weights (`cv_mean_r ≈ 0.018`) — plumbing only
- **Run dual_track AFTER analyze** to keep shadow track in bundle (BUG-004)

### Blockers hit today

| ID | Issue | Fix |
|----|-------|-----|
| BUG-002 | Playwright browsers missing in sandbox | `playwright install chromium` (needs network to Microsoft CDN) |
| BUG-003 | Modal `getaddrinfo failed` mid-run | Retry `modal run tribe.py::record_session --no-visualize` |
| BUG-004 | `analyze_session` strips `emotion_decoder_track` | Run `dual_track` last after analyze |

---

## Tomorrow — priority order

### 1. Real Horikawa training (FEATURE-003 — start here)

Bootstrap plumbing is done. Next work is the real model per plan phases 0–6.

Follow `.cursor/plans/horikawa_real_tribe_training_1327d1e4.plan.md`:

1. Figshare labels + `download_horikawa_labels.py`
2. `pilot_150.json` manifest + Modal batch `generate_tribev2_features.py`
3. Clip-level `train.npz` + **LOVO by stimulus_id** (not subject LOSO)
4. Train `horikawa_ridge_v1` + `horikawa_ridge_dims_v1`
5. Eval ablation + tests in `tests/horikawaCode/`

Review each phase **twice** with `.cursor/agents/ml-training-specialist.md`.

### 3. Close production gate (ISSUE-001)

Only after real training:

- `meta.status` → `pilot` then `full`
- UX study N≥10, SAM ratings → `aggregate_validation_metrics.py`
- `transfer_r_valence >= 0.25` → then consider `emotion.mode: decoder`

---

## Key files

| Path | Role |
|------|------|
| `tribe.py` | Modal `record_session`, `predict_brain_both_npz` |
| `scout_core/affect_features.py` | 808-dim fused features |
| `scout_core/mvpa_engine.py` | Decoder inference |
| `scripts/run_dual_track.py` | Shadow Track 2b |
| `scripts/horikawaCode/*` | Training pipeline |
| `configs/dual_track.yaml` | Production vs shadow |
| `configs/horikawa_decoding.yaml` | Ridge training config |
| `sessions/CS-20260620-EMOTION-ACCURACY-ADR7/` | Ledger, issues, features |

---

## Do NOT do tomorrow

- Do not flip `emotion.mode: decoder` on bootstrap weights
- Do not run UX correlation study on bootstrap decoder (misleading)
- Do not interpret `emotion_decoder_track` probabilities as real emotion

---

## Tests to run after real training

```powershell
python -m pytest tests/horikawaCode/ tests/test_affect_features.py tests/test_mvpa_engine.py tests/test_subcortical_io.py -v --tb=short
```

---

## Related transcript

Emotion work: agent transcript `71a20e5e-1cd7-4e08-898d-9fb6b281f518` (not demographic `f2547e19`).
