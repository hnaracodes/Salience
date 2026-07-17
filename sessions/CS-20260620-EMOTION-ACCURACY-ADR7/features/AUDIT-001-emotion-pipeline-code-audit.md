---
id: AUDIT-001
title: Code audit — emotion accuracy pipeline
status: completed
severity_overall: medium
verdict: approve_with_caveats
auditor: code-auditor@composer-2.5-fast
date: 2026-06-20
---

# Code Audit — Emotion Accuracy Pipeline

## 1. Executive summary

ADR-7 adds subcortical TRIBE predictions (`preds_subcortical.npz`), fused affect features, and a Horikawa ridge decoder with config-gated rollout. Overall risk is **Medium**: architecture is sound and tests cover I/O + feature shapes, but bootstrap decoder metrics on synthetic demo data are not production-ready, and subcortical inference silently falls back to cortical-derived proxies when the dedicated checkpoint fails. Safe to run as **shadow track** (`parallel_decoder: true`, `mode: template`); do not flip production `emotion.mode: decoder` until real Horikawa training + UX validation pass.

## 2. Change inventory

| Path | Type | Note |
|------|------|------|
| `tribe.py` | modified | `predict_brain_both_npz`, subcortical checkpoint env |
| `activation_store.py` | modified | `save_subcortical_timeseries` |
| `services/pipeline/runner.py` | modified | FAKE_TRIBE + prod both-NPZ path |
| `scout_core/subcortical/*` | added | atlas, io, tribev2_adapter |
| `scout_core/affect_features.py` | added | fused cortical+subcortical features |
| `scout_core/mvpa_engine.py` | added | decoder inference + bundle export |
| `scout_core/horikawaCode/*` | added | labels, constants |
| `scout_core/dual_track.py` | modified | decoder + auto fallback |
| `scripts/run_dual_track.py` | modified | parallel shadow decoder track |
| `scripts/horikawaCode/*` | added | prepare/train/evaluate |
| `scripts/validation/*` | added | SAM UX study |
| `configs/*.yaml` | added/modified | subcortical, horikawa, validation, dual_track |
| `tests/test_*` | added | subcortical, affect, mvpa, horikawa |

## 3. Mental model

Playwright capture → Modal TRIBE cortical (`preds.npz` T×20484) + subcortical head (`preds_subcortical.npz` T×8802) → `affect_features.build_affect_features` (Schaefer-400 mean/std + 8 Harvard-Oxford ROI means) → `mvpa_engine.predict_emotion_track` (RidgeCV bundle) → shadow `emotion_decoder_track` in bundle while production Track 2 stays Kragel template cosines.

## 4. Findings by severity

### Medium — Silent subcortical fallback masks checkpoint failures

- **Location:** `scout_core/subcortical/tribev2_adapter.py` lines 38–56
- **What's wrong:** Broad `except Exception: pass` then cortical-derived proxy if subcortical `TribeModel.from_pretrained` fails.
- **Why it matters:** Production sessions may run with fake limbic signal; decoder trained on real subcortical will not transfer.
- **Suggested fix:** Log warning + set `meta_json.subcortical_source=proxy`; fail hard when `require_subcortical` in prod.

### Medium — Bootstrap model not validated for product switch

- **Location:** `scout_models/horikawa_ridge_v1/meta.json` (`status: bootstrap`, `cv_mean_r ≈ 0.018`)
- **What's wrong:** Demo synthetic training data yields near-zero LOVO correlations.
- **Why it matters:** Decoder probabilities are not meaningful until real ds002425 TRIBE features + human ratings.
- **Suggested fix:** Retrain on real corpus; gate `emotion.mode: decoder` on `validation_study.yaml` transfer metrics.

### Medium — `model.joblib` missing until train script runs

- **Location:** `scout_models/horikawa_ridge_v1/`
- **What's wrong:** Only `meta.json` + `label_map.json` present if bootstrap train not executed.
- **Why it matters:** `parallel_decoder` and decoder mode raise `FileNotFoundError`.
- **Suggested fix:** Run bootstrap train commands in CI or deploy checklist.

### Low — Probability mapping ambiguity

- **Location:** `scout_core/mvpa_engine.py` lines 94–96
- **What's wrong:** Ridge outputs clipped to [0,1] unless out of range, then sigmoid applied.
- **Why it matters:** Probabilities are not calibrated posteriors; UI should label as model scores.
- **Suggested fix:** Document in `CLAIMS.md`; optional Platt scaling after UX study.

### Info — Parallel track does not write SQLite decoder trace

- **Location:** `scripts/run_dual_track.py`
- **Observation:** Shadow decoder only in `analysis_bundle.json`; SQLite `session_emotion_trace` stays template.
- **Impact:** Intentional separation for A/B comparison before production switch.

## 5. Accuracy and assumptions

- Horikawa demo bootstrap uses synthetic X/y — metrics are placeholders only.
- Subcortical voxels (8802) match Meta `tribev2-subcortical` Harvard-Oxford layout per `subcortical_manifest.yaml`.
- Domain gap: decoder trained on movies; website stimuli need UX validation (Phase 4).

## 6. Usage checklist

- [ ] `python scripts/horikawaCode/prepare_horikawa_tribev2.py --demo`
- [ ] `python scripts/horikawaCode/train_horikawa_ridge_decoder.py`
- [ ] `modal deploy tribe.py` (enables `predict_brain_both_npz`)
- [ ] Run pipeline e2e with FAKE_TRIBE; confirm `preds_subcortical.npz` exists
- [ ] `run_dual_track.py` on session; confirm `emotion_decoder_track` in bundle
- [ ] Record UX validation sessions + `aggregate_validation_metrics.py`

## 7. Verdict

**Approve with caveats** — shadow deployment OK; production decoder mode blocked until real training + UX gate.
