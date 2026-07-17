---
session_id: CS-20260620-EMOTION-ACCURACY-ADR7
title: Emotion accuracy pipeline — subcortical TRIBE + Horikawa decoder
date: 2026-06-20
model: claude-sonnet-4-5
signature: claude-sonnet-4-5@CS-20260620-EMOTION-ACCURACY-ADR7
status: in_progress
track: website
related:
  - CS-20260528-WEBSITE-CONSOLIDATED
  - CS-20260602-EEV-PIPELINE-PILOT
files_touched:
  - tribe.py
  - activation_store.py
  - services/pipeline/runner.py
  - scout_core/subcortical/
  - scout_core/affect_features.py
  - scout_core/mvpa_engine.py
  - scout_core/dual_track.py
  - scripts/run_dual_track.py
  - scripts/horikawaCode/
  - scripts/validation/
  - configs/dual_track.yaml
  - configs/horikawa_decoding.yaml
  - configs/validation_study.yaml
---

# Emotion Accuracy Pipeline (ADR-7)

## Summary

Implements cortical + subcortical TRIBE inference, Horikawa ridge decoder training, and a **shadow decoder track** (`emotion_decoder_track`) that runs alongside production Kragel template Track 2 until UX validation passes.

See [adr-7-subcortical-decoder.md](./adr-7-subcortical-decoder.md) for architecture decisions.

## Important Files

- `tribe.py` — `predict_brain_both_npz` Modal method
- `scout_core/subcortical/tribev2_adapter.py` — `facebook/tribev2-subcortical` inference
- `scout_core/affect_features.py` — fused Schaefer-400 + 8 subcortical ROI features
- `scout_core/mvpa_engine.py` — decoder inference
- `scripts/horikawaCode/` — bootstrap training on ds002425 (demo synthetic when dataset absent)
- `configs/dual_track.yaml` — `emotion.parallel_decoder: true`, `mode: template` (production)

## Validation

- Unit tests: `tests/test_subcortical_io.py`, `tests/test_affect_features.py`, `tests/test_mvpa_engine.py`, `tests/horikawaCode/`
- Bootstrap: `prepare_horikawa_tribev2.py --demo` → `train_horikawa_ridge_decoder.py`
- Modal redeploy: `modal deploy tribe.py`
- UX gate: `scripts/validation/aggregate_validation_metrics.py` vs `validation_study.yaml` thresholds

## How To Continue

**Start here (2026-07-16 dual-track handoff):** [`../CS-20260716-DAILY-HANDOFF/ledger.md`](../CS-20260716-DAILY-HANDOFF/ledger.md)

Also: [`AGENT-HANDOFF-EMOTION-PIPELINE.md`](../../AGENT-HANDOFF-EMOTION-PIPELINE.md) (may lag today’s `ready_all` restore).

### Current state (2026-07-16)
- Phase 0 consistent on **371** `ready_all` clips; `cv_mean_r ≈ 0.242`
- `eval_ablation_v1.json` / dims eval / `shadow_inference_smoke.json` OK
- **Still missing:** frozen ablation-subset report on disk + `reframing_findings.md`

### Next
1. Run `scripts/horikawaCode/run_ablation_subset_frozen.py` then `write_reframing_findings.py`.
2. Record Modal go/no-go in ADR-7 ledger.
3. UX validation pilot after findings; flip `emotion.mode` only when ISSUE-001 gate passes.

## Open items

| Type | ID | Title |
|------|-----|-------|
| Issue | ISSUE-001 | Decoder production gate (UX + real training) |
| Bug | BUG-002 | Playwright browsers missing |
| Bug | BUG-003 | Modal network interrupt during tribe |
| Feature | FEATURE-002 | Bootstrap full pipeline run (in progress) |
| Feature | FEATURE-003 | Real Horikawa TRIBE training |
| Sprint | SPRINT-001 | Complete emotion pipeline by 2026-07-01 |

## Signature

Signed-off-by: composer on 2026-06-30 (bootstrap run + handoff)
