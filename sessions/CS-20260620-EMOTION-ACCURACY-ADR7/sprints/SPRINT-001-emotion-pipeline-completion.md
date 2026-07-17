---
id: SPRINT-001
session_id: CS-20260620-EMOTION-ACCURACY-ADR7
status: in_progress
owner: agent
target_date: 2026-07-01
---

# Sprint: Complete emotion analysis pipeline

## Objective

Ship validated subcortical + Horikawa decoder path with production-safe gating.

## Day 1 (2026-06-30) — done / in progress

- [x] Bootstrap decoder train (`model.joblib` exported)
- [x] Handoff doc: `AGENT-HANDOFF-EMOTION-PIPELINE.md`
- [x] Ledger bugs/features/sprint logged
- [x] Bootstrap full run session `CS-20260620-BOOTSTRAP-FULL` (Modal + dual_track + analyze + viewer)

## Day 2 (2026-07-01) — agent tasks

- [ ] Confirm bootstrap session artifacts (see FEATURE-002)
- [ ] Phase 0: `download_horikawa_labels.py` + constants (FEATURE-003)
- [ ] Phase 1–2: pilot manifest + Modal feature batch (~150 clips)
- [ ] Phase 3–4: prepare NPZ + train both ridge bundles (LOVO fix)
- [ ] Phase 5: eval reports + pytest
- [ ] Re-run dual_track on website session; compare template vs real decoder shadow track
- [ ] UX validation pilot (optional until real model exists)

## Exit criteria

1. `preds_subcortical.npz` on all production pipeline paths (no silent proxy in prod)
2. `horikawa_ridge_v1` with `meta.status: pilot` or `full`
3. ISSUE-001 remains open until UX `transfer_r_valence >= 0.25`

## Handoff entry point

Read **`TribeV2/AGENT-HANDOFF-EMOTION-PIPELINE.md`** first.
