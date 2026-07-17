---
id: FEATURE-003
session_id: CS-20260620-EMOTION-ACCURACY-ADR7
status: in_progress
owner: agent
files:
  - scripts/horikawaCode/
  - scout_core/horikawaCode/
  - configs/horikawa_decoding.yaml
  - scout_models/horikawa_ridge_v1/
  - scout_models/horikawa_ridge_dims_v1/
---

# Real Horikawa TRIBE training (ds002425)

## Goal

Replace bootstrap synthetic decoder with models trained on **real TRIBE features** from Horikawa/Cowen video corpus + human ratings.

## Implementation Notes

Follow plan: `.cursor/plans/horikawa_real_tribe_training_1327d1e4.plan.md`

Phases:

0. Figshare label loader (34 categories + 14 dimensions)
1. Pilot manifest ~150 clips
2. Modal batch `generate_tribev2_features.py` via `predict_brain_both_npz`
3. Clip-level `train.npz` / `train_dims.npz` with **LOVO groups = stimulus_id**
4. Train `horikawa_ridge_v1` + `horikawa_ridge_dims_v1`
5. Eval ablation + tests
6. Scale to full 2181 clips

ML specialist review ×2 per phase (`.cursor/agents/ml-training-specialist.md`).

## Status (2026-07-01)

| Phase | Code | Data / run |
|-------|------|------------|
| 0 Labels | done | `ratings_cache.npz` n=2196 |
| 1 Manifest | done (pilot) | `pilot_150.json` 118 clips; full_2181 not built |
| 2 Modal features | done | blocked — no MP4s in `scout_data/horikawaCode/videos/` |
| 3 Prepare | done | waiting on intermediates |
| 4 Train | done | bootstrap model only; real train pending |
| 5 Eval/tests | done | 9/9 pytest; website shadow not run |
| 6 Full scale | pending | after pilot |

**Video folder:** `TribeV2/scout_data/horikawaCode/videos/`  
**Orchestrator:** `scripts/horikawaCode/run_pilot_training.py` ($20 Modal cap)

## Validation

- `meta.status`: `pilot` → `full`
- `cv_mean_r` beats permutation baseline; `subcortical_lift > 0`
- `pytest tests/horikawaCode/` green
