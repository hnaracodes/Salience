---
session_id: CS-20260618-DEMOGRAPHIC-MUX-PLAN
title: Demographic Multiplexer Implementation Plan Rework
date: 2026-06-18
model: composer-2.5
signature: composer-2.5@CS-20260618-DEMOGRAPHIC-MUX-PLAN
status: in_progress
track: neuroemo
related:
  - CS-20260602-EEV-PIPELINE-PILOT
  - CS-20260528-WEBSITE-CONSOLIDATED
  - CS-20260522-NEUROEMO-DATASET-PREP
files_touched:
  - docs/implementation-plans/demographic-multiplexer-implementation-plan.md
  - configs/cluster_schema.yaml
  - configs/clusters.yaml
  - data_prep/
  - salience/
  - scripts/run_demographic_viability.py
  - scripts/run_demographic_data_prep.py
  - scripts/run_demographic_mux_session.py
  - scripts/train_demographic_mux_local.py
  - services/pipeline/runner.py
  - tests/test_demographic_viability.py
  - tests/test_demographic_data_prep.py
  - tests/test_tribe_mux.py
  - tests/test_neural_barrier.py
  - requirements.txt
---

# Demographic Multiplexer Implementation Plan Rework

## Summary

Reworked the demographic subagent multiplexer implementation plan to correct factually wrong architecture assumptions, confront data-availability and movie→UI domain-shift problems honestly, and add gating milestones (M0 viability experiment, M3 eval gate with kill switch) before any Modal training infra is built.

Key corrections applied to the plan:

- TRIBE Transformer hidden dim is **1152**, not 4096.
- Readout should be a **low-rank delta on TRIBE's population/unseen-subject head**, not a from-scratch 53M-param MLP to 20484 vertices.
- Primary training target is **Schaefer-400 / Yeo-7 parcel/network space** with **correlation + noise-ceiling loss**, not MSE on raw vertices.
- **Ethnicity cut** from cluster schema; marketed age×sex grid rescoped to coarse age bands with site covariate.
- **M0 gating experiment** (variance partition + permutation null on public movie-fMRI) must pass before building training infra.
- **M3 eval gate** (`delta_r`, demographic-shuffle null, LODO) with explicit kill switch before product integration.

No production Python code was written in this session — documentation and ledger only.

## Important Files

- [docs/implementation-plans/demographic-multiplexer-implementation-plan.md](../../docs/implementation-plans/demographic-multiplexer-implementation-plan.md) — full rework
- [sessions/CS-20260618-DEMOGRAPHIC-MUX-PLAN/features/FEATURE-001-demographic-multiplexer.md](./features/FEATURE-001-demographic-multiplexer.md)
- [sessions/CS-20260618-DEMOGRAPHIC-MUX-PLAN/issues/ISSUE-001-demographic-data-availability.md](./issues/ISSUE-001-demographic-data-availability.md)

## Validation

- Plan document rewritten per attached rework specification.
- Ledger session created with feature and open issue files.
- `sessions/INDEX.md` updated with new session row.

## How To Continue

1. **User action required:** Obtain HCP 7T movie or Cam-CAN parcellated subject CSV and run real M0:
   `python scripts/run_demographic_viability.py --subjects-csv <path> --n-permutations 1000`
2. If M0 passes, run R1 TRIBE hidden-state spike on Modal and pin commit in `salience/mux/tribe_commit.txt`.
3. Prepare raw metadata in `scout_data/demographic/raw_metadata/` and run data prep.
4. Train mux (local smoke: `python scripts/train_demographic_mux_local.py`; production: Modal `train_clusters.py`).
5. Enable pipeline: `DEMOGRAPHIC_MUX=1`, `DEMOGRAPHIC_MUX_CHECKPOINT=<path>`, `DEMOGRAPHIC_MUX_CLUSTER_IDS=0,1`.

## Signature

Signed-off-by: Composer 2.5 (composer-2.5) on 2026-06-18
