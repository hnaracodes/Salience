---
id: FEATURE-001
session_id: CS-20260618-DEMOGRAPHIC-MUX-PLAN
status: open
owner: agent
files:
  - docs/implementation-plans/demographic-multiplexer-implementation-plan.md
  - salience/mux/tribe_mux.py
  - salience/modal_app/train_clusters.py
  - salience/modal_app/inference_mux.py
  - salience/mux/neural_barrier.py
  - services/pipeline/runner.py
  - scripts/run_demographic_mux_session.py
---

# Demographic Subagent Multiplexer

## Goal

Enable per-demographic-cluster cortical activation predictions on UI walkthrough videos by branching from a frozen TRIBE universal encoder with a low-rank cluster-conditional readout delta. Product users see side-by-side network traces for selected demographic prototypes (e.g. young adult female vs. male) with explicit uncertainty disclaimers.

**Gated:** Feature must pass M0 viability experiment and M3 eval gate before any product UI exposure.

## Implementation Notes

- **Plan:** [demographic-multiplexer-implementation-plan.md](../../../docs/implementation-plans/demographic-multiplexer-implementation-plan.md) (revision 2026-06-18)
- **Architecture:** `X_univ[T,1152]` → cluster embedding → low-rank delta on frozen population readout → `parcel_ts[K,T,400]`
- **Loss:** Temporal Pearson correlation in parcel/network space; not MSE on 20484 vertices
- **Fallback:** If hidden state unavailable, use TRIBE `preds[T,V]` as input features
- **Integration point:** Optional stage in `services/pipeline/runner.py` behind `DEMOGRAPHIC_MUX=1` feature flag (default off)

### Milestone sequence

| Milestone | Gate | Deliverable |
|-----------|------|-------------|
| M0 | **Must pass first** | `m0_report.json` — demographic variance above permutation null |
| M1 | After M0 | TRIBE hidden-state hook + commit pin |
| M2 | After M0 | Parcel-space centroids from HCP 7T / Cam-CAN |
| M3 | **Eval gate** | Trained mux + `delta_r > 0.02`, `p_perm < 0.01` |
| M4 | After M3 pass | `infer_mux` in pipeline + analysis_bundle per cluster |
| M5 | After M3 pass | Neural barrier / divergence windows in viewer |

### Kill switch

If M0 or M3 eval gate fails: cancel feature, ship population-only mode, set `DEMOGRAPHIC_MUX=0` permanently until new data or architecture.

## Validation

- M0: `python scripts/run_demographic_viability.py` produces passing `scout_data/demographic/viability/m0_report.json`
- Unit: `pytest tests/test_tribe_mux.py tests/test_demographic_viability.py`
- Eval gate metrics logged in training checkpoint `metrics.json`
- Product: analysis_bundle includes `provenance.disclaimer` for all cluster outputs
