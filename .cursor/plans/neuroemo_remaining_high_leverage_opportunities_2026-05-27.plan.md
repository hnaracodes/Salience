---
name: NeuroEmo Remaining High-Leverage Opportunities
overview: Finish the remaining very-high and high-leverage NeuroEmo model improvements after the surface-native Schaefer atlas and mesh-identity provenance work. The next work should separate trust/provenance completion from true signal-gain experiments, using the current projected postfix best and the first surface-native logistic rerun as fixed comparison anchors.
todos:
  - id: projection-equivalence-proof
    content: Upgrade vertex proof from mesh_identity_verified to projection_equivalence_verified and fix any overclaiming status labels/docs.
    status: pending
  - id: full-surface-postfix-matrix
    content: Rerun the full projected-postfix-equivalent model matrix on the surface-native atlas.
    status: pending
  - id: timing-grid
    content: Sweep BOLD lag, block-edge transition trimming, and temporal stride under a frozen model/atlas/fold contract.
    status: pending
  - id: dynamic-window-features
    content: Add dynamic within-window ROI features and compare them against the current static reducers.
    status: pending
  - id: preprocessing-roi-ablation
    content: Run raw-vs-preprocessed and ROI strategy A/Bs after proof and matrix results settle.
    status: pending
  - id: structured-decoding
    content: Test valence-first or coarse-to-fine decoding plus targeted specialist refiners.
    status: pending
isProject: false
---

# NeuroEmo Remaining High-Leverage Opportunities

## Current Baseline Anchors

Use these as fixed reference points until superseded by a new matrix report.

- **Projected postfix best:** `logistic_saga_5class_10tr` from `scout_data/neuroemo/models/2026-05-25_postfix_matrix/`
  - Mean accuracy: `0.445`
  - Mean macro F1: `0.433494`
  - Pooled log loss: `2.279231`
- **Projected postfix binary valence best:** `mlp_valence_binary_10tr`
  - Mean accuracy: `0.659375`
  - Mean macro F1: `0.655723`
- **First surface-native matched logistic run:** `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/logistic_saga_5class_10tr_metrics.json`
  - Mean accuracy: `0.420`
  - Mean macro F1: `0.412`
  - Pooled log loss: `2.244`
  - Interpretation: provenance improved; 5-class top-1 and macro F1 did not.

## Guiding Rules

1. Keep every experiment subject-held-out and fold-compatible with the postfix matrix unless the plan explicitly says otherwise.
2. Treat macro F1, balanced accuracy, log loss, and pooled confusion as first-class metrics. Accuracy alone is not enough.
3. Promote a change only if it beats the fixed baseline or improves a hard-class confusion pattern without worsening calibration materially.
4. Separate generated data/model artifacts from code/docs/tests when preparing commits or PRs.

## Implementation Support Added

The phase 2-end execution machinery is now implemented while phase 1 remains deliberately separate from this runner:

- `scripts/run_neuroemo_experiment_matrix.py` writes reproducible commands for phase 2-end and can execute them with `--execute`.
- `scripts/train_neuroemo_emotion_model.py` now supports dynamic temporal reducers shared by the linear/SVM, MLP, specialist, and hierarchical paths.
- `scripts/train_neuroemo_hierarchical_model.py` adds valence-style coarse-to-fine decoding without changing the upstream Nilearn fsaverage5/ROI training contract.
- `scout_data/neuroemo/models/neuroemo_remaining_high_leverage_matrix_manifest.json` is a dry-run manifest for the full phase 2-end command set.

Run all phase 2-end commands locally with:

```powershell
.venv\Scripts\python.exe scripts/run_neuroemo_experiment_matrix.py --execute --skip-existing
```

Run only the full surface-native postfix matrix with:

```powershell
.venv\Scripts\python.exe scripts/run_neuroemo_experiment_matrix.py --phase surface-postfix --execute --skip-existing
```

## Phase 1 - Complete Vertex And Projection Equivalence

**Goal:** close the remaining trust gap before running another large experiment matrix.

Current status:

- `configs/parcellation_manifest.yaml` records `proof_status: mesh_identity_verified`.
- `scripts/run_vertex_equivalence_verification.ps1` is set up to require `projection_equivalence_verified`.
- The surface-native evaluation doc correctly distinguishes mesh identity from projection agreement, but some status wording around annot-vs-projection agreement can still be misread.

Implementation steps:

1. Run `scripts/run_vertex_equivalence_verification.ps1` in the project `.venv`.
2. Inspect `scout_data/neuroemo/vertex_equivalence_report.json` and verify:
   - `proof_status == "projection_equivalence_verified"`;
   - a concrete BOLD sample path or sample fingerprint is recorded;
   - numeric tolerance, max error, and pass/fail details are present;
   - `report_sha256` changes and is propagated.
3. Regenerate or update artifacts that embed the proof reference:
   - `configs/parcellation_manifest.yaml`;
   - combined and subject-level NeuroEmo NPZ metadata if needed;
   - any training metrics regenerated after this phase.
4. Rename or document any status that currently sounds like volume-projection success when it only proves mesh-coordinate identity.
5. Add or tighten tests in `tests/test_vertex_equivalence.py`, `tests/test_neuroemo_training.py`, and `tests/test_build_schaefer_surface_annot.py`.

Exit criteria:

- Tests pass for vertex equivalence and NeuroEmo training contract.
- The manifest, training NPZ, and metrics all reference the same proof report hash.
- Docs can honestly say projection equivalence is verified, or they explicitly say it remains unavailable.

## Phase 2 - Full Surface-Native Postfix Matrix

**Goal:** decide whether the surface-native atlas helps modeling when compared fairly across all previously useful model families.

Run the surface-native equivalents of the May 25 postfix matrix under the same contract:

- train NPZ: `scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz`
- atlas: current `configs/vertex_regions.csv` and `configs/parcellation_manifest.yaml`
- labels: exclude `neutral`
- temporal window: `10`
- temporal contiguity: `contiguous`
- ROI reducers: `mean,std,mean_abs`
- CV: subject-held-out 5-fold

Required runs:

- `logistic_saga_5class_10tr`
- `mlp_5class_10tr`
- `sgd_logistic_5class_10tr`
- `linear_svc_5class_10tr`
- `specialist_none_5class_10tr`
- `specialist_sigmoid_5class_10tr`
- `mlp_valence_binary_10tr`

Implementation steps:

1. Add a small matrix runner script or document exact commands in a session note to avoid ad hoc command drift.
2. Write outputs under a single dated directory, for example `scout_data/neuroemo/models/2026-05-27_surface_annot_matrix/`.
3. Generate a comparison report against `2026-05-25_postfix_matrix`.
4. Include:
   - mean accuracy;
   - mean balanced accuracy;
   - mean macro F1;
   - pooled log loss;
   - pooled confusion matrix;
   - per-class precision/recall/F1;
   - proof and manifest hashes.

Exit criteria:

- A single report answers whether surface-native should become the modeling default.
- The report explicitly states if the choice is a metric win, a provenance tradeoff, or inconclusive.

## Phase 3 - Timing Grid

**Goal:** reduce label noise before adding more model complexity.

The prep path already records `bold_lag_s`, `drop_transition_trs`, `t_idx`, and `time_s`. Use that support to test timing assumptions directly.

Suggested grid:

- `--bold-lag-s`: `2`, `4`, `6`, `8`
- `--drop-transition-trs`: `0`, `1`, `2`
- `--temporal-stride-trs`: `1`, `2`, `5`, `10`

Keep fixed:

- best atlas choice from Phase 2;
- `logistic_saga` first, then the best MLP only for finalist timing contracts;
- subject-held-out folds;
- `mean,std,mean_abs` reducers until dynamic features are introduced.

Implementation steps:

1. Add a reproducible timing matrix runner or config file.
2. Ensure each prepared NPZ stores timing metadata and each training run refuses accidental re-windowing unless explicitly requested.
3. Rank timing settings by macro F1 and hard-class confusion, especially `afraid`, `calm`, and delighted/excited confusion.
4. Promote one timing contract only if it is stable across repeated seeds or fold-level inspection.

Exit criteria:

- A timing report identifies the chosen lag/drop/stride contract or recommends staying with the current contract.
- The chosen timing contract is documented in the next model artifact metadata.

## Phase 4 - Dynamic Window Features

**Goal:** stop discarding useful within-window shape.

The trainer currently reduces a 10-TR window with `mean`, `median`, or `last` before ROI reduction. Add feature modes that preserve simple dynamics without requiring a large sequence model.

Candidate temporal reducers/features:

- `early_mean`
- `late_mean`
- `late_minus_early`
- `slope`
- `within_window_std`
- `last_minus_first`
- optional concatenated static + dynamic bundle

Implementation steps:

1. Extend `scripts/train_neuroemo_emotion_model.py` temporal reduction with named dynamic bundles.
2. Record exact temporal reducer names and resulting feature dimensionality in metrics JSON.
3. Add tests for:
   - output shape;
   - deterministic reducer math;
   - metadata recording;
   - backwards-compatible current static baseline.
4. Compare dynamic features against the static baseline using the best timing/atlas contract.

Exit criteria:

- Dynamic features can reproduce the old static baseline and add new provenance.
- At least one dynamic bundle is tested under the frozen comparison matrix.

## Phase 5 - Preprocessing And ROI Strategy A/B

**Goal:** test representation choices after the atlas/proof/timing contract is stable.

Candidate A/Bs:

- raw vs `--preprocess-bold`;
- medial-wall nearest-neighbor fill vs excluding unassigned vertices from ROI reducers;
- Schaefer 200 vs 400 vs 600 if annot files are available;
- 7-network vs 17-network variants;
- network-level aggregates;
- left-right asymmetry features.

Implementation steps:

1. Prioritize low-blast-radius toggles first: raw/preprocessed and medial-wall exclude vs fill.
2. Add manifest fields for any ROI strategy that changes how vertices contribute to parcels.
3. Keep model, timing, folds, and labels fixed for each A/B.
4. Report changes by class, not only overall metrics.

Exit criteria:

- The project has a documented ROI/preprocessing default with evidence.
- Any promoted ROI strategy is represented in manifest metadata and tests.

## Phase 6 - Structured Decoding And Specialists

**Goal:** use the observed binary-valence advantage instead of forcing every decision through one flat 5-way argmax.

Candidate designs:

- valence-first classifier, then class refinement within broad groups;
- one-vs-rest or pairwise refiners only for high-confusion pairs;
- specialist logits/probabilities as second-stage features rather than direct argmax replacement;
- optional joint report that compares flat and hierarchical predictions on identical folds.

Implementation steps:

1. Define a stable class grouping, for example calm / negative / positive, with explicit handling for `afraid`, `depressed`, `delighted`, and `excited`.
2. Train the coarse classifier and within-group refiners under the best upstream contract.
3. Compare against flat 5-way on the same held-out subjects.
4. Use specialists only where confusion supports them, such as calm-afraid or delighted-excited/depressed leakage.

Exit criteria:

- A structured-decoding report shows whether hierarchy improves macro F1 or hard-class recall.
- If hierarchy loses overall accuracy, keep the flat model and retain binary valence as a separate useful endpoint.

## Final Promotion Checklist

Before replacing the current best artifact:

- Metrics beat the fixed projected postfix baseline or justify a provenance-over-metric tradeoff.
- The model artifact records atlas manifest hash, vertex proof hash, timing contract, ROI reducer contract, and preprocessing flag.
- Tests cover any new metadata gates or feature reducers.
- Docs include a short model-card-style caveat that NeuroEmo outputs are emotion-state hypotheses, not clinical measurements.
