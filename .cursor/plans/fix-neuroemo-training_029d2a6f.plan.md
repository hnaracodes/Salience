---
name: fix-neuroemo-training
overview: Harden the NeuroEmo ROI training path around semantic vertex alignment, surface-native Schaefer labels, explicit temporal-contract enforcement, atlas metadata correctness, and stronger linear-model baselines without changing code yet.
todos:
  - id: vertex-order-enforcement
    content: Add semantic vertex-order and hemisphere-boundary validation for parcellation artifacts.
    status: pending
  - id: surface-native-schaefer
    content: Switch the Schaefer builder to surface-native labels and regenerate atlas artifacts with full provenance.
    status: pending
  - id: temporal-contract
    content: Make dataset prep/trainer temporal settings explicit and reject silent double-windowing.
    status: pending
  - id: atlas-metadata-cleanup
    content: Fix atlas label/network parsing and remove unstable legacy parcellation fallbacks.
    status: pending
  - id: linear-baselines
    content: Add LinearSVC-style baselines and standardize experiment reporting for fair comparisons.
    status: pending
isProject: false
---

# NeuroEmo Training Fix Plan

## Goal
Fix the five training issues we discussed while preserving your teammate's existing block-level setup: treat temporal aggregation as a contract to validate and document, not as a new default to impose.

## Scope
Target the current ROI training stack centered on [scripts/build_schaefer_vertex_regions.py](scripts/build_schaefer_vertex_regions.py), [scout_core/parcellation.py](scout_core/parcellation.py), [scout_core/roi_features.py](scout_core/roi_features.py), [scripts/prepare_neuroemo_tribev2.py](scripts/prepare_neuroemo_tribev2.py), and [scripts/train_neuroemo_emotion_model.py](scripts/train_neuroemo_emotion_model.py).

## Workstreams
### 1. Enforce real vertex-order compatibility
Problem addressed: the current validator proves only `0..20483` contiguity and `lh/rh` counts; it does not prove the CSV matches TribeV2's semantic vertex order.

Implementation steps:
- Add a stricter validation layer in [scout_core/parcellation.py](scout_core/parcellation.py) that checks hemisphere boundary ordering, not just counts.
- Add a mesh-order verification utility that compares the atlas/source surface ordering against the same `fsaverage5` ordering assumed by Tribe outputs and records the result in the validation summary.
- Fail training early in [scout_core/roi_features.py](scout_core/roi_features.py) / [scripts/train_neuroemo_emotion_model.py](scripts/train_neuroemo_emotion_model.py) if the parcellation cannot prove compatibility with the expected Tribe ordering.
- Expose this check in a small CLI or validation mode so atlas generation can be verified before training runs.

### 2. Replace projected Schaefer labels with surface-native labels
Problem addressed: [configs/parcellation_manifest.yaml](configs/parcellation_manifest.yaml) shows the current atlas came from `vol_to_surf` plus nearest-neighbor fill, including `filled_unassigned_vertices: 2487`.

Implementation steps:
- Update [scripts/build_schaefer_vertex_regions.py](scripts/build_schaefer_vertex_regions.py) to support a surface-native Schaefer input path (`.label.gii` left/right hemisphere labels) as the primary route.
- Keep the current volumetric projection path only as an explicit fallback/debug mode, not the default.
- Regenerate [configs/vertex_regions.csv](configs/vertex_regions.csv) and [configs/parcellation_manifest.yaml](configs/parcellation_manifest.yaml) from the surface-native atlas, including source file hashes and the exact label source used.
- Propagate atlas provenance into the model artifact via [scout_core/roi_features.py](scout_core/roi_features.py) so downstream consumers can see whether training used surface-native or projected labels.

### 3. Lock the temporal contract instead of re-windowing blindly
Problem addressed: temporal handling currently exists in both [scripts/prepare_neuroemo_tribev2.py](scripts/prepare_neuroemo_tribev2.py) and [scripts/train_neuroemo_emotion_model.py](scripts/train_neuroemo_emotion_model.py). Since your teammate already trained on 10-TR block-level samples and lag handling is already accounted for, the fix is to prevent silent double-windowing or fake temporal assumptions.

Implementation steps:
- Extend the combined NPZ metadata produced by [scripts/prepare_neuroemo_tribev2.py](scripts/prepare_neuroemo_tribev2.py) so it explicitly records `window_trs`, whether samples are already block/window aggregated, lag settings, transition-drop settings, and any block/run identifiers available.
- Update `_load_training_data()` and `_apply_temporal_windows()` in [scripts/train_neuroemo_emotion_model.py](scripts/train_neuroemo_emotion_model.py) to detect pre-windowed data and either:
  - reject incompatible second-stage temporal aggregation, or
  - require an explicit override flag for diagnostics.
- Stop synthesizing temporal context when temporal aggregation is requested. If the trainer is asked to aggregate but true `t_idx` / `time_s` / block metadata are missing, fail loudly.
- Add artifact/metrics fields that preserve both prep-time and train-time temporal settings so experiment outputs show whether the model was trained on single TRs, 3-TR summaries, 10-TR blocks, or an additional reduction step.

### 4. Clean up atlas metadata bugs and validator gaps
Problem addressed: current atlas metadata is fragile in several places: the Schaefer builder's network parsing is 7-network specific, legacy CSV fallback uses unstable Python `hash()`, and validators do not detect conflicting parcel labels or mixed hemisphere ordering.

Implementation steps:
- Fix the label/network parsing path in [scripts/build_schaefer_vertex_regions.py](scripts/build_schaefer_vertex_regions.py) so atlas metadata is deterministic and correct for the supported atlas variants.
- Tighten [scout_core/parcellation.py](scout_core/parcellation.py) to detect conflicting `parcel_id -> parcel_label` or `parcel_id -> network` mappings instead of silently accepting the first row.
- Remove or quarantine unstable legacy behavior in `load_vertex_table()` for old two-column CSVs, especially the process-randomized `hash(region)` fallback.
- Add validation checks that the first 10,242 vertices are `lh` and the second 10,242 are `rh`, since the current validator only checks totals.

### 5. Add the intended linear-model baselines and comparison gates
Problem addressed: the main trainer currently supports only `sgd_logistic` and `logistic_saga`, even though the migration direction and past discussion pointed toward stronger linear SVM-style baselines.

Implementation steps:
- Extend `_make_estimator()` in [scripts/train_neuroemo_emotion_model.py](scripts/train_neuroemo_emotion_model.py) to support at least:
  - `linear_svc` (with calibration if probabilities are required),
  - `logistic_saga`,
  - existing `sgd_logistic` as a baseline.
- Keep ROI features fixed while comparing models so classifier changes are isolated from atlas/feature changes.
- Standardize experiment outputs in the metrics JSON: balanced accuracy, macro F1, per-class F1, confusion matrix, fold variance, temporal settings, atlas id, and ROI reducer set.
- Define a single benchmark matrix that compares old projected-Schaefer + SGD against fixed-atlas + linear baselines before promoting any new default.

## Validation
- Add focused tests around:
  - parcellation semantic-order validation,
  - surface-native label ingestion,
  - rejection of double temporal aggregation,
  - artifact metadata completeness,
  - classifier selection and probability output behavior.
- Run controlled experiment sets where only one axis changes at a time:
  1. current atlas/current model,
  2. fixed atlas/current model,
  3. fixed atlas + `linear_svc`,
  4. fixed atlas + alternate ROI reducers.

## Suggested execution order
1. Vertex-order enforcement in [scout_core/parcellation.py](scout_core/parcellation.py).
2. Surface-native Schaefer builder in [scripts/build_schaefer_vertex_regions.py](scripts/build_schaefer_vertex_regions.py).
3. Temporal-contract enforcement across [scripts/prepare_neuroemo_tribev2.py](scripts/prepare_neuroemo_tribev2.py) and [scripts/train_neuroemo_emotion_model.py](scripts/train_neuroemo_emotion_model.py).
4. Atlas metadata/legacy cleanup in [scout_core/parcellation.py](scout_core/parcellation.py).
5. Linear-model baseline expansion and experiment reruns in [scripts/train_neuroemo_emotion_model.py](scripts/train_neuroemo_emotion_model.py).

## Expected outcome
After this plan, low accuracy should no longer be confounded by unknown vertex misalignment, noisy projected parcel labels, ambiguous temporal windowing, fragile atlas metadata, or an underpowered/default-only classifier choice. The remaining model quality will be attributable to the data and feature representation rather than hidden pipeline errors.