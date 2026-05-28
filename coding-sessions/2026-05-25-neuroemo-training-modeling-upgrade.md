# Coding Session - May 25, 2026

## NeuroEmo Training Pipeline Modeling Upgrade

**Scope:** Changes made after `coding-sessions/2026-05-23-neuroemo-training-roi-neutral.md`.

This session continued the NeuroEmo-to-TribeV2 modeling work after the first ROI/neutral-class training setup. The main theme was moving from a simple ROI mean classifier toward a more realistic experimentation pipeline: atlas-backed ROIs, richer ROI summaries, temporal windows, optional BOLD preprocessing, MLP experiments, label-merging experiments, and one-vs-rest specialist classifiers.

The work also clarified several important modeling assumptions, especially around subject-level validation, temporal leakage, neutral/unlabeled labels, and why high validation scores can be misleading if samples from the same subject are split across train and validation.

---

## 1. Starting Point

The previous session left the project with:

- prepared NeuroEmo surface data in TribeV2-style fsaverage5 vertex order;
- a baseline training script at `scripts/train_neuroemo_emotion_model.py`;
- ROI support using `configs/vertex_regions.csv`;
- a balanced synthetic `neutral` class derived from white-noise TRs;
- subject-held-out cross-validation using subject groups;
- model metrics written to `scout_data/neuroemo/models/neuroemo_emotion_metrics.json`.

At that point, the ROI mapping was still a temporary compatibility mapping. The main open question was whether stronger ROI extraction and temporal aggregation could improve the low emotion-classification accuracy.

---

## 2. Atlas-Backed ROI Extraction Plan

Created:

```text
docs/implementation-plans/improved-roi-extraction-pipeline-plan.md
```

This plan documented the next upgrade for ROI extraction.

The key decision was to replace the synthetic/demo ROI mapping with a real surface-compatible atlas mapping. The recommended atlas was:

```text
Schaefer 2018
400 parcels
7 Yeo networks
fsaverage5 surface space
```

The plan covered:

- generating a stable vertex-to-parcel table;
- recording atlas metadata in a manifest;
- validating vertex counts and ordering;
- extracting richer ROI features;
- preserving compatibility with future TribeV2 inference;
- comparing model accuracy before and after the ROI upgrade.

Simple definition:

**Schaefer atlas:** A published brain parcellation that divides cortex into many functional regions. Instead of treating 20,484 surface vertices independently, the model can summarize activity inside known functional parcels.

---

## 3. Schaefer Vertex Region Generation

Created:

```text
scripts/build_schaefer_vertex_regions.py
```

This script generates the atlas-backed vertex mapping used by the training pipeline.

It uses Nilearn to fetch the Schaefer 2018 atlas, then projects the volumetric atlas to fsaverage5 surfaces.

Important behavior:

- fetches Schaefer 2018 from Nilearn;
- uses 400 parcels by default;
- uses 7 Yeo networks by default;
- projects to fsaverage5 left and right pial surfaces;
- preserves TribeV2 vertex order: left hemisphere first, right hemisphere second;
- writes the generated table to `configs/vertex_regions.csv`;
- writes atlas metadata to `configs/parcellation_manifest.yaml`;
- fills background/unassigned projected vertices by nearest assigned vertex within the same hemisphere.

The command used for the default mapping:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/build_schaefer_vertex_regions.py --n-rois 400 --yeo-networks 7
```

Generated/updated:

```text
configs/vertex_regions.csv
configs/parcellation_manifest.yaml
```

Simple definition:

**Vertex region table:** A CSV that says which atlas parcel each fsaverage5 surface vertex belongs to.

Simple definition:

**Manifest:** A small metadata file that records where the atlas came from, how many vertices/parcels are expected, and enough hashes/settings to catch accidental mismatches later.

---

## 4. Atlas Cache Ignore Rule

Updated:

```text
.gitignore
```

Added:

```text
scout_data/atlases/
```

Reason:

Nilearn atlas downloads and generated cache files are large, reproducible artifacts. They should not be committed to the repository.

---

## 5. Parcellation Validation Utilities

Updated:

```text
scout_core/parcellation.py
```

This file now has stronger support for validating and loading atlas-based vertex mappings.

Added concepts:

```text
EXPECTED_FSAVERAGE5_VERTICES = 20484
EXPECTED_FSAVERAGE5_HEMI_VERTICES = 10242
```

Added helpers:

- `file_sha256`
- `load_parcellation_manifest`
- `validate_vertex_table`
- `validate_manifest_matches_table`

Expanded `VertexParcellationTable` with:

- `parcel_ids`
- `n_parcels`
- `parcel_labels_by_id`
- `network_by_parcel_id`

Why this matters:

The model only makes sense if the NeuroEmo prepared surfaces and TribeV2 outputs refer to the same vertex order. These checks make it harder to silently train against the wrong shape, wrong atlas, or wrong vertex table.

Simple definition:

**Vertex compatibility:** The same row/column index must mean the same brain surface point in both training data and TribeV2 inference data.

Update after the vertex-equivalence hardening pass:

- `scout_core.vertex_equivalence` now defines a canonical proof schema and report reference contract.
- `scripts/verify_tribe_vertex_equivalence.py` is Modal-first and is intended to generate the proof artifact once per dependency/runtime revision.
- `scripts/prepare_neuroemo_tribev2.py` now stores `vertex_equivalence` report references in subject and combined NPZs by consuming the pinned artifact.
- `configs/parcellation_manifest.yaml` now includes mesh fingerprints plus a `vertex_equivalence` block.
- `scripts/train_neuroemo_emotion_model.py` now rejects missing or contract-only proof metadata by default, with an explicit legacy override flag for older artifacts.

---

## 6. Shared ROI Feature Module

Created:

```text
scout_core/roi_features.py
```

This extracted ROI feature logic out of the training script and into reusable core code.

Main pieces:

- `RoiFeatureSpec`
- `SUPPORTED_REDUCERS`
- `parse_reducers`
- `load_roi_feature_spec`
- `reduce_vertices_to_rois`
- `summarise_roi_feature_spec`

Supported reducers:

```text
mean
std
mean_abs
max_abs
```

This lets the model represent each parcel with more than one number.

For example, with 400 parcels and three reducers:

```text
400 parcels * 3 reducers = 1200 ROI features per TR/window
```

Simple definitions:

**ROI reducer:** A function that turns many vertex values inside one parcel into one summary value.

**mean:** The average activation in the parcel.

**std:** How spread out the activation values are inside the parcel.

**mean_abs:** The average absolute activation. This captures signal strength regardless of whether it is positive or negative.

**max_abs:** The strongest absolute activation in the parcel.

---

## 7. Linear Training Script Upgrades

Updated:

```text
scripts/train_neuroemo_emotion_model.py
```

This remains the main classical ML training script.

Important changes:

- switched ROI logic to `scout_core.roi_features`;
- added atlas manifest support;
- added multiple ROI reducers;
- added temporal window aggregation;
- added label exclusion;
- disabled `neutral` by default;
- recorded richer model and feature metadata in metrics/model artifacts;
- updated Modal packaging to include atlas configuration and `pyyaml`.

New/default configuration fields included:

```text
roi_reducers
atlas_manifest
temporal_window_trs
temporal_stride_trs
temporal_reducer
temporal_contiguity
temporal_allow_class_drop
exclude_labels
```

The old single-reducer flag was kept as a compatibility alias:

```text
--roi-reducer
```

but the preferred flag is now:

```text
--roi-reducers mean,std,mean_abs
```

Simple definition:

**Temporal window:** Instead of training on one TR at a time, combine several nearby TRs into one sample.

Simple definition:

**TR:** Repetition time. In fMRI, this is one time step of the BOLD recording.

Simple definition:

**Contiguous window:** A window where the TRs are next to each other in time, such as TR 40 through TR 49.

Simple definition:

**Same-label window:** A window where all TRs have the same label, even if they are not directly adjacent. This is less strict and can include more data, but it is less faithful to actual time flow.

Useful commands:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_emotion_model.py --max-iter 3000
```

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_emotion_model.py --temporal-window-trs 10 --temporal-contiguity contiguous --max-iter 3000
```

---

## 8. Neutral Label Disabled For Training

The `neutral` class was disabled by default during training:

```text
exclude_labels = neutral
```

Reason:

The neutral class was not part of the original NeuroEmo emotion labels. It was created from white-noise blocks. That made it useful as an experiment, but not a clean discrete emotion label.

Important distinction:

**Raw emotion labels:** The stimulus-condition labels from the NeuroEmo task, such as calm, afraid, delighted, depressed, and excited.

**Neutral:** A synthetic training class we created from white-noise blocks.

**Unlabeled:** TRs that exist in the scan but are not used as supervised training samples.

**White-noise:** A stimulus block in the task that can be used as a candidate non-emotion baseline, but it is not the same as a measured emotional state.

---

## 9. Temporal Windowing Added In Training, Not Data Preparation

Temporal windowing was implemented inside the training script instead of rewriting the prepared dataset.

This was intentional.

The prepared `.npz` files remain flexible TR-by-TR data:

```text
X[t, 20484]
labels[t]
subjects[t]
t_idx[t]
```

Then the trainer can try different window sizes without rerunning the expensive volume-to-surface projection.

This allows experiments such as:

```bash
--temporal-window-trs 2
--temporal-window-trs 6
--temporal-window-trs 8
--temporal-window-trs 10
```

Tradeoff:

Larger windows can improve signal-to-noise because BOLD responses are slow and noisy. But larger windows also reduce the number of samples and blur timing. If windows overlap heavily, the train/test split must be subject-held-out to avoid inflated metrics.

---

## 10. Optional BOLD Preprocessing Before Surface Projection

Updated:

```text
scripts/prepare_neuroemo_tribev2.py
```

Added optional preprocessing before projecting volume data to the fsaverage5 surface.

New flags:

```text
--preprocess-bold
--preprocessed-dir
--force-preprocess
--smooth-fwhm
--no-detrend
--no-temporal-standardize
--high-pass
--low-pass
```

Default preprocessing choices:

```text
smooth_fwhm = 5mm
detrend = true
temporal_standardize = true
high_pass = 0.008 Hz
low_pass = 0.1 Hz
```

The script now stores cached preprocessed NIfTI files under:

```text
scout_data/neuroemo/preprocessed/
```

and can prepare a separate output dataset, for example:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/prepare_neuroemo_tribev2.py \
  --subjects 1-40 \
  --skip-download \
  --raw-dir scout_data/neuroemo/raw \
  --out-dir scout_data/neuroemo/tribev2_surface_preprocessed \
  --preprocess-bold \
  --preprocessed-dir scout_data/neuroemo/preprocessed
```

Training against the preprocessed output:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_emotion_model.py \
  --train-npz scout_data/neuroemo/tribev2_surface_preprocessed/neuroemo_tribev2_train.npz \
  --temporal-window-trs 10 \
  --temporal-contiguity contiguous \
  --max-iter 3000
```

Simple definitions:

**Spatial smoothing:** Blurs nearby brain locations together slightly. It can reduce noise, but too much smoothing can erase fine spatial patterns.

**Temporal detrending:** Removes slow upward or downward drift over time.

**Temporal standardization:** Puts each voxel/time series on a comparable scale, often by converting it to z-scores.

**High-pass filtering:** Removes very slow signal changes. In this setup, `0.008 Hz` means changes slower than roughly 125 seconds are reduced.

**Low-pass filtering:** Removes fast signal changes. In this setup, `0.1 Hz` means changes faster than roughly 10 seconds are reduced.

Why these filter values were conservative:

The emotion blocks are much slower than single-frame noise, so the low-pass filter keeps the slow BOLD-scale signal while reducing fast scanner/physiological noise. The high-pass filter is low enough to preserve task-block variation while removing very slow drift.

---

## 11. Preprocessing Path Bug Fix

Updated:

```text
scripts/prepare_neuroemo_tribev2.py
```

Added a helper to display paths safely:

```text
_display_path
```

This fixed an error seen after stopping preprocessing early and trying to complete metadata:

```text
ValueError: 'scout_data/neuroemo/preprocessed/sub-01/func/sub-01_task-fe_bold_preproc.nii.gz'
is not in the subpath of '/Users/cameron/TribeV2'
```

The issue was not the data itself. It was caused by trying to display a relative path as if it were guaranteed to be below the resolved project root. The helper now handles both absolute and relative paths more gracefully.

---

## 12. MLP Training Script

Created:

```text
scripts/train_neuroemo_mlp_model.py
```

This file trains a small neural network classifier using scikit-learn's `MLPClassifier`.

It reuses the same dataset loading and feature extraction path as the linear model:

- subject-held-out folds;
- Schaefer ROI features;
- temporal windows;
- label exclusion;
- optional label merging;
- metrics JSON output.

Default intent:

The MLP was intentionally small to reduce overfitting. This dataset is not large enough for a big deep learning model to be a clear win.

Default temporal behavior:

```text
temporal_window_trs = 2
temporal_contiguity = contiguous
```

Common stronger 5-class experiment:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_mlp_model.py \
  --temporal-window-trs 10 \
  --alpha 0.003 \
  --hidden-layers 32 \
  --roi-reducers mean,std,mean_abs
```

Simple definitions:

**MLP:** Multi-layer perceptron. A feed-forward neural network.

**Hidden layer:** A layer of learned intermediate features between the input and output.

**Alpha:** L2 regularization strength. Larger values penalize large weights more strongly and can reduce overfitting.

**Epoch:** One pass through the training data.

**Batch:** A smaller chunk of samples used for one weight update. Scikit-learn handles the batching internally for `MLPClassifier`.

---

## 13. MLP Epoch Metrics And Subject-Level Validation

Updated:

```text
scripts/train_neuroemo_mlp_model.py
```

The MLP script originally used sample-level validation behavior, which caused suspiciously high validation scores above 0.9 while held-out subject accuracy stayed low.

That was a leakage problem.

If samples from the same subject appear in both training and validation, the model can learn subject-specific patterns instead of emotion-specific patterns.

The fix was to make validation subject-level:

- split validation subjects out of each training fold;
- train only on training subjects;
- validate on held-out validation subjects;
- keep final fold testing subject-held-out as before.

Added helpers:

```text
_subject_validation_split
_fit_estimator
```

Epoch logs now print in real time, for example:

```text
[fold 1] Epoch 001/200: loss=..., subject_validation_score=...
```

The metrics JSON records the fit history so each run can be inspected later.

Simple definition:

**Subject-held-out validation:** The model is evaluated on people it did not see during fitting. This is much stricter and more honest for fMRI classification.

---

## 14. Label Merge Experiments

Updated:

```text
scripts/train_neuroemo_mlp_model.py
```

Added label-merging support.

New flags:

```text
--label-merge-preset none|valence3|arousal3
--label-merge
```

Example:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_mlp_model.py \
  --temporal-window-trs 10 \
  --alpha 0.01 \
  --hidden-layers 32 \
  --roi-reducers mean,std,mean_abs \
  --label-merge 'positive=delighted,excited;negative=afraid,depressed;calm=calm'
```

Binary valence experiment:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_mlp_model.py \
  --temporal-window-trs 10 \
  --alpha 0.01 \
  --hidden-layers 32 \
  --roi-reducers mean,std,mean_abs \
  --exclude-labels neutral,calm \
  --label-merge 'positive=delighted,excited;negative=afraid,depressed'
```

Why this was useful:

The 5-way discrete emotion task remained difficult. The merged-label experiments tested whether the model could at least recover broader emotional structure, such as positive vs negative valence.

Observed result:

The positive-vs-negative experiment reached roughly:

```text
mean_accuracy ~= 0.64
```

with a confusion matrix like:

```text
labels: positive, negative

[[ 96,  64],
 [ 51, 109]]
```

Interpretation:

The model was better than chance at valence classification, but that does not solve the original discrete-emotion problem. It shows there is some usable emotional signal, while also showing that fine-grained labels are harder.

---

## 15. Five-Class MLP Findings

The best repeated 5-class MLP settings found so far were approximately:

```bash
--temporal-window-trs 10
--alpha 0.003
--hidden-layers 32
--roi-reducers mean,std,mean_abs
```

Across a few random states, this averaged around:

```text
mean_accuracy ~= 0.40
```

This was better than early baselines but still low for reliable discrete emotion recognition.

General confusion-matrix pattern:

- depressed and excited were usually easier;
- afraid and delighted were weaker;
- calm was often confused with other classes;
- positive/negative grouping was easier than individual emotion classification.

Interpretation:

The model may be learning broad affective dimensions more readily than exact discrete categories.

---

## 16. Specialist Classifier Pipeline

Created:

```text
scripts/train_neuroemo_specialist_models.py
```

This script trains one binary model per emotion:

```text
calm vs other
afraid vs other
delighted vs other
depressed vs other
excited vs other
```

Then it combines them by choosing the label whose specialist assigns the highest positive probability.

Default configuration:

```text
exclude_labels = neutral
temporal_window_trs = 10
temporal_contiguity = contiguous
roi_reducers = mean,std,mean_abs
model_type = logistic
class_weight = balanced
```

Simple definition:

**Specialist classifier:** A binary model that only answers one question, such as "is this afraid or not afraid?"

Why try this:

The 5-class model has to learn all class boundaries at once. Specialist models can sometimes work better when each class has a different pattern of mistakes.

Initial result:

The combined specialist model was not clearly better than the MLP. It landed around:

```text
mean_accuracy ~= 0.40
```

Example combined confusion matrix:

```text
[[26,25, 3,17, 9],
 [30,22, 4,10,14],
 [ 8, 4,28,22,18],
 [10, 7, 9,44,10],
 [10,10,11, 9,40]]
```

Interpretation:

The specialist idea was structurally useful, but the raw probabilities were not calibrated well enough for a clean argmax combination.

---

## 17. Sigmoid Probability Calibration For Specialists

Updated:

```text
scripts/train_neuroemo_specialist_models.py
```

Added sigmoid probability calibration.

New flag:

```text
--calibration sigmoid
```

Optional comparison:

```text
--calibration none
```

The specialist pipeline now fits each binary classifier and then calibrates its scores using a held-out subject calibration split from inside the training fold.

Added helpers:

```text
_subject_calibration_split
_raw_positive_score
_fit_specialist_estimator
```

Each specialist now stores:

```text
estimator
calibrator
```

The metrics include calibration summaries such as:

- calibration subjects;
- number of calibration samples;
- calibrator coefficients;
- calibrator intercepts.

Simple definition:

**Sigmoid calibration:** A small probability-correction step. It learns how to turn a model's raw confidence score into a better-calibrated probability.

Simple definition:

**Platt scaling:** Another name for sigmoid calibration. It fits a logistic curve on validation data.

Why this matters:

The final specialist combo model compares probabilities across different binary classifiers. If one specialist is naturally overconfident, it can dominate the argmax even when it is wrong. Calibration tries to make the probabilities more comparable.

---

## 18. Inference Status

The previous inference script had been deleted intentionally:

```text
scripts/run_neuroemo_emotion_inference.py
```

It was not restored in this session.

Reason:

The training side was still changing rapidly. The safer path was to finish the feature, temporal, label, and model artifact design first, then add inference once the model format stabilizes.

Future inference should load the model artifact and obey the saved metadata:

- atlas manifest;
- vertex table hash;
- ROI reducers;
- temporal window size;
- temporal reducer;
- label merge map;
- excluded labels;
- class order.

This is important because inference must reproduce the exact same feature pipeline used during training.

---

## 19. Main Accuracy Lessons

The session produced several useful conclusions.

First, subject-level validation is required. Sample-level validation is too optimistic for this dataset.

Second, larger temporal windows helped up to around 10 TRs in current experiments. This is plausible because BOLD signals are slow and noisy.

Third, adding richer ROI reducers helped enough to keep using:

```text
mean,std,mean_abs
```

Fourth, the synthetic neutral class should stay disabled for now when the goal is original NeuroEmo discrete emotion classification.

Fifth, binary valence is much easier than 5-way discrete emotion classification. The model sees some broad emotional signal, but exact emotion labels remain hard.

Sixth, specialist models are worth keeping as an experimental path, especially after calibration, but they did not immediately solve the 5-way accuracy ceiling.

---

## 20. Current Recommended Commands

Build or rebuild the Schaefer vertex mapping:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/build_schaefer_vertex_regions.py --n-rois 400 --yeo-networks 7
```

Run the linear baseline:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_emotion_model.py --max-iter 3000
```

Run the linear baseline with 10-TR contiguous windows:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_emotion_model.py \
  --temporal-window-trs 10 \
  --temporal-contiguity contiguous \
  --max-iter 3000
```

Run the strongest current 5-class MLP experiment:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_mlp_model.py \
  --temporal-window-trs 10 \
  --alpha 0.003 \
  --hidden-layers 32 \
  --roi-reducers mean,std,mean_abs
```

Run the binary valence MLP experiment:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_mlp_model.py \
  --temporal-window-trs 10 \
  --alpha 0.01 \
  --hidden-layers 32 \
  --roi-reducers mean,std,mean_abs \
  --exclude-labels neutral,calm \
  --label-merge 'positive=delighted,excited;negative=afraid,depressed'
```

Run the calibrated specialist pipeline:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_specialist_models.py
```

Compare against uncalibrated specialists:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_specialist_models.py --calibration none
```

Prepare a preprocessed surface dataset:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/prepare_neuroemo_tribev2.py \
  --subjects 1-40 \
  --skip-download \
  --raw-dir scout_data/neuroemo/raw \
  --out-dir scout_data/neuroemo/tribev2_surface_preprocessed \
  --preprocess-bold \
  --preprocessed-dir scout_data/neuroemo/preprocessed
```

Train from the preprocessed dataset:

```bash
/Users/cameron/.pyenv/versions/tribev2/bin/python scripts/train_neuroemo_emotion_model.py \
  --train-npz scout_data/neuroemo/tribev2_surface_preprocessed/neuroemo_tribev2_train.npz \
  --temporal-window-trs 10 \
  --temporal-contiguity contiguous \
  --max-iter 3000
```

---

## 21. Files Changed Or Added Since The Previous Session

Created:

```text
docs/implementation-plans/improved-roi-extraction-pipeline-plan.md
scripts/build_schaefer_vertex_regions.py
scout_core/roi_features.py
scripts/train_neuroemo_mlp_model.py
scripts/train_neuroemo_specialist_models.py
coding-sessions/2026-05-25-neuroemo-training-modeling-upgrade.md
```

Updated:

```text
.gitignore
configs/vertex_regions.csv
configs/parcellation_manifest.yaml
scout_core/parcellation.py
scripts/train_neuroemo_emotion_model.py
scripts/prepare_neuroemo_tribev2.py
```

Generated or updated during experiments:

```text
scout_data/neuroemo/models/neuroemo_emotion_metrics.json
scout_data/neuroemo/models/neuroemo_mlp_emotion_metrics.json
scout_data/neuroemo/models/neuroemo_specialist_emotion_metrics.json
```

---

## 22. Recommended Next Step

The next practical step is to choose one primary 5-class training configuration and make it reproducible across several random states.

Suggested current candidate:

```text
MLP
Schaefer 400
ROI reducers: mean,std,mean_abs
temporal window: 10 contiguous TRs
hidden layers: 32
alpha: 0.003
exclude labels: neutral
subject-held-out validation
```

Then compare it against:

- calibrated specialist classifiers;
- logistic/linear baseline with identical features;
- preprocessed vs non-preprocessed surfaces on the same subject subset;
- confusion matrices, not only mean accuracy.

The most important metric is not just whether one run gets a higher number. The goal is to find a stable configuration whose confusion matrix shows real improvement on held-out subjects.
