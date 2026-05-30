# Coding Session - May 23, 2026

## NeuroEmo Training Pipeline Upgrade

**Scope:** End-to-end summary of the NeuroEmo supervised training pipeline upgrades so far: dataset repair, neutral-label handling, Schaefer ROI extraction, temporal windowing, and optional lightweight fMRI preprocessing.

This note is written as a practical handoff. It includes simple definitions for the neuroscience/ML terms used in the implementation.

---

## 1. Goal

The project goal is to train a supervised emotion-recognition model from NeuroEmo fMRI data and use it with TribeV2-style outputs.

The shared feature shape is:

```text
T x 20484
```

where:

- `T` is time;
- `20484` is the number of `fsaverage5` cortical surface vertices;
- left hemisphere vertices come first, then right hemisphere vertices.

The model currently predicts NeuroEmo emotion classes:

```text
calm
afraid
delighted
depressed
excited
```

`neutral` exists in the prepared dataset, but is excluded during training by default.

---

## 2. Data Flow

The pipeline is:

```text
raw NeuroEmo BOLD NIfTI
  -> optional lightweight preprocessing
  -> volume-to-surface projection
  -> TribeV2-compatible surface arrays
  -> ROI feature extraction
  -> optional temporal window aggregation
  -> supervised classifier training
```

Key files:

```text
scripts/prepare_neuroemo_tribev2.py
scripts/train_neuroemo_emotion_model.py
scripts/build_schaefer_vertex_regions.py
scout_core/roi_features.py
scout_core/parcellation.py
configs/vertex_regions.csv
configs/parcellation_manifest.yaml
```

---

## 3. Simple Definitions

### BOLD

`BOLD` is the fMRI signal. It measures blood oxygen changes that are indirectly related to neural activity.

Simple idea:

```text
brain activity changes blood oxygen
scanner measures that oxygen-related signal
```

### TR

`TR` means repetition time.

In this dataset:

```text
TR = 3 seconds
```

So one TR is one whole-brain fMRI timepoint.

### Surface Projection

Surface projection converts 3D brain volume data into values on the cortical surface mesh.

Simple idea:

```text
3D voxel brain image -> 20484 surface vertex values
```

This makes NeuroEmo data match the TribeV2 output format.

### ROI

`ROI` means region of interest.

Instead of feeding all 20,484 vertex values into the model, vertices are grouped into brain parcels.

Simple idea:

```text
many nearby/related vertices -> one region value
```

### Parcellation

A parcellation is a map that assigns each surface vertex to a brain parcel.

Simple idea:

```text
vertex 0 belongs to parcel 57
vertex 1 belongs to parcel 81
...
```

### Schaefer Atlas

The Schaefer 2018 atlas is a commonly used functional brain parcellation.

The current upgraded ROI setup uses:

```text
Schaefer2018 400 parcels, 7 networks
```

### Spatial Smoothing

This blurs nearby brain voxels together a little.

Simple idea:

```text
replace each voxel with an average of itself and nearby voxels
```

Why: fMRI is noisy, and neighboring brain locations often carry related signal.

### Temporal Detrending

This removes slow drift over time.

Simple idea:

```text
if the signal slowly rises or falls during the scan, subtract that trend
```

Why: scanner signal can drift for reasons unrelated to emotion.

### Temporal Standardization

This puts each voxel time series on a comparable scale.

Simple idea:

```text
subtract the voxel average
divide by the voxel variation
```

Why: it turns raw values into "how unusually high or low is this signal right now?"

### High-Pass Filtering

This removes very slow changes.

Simple idea:

```text
ignore changes that happen too slowly
```

Why: very slow fMRI waves are often drift or nuisance signal.

### Low-Pass Filtering

This removes very fast changes.

Simple idea:

```text
ignore changes that happen too quickly
```

Why: BOLD fMRI is slow, so very fast fluctuations are usually noise.

---

## 4. Dataset Repairs

The dataset was repaired after interrupted NeuroEmo preparation runs.

At one point, only subjects `1-10` were represented in the combined training file. Later, subjects `26-40` overwrote aggregate metadata/training files. The per-subject surface files were still present, so the aggregate files were rebuilt from existing subject NPZs.

Final repaired subject coverage:

```text
sub-01 through sub-40
```

With neutral included in the prepared dataset:

```text
4800 samples
6 classes
800 samples per class
```

With neutral excluded during training:

```text
4000 samples
5 classes
800 samples per class
```

---

## 5. Neutral Label Handling

`neutral` was not a raw NeuroEmo emotion condition.

The raw task schedule includes:

```text
calm
afraid
delighted
depressed
excited
white_noise
```

The pipeline created `neutral` by selecting a balanced subset of white-noise TRs.

Simple distinction:

```text
white_noise = original task condition
neutral = derived training label from selected white-noise TRs
unlabeled = ignored during supervised training
```

Because `neutral` is more like a white-noise baseline than a true felt-emotion label, training now excludes it by default:

```bash
--exclude-labels neutral
```

To include it again:

```bash
python scripts/train_neuroemo_emotion_model.py --exclude-labels ""
```

---

## 6. ROI Upgrade

The original ROI file was synthetic/demo:

```text
configs/vertex_regions.csv
```

It had the right shape but was not a real atlas.

The upgraded pipeline added:

```text
scripts/build_schaefer_vertex_regions.py
scout_core/roi_features.py
```

The current ROI map is:

```text
Schaefer2018 400 parcels, 7 networks, fsaverage5
```

The trainer now validates:

```text
20484 vertices
10242 left hemisphere vertices
10242 right hemisphere vertices
400 parcels
CSV hash matches manifest
```

The trained model artifact stores ROI metadata so inference can later reproduce the same feature extraction.

---

## 7. ROI Feature Extraction

Default ROI reducer:

```text
mean
```

Simple idea:

```text
average all vertex values inside each parcel
```

So:

```text
20484 vertices -> 400 parcel means
```

The trainer also supports richer ROI features:

```bash
--roi-reducers mean,std,mean_abs
```

That produces:

```text
400 parcels x 3 summaries = 1200 features
```

Simple meanings:

```text
mean = average activation in parcel
std = how spread out the parcel values are
mean_abs = average signal magnitude, ignoring sign
```

---

## 8. Temporal Windowing

Training can now aggregate TR-level rows into larger windows without rebuilding the dataset.

Example:

```bash
python scripts/train_neuroemo_emotion_model.py \
  --temporal-window-trs 10 \
  --temporal-contiguity contiguous
```

Simple idea:

```text
10 consecutive same-label TRs -> one averaged training sample
```

Since NeuroEmo blocks are 30 seconds and `TR = 3s`:

```text
10 TRs = 30 seconds = one full task block
```

With neutral excluded, a 10-TR contiguous window produces roughly:

```text
40 subjects x 5 emotions x 2 blocks = 400 samples
```

This is effectively block-level training.

---

## 9. Temporal Contiguity Modes

The trainer supports:

```text
contiguous
same_label
```

### contiguous

Requires:

```text
same subject
same label
consecutive TR indices
```

This is the scientifically cleaner option.

### same_label

Requires:

```text
same subject
same label
```

but does not require consecutive TRs.

This was useful when `neutral` was included, because neutral samples were sparse. With neutral removed, `contiguous` is preferred.

---

## 10. Optional Lightweight Preprocessing

The prep script now supports optional preprocessing before surface projection:

```bash
python scripts/prepare_neuroemo_tribev2.py \
  --subjects 1-40 \
  --skip-download \
  --raw-dir scout_data/neuroemo/raw \
  --out-dir scout_data/neuroemo/tribev2_surface_preprocessed \
  --preprocess-bold \
  --preprocessed-dir scout_data/neuroemo/preprocessed
```

This does:

```text
spatial smoothing
temporal detrending
temporal standardization
high-pass filtering
low-pass filtering
```

The smoke test succeeded for `sub-01` and produced:

```text
surface shape: (200, 20484)
labeled TRs: 100
```

Important caveat:

This is not full fMRI preprocessing. It does not replace motion correction, slice timing correction, anatomical coregistration, or full normalization. It is a lightweight improvement available from the current raw BOLD files.

---

## 11. Current Model

The model is a supervised linear classifier.

Default pipeline:

```text
surface sample
  -> ROI reduction
  -> StandardScaler
  -> SGDClassifier(loss="log_loss")
```

Simple idea:

```text
learn a weighted pattern of brain-region values for each emotion class
```

Default training now excludes neutral:

```text
calm
afraid
delighted
depressed
excited
```

---

## 12. Key Commands

Train default five-class ROI model:

```bash
python scripts/train_neuroemo_emotion_model.py --max-iter 3000
```

Train with 10-TR contiguous block-like windows:

```bash
python scripts/train_neuroemo_emotion_model.py \
  --max-iter 3000 \
  --temporal-window-trs 10 \
  --temporal-contiguity contiguous
```

Train with richer ROI summaries:

```bash
python scripts/train_neuroemo_emotion_model.py \
  --max-iter 3000 \
  --temporal-window-trs 10 \
  --temporal-contiguity contiguous \
  --roi-reducers mean,std,mean_abs
```

Include neutral again:

```bash
python scripts/train_neuroemo_emotion_model.py \
  --exclude-labels ""
```

Build Schaefer ROI table:

```bash
python scripts/build_schaefer_vertex_regions.py \
  --n-rois 400 \
  --yeo-networks 7
```

Prepare preprocessed surface dataset:

```bash
python scripts/prepare_neuroemo_tribev2.py \
  --subjects 1-40 \
  --skip-download \
  --raw-dir scout_data/neuroemo/raw \
  --out-dir scout_data/neuroemo/tribev2_surface_preprocessed \
  --preprocess-bold \
  --preprocessed-dir scout_data/neuroemo/preprocessed
```

Train against preprocessed dataset:

```bash
python scripts/train_neuroemo_emotion_model.py \
  --train-npz scout_data/neuroemo/tribev2_surface_preprocessed/neuroemo_tribev2_train.npz \
  --temporal-window-trs 10 \
  --temporal-contiguity contiguous \
  --max-iter 3000
```

---

## 13. Current Accuracy Interpretation

Accuracy is still modest.

This is not surprising because:

- fMRI BOLD is noisy;
- the original data are raw BIDS task files;
- labels are stimulus labels, not direct reports of felt emotion;
- cross-subject decoding is difficult;
- emotion classes can overlap;
- current preprocessing is still lightweight.

The best next comparison is:

```text
raw surface dataset vs preprocessed surface dataset
```

using the same training command:

```text
neutral excluded
Schaefer 400 ROI
10-TR contiguous windows
subject-held-out CV
```

---

## 14. Next Recommended Experiments

1. Finish generating the preprocessed surface dataset.
2. Train the same 10-TR contiguous model on raw vs preprocessed surfaces.
3. Compare:

```text
pooled balanced accuracy
macro F1
confusion matrix
per-class F1
```

4. If preprocessing helps, try:

```text
Schaefer 200
Schaefer 400
Schaefer 600
mean vs mean,std,mean_abs
```

5. If preprocessing does not help, the next big step is full fMRI preprocessing with motion correction and proper normalization before surface projection.
