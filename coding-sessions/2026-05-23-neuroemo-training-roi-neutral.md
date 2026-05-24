# Coding Session - May 23, 2026

## NeuroEmo Training, ROI Features, and Neutral Class

**Scope:** NeuroEmo supervised model training setup, ROI feature pipeline, dataset repair after interrupted preparation, and balanced neutral-class addition.

---

## 1. Goal

The goal was to move from prepared NeuroEmo surface data toward a trainable emotion-recognition model compatible with TribeV2 outputs.

The training target is a classifier that consumes TribeV2-style cortical arrays:

```text
X[t, 20484]
```

and predicts NeuroEmo-derived emotion labels.

---

## 2. Training Script

The main training file is:

```text
scripts/train_neuroemo_emotion_model.py
```

It reads:

```text
scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz
```

and writes:

```text
scout_data/neuroemo/models/neuroemo_emotion_model.joblib
scout_data/neuroemo/models/neuroemo_emotion_metrics.json
```

The trainer performs subject-held-out cross-validation using `GroupKFold`, then fits a final model on all training samples.

Default model:

```text
SGDClassifier(loss="log_loss")
```

Alternative model:

```bash
python3 scripts/train_neuroemo_emotion_model.py --model-type logistic_saga
```

---

## 3. ROI Pipeline

The training file now defaults to ROI features instead of full-vertex features.

Default behavior:

```text
20484 fsaverage5 vertices -> ROI features
```

Current ROI source:

```text
configs/vertex_regions.csv
```

Current reducer:

```text
mean activation per parcel
```

Default command:

```bash
python3 scripts/train_neuroemo_emotion_model.py
```

Full-vertex baseline:

```bash
python3 scripts/train_neuroemo_emotion_model.py --feature-mode vertices
```

ROI caveat:

The current `configs/vertex_regions.csv` is a compatibility/demo mapping, not a true anatomical or functional atlas. The recommended upgrade is to replace it with a real `fsaverage5` atlas mapping, likely Schaefer 2018:

```text
Schaefer2018 200 parcels
Schaefer2018 400 parcels
Schaefer2018 600 parcels
```

The first recommended experiment is Schaefer 400 parcels with mean ROI activation.

---

## 4. Dataset Repair To 25 Subjects

The NeuroEmo download/preparation process had been stopped before the aggregate metadata and training files were updated.

Per-subject prepared surface files existed for:

```text
sub-01 through sub-25
```

but the aggregate training NPZ still contained only:

```text
sub-01 through sub-10
```

The aggregate files were rebuilt from the existing per-subject surface NPZs, without re-running the expensive volume-to-surface projection.

After the first repair, the five-class dataset became:

```text
X: (2500, 20484)
subjects: 25
class counts:
  calm: 500
  afraid: 500
  delighted: 500
  depressed: 500
  excited: 500
```

Updated files:

```text
scout_data/neuroemo/tribev2_surface/metadata.json
scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz
scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_labels.csv
```

---

## 5. Balanced Neutral Class

A new neutral class was added from the white-noise blocks.

The preparation script now supports:

```bash
python3 scripts/prepare_neuroemo_tribev2.py --include-neutral
```

Neutral construction rule:

```text
Use a balanced subset of white-noise TRs.
Keep neutral count equal to each emotion class.
Leave extra white-noise TRs unlabeled.
```

The current generated dataset was rebuilt in place from the existing 25 subject surface files.

Current six-class dataset:

```text
X: (3000, 20484)
subjects: 25
labels:
  calm
  afraid
  delighted
  depressed
  excited
  neutral
```

Balanced class counts:

```text
calm: 500
afraid: 500
delighted: 500
depressed: 500
excited: 500
neutral: 500
```

Each subject contributes:

```text
20 samples per class
120 labeled samples total
```

---

## 6. Notes On Accuracy

Accuracy decreased after adding more subjects. This is plausible and not necessarily a regression in the code.

Likely reasons:

- subject-held-out evaluation became harder with 25 subjects;
- raw BOLD projection still contains substantial subject-specific noise;
- the current ROI mapping is not a real atlas;
- labels are stimulus-condition labels, not direct measured emotional states;
- white-noise/neutral may introduce a different and potentially confounding class boundary.

The most likely improvement path is:

```text
real fMRI preprocessing
real fsaverage5 atlas ROI mapping
block-level or GLM beta features
then model tuning
```

---

## 7. Current Inference Status

The separate inference script was intentionally deleted for now.

Inference should be reintroduced after the training portion stabilizes. When it returns, it should load the saved training artifact and apply the exact same feature pipeline:

```text
vertex order
ROI atlas
ROI reducer
label order
scaling/model pipeline
```

This is especially important once the ROI mapping is upgraded to a real atlas.

---

## 8. Useful Commands

Train default ROI model:

```bash
python3 scripts/train_neuroemo_emotion_model.py
```

Increase optimizer iterations:

```bash
python3 scripts/train_neuroemo_emotion_model.py --max-iter 3000
```

Train full-vertex baseline:

```bash
python3 scripts/train_neuroemo_emotion_model.py --feature-mode vertices
```

Use logistic regression SAGA solver:

```bash
python3 scripts/train_neuroemo_emotion_model.py --model-type logistic_saga --max-iter 3000
```

Quick smoke test:

```bash
python3 scripts/train_neuroemo_emotion_model.py \
  --max-samples 60 \
  --no-cv \
  --max-iter 20 \
  --output-model /private/tmp/neuroemo_smoke_model.joblib \
  --metrics-json /private/tmp/neuroemo_smoke_metrics.json
```

---

## 9. Recommended Next Step

Replace the demo ROI mapping with a real Schaefer `fsaverage5` vertex-to-parcel file, then compare:

```text
Schaefer 200 mean
Schaefer 400 mean
Schaefer 400 mean + std + mean_abs
```

This should happen before investing much more effort into deeper models.
