# Coding Session — May 22, 2026

## NeuroEmo Dataset Download & TribeV2-Compatible Formatting

**Audience:** Project team members who need to understand the NeuroEmo data-prep work without reading the script line by line.  
**Scope:** New NeuroEmo preparation script, output contracts, label construction, compatibility assumptions, and next validation steps.

---

## 1. Why This Was Added

The current TribeV2 emotional recognition path uses zero-shot template matching against downloaded NeuroVault emotion maps. That gives useful emotion hypotheses, but it is not a supervised model trained on labeled emotional fMRI.

NeuroEmo (`OpenNeuro ds005700`, version `1.2.0`) provides labeled fMRI responses to affective video blocks:

- `calm`
- `afraid`
- `delighted`
- `depressed`
- `excited`

The goal of this session was to create a preparation path that turns NeuroEmo emotion-task fMRI into arrays that can train a downstream classifier compatible with TribeV2 outputs.

TribeV2 outputs cortical surface predictions shaped:

```text
preds[T, 20484]
```

The new formatter aims to produce training samples with the same feature dimension:

```text
X[N, 20484]
y[N]
```

or, for temporal models:

```text
X[N, window_trs, 20484]
y[N]
```

---

## 2. New File

| Path | Purpose |
|------|---------|
| `scripts/prepare_neuroemo_tribev2.py` | Downloads NeuroEmo emotion-task BOLD runs, projects 4D NIfTI volumes to `fsaverage5`, creates per-TR emotion labels, and writes ML-ready `.npz` artifacts. |

No existing dual-track or TribeV2 inference files were changed for this work.

---

## 3. What The Script Does

### Download

The script downloads NeuroEmo files from the OpenNeuro CRN endpoint:

```text
https://openneuro.org/crn/datasets/ds005700/snapshots/1.2.0/files/...
```

For each subject, it downloads:

```text
sub-XX/func/sub-XX_task-fe_bold.nii.gz
sub-XX/func/sub-XX_task-fe_bold.json
```

Default raw output:

```text
scout_data/neuroemo/raw/
```

### Volume-To-Surface Projection

The script loads each `task-fe` BOLD run as a 4D image and uses:

```python
nilearn.datasets.fetch_surf_fsaverage(mesh="fsaverage5")
nilearn.surface.vol_to_surf(...)
```

It samples both hemispheres from the `pial` surfaces:

```text
left hemisphere:  10242 vertices
right hemisphere: 10242 vertices
combined:         20484 vertices
```

The output convention is:

```text
left hemisphere vertices, then right hemisphere vertices
```

### Label Construction

The NeuroEmo emotion task is represented as 30-second emotion clips interleaved with 30-second white-noise blocks.

The script encodes the published schedule into `TASK_EVENTS`, then assigns one label per TR after applying a configurable BOLD lag:

```bash
--bold-lag-s 6.0
```

By default, only the five emotion classes are included. White-noise TRs are treated as unlabeled unless this flag is provided:

```bash
--include-white-noise
```

### Feature Scaling

Default scaling is per subject and per vertex:

```bash
--standardize per_subject_vertex
```

Other options:

```bash
--standardize per_subject_global
--standardize none
```

---

## 4. Output Artifacts

Default formatted output directory:

```text
scout_data/neuroemo/tribev2_surface/
```

Per-subject artifact:

```text
subjects/sub-XX_task-fe_fsaverage5.npz
```

Contains:

```text
surface     (T, 20484) float32
labels      (T,) string
label_ids   (T,) int64
time_s      (T,) float32
class_names (...) string
```

Combined training artifact:

```text
neuroemo_tribev2_train.npz
```

Contains:

```text
X                  (N, 20484) or (N, window_trs, 20484)
y                  (N,)
subject            (N,)
t_idx              (N,)
time_s             (N,)
labels             class-name lookup ordered by y id
dataset_id         ds005700
snapshot_version   1.2.0
```

Human-readable label table:

```text
neuroemo_tribev2_labels.csv
```

Metadata and warnings:

```text
metadata.json
```

---

## 5. How To Run

Format all 40 NeuroEmo subjects:

```bash
python3 scripts/prepare_neuroemo_tribev2.py --subjects 1-40
```

Include white-noise as a sixth class:

```bash
python3 scripts/prepare_neuroemo_tribev2.py --subjects 1-40 --include-white-noise
```

Create 3-TR same-label temporal windows:

```bash
python3 scripts/prepare_neuroemo_tribev2.py --subjects 1-40 --window-trs 3
```

Use already-downloaded or externally preprocessed BIDS files:

```bash
python3 scripts/prepare_neuroemo_tribev2.py \
  --skip-download \
  --raw-dir path/to/preprocessed_bids
```

---

## 6. Compatibility With TribeV2

The script currently guarantees **shape-level compatibility**:

```text
NeuroEmo formatted sample: 20484 fsaverage5 vertices
TribeV2 prediction:        20484 fsaverage5 vertices
```

It also uses the same broad convention used elsewhere in the repo:

```text
fsaverage5 cortical surface
left hemisphere then right hemisphere
```

However, this does **not yet prove exact vertex-index identity** with TribeV2's original model training pipeline.

Current guarantee:

```text
T x 20484, fsaverage5, lh-then-rh Nilearn vertex order
```

Still to verify:

```text
TribeV2 vertex j == Nilearn fsaverage5 vertex j for all j
```

Recommended next compatibility check:

1. Load TribeV2's own surface projection helper, if available, such as `TribeSurfaceProjector`.
2. Project the same NeuroEmo BOLD sample through both:
   - `scripts/prepare_neuroemo_tribev2.py`
   - TribeV2's projector
3. Assert the resulting arrays match within floating-point tolerance.
4. Hash the mesh coordinate and face arrays for:
   - left pial
   - right pial
   - left faces
   - right faces
5. Write those hashes into `metadata.json`.

This would turn the current compatibility assumption into a testable contract.

---

## 7. Scientific Caveat

NeuroEmo files on OpenNeuro are raw BIDS fMRI. The script can project those volumes directly to `fsaverage5`, but this is best understood as a **format bridge for model development**.

For serious model training, use preprocessed fMRI first:

- motion correction
- slice-timing correction
- coregistration
- normalization to a common space
- optional smoothing
- nuisance regression / filtering as appropriate

Then run:

```bash
python3 scripts/prepare_neuroemo_tribev2.py \
  --skip-download \
  --raw-dir path/to/preprocessed_bids
```

---

## 8. Verification Performed

The script was checked without downloading the full dataset:

```bash
python3 -m py_compile scripts/prepare_neuroemo_tribev2.py
python3 scripts/prepare_neuroemo_tribev2.py --help
```

A lightweight label schedule check confirmed that, with the default five-class setup and a 6-second BOLD lag, a 200-TR emotion run yields balanced emotion labels:

```text
calm       20 TRs
afraid     20 TRs
delighted  20 TRs
depressed  20 TRs
excited    20 TRs
unlabeled 100 TRs
```

Across 40 subjects, this corresponds to approximately:

```text
4,000 emotion-labeled TR samples
```

before any transition dropping or temporal windowing.

---

## 9. Next Steps

1. Add a `--verify-tribev2-compat` mode that compares against TribeV2's own projector and writes mesh hashes.
2. Add a small training script for a baseline classifier:
   - `StandardScaler`
   - linear model or calibrated SVM/logistic regression
   - subject-held-out cross-validation
3. Add a new inference track that reads TribeV2 `preds.npz` and emits:

```json
{
  "neuroemo_emotion_track": {
    "class_names": ["calm", "afraid", "delighted", "depressed", "excited"],
    "probabilities": [[...]],
    "predicted_label": [...]
  }
}
```

4. Compare the trained NeuroEmo classifier against the existing Kragel template emotion track before replacing current dual-track behavior.
