# Implementation Plan - Improved ROI Extraction Pipeline

## Goal

Replace the current synthetic/demo ROI mapping with a reproducible atlas-backed ROI extraction pipeline for NeuroEmo training and future TribeV2 inference.

The upgraded pipeline should turn dense surface activations:

```text
X[N, 20484]
```

into validated, versioned ROI features:

```text
X_roi[N, R]
```

or richer summary features:

```text
X_roi[N, R * S]
```

where:

- `R` is the number of atlas parcels;
- `S` is the number of summary statistics per parcel;
- vertex order remains compatible with TribeV2 `fsaverage5` outputs.

---

## Current State

Current training defaults to ROI mode in:

```text
scripts/train_neuroemo_emotion_model.py
```

It reads:

```text
configs/vertex_regions.csv
```

and computes parcel means from columns:

```text
vertex_index
parcel_id
parcel_label
yeo_network_id
yeo_network_name
hemisphere
```

Current limitation:

```text
configs/vertex_regions.csv
```

is a synthetic/demo mapping, not a real anatomical or functional atlas. It is good enough to test the mechanics of ROI training, but it should not be treated as a scientifically meaningful parcellation.

---

## Recommended Atlas

Use Schaefer 2018 as the first real ROI source.

Initial atlas variants:

```text
Schaefer2018 200 parcels, 7 networks, fsaverage5
Schaefer2018 400 parcels, 7 networks, fsaverage5
Schaefer2018 600 parcels, 7 networks, fsaverage5
```

Default candidate:

```text
Schaefer2018 400 parcels, 7 networks
```

Reason:

- close to the current 400-feature ROI setup;
- widely used functional parcellation;
- compatible with Yeo network labels;
- low enough dimensionality for subject-held-out NeuroEmo training.

---

## Target Artifacts

### 1. Parcellation Manifest

Update:

```text
configs/parcellation_manifest.yaml
```

Target fields:

```yaml
mesh: fsaverage5
tribe_vertex_order: facebook/tribev2
atlas_id: schaefer2018_400_7networks_fsaverage5
n_vertices_expected: 20484
n_parcels: 400
hemisphere_order: lh_then_rh
source:
  name: Schaefer2018
  parcels: 400
  networks: 7
  mesh: fsaverage5
artifact_sha256:
  vertex_regions_csv: "<sha256>"
  labels_gii_lh: "<sha256-or-empty>"
  labels_gii_rh: "<sha256-or-empty>"
inputs:
  labels_gii_lh: "<path-or-url>"
  labels_gii_rh: "<path-or-url>"
```

### 2. Vertex-To-Parcel CSV

Replace or generate:

```text
configs/vertex_regions.csv
```

Required columns:

| Column | Type | Requirement |
|--------|------|-------------|
| `vertex_index` | int | contiguous `0..20483` |
| `parcel_id` | int | stable atlas parcel id |
| `parcel_label` | str | atlas label |
| `yeo_network_id` | int | 1-7 where known |
| `yeo_network_name` | str | Yeo 7 network label |
| `hemisphere` | str | `lh` or `rh` |

The first 10,242 rows should correspond to left hemisphere vertices. The next 10,242 rows should correspond to right hemisphere vertices.

### 3. ROI Extractor Metadata

Every trained model artifact should store:

```json
{
  "feature_mode": "roi",
  "atlas_id": "schaefer2018_400_7networks_fsaverage5",
  "n_vertices": 20484,
  "n_rois": 400,
  "roi_reducer": "mean",
  "parcel_ids": ["..."],
  "parcel_labels": ["..."],
  "parcel_id_by_vertex": ["..."]
}
```

This ensures future inference uses the exact same ROI mapping as training.

---

## Implementation Phases

## Phase 1 - Atlas Acquisition

Create a script:

```text
scripts/build_schaefer_vertex_regions.py
```

Responsibilities:

1. Fetch or load Schaefer 2018 labels for `fsaverage5`.
2. Parse left and right hemisphere label arrays.
3. Convert them into the TribeV2 vertex order:

```text
lh vertices first, rh vertices second
```

4. Write `configs/vertex_regions.csv`.
5. Update `configs/parcellation_manifest.yaml`.

Preferred source (current default):

```text
CBIG FreeSurfer5.3 fsaverage5 .annot files (two files only)
docs/atlas-setup/cbig-schaefer2018-fsaverage5.md
```

Volumetric projection via `nilearn.datasets.fetch_atlas_schaefer_2018` is **deprecated** and only available for diagnostics (`--allow-volume-projection-fallback` on the builder).

Deliverable:

```text
configs/vertex_regions.csv
configs/parcellation_manifest.yaml
```

---

## Phase 2 - Validation Utilities

Add validation logic, ideally reusable by both the builder and trainer:

```text
scout_core/parcellation.py
```

Core checks:

```text
vertex_index is contiguous 0..20483
exactly 20484 rows
hemisphere counts are 10242 lh and 10242 rh
no missing parcel_id values
parcel ids are stable and deterministic
parcel labels are non-empty
manifest atlas_id matches CSV metadata
```

Optional checks:

```text
expected parcel count matches manifest
Yeo network names are from an allowed set
CSV sha256 matches manifest
```

The trainer should fail loudly if these checks fail.

---

## Phase 3 - ROI Feature Extractor

Move ROI extraction out of the training script into a reusable module:

```text
scout_core/roi_features.py
```

Suggested public functions:

```python
def load_roi_table(path: Path, *, n_vertices: int = 20484) -> RoiTable: ...

def reduce_vertices_to_rois(
    X: np.ndarray,
    roi_table: RoiTable,
    reducers: tuple[str, ...] = ("mean",),
) -> np.ndarray: ...
```

Supported reducers:

```text
mean
std
mean_abs
max_abs
```

Initial default:

```text
mean
```

Experiment candidates:

```text
mean
mean + std
mean + std + mean_abs
```

Feature naming should be deterministic:

```text
<parcel_label>__mean
<parcel_label>__std
<parcel_label>__mean_abs
```

---

## Phase 4 - Trainer Integration

Update:

```text
scripts/train_neuroemo_emotion_model.py
```

Desired flags:

```bash
--feature-mode roi
--vertex-csv configs/vertex_regions.csv
--roi-reducers mean
--roi-reducers mean,std,mean_abs
--atlas-manifest configs/parcellation_manifest.yaml
```

Training output should include:

```text
atlas_id
n_rois
roi_reducers
roi_feature_names
parcel_id_by_vertex
vertex_csv_sha256
manifest_sha256
```

Metrics JSON should include the same atlas summary, but not necessarily the full `parcel_id_by_vertex` array.

---

## Phase 5 - Experiments

Run comparable subject-held-out experiments:

```bash
python3 scripts/train_neuroemo_emotion_model.py \
  --feature-mode roi \
  --roi-reducers mean
```

Then compare:

```text
synthetic demo 400 mean
Schaefer 200 mean
Schaefer 400 mean
Schaefer 600 mean
Schaefer 400 mean,std
Schaefer 400 mean,std,mean_abs
```

Primary metrics:

```text
pooled balanced accuracy
macro F1
per-class F1
confusion matrix
held-out subject fold variance
```

Do not optimize only for overall accuracy. The neutral class and similar positive-valence classes may create imbalance in interpretability even when accuracy improves.

---

## Phase 6 - Future Inference Compatibility

The inference file has been intentionally deferred until training stabilizes.

When inference is restored, it must not independently load `configs/vertex_regions.csv` as its source of truth. Instead, it should load the trained model artifact and use the embedded ROI metadata:

```text
parcel_id_by_vertex
parcel_ids
roi_reducers
feature_names
label order
model pipeline
```

This prevents training/inference ROI drift.

---

## Acceptance Criteria

The ROI upgrade is complete when:

1. `configs/vertex_regions.csv` is generated from a real atlas, not synthetic/demo labels.
2. `configs/parcellation_manifest.yaml` records atlas identity, mesh, vertex order, counts, hashes, and vertex-equivalence proof metadata.
3. A validation command confirms all 20,484 vertices are mapped exactly once.
4. Training can run with the new atlas without changing the NeuroEmo dataset NPZ.
5. Model artifacts store enough ROI metadata for future inference to reproduce the same features.
6. Metrics compare at least synthetic demo ROI vs Schaefer 400 ROI under the same train/CV settings.

---

## Risks

### Vertex Order Mismatch

The biggest risk is silently using an atlas in a different vertex order than TribeV2 outputs. This would produce plausible shapes but meaningless ROI features.

Mitigation:

- validate hemisphere counts;
- verify known visual/somatomotor parcels occupy plausible vertex ranges;
- store atlas hashes;
- require a vertex-equivalence report or explicit legacy override before training;
- generate the proof artifact in Modal once, then gate all consumers against its pinned hash;
- keep a small visual inspection notebook or script.

### Atlas Projection Error

If a volumetric atlas is projected to surface poorly, parcels may be noisy or spatially distorted.

Mitigation:

- prefer surface-native `fsaverage5` labels;
- if projection is necessary, document the method and parameters;
- compare parcel size distributions against expected ranges.

### Over-Averaging

ROI means may remove discriminative within-parcel patterns.

Mitigation:

- test `std` and `mean_abs` reducers;
- compare against full-vertex baseline;
- consider network-restricted MVPA later if ROI means underperform.

---

## Recommended First Task

Build and validate:

```text
Schaefer2018 400 parcels, 7 networks, fsaverage5
```

Then retrain the current NeuroEmo model with:

```bash
python3 scripts/train_neuroemo_emotion_model.py \
  --feature-mode roi \
  --roi-reducers mean
```

This is the smallest change likely to improve scientific validity without changing the NeuroEmo training data format.
