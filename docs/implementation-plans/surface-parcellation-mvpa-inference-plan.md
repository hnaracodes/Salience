# Microplan: atlas-backed surface masks and MVPA inference

Companion doc for **[salience-architecture-plan.md](salience-architecture-plan.md)** and **[salience-phased-delivery-plan.md](salience-phased-delivery-plan.md)**. Execution blueprint for the **strong neuroscience-facing layer** on top of scaffolding in [`activation_store.py`](../../activation_store.py) (within-session percentiles + coarse `vertex_fraction` bins).

---

## Goal

Turn dense **`preds[T, V]`** into:

1. **Anatomically meaningful surface masks** via a versioned vertex?parcel/network mapping (replacing mesh-index proxies).
2. **Network-level features** that preserve the spatial activation pattern inside each network, rather than averaging vertices.
3. **Continuous Z-score engagement and cosine-similarity emotion profiles produced without any trained model**, using surface network masks and downloadable NeuroVault templates.

Guardrail (product copy): output language remains **model-relative hypotheses**, aligned with Salience §3 in the main implementation plan.

---

## Current integration points (do not break)

| Artifact | Role |
|----------|------|
| `scout_data/sessions/<id>/preds.npz` | Source of truth matrix `preds` `(T, V)` |
| [`configs/vertex_regions.csv`](../../configs/vertex_regions.csv) | Optional legacy two-column form or extended atlas-backed columns |
| [`scout_data/activations.sqlite`](../../scout_data/activations.sqlite) | Append neuro tables; keep existing `sessions`, `activation_peak`, lookup bands |

---

## Data artifacts (versioned, reproducible)

### A. `configs/parcellation_manifest.yaml`

Single source of truth for reproducibility:

```yaml
mesh: fsaverage5
tribe_vertex_order: facebook/tribev2  # confirm against Tribev2 plotting mesh
atlas_id: schaefer2018_400_fsaverage5  # example
yeo7_map_version: 7Networks_Labels_v1   # example
artifact_sha256:
  labels_gii_lh: "<hex>"
  labels_gii_rh: "<hex>"
```

### B. `configs/vertex_regions.csv` (extended columns)

Minimum columns after upgrade:

| Column | Type | Purpose |
|--------|------|---------|
| `vertex_index` | int | 0..V-1, Tribev2 order |
| `parcel_id` | int | Stable atlas parcel id |
| `parcel_label` | string | Human label (e.g. Schaefer name) |
| `yeo_network_id` | int (nullable) | 1–7 or NULL if unknown |
| `yeo_network_name` | string | e.g. `SomMot`, `SalVentAttn` |
| `hemisphere` | `lh`/`rh` | From atlas |

**Generation path (offline script, not hand-edited):**

- Input: FreeSurfer/fsaverage5-compatible label GIFTI (lh/rh) **or** nilearn-fetchable volumetric labels projected to surface (document chosen path).
- Script: [`scripts/build_vertex_regions_csv.py`](../../scripts/build_vertex_regions_csv.py)
  - Validate `len(unique(vertex_index)) == V` and contiguous `0..V-1`.
  - Emit CSV + update manifest counts/shas.

### C. `configs/emotion_templates/`

| File | Contents |
|------|----------|
| `template_anger.npy` | Kragel (2015) anger template, resampled to fsaverage5, L2-normalised, shape `(20484,)` float32 |
| `template_disgust.npy` | Kragel (2015) disgust template |
| `template_fear.npy` | Kragel (2015) fear template |
| `template_happy.npy` | Kragel (2015) happy template |
| `template_neutral.npy` | Kragel (2015) neutral template |
| `template_sad.npy` | Kragel (2015) sad template |
| `template_negative_affect.npy` | PINES (2015) negative affect template, resampled to fsaverage5, L2-normalised |

**How templates are produced (one-time offline; no training):**

- Download NIfTI files from NeuroVault (collection #503 for Kragel; image #10704 for PINES) using `scripts/download_emotion_templates.py`.
- Resample each to fsaverage5 surface via `nilearn.surface.vol_to_surf` (pial surface, ball interpolation).
- L2-normalise: `template /= np.linalg.norm(template)`.
- Save as `.npy`. Gitignored; regenerate with `python scripts/download_emotion_templates.py`.

---

## Low-level code layout

New package directory:

```text
scout_core/
  __init__.py
  parcellation.py       # load vertex_regions.csv ? dense parcel_id[V]
  mvpa_engine.py        # SurfaceMasker/network masks ? windows ? sklearn probability traces
  schemas.py            # pydantic models
  storage_migrations.py # neuro SQLite DDL + inserts

scripts/
  build_vertex_regions_csv.py   # atlas ? CSV
  train_mvpa_model.py           # labeled corpus preds.npz ? scout_models/<model_id>/model.pkl
  analyze_session.py            # session id ? SQLite augment + analysis_bundle.json
```

Deprecated/deleted modules:

- [`scout_core/aggregate.py`](../../scout_core/aggregate.py) — delete or keep only as a legacy helper. Averaging vertices into parcel/network means destroys the spatial “barcode” MVPA needs to detect cognitive states.
- [`scout_core/threshold_engine.py`](../../scout_core/threshold_engine.py) — delete/deprecate. The inference output is a probability trace from a trained model, not a rules engine event.
- [`configs/calibrated_rules.yaml`](../../configs/calibrated_rules.yaml) — delete/deprecate. YAML thresholds should not be the source of truth for cognitive-state detection.

---

## Core functions (signatures-level contract)

**[`scout_core/parcellation.py`](../../scout_core/parcellation.py)**

```python
def load_vertex_table(path: Path) -> VertexParcellationTable: ...
def dense_parcel_labels(table: VertexParcellationTable, n_vertices: int) -> np.ndarray: ...  # shape (V,)
def parcel_to_network_map(table: VertexParcellationTable) -> dict[int, int]: ...  # parcel_id -> yeo_network_id
```

**[`scout_core/mvpa_engine.py`](../../scout_core/mvpa_engine.py)**

```python
def build_surface_masker(vertex_table: VertexParcellationTable, *, network_name: str) -> SurfaceMasker: ...
def mask_network_vertices(preds: np.ndarray, masker: SurfaceMasker) -> np.ndarray: ...  # shape (T, P_network)
def sliding_window_features(masked_ts: np.ndarray, *, window_seconds: int = 3) -> np.ndarray: ...  # shape (T-2, 3 * P_network)
def load_mvpa_model(model_path: Path) -> sklearn.pipeline.Pipeline: ...
def predict_probability_trace(model, X: np.ndarray) -> np.ndarray: ...  # shape (T-2, n_classes)
```

---

## MVPA inference

[`scout_core/mvpa_engine.py`](../../scout_core/mvpa_engine.py) owns inference:

1. Load live `preds[T, V]`.
2. Use `nilearn.maskers.SurfaceMasker` plus [`configs/vertex_regions.csv`](../../configs/vertex_regions.csv) to isolate the target vertices, such as Yeo-7 Frontoparietal, without reducing them to a mean.
3. For each timestep `t`, take rows `[t-2, t-1, t]` from the masked matrix and flatten to a 1D vector of length `3 * P_network`.
4. Pass the feature matrix into a pre-trained `scikit-learn` `.pkl` model, for example a calibrated `LinearSVC`.
5. Emit `probability_trace[T-2, n_classes]`, optionally with timestamps aligned back to the source video.

---

## SQLite extensions (neuro tables)

- **`mvpa_model_bundle`** — `model_id`, model path, `label_map_json`, `meta_json`, `created_at`
- **`session_masked_feature_meta`** — network mask id, selected vertex count, window size, timestep alignment
- **`mvpa_probability_trace`** — per timestep class probabilities + model id + network mask metadata

DDL lives in [`scout_core/storage_migrations.py`](../../scout_core/storage_migrations.py); [`activation_store.py`](../../activation_store.py) calls `ensure_neuro_schema()` on connect.

---

## Execution pipeline

```mermaid
flowchart TD
  subgraph offline [Offline_prep]
    A1[build_vertex_regions_csv]
    A2[train_mvpa_model_LOVO]
    A3[register_model_bundle_sql]
  end
  subgraph per_session [Per_session]
    B1[Load_preds_npz]
    B2[SurfaceMasker_network_vertices]
    B3[sliding_window_flatten_3s]
    B4[sklearn_model_predict_proba]
    B5[probability_trace_events]
    B6[persist_SQL_and_analysis_bundle_json]
  end
  A1 --> B2
  A2 --> B4
  B1 --> B2 --> B3 --> B4 --> B5 --> B6
```

---

## Integration with `modal run tribe.py::record`

- **Modal / GPU**: inference unchanged; writes `preds.npz` + SQLite peaks (UX scaffolding).
- **Local CPU post-step**: `python scripts/run_dual_track.py --session-id …` reads `preds.npz` + `preds_baseline.npz` + precomputed templates, computes engagement and emotion scores, fills neuro tables + **`scout_data/sessions/<id>/analysis_bundle.json`**.

---

## Validation checklist

- **Parcellation**: every `vertex_index` in `[0, V-1]` appears exactly once; `V` matches `preds.shape[1]`.
- **Masking**: target network masks select the expected vertices and do not average or reorder them.
- **Windowing**: 3-second flattened windows have shape `(T-2, 3 * P_network)` with deterministic handling of the first two timesteps.
- **Model**: `.pkl` bundle exposes probability output, directly or through calibration.
- **Leakage**: model training/evaluation uses Leave-One-Video-Out splits.

---

## Dependencies (local)

See [`requirements.txt`](../../requirements.txt): `numpy`, `pandas`, `pydantic`, `pyarrow`, `nibabel`, `nilearn`, `requests`, `pytest`, plus `modal` as needed. `scikit-learn` and `joblib` are **not required** for inference under this architecture.

---

## Estimated effort ordering

1. Parcellation CSV builder + manifest.
2. `mvpa_engine.py` network masking + 3-second sliding-window tests.
3. `train_mvpa_model.py` + LOVO validation + `model.pkl` bundle.
4. MVPA SQLite migrations + inserts.
5. `analyze_session.py` probability trace integration.
6. `analysis_bundle.json` + dashboard/agent consumers.
