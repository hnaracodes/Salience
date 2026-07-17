# Demographic Subagent Multiplexer Implementation Plan

**Revision:** 2026-06-18 (CS-20260618-DEMOGRAPHIC-MUX-PLAN)  
**Status:** Planning — gated on M0 viability experiment before any Modal training infra.

## Intent

This plan implements the Salience demographic multiplexer as a late-stage feature branching system on top of Meta TRIBE v2. The design freezes the expensive universal stimulus encoder and trains only a demographic prototype embedding table plus a **low-rank cluster-conditional readout delta** on TRIBE's existing population ("unseen-subject") head.

**Critical constraint:** The app is already inaccurate (TRIBE on UI walkthrough video is out-of-distribution vs. naturalistic movie training). The multiplexer must not pretend to fix that. It can only learn **demographic deltas relative to the population average**, and only if those deltas are detectable above noise in available fMRI movie datasets — a hypothesis that must be validated in **M0** before building training infra.

The target production path is:

```text
UI walkthrough video
  -> frozen TRIBE universal encoders (once per video)
  -> X_univ[T, 1152]   # Transformer output after 2Hz->1Hz pool (R0 measurement)
  -> demographic embedding lookup C[K, D_emb]
  -> low-rank cluster-conditional readout delta on population head
  -> parcel_ts[K, T, 400] or network_ts[K, T, 7]   # primary training target
  -> optional upsample to vertex_ts[K, T, 20484] for viewer compatibility
  -> existing Yeo-7 parcellation / norms / threshold mapper
  -> per-cluster cognitive and UX analysis (with explicit uncertainty provenance)
```

This must not fork TRIBE into separate demographic models. The only trainable demographic state is the cluster embedding matrix and readout delta parameters.

**No-hidden-state fallback:** If R0 cannot cleanly expose `X_univ`, use TRIBE `preds[T, 20484]` (average-subject output from the public demo) as input features and learn only the demographic delta on top. This is lower-confidence but unblocks experimentation.

## Scientific Preconditions (Read First)

| Fact | Implication |
|------|-------------|
| TRIBE Transformer hidden dim is **1152** (3 × 384 modality streams), not 4096 | All configs must use `hidden_dim=1152` until R0 confirms otherwise |
| TRIBE public demo predicts the **average/unseen subject** via subject dropout during training | Demographic branch is a delta on population readout, not a from-scratch vertex regressor |
| TRIBE subject block is a **low-rank conditional linear** layer `(S, D_model, N_targets)` | Mirror this structure for clusters; do not use a 53M-param from-scratch MLP |
| TRIBE preds are offset **5 s into the past** (hemodynamic lag) | Temporal alignment in data prep must respect this |
| Available movie-fMRI datasets are **small, demographically skewed, and site-confounded** | The marketed age×sex×ethnicity grid is not achievable at N≥20 per cell |
| Subjects watched **Hollywood movies**, not **UI screen-capture** | Demographic deltas learned on movies are double-extrapolated when applied to Salience walkthroughs |

## Directory Structure

The current repo has root-level `tribe.py`, `scout_core/`, `scripts/`, `configs/`, and `tests/`. Add the demographic multiplexer under a new `salience/` package while keeping the existing baseline code stable.

```text
TribeV2/
  docs/implementation-plans/demographic-multiplexer-implementation-plan.md
  tribe.py                              # existing Modal TRIBE integration
  activation_store.py
  configs/
    cluster_schema.yaml                 # new: allowed demographic fields, min-n policy
    clusters.yaml                       # new: cluster_id -> label, rule, n_subjects
    parcellation_manifest.yaml
    vertex_regions.csv
  data_prep/
    __init__.py
    metadata_harmonize.py               # ingest HCP/CNeuroMod/NNDb metadata
    segment_clusters.py                 # rule-based age/sex cluster assignment
    align_timeseries.py                 # temporal + surface-space alignment
    aggregate_centroids.py              # group subject tensors into Y_target
    package_webdataset.py               # video + centroid tensor tar sharding
    schemas.py                          # pydantic contracts
    viability_partition.py              # new: M0 variance-partition / permutation null
  salience/
    __init__.py
    mux/
      __init__.py
      tribe_mux.py                      # TribeDemographicMux (low-rank delta readout)
      losses.py                         # correlation + noise-ceiling loss
      checkpoints.py
      inference.py
    modal_app/
      __init__.py
      image_defs.py
      train_clusters.py                 # Modal A100 training loop (after M0 pass)
      inference_mux.py
      stream_protocol.py
  scout_core/
    aggregate.py, norms.py, parcellation.py, threshold_engine.py, ...
  scout_data/
    demographic/
      metadata_harmonized.parquet
      cluster_members.parquet
      aligned_subject_ts/
      centroids/                        # Y_target in parcel or network space
      viability/                        # M0 reports and permutation nulls
      webdataset/
  scripts/
    run_demographic_data_prep.py
    run_demographic_viability.py        # new: M0 entry point
  tests/
    test_tribe_mux.py
    test_demographic_data_prep.py
    test_demographic_viability.py       # new
```

## Architecture

### TRIBE v2 Internal Geometry (from paper + R0 targets)

TRIBE v2 pipeline (relevant stages):

```text
Video/Audio/Text -> frozen encoders (384-dim each)
  -> concat -> [T_2hz, 1152]
  -> Transformer encoder (8 layers, 100s context)
  -> adaptive average pool 2Hz -> 1Hz
  -> X_univ[T, 1152]                    # hook point for multiplexer
  -> subject block OR unseen-subject layer
  -> preds[T, 20484] fsaverage5
```

The public `TribeModel.predict(events=df)` in [tribe.py](../../tribe.py) returns only final `preds[T, 20484]`. R0 must locate and expose `X_univ` from the pinned TRIBE commit.

### Frozen Universal Encoder

The frozen path should be isolated behind an adapter because TRIBE's public demo API currently exposes `predict(events=df)` rather than a clean "return universal hidden state" function.

Required R1 spike (was "R0" in v1 plan; M0 runs first without this):

1. Pin TRIBE commit hash in `salience/mux/tribe_commit.txt`.
2. Locate the Transformer output tensor after the 2Hz→1Hz adaptive pool in `tribev2/model.py` / `demo_utils.py`.
3. Confirm tensor rank: `X_univ: FloatTensor[T, 1152]` or `FloatTensor[B, T, 1152]`.
4. Prove that `population_readout(X_univ) ≈ predict(events=df)` for K=1 within tolerance.
5. Record hemodynamic offset (5 s) and TR alignment behavior.
6. Freeze all upstream TRIBE parameters:

```python
for p in tribe_backbone.parameters():
    p.requires_grad_(False)
tribe_backbone.eval()
```

Do not call the video/audio/text encoders once per cluster. Encoders run exactly once per video; only the small demographic branch scales with `K`.

### Demographic Prototypes

Represent demographic segments as integer cluster ids. **Rescoped cluster labels** (ethnicity removed — not achievable at N≥20 without confounding):

```text
cluster_id: int64 in [0, num_clusters)
cluster label examples (only if N >= min_subjects_per_cluster):
  young_adult_female      # 22-35, pooled across datasets with site covariate
  young_adult_male
  middle_adult_female     # 36-55, Cam-CAN / NNDb only
  middle_adult_male
  older_adult_pooled      # 56+, Cam-CAN only; sex may need pooling
```

Each cluster id indexes a trainable embedding table:

```text
C = Embedding(num_clusters=K_total, embedding_dim=D_emb)   # D_emb default 64-128
```

The embedding vector is not a demographic claim by itself. It is a learned prototype coordinate optimized to match cluster-average fMRI targets **in parcel/network space**.

Clusters that fail M0 power analysis or eval gate are marked `suppressed=true` and never shown in the product UI.

### Multiplexed Readout — Recommended: Low-Rank Delta on Population Head

**Do not** train a from-scratch 3-layer MLP to 20484 vertices (~53M parameters). Instead mirror TRIBE's subject block:

```text
W_pop:     [1152, N_out]           # frozen copy of unseen-subject readout (R1)
Delta_k:   low-rank factorization  # U_k @ V_k, rank r << min(1152, N_out)
y_k(t)  =  W_pop @ x_univ(t)  +  Delta_k @ x_univ(t)     # per cluster k

N_out primary:   400 (Schaefer parcels) or 7 (Yeo networks)
N_out optional:  20484 (vertices, lower confidence, viewer only)
```

Alternative: **FiLM modulation** — cluster embedding generates `(gamma_k, beta_k)` to modulate `W_pop @ x` before output. Simpler, fewer params, good baseline.

For selected `K` cluster ids at inference:

```text
X_univ:           [T, 1152]
cluster_ids:      [K]
y_base:           [T, N_out]              # population readout, computed once
y_delta:          [K, T, N_out]           # cluster-conditional delta
y_cluster:        [K, T, N_out]           # y_base broadcast + y_delta
```

Output shapes:

```text
parcel_ts:   FloatTensor[K, T, 400]   # primary training/inference target
network_ts:  FloatTensor[K, T, 7]      # derived via scout_core parcellation
vertex_ts:   FloatTensor[K, T, 20484] # optional, via fixed vertex projection matrix
```

### Legacy MLP Variant (Not Recommended)

The v1 plan's concat+MLP design is retained only as a **synthetic smoke-test baseline** for unit tests. Do not use it for real fMRI training — it overfits, ignores TRIBE readout structure, and optimizes the wrong metric at 20484 dimensions.

## Loss and Metrics

### Primary Training Loss

Replace raw MSE on vertices with **temporal Pearson correlation loss** in parcel/network space:

```python
def correlation_loss(y_pred, y_target, dim=-1, eps=1e-8):
    """Mean Pearson r across parcels/networks, negated for minimization."""
    pred_c = y_pred - y_pred.mean(dim=dim, keepdim=True)
    targ_c = y_target - y_target.mean(dim=dim, keepdim=True)
    r = (pred_c * targ_c).sum(dim=dim) / (
        pred_c.pow(2).sum(dim=dim).sqrt() * targ_c.pow(2).sum(dim=dim).sqrt() + eps
    )
    return 1.0 - r.mean()
```

Add optional terms:

- **Noise-ceiling weighting:** down-weight parcels where split-half reliability `< 0.3`.
- **Delta regularization:** `lambda * ||Delta_k||_F` to keep cluster deltas small relative to population.
- **Temporal smoothness:** penalize `||y(t) - y(t-1)||` on delta channel only (not base).

### Evaluation Metrics (Eval Gate)

Report on held-out subjects and leave-one-dataset-out folds:

| Metric | Definition | Notes |
|--------|------------|-------|
| `r_pop` | Pearson r, population readout vs. held-out subject | TRIBE baseline ceiling |
| `r_cluster` | Pearson r, cluster readout vs. held-out subject in assigned cluster | Primary |
| `delta_r` | `r_cluster - r_pop` | Must be positive and significant |
| `r_norm` | `delta_r / noise_ceiling` | Normalized gain |
| `p_perm` | Demographic-shuffle permutation null | Cluster labels permuted within site |

**Eval gate (go/no-go after M3):**

- `delta_r > 0.02` mean across held-out subjects (parcel space, network-aggregated).
- `p_perm < 0.01` (demographic label explains variance beyond site/scanner).
- Cluster readout beats single-cluster baseline on LOVO (leave-one-video-out) for ≥ 2 of 3 held-out datasets.
- If gate fails: **kill feature** or ship population-only mode with explicit "demographic comparison unavailable" in UI.

## `tribe_mux.py`

Place at `salience/mux/tribe_mux.py`. Recommended implementation:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TribeMuxConfig:
    hidden_dim: int = 1152          # TRIBE Transformer output dim (R1 confirm)
    num_clusters: int = 8
    cluster_emb_dim: int = 64
    readout_rank: int = 16          # low-rank delta rank
    n_parcels: int = 400            # primary output (Schaefer @ fsaverage5)
    n_networks: int = 7             # optional derived target
    n_vertices: int = 20484         # optional vertex projection
    use_film: bool = False          # if True, FiLM instead of low-rank delta
    target_space: str = "parcel"    # "parcel" | "network" | "vertex"


class TribeDemographicMux(nn.Module):
    """
    Late-stage demographic multiplexer: population readout + cluster delta.

    Expects TRIBE encoders to run outside this class exactly once per video.
    """

    def __init__(self, config: TribeMuxConfig) -> None:
        super().__init__()
        self.config = config
        n_out = {"parcel": config.n_parcels, "network": config.n_networks, "vertex": config.n_vertices}[config.target_space]

        self.cluster_embedding = nn.Embedding(config.num_clusters, config.cluster_emb_dim)

        # Frozen population readout — loaded from TRIBE unseen-subject weights (R1)
        self.pop_weight = nn.Linear(config.hidden_dim, n_out, bias=True)
        self.pop_weight.weight.requires_grad_(False)
        self.pop_weight.bias.requires_grad_(False)

        # Low-rank cluster delta: for each cluster k, Delta_k = U_k @ V_k
        self.delta_u = nn.Parameter(torch.zeros(config.num_clusters, config.hidden_dim, config.readout_rank))
        self.delta_v = nn.Parameter(torch.zeros(config.num_clusters, config.readout_rank, n_out))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cluster_embedding.weight, std=0.02)
        nn.init.normal_(self.delta_u, std=0.01)
        nn.init.normal_(self.delta_v, std=0.01)

    def forward(self, x_univ: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        """
        x_univ: [T, D] or [B, T, D]
        cluster_ids: [K] or [B, K]
        returns: [K, T, N_out] or [B, K, T, N_out]
        """
        if x_univ.ndim == 2:
            return self._forward_single(x_univ, cluster_ids)
        if x_univ.ndim == 3:
            return self._forward_batched(x_univ, cluster_ids)
        raise ValueError(f"x_univ must be [T,D] or [B,T,D], got {tuple(x_univ.shape)}")

    def _forward_single(self, x_univ: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        T, D = x_univ.shape
        cluster_ids = cluster_ids.to(device=x_univ.device, dtype=torch.long)
        K = cluster_ids.shape[0]

        y_base = self.pop_weight(x_univ)  # [T, N_out]

        u = self.delta_u[cluster_ids]     # [K, D, r]
        v = self.delta_v[cluster_ids]     # [K, r, N_out]
        # delta: [K, T, N_out] = einsum over D,r
        y_delta = torch.einsum("ktd,kdr,kro->kto", x_univ.unsqueeze(0).expand(K, -1, -1), u, v)

        return y_base.unsqueeze(0) + y_delta  # [K, T, N_out]

    def _forward_batched(self, x_univ: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        B, T, D = x_univ.shape
        if cluster_ids.ndim == 1:
            cluster_ids = cluster_ids.unsqueeze(0).expand(B, -1)
        K = cluster_ids.shape[1]
        outs = []
        for b in range(B):
            outs.append(self._forward_single(x_univ[b], cluster_ids[b]))
        return torch.stack(outs, dim=0)  # [B, K, T, N_out]

    def trainable_parameter_groups(self, *, emb_wd: float = 0.1, delta_wd: float = 0.01) -> list[dict]:
        return [
            {"params": [self.cluster_embedding.weight], "weight_decay": emb_wd, "name": "cluster_embedding"},
            {"params": [self.delta_u, self.delta_v], "weight_decay": delta_wd, "name": "delta_readout"},
        ]
```

## Data Extraction Plan

### Dataset Reality Check

**Brutally honest inventory** of candidate naturalistic movie-fMRI datasets:

| Dataset | N subjects | Age range | Sex balance | Stimulus | Surface space | TR | Demographic cells at N≥20? |
|---------|-----------|-----------|-------------|----------|---------------|-----|---------------------------|
| **CNeuroMod** | 6 (4-6 with DUA) | ~25-35 | mixed | Hollywood movies (movie10) | fsLR → fsaverage5 | 1.49s | **No** — entire dataset < 20 |
| **HCP 7T Movie** | 184 | 22-36 only | ~balanced | Movie clips (Wolf/Mario/etc.) | fsLR_32k | 1.0s | **Age×sex partially** — all "young adult"; no genx/boomer |
| **NNDb** (OpenNeuro ds002837) | 86 | 18-58 | 42F/44M | 10 full-length movies | MNI vol → surf | 2.0s | **~8 per movie**; age bands poolable but site=single |
| **StudyForrest 7T** | 20 | 21-38 | 12M/8F | Forrest Gump | fsLR | 2.0s | **No** per cell |
| **Narratives** (OpenNeuro) | 345 | 18-53 | 204F/141M | Audio stories (no video) | MNI vol | 1.5s | Audio-only; **not valid for TRIBE video path** |
| **Cam-CAN** | ~650 | 18-88 | balanced | Single 8-min movie clip | MNI vol | 0.74s | **Age bands yes, sex yes** — but one short clip, single site |

**Conclusions:**

1. **Ethnicity is cut** from `cluster_schema.yaml` and `clusters.yaml`. No public movie-fMRI dataset provides ethnicity at scale with matched stimuli.
2. The v1 marketing grid (`genz_female`, `millennial_male`, …) is **not achievable**. Rescope to coarse age bands with sex, accepting that "young adult" ≈ HCP/NNDb young subjects only.
3. **Site/scanner confound** is unavoidable: HCP ≠ Cam-CAN ≠ CNeuroMod. Mandate leave-one-dataset-out (LODO) in all eval.
4. **Access gates:** CNeuroMod requires registered access + DTA; HCP requires ConnectomeDB credentials; NNDb and Cam-CAN are more open.

### Recommended Data Strategy (Phased)

**Phase A — M0 only (no video, no TRIBE training):**

- Download HCP 7T movie or Cam-CAN preprocessed parcellated timeseries (public derivatives).
- Run variance partition: `response ~ age + sex + site + (1|subject)` on network timeseries.
- Permutation null: shuffle age/sex labels within site, recompute explained variance.
- **Decision:** proceed only if demographic variance exceeds permutation null at p < 0.01 and is > site variance for ≥ 1 network.

**Phase B — Centroid build (if M0 passes):**

- Primary: HCP 7T movie (N=184, young adult, best-powered).
- Secondary: Cam-CAN (age diversity, single clip — age effects only).
- Tertiary: NNDb (movie diversity, small N per movie).
- CNeuroMod: use for TRIBE-native validation only (N=6, not for cluster centroids).

### Dependencies

```text
torch, webdataset, nilearn, nibabel, pandas, pyarrow, numpy, scipy,
scikit-learn, pydantic, pyyaml, tqdm, statsmodels
```

Optional: `neuromaps`, `templateflow`, `wb_command` (system binary).

### 1. Metadata Harmonization

Script: `data_prep/metadata_harmonize.py`

Unified schema (**ethnicity removed**):

```text
subject_id: str
source_subject_id: str
dataset: Literal["hcp_7t", "cneuromod", "nndb", "camcan", "forrest"]
age: float | null
age_band: str          # young_adult | middle_adult | older_adult
sex: Literal["female", "male", "unknown"]
site_id: str           # scanner/site identifier (critical covariate)
handedness: str | null
metadata_quality: str
consent_scope: str | null
dua_version: str | null
```

Age bands (rescaled):

```text
18-35 -> young_adult
36-55 -> middle_adult
56+   -> older_adult
```

Output: `scout_data/demographic/metadata_harmonized.parquet`

### 2. Rule-Based Segmentation

Config: `configs/clusters.yaml`

```yaml
schema_version: 2
min_subjects_per_cluster: 20
require_site_covariate: true
clusters:
  - cluster_id: 0
    label: young_adult_female
    where:
      age_band: young_adult
      sex: female
    min_datasets: 1          # at least 1 dataset contributes
  - cluster_id: 1
    label: young_adult_male
    where:
      age_band: young_adult
      sex: male
  - cluster_id: 2
    label: middle_adult_pooled
    where:
      age_band: middle_adult
    notes: "Sex pooled — insufficient N when split"
  - cluster_id: 3
    label: older_adult_pooled
    where:
      age_band: older_adult
    notes: "Cam-CAN only; sex pooled"
```

Leakage guards:

```python
from sklearn.model_selection import GroupShuffleSplit

# Subject-held-out
splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
# groups = source_subject_id

# Leave-one-dataset-out (mandatory secondary eval)
for held_out_dataset in datasets:
    train = df[df["dataset"] != held_out_dataset]
    test = df[df["dataset"] == held_out_dataset]
```

### 3. Spatiotemporal Alignment

Script: `data_prep/align_timeseries.py`

Primary target shape (preferred):

```text
aligned subject response: float32 [T_1hz, 400]   # Schaefer parcels
derived network:        float32 [T_1hz, 7]      # Yeo-7 aggregation
optional vertex:        float32 [T_1hz, 20484]  # lower confidence
```

Temporal alignment:

- Respect dataset TR (HCP 0.72s/1.0s, CNeuroMod 1.49s, NNDb 2.0s, Cam-CAN 0.74s).
- Apply **5 s hemodynamic shift** when aligning video events to fMRI (match TRIBE convention).
- Interpolate to 1 Hz with `scipy.interpolate.interp1d` (auditable).

Spatial alignment:

- Prefer dataset-provided fsLR/fsaverage5 derivatives.
- Use `wb_command -metric-resample` for fsLR_32k → fsaverage5 when needed.
- Parcellate early via `configs/vertex_regions.csv` — do not train on raw 20484 unless necessary.

Output:

```text
scout_data/demographic/aligned_subject_ts/
  <dataset>/<video_id>/<subject_id>.npz
    parcel_ts: [T, 400]
    network_ts: [T, 7]       # optional pre-aggregated
    vertex_ts: [T, 20484]    # optional
```

### 4. Centroid Aggregation

Script: `data_prep/aggregate_centroids.py`

For each `video_id` and cluster:

```python
stack = np.stack(subject_tensors, axis=0)  # [N_cluster, T, P]
centroid = stack.mean(axis=0).astype("float32")  # [T, P]
```

Recommended:

1. Z-score each subject per parcel before averaging (cross-dataset scale compatibility).
2. Average within dataset first, then across datasets (prevent HCP dominance).
3. Store `n_subjects`, `dataset_ids`, and `site_ids` in centroid metadata.

Output:

```text
scout_data/demographic/centroids/<video_id>.npz
  y_target: float32 [K_trainable, T, P]     # P=400 parcels (primary)
  cluster_ids: int64 [K_trainable]
  fps: 1.0
  provenance_json: {...}
```

### 5. WebDataset Packaging

Script: `data_prep/package_webdataset.py`

Shard keys include `y_target.npy` (parcel space), `cluster_ids.npy`, and provenance JSON. Store on Modal Volume or object storage.

## M0 Viability Experiment (Gating — Run Before Any Modal Infra)

Script: `scripts/run_demographic_viability.py`  
Core logic: `data_prep/viability_partition.py`

**Goal:** Determine whether demographic group explains any variance in network-level movie-fMRI responses above a site-confound and permutation null — **without TRIBE, without Modal, without WebDataset**.

**Procedure:**

1. Load public parcellated timeseries from HCP 7T movie and/or Cam-CAN (whichever is faster to obtain).
2. Aggregate vertices → Yeo-7 network timeseries per subject per run.
3. Fit mixed model or variance partition per network:
   `network_ts ~ age_band + sex + site + (1|subject)`
4. Permutation null (1000 shuffles): shuffle `age_band` and `sex` within `site_id`, recompute demographic explained variance.
5. Compare demographic variance to site variance and to permutation distribution.

**Go criterion (all required):**

- Demographic factors explain ≥ 1% unique variance in ≥ 2 Yeo-7 networks (Vis, Default, or SalVentAttn).
- Demographic explained variance exceeds 95th percentile of permutation null (p < 0.05) in those networks.
- Demographic variance is not entirely explained by site (partial R² age/sex > partial R² site alone for at least one network).

**No-go:** Kill or rescope feature to "population-only + explicit disclaimer." Do not build `salience/modal_app/train_clusters.py` until M0 passes.

Output: `scout_data/demographic/viability/m0_report.json`

## `train_clusters.py`

Place at `salience/modal_app/train_clusters.py`. **Build only after M0 pass and R1 hidden-state spike.**

Key changes from v1:

- `hidden_dim=1152` (not 4096).
- `TribeDemographicMux` with low-rank delta readout.
- Loss: `correlation_loss` in parcel space (not `F.mse_loss` on vertices).
- Eval hook: compute `delta_r` on val split each epoch; early-stop if `delta_r` plateaus below 0.

```python
config = TribeMuxConfig(
    hidden_dim=1152,
    num_clusters=8,
    cluster_emb_dim=64,
    readout_rank=16,
    n_parcels=400,
    target_space="parcel",
)
# ...
loss = correlation_loss(y_pred, y_target, dim=-1)  # not F.mse_loss
```

## Inference Contract

Place Modal inference in `salience/modal_app/inference_mux.py`.

### API

```python
def infer_mux(
    video_bytes: bytes,
    cluster_ids: list[int],
    *,
    k_chunk: int = 8,
    target_space: str = "parcel",
    return_format: str = "npz",
) -> dict:
    ...
```

### Output

```text
{
  "schema_version": 2,
  "mesh": "fsaverage5",
  "target_space": "parcel",
  "n_parcels": 400,
  "fps": 1.0,
  "cluster_ids": [0, 1, ...],
  "provenance": {
    "mux_checkpoint": "...",
    "m0_passed": true,
    "eval_gate_passed": true,
    "disclaimer": "Model-relative demographic prototypes; not measured emotion."
  },
  "parcel_ts_uri": "scout_data/sessions/<id>/mux_parcel_ts.npz",
  "chunks": [...]
}
```

### Runtime Flow

```text
video_bytes
  -> TRIBE get_events_dataframe (once)
  -> frozen latent extractor -> X_univ[T, 1152]
  -> y_base = pop_readout(X_univ)                    # population, once
  -> for cluster_ids chunks:
       y_delta = mux_delta(X_univ, cluster_ids_chunk)
       y_cluster = y_base + y_delta
       persist parcel_ts chunk
  -> scout_core parcellation/network aggregation if needed
  -> per-cluster analysis_bundle sections
```

### Integration With Existing Yeo-7 Mapper

For each cluster `k` (parcel_ts already `[T, 400]` or derive network_ts):

```python
net_ts, net_ids = network_timeseries(parcel_ts[k], parcel_ids, parcel_to_net)
z_net = zscore_network(net_ts, network_norms, net_ids)
hits = evaluate_rules(ctx, rules_path, insight_catalog_path)
```

## Training Validation

### Unit Tests (`tests/test_tribe_mux.py`)

1. `x_univ[T,1152] + cluster_ids[K] -> [K,T,P]`.
2. Batched shapes and gradient flow to `delta_u`, `delta_v`, `cluster_embedding` only.
3. `pop_weight` gradients are zero (frozen).
4. K=1 output equals `pop_readout + delta_0` within tolerance.

### Training Checks

1. Synthetic shard: known linear delta from cluster ids; correlation loss decreases in < 100 steps.
2. M0 report exists and `passed: true` before Modal training run.
3. Eval gate metrics logged each epoch.

## Milestones

### M0: Viability Experiment (GATING — must pass first)

- Run `scripts/run_demographic_viability.py` on HCP 7T or Cam-CAN parcellated data.
- Produce `scout_data/demographic/viability/m0_report.json`.
- **Go/no-go decision documented in ledger.**
- If no-go: stop; do not proceed to M1-M5.

### M1: TRIBE Hidden-State Spike

- Pin TRIBE commit.
- Locate `X_univ[T, 1152]` after 2Hz→1Hz pool.
- Confirm `pop_readout(X_univ) ≈ predict(events=df)`.
- Document 5 s hemodynamic offset.
- Implement no-hidden-state fallback path if hook is blocked.

### M2: Data Prep Dry Run

- Harmonize metadata for HCP 7T + Cam-CAN (skip ethnicity).
- Build clusters with `N >= 20` or explicit `suppressed=true`.
- Align one subject/video to `[T, 400]` parcel space.
- Package one WebDataset shard with provenance.

### M3: Mux Training Smoke Test + Eval Gate

- Train `TribeDemographicMux` with correlation loss on real centroids.
- Subject-held-out + LODO eval.
- **Eval gate:** `delta_r > 0.02`, `p_perm < 0.01`, beats single-cluster baseline.
- If gate fails: **kill switch** — do not proceed to M4.

### M4: End-to-End Demographic Inference

- `infer_mux(video, cluster_ids)` emits `[K,T,P]` parcel timeseries.
- `scout_core` produces per-cluster `network_ts[K,T,7]`.
- `analysis_bundle` contains per-cluster traces with provenance disclaimer.
- Integrate into `services/pipeline/runner.py` as optional stage (behind feature flag).

### M5: Neural Barrier Layer

- Compare cluster pairs over network z-scores.
- Detect divergence windows.
- Link to DOM/video timeline.
- Only ship if M3 eval gate passed.

## Risks And Guardrails

### Accuracy and Scientific Integrity

- **Do not market clusters as measuring real demographic emotion.** They are model-relative prototype predictions trained on movie-watching fMRI, applied to UI walkthrough video.
- **Population baseline is already OOD** for UI capture. Demographic deltas compound extrapolation (movie → UI, group average → individual segment).
- **MSE on 20484 vertices** optimizes scale/variance, not demographic signal. Use correlation + noise-ceiling in parcel/network space only.

### Data Risks

- **Small N per cell:** Most age×sex cells cannot reach N≥20. Suppress underpowered clusters; never show in UI.
- **Site/scanner confound:** HCP, Cam-CAN, CNeuroMod use different scanners. Age correlates with dataset. Mandate LODO and site covariate in M0 and eval.
- **Ethnicity removed:** Do not re-add without a powered, consented dataset.

### Domain Shift (Movie → UI)

- Training data: subjects watched Hollywood movies / short clips.
- Inference data: Playwright UI screen-capture walkthroughs.
- **Validation:** Compare multiplexer delta magnitude on movie vs. UI sessions. If UI deltas are uncorrelated with movie-trained deltas or are larger than movie deltas (noise amplification), suppress demographic comparison in product.
- **Product copy:** Always show `provenance.disclaimer` in viewer and API responses.

### Engineering Guardrails

- Suppress clusters below `N=20` or that fail eval gate.
- Subject-held-out + LODO splits mandatory; no sample-level k-fold on windowed timeseries.
- Preserve dataset provenance, DUA version, and consent scope in every centroid artifact.
- Do not train or fine-tune DINOv2, V-JEPA2, LLaMA, or other TRIBE backbone components.
- Keep `fsaverage5` vertex count at 20484 for optional viewer path; primary path is 400 parcels / 7 networks.

## Definition Of Done

The demographic multiplexer is ready for the application **only when all of the following hold**:

1. **M0 passed:** `m0_report.json` documents demographic variance above permutation null.
2. **M3 eval gate passed:** `delta_r > 0.02`, `p_perm < 0.01`, LODO stable.
3. One video produces `parcel_ts[K,T,400]` (or `network_ts[K,T,7]`) for selected, non-suppressed cluster ids.
4. Frozen TRIBE encoder runs once per video, not once per cluster.
5. Training optimizes only mux delta parameters (population readout frozen).
6. Cluster centroids have documented metadata provenance including site and DUA.
7. Existing Yeo-7 analysis runs per cluster with explicit uncertainty disclaimer in `analysis_bundle`.
8. Viewer shows side-by-side cluster network traces only for eval-gate-passed clusters.
9. Feature flag `DEMOGRAPHIC_MUX=1` defaults off until eval gate signed off in ledger.

If M0 or M3 eval gate fails, Definition of Done is **not met** and the feature ships as population-only mode.
