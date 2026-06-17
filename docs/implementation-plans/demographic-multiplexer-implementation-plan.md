# Demographic Subagent Multiplexer Implementation Plan

## Intent

This plan implements the Salience demographic multiplexer as a late-stage feature branching system on top of Meta TRIBE v2. The design freezes the expensive universal stimulus encoder and trains only a demographic prototype embedding table plus a shared readout head.

The target production path is:

```text
UI walkthrough video
  -> frozen TRIBE universal encoders
  -> X_univ[T, D_hidden]
  -> demographic embedding lookup C[K, D_emb]
  -> shared multiplexed readout MLP
  -> vertex_ts[K, T, 20484]
  -> existing Yeo-7 parcellation / norms / threshold mapper
  -> per-demographic cognitive and UX analysis
```

This must not fork TRIBE into separate demographic models. The only trainable demographic state is the cluster embedding matrix and readout head.

## Directory Structure

The current repo has root-level `tribe.py`, `scout_core/`, `scripts/`, `configs/`, and `tests/`. Add the demographic multiplexer under a new `salience/` package while keeping the existing baseline code stable.

```text
TribeV2Application/
  docs/implementation-plans/demographic-multiplexer-implementation-plan.md
  tribe.py
  activation_store.py
  configs/
    calibrated_rules.yaml
    cluster_schema.yaml                 # new: allowed demographic fields, min-n policy
    clusters.yaml                       # new: cluster_id -> label, rule, n_subjects
    parcellation_manifest.yaml
    vertex_regions.csv
  data_prep/
    __init__.py                         # new
    metadata_harmonize.py               # new: ingest HCP/CNeuroMod/Algonauts metadata
    segment_clusters.py                 # new: rule-based age/sex/dataset cluster assignment
    align_timeseries.py                 # new: temporal + surface-space alignment
    aggregate_centroids.py              # new: group subject tensors into Y_target[K,T,V]
    package_webdataset.py               # new: video + centroid tensor tar sharding
    schemas.py                          # new: pydantic contracts for subject/video/cluster rows
  salience/
    __init__.py                         # new
    mux/
      __init__.py                       # new
      tribe_mux.py                      # new: TribeDemographicMux nn.Module
      losses.py                         # new: MSE + optional smoothness/cluster regularizers
      checkpoints.py                    # new: save/load mux-only state_dict
      inference.py                      # new: infer_mux contract + K_chunk streaming
    modal_app/
      __init__.py                       # new
      image_defs.py                     # new: shared Modal image with TRIBE/WebDataset deps
      train_clusters.py                 # new: Modal A100 training loop
      inference_mux.py                  # new: Modal inference API returning K,T,V chunks
      stream_protocol.py                # new: chunked vertex_ts / Yeo-7 payload framing
  scout_core/
    aggregate.py
    norms.py
    parcellation.py
    threshold_engine.py
    storage_migrations.py
  scout_data/
    demographic/
      metadata_harmonized.parquet       # generated
      cluster_members.parquet           # generated
      aligned_subject_ts/               # generated: subject/video tensors
      centroids/                        # generated: Y_target per video
      webdataset/
        train-000000.tar                # generated
        val-000000.tar                  # generated
        test-000000.tar                 # generated
  scripts/
    analyze_session.py
    build_vertex_regions_csv.py
    compute_norms.py
    run_demographic_data_prep.py        # new: orchestrates 5-step centroid build
  tests/
    test_scout_core.py
    test_tribe_mux.py                   # new: tensor-shape and gradient-freeze tests
    test_demographic_data_prep.py       # new: metadata/cluster leakage checks
```

## Architecture

### Frozen Universal Encoder

The frozen path should be isolated behind an adapter because TRIBE's public demo API currently exposes `predict(events=df)` rather than a clean "return universal hidden state" function.

Required R0 spike:

1. Locate TRIBE's internal module that produces the final pre-subject or pre-readout hidden state.
2. Confirm tensor rank and meaning, expected as `X_univ: FloatTensor[T, D_hidden]` or `FloatTensor[B, T, D_hidden]`.
3. Prove that calling this path once per video is equivalent to the baseline public `predict()` path for `K=1` after attaching a compatible readout.
4. Freeze all upstream TRIBE parameters:

```python
for p in tribe_backbone.parameters():
    p.requires_grad_(False)
tribe_backbone.eval()
```

Do not call the video/audio/text encoders once per cluster. The encoders run exactly once per video; only the small demographic branch scales with `K`.

### Demographic Prototypes

Represent demographic segments as integer cluster ids:

```text
cluster_id: int64 in [0, num_clusters)
cluster label examples:
  genz_female
  millennial_male
  genx_female
  boomer_male
```

Each cluster id indexes a trainable embedding table:

```text
C = Embedding(num_clusters=K_total, embedding_dim=D_emb)
```

The embedding vector is not a demographic claim by itself. It is a learned prototype coordinate optimized to match cluster-average fMRI targets.

### Multiplexed Readout

For selected `K` cluster ids:

```text
X_univ:      [T, D_hidden]
C(ids):     [K, D_emb]
X_broadcast [K, T, D_hidden]
C_broadcast [K, T, D_emb]
concat:     [K, T, D_hidden + D_emb]
MLP:        [K, T, 20484]
```

Output:

```text
vertex_ts: FloatTensor[K, T, V]
```

Where `V = 20484` for TRIBE's `fsaverage5` surface output.

## `tribe_mux.py`

Place this file at `salience/mux/tribe_mux.py`.

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TribeMuxConfig:
    hidden_dim: int
    num_clusters: int
    cluster_emb_dim: int = 128
    readout_hidden_dim: int = 2048
    n_vertices: int = 20484
    dropout: float = 0.1
    layer_norm: bool = True


class TribeDemographicMux(nn.Module):
    """
    Late-stage demographic multiplexer for TRIBE universal hidden states.

    The module expects TRIBE's expensive multimodal encoders to be run outside
    this class exactly once per video. This class only performs demographic
    branching from a universal hidden tensor X_univ and selected cluster ids.

    Shapes
    ------
    x_univ:
        [T, D_hidden] or [B, T, D_hidden]
    cluster_ids:
        [K] for unbatched x_univ, or [B, K] / [K] for batched x_univ
    returns:
        [K, T, V] for unbatched input, or [B, K, T, V] for batched input
    """

    def __init__(self, config: TribeMuxConfig) -> None:
        super().__init__()
        self.config = config

        self.cluster_embedding = nn.Embedding(
            num_embeddings=config.num_clusters,
            embedding_dim=config.cluster_emb_dim,
        )

        in_dim = config.hidden_dim + config.cluster_emb_dim
        layers: list[nn.Module] = []
        if config.layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.extend(
            [
                nn.Linear(in_dim, config.readout_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.readout_hidden_dim, config.readout_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.readout_hidden_dim, config.n_vertices),
            ]
        )
        self.readout = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cluster_embedding.weight, mean=0.0, std=0.02)
        for module in self.readout.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_univ: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Broadcast universal stimulus features over demographic prototypes.

        Parameters
        ----------
        x_univ:
            Universal TRIBE latent, shape [T, D] or [B, T, D].
        cluster_ids:
            Cluster ids. For [T, D], pass [K]. For [B, T, D], pass [K] to
            share clusters across batch or [B, K] for per-example cluster sets.

        Returns
        -------
        torch.Tensor
            Vertex activations shaped [K, T, V] or [B, K, T, V].
        """
        if x_univ.ndim == 2:
            return self._forward_single(x_univ, cluster_ids)
        if x_univ.ndim == 3:
            return self._forward_batched(x_univ, cluster_ids)
        raise ValueError(f"x_univ must be [T,D] or [B,T,D], got {tuple(x_univ.shape)}")

    def _forward_single(
        self,
        x_univ: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        if cluster_ids.ndim != 1:
            raise ValueError(f"cluster_ids must be [K] for single input, got {tuple(cluster_ids.shape)}")

        T, D = x_univ.shape
        if D != self.config.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.config.hidden_dim}, got {D}")

        cluster_ids = cluster_ids.to(device=x_univ.device, dtype=torch.long)
        cluster_emb = self.cluster_embedding(cluster_ids)  # [K, D_emb]
        K = cluster_emb.shape[0]

        x_k = x_univ.unsqueeze(0).expand(K, T, D)  # [K, T, D_hidden]
        c_k = cluster_emb[:, None, :].expand(K, T, self.config.cluster_emb_dim)
        mux_in = torch.cat([x_k, c_k], dim=-1)  # [K, T, D_hidden + D_emb]

        flat = mux_in.reshape(K * T, -1)
        pred = self.readout(flat)
        return pred.reshape(K, T, self.config.n_vertices)

    def _forward_batched(
        self,
        x_univ: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, T, D = x_univ.shape
        if D != self.config.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.config.hidden_dim}, got {D}")

        cluster_ids = cluster_ids.to(device=x_univ.device, dtype=torch.long)
        if cluster_ids.ndim == 1:
            cluster_ids = cluster_ids.unsqueeze(0).expand(B, -1)
        if cluster_ids.ndim != 2 or cluster_ids.shape[0] != B:
            raise ValueError(
                "For batched input, cluster_ids must be [K] or [B,K], "
                f"got {tuple(cluster_ids.shape)} for B={B}"
            )

        cluster_emb = self.cluster_embedding(cluster_ids)  # [B, K, D_emb]
        K = cluster_emb.shape[1]

        x_k = x_univ[:, None, :, :].expand(B, K, T, D)
        c_k = cluster_emb[:, :, None, :].expand(B, K, T, self.config.cluster_emb_dim)
        mux_in = torch.cat([x_k, c_k], dim=-1)  # [B, K, T, D_hidden + D_emb]

        flat = mux_in.reshape(B * K * T, -1)
        pred = self.readout(flat)
        return pred.reshape(B, K, T, self.config.n_vertices)

    def trainable_parameter_groups(
        self,
        *,
        embedding_weight_decay: float = 0.1,
        readout_weight_decay: float = 0.01,
    ) -> list[dict]:
        """
        Optimizer groups with stronger regularization on demographic embeddings.
        """
        readout_params = [p for p in self.readout.parameters() if p.requires_grad]
        emb_params = [p for p in self.cluster_embedding.parameters() if p.requires_grad]
        return [
            {
                "params": emb_params,
                "weight_decay": embedding_weight_decay,
                "name": "cluster_embedding",
            },
            {
                "params": readout_params,
                "weight_decay": readout_weight_decay,
                "name": "readout",
            },
        ]
```

## Data Extraction Plan

### Dependencies

Add the following dependencies once implementation begins:

```text
torch
webdataset
nilearn
nibabel
pandas
pyarrow
numpy
scipy
scikit-learn
pydantic
pyyaml
tqdm
```

Optional but likely required for serious surface conversion:

```text
neuromaps
templateflow
```

Workbench is usually installed as a system binary, not a Python package:

```text
wb_command
```

### 1. Metadata Harmonization

New script: `data_prep/metadata_harmonize.py`

Input examples:

```text
raw_metadata/
  hcp_subjects.csv
  cneuromod_participants.tsv
  algonauts_subjects.tsv
```

Unified schema:

```text
subject_id: str
source_subject_id: str
dataset: Literal["hcp", "cneuromod", "algonauts"]
age: float | null
age_band: str
sex: Literal["female", "male", "unknown"]
race_ethnicity: str | null
handedness: str | null
metadata_quality: str
consent_scope: str | null
```

Implementation details:

1. Use `pandas.read_csv()` with `sep="\t"` for TSV datasets.
2. Normalize column names through dataset-specific mapping dictionaries.
3. Convert age ranges to midpoint or controlled band:

```text
18-24 -> genz
25-40 -> millennial
41-56 -> genx
57+   -> boomer_plus
```

4. Normalize sex values to `female`, `male`, `unknown`.
5. Persist:

```text
scout_data/demographic/metadata_harmonized.parquet
```

6. Validate:

```python
assert df["subject_id"].is_unique
assert {"subject_id", "age_band", "sex", "dataset"}.issubset(df.columns)
```

### 2. Rule-Based Segmentation

New script: `data_prep/segment_clusters.py`

Config: `configs/clusters.yaml`

```yaml
schema_version: 1
min_subjects_per_cluster: 20
clusters:
  - cluster_id: 0
    label: genz_female
    where:
      age_band: genz
      sex: female
  - cluster_id: 1
    label: genz_male
    where:
      age_band: genz
      sex: male
  - cluster_id: 2
    label: millennial_female
    where:
      age_band: millennial
      sex: female
```

Implementation details:

1. Load harmonized metadata with `pandas.read_parquet()`.
2. Apply exact-match rules from YAML.
3. Enforce `N >= 20` before allowing a cluster into training.
4. Mark underpowered clusters as `suppressed=true`, never as trainable labels.
5. Persist:

```text
scout_data/demographic/cluster_members.parquet
scout_data/demographic/cluster_demographics.json
```

Leakage guard:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
# group by source subject id so the same person never crosses train/val/test
```

### 3. Spatiotemporal Alignment

New script: `data_prep/align_timeseries.py`

Goal:

```text
raw subject fMRI/video tensor
  -> aligned subject response tensor [T_1hz, 20484]
```

Temporal alignment:

1. Load subject response time series as `float32`.
2. Read dataset TR:

```text
HCP       TR = 0.72s
CNeuroMod TR = 1.49s
Algonauts dataset-specific TR
```

3. Convert source sample index to seconds:

```python
source_t = np.arange(n_source_t) * source_tr
target_t = np.arange(0.0, video_duration_s, 1.0)
```

4. Use SciPy interpolation:

```python
from scipy.interpolate import interp1d

interp = interp1d(
    source_t,
    y_source,
    axis=0,
    kind="linear",
    bounds_error=False,
    fill_value="extrapolate",
)
y_1hz = interp(target_t).astype("float32")
```

Use `scipy.signal.resample_poly()` only when the ratio is clean and the series is long enough. For variable TR and explicit video-duration alignment, `interp1d()` is easier to audit.

Spatial alignment:

Preferred path:

1. Convert all subjects to a known surface space with dataset-provided preprocessing when available.
2. If input is `fsLR_32k`, transform to `fsaverage5`.
3. Use Workbench for surface resampling where surfaces/spheres are available:

```text
wb_command -metric-resample \
  input.func.gii \
  source.sphere.gii \
  target.fsaverage5.sphere.gii \
  ADAP_BARY_AREA \
  output.fsaverage5.func.gii \
  -area-surfs source.midthickness.surf.gii target.midthickness.surf.gii
```

4. Use `nibabel.load()` to read GIFTI:

```python
import nibabel as nib
arr = np.asarray(nib.load(path).agg_data(), dtype=np.float32)
```

5. Concatenate hemispheres into TRIBE order:

```text
lh vertices first, rh vertices second
```

6. Validate exact target shape:

```python
assert y_aligned.shape == (T_1hz, 20484)
```

Nilearn functions to use:

```python
from nilearn import datasets, surface

fsaverage = datasets.load_fsaverage(mesh="fsaverage5")
surface.load_surf_data(...)
surface.vol_to_surf(...)  # only if starting from volumetric NIfTI
```

If starting from MNI volumetric NIfTI, use `nilearn.surface.vol_to_surf()` onto fsaverage5 pial surfaces, but treat this as a lower-confidence path because interpolation can blur cortical signals.

Output:

```text
scout_data/demographic/aligned_subject_ts/
  <dataset>/<video_id>/<subject_id>.npy   # float32 [T, 20484]
```

### 4. Centroid Aggregation

New script: `data_prep/aggregate_centroids.py`

For each `video_id` and cluster:

```python
stack = np.stack(subject_tensors, axis=0)  # [N_cluster, T, V]
centroid = stack.mean(axis=0).astype("float32")  # [T, V]
```

Robust alternative:

```python
centroid = np.median(stack, axis=0).astype("float32")
```

Recommended baseline:

1. Z-score each subject per vertex before averaging if datasets have incompatible response scales.
2. Average within dataset first, then across datasets to prevent a larger dataset from dominating:

```text
subject -> dataset_cluster_mean -> cross_dataset_cluster_mean
```

3. Smooth over time with a small rolling window only after preserving the raw centroid:

```python
from scipy.ndimage import uniform_filter1d
centroid_smooth = uniform_filter1d(centroid, size=3, axis=0, mode="nearest")
```

Output:

```text
scout_data/demographic/centroids/
  <video_id>.npz
    y_target: float32 [K_trainable, T, 20484]
    cluster_ids: int64 [K_trainable]
    fps: float32 scalar = 1.0
```

### 5. WebDataset Packaging

New script: `data_prep/package_webdataset.py`

Shard structure:

```text
train-000000.tar
  000000.json
  000000.mp4
  000000.y_target.npy
  000000.cluster_ids.npy
  000001.json
  000001.mp4
  000001.y_target.npy
  000001.cluster_ids.npy
```

Use:

```python
import webdataset as wds

with wds.TarWriter("train-000000.tar") as sink:
    sink.write({
        "__key__": sample_id,
        "mp4": video_bytes,
        "y_target.npy": y_target.astype("float32"),
        "cluster_ids.npy": cluster_ids.astype("int64"),
        "json": {
            "video_id": video_id,
            "duration_s": float(duration_s),
            "fps": 1.0,
            "mesh": "fsaverage5",
            "n_vertices": 20484,
        },
    })
```

Training loader:

```python
dataset = (
    wds.WebDataset(shard_urls)
    .decode()
    .to_tuple("mp4", "y_target.npy", "cluster_ids.npy", "json")
)
```

For Modal, store shards on a `modal.Volume` or external object storage. Keep sample-level metadata in each tar to make training reproducible even after re-sharding.

## `train_clusters.py`

Place this file at `salience/modal_app/train_clusters.py`.

This is a skeletal script. The `TribeLatentExtractor` adapter must be implemented after the R0 source-code spike finds TRIBE's internal universal hidden state.

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import modal


app = modal.App("tribe-demographic-mux-train")

volume = modal.Volume.from_name("tribe-demographic-mux-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libegl1-mesa",
        "libosmesa6",
    )
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pandas",
        "pyarrow",
        "webdataset",
        "scipy",
        "nilearn",
        "nibabel",
        "pyyaml",
        "tqdm",
        "modal",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/tribev2.git /root/tribev2",
        'pip install -e "/root/tribev2[plotting]"',
    )
)


class TribeLatentExtractor:
    """
    Adapter around TRIBE v2 internals.

    TODO after R0:
    - locate universal hidden-state function/module
    - return X_univ as [B, T, D_hidden]
    - guarantee frozen encoder path and no cluster-dependent compute
    """

    def __init__(self, cache_folder: str) -> None:
        from tribev2.demo_utils import TribeModel

        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=cache_folder,
        )

        # If TribeModel exposes a torch module, freeze it here. The exact
        # attribute must be verified against the pinned TRIBE commit.
        for maybe_name in ("model", "net", "module"):
            module = getattr(self.model, maybe_name, None)
            if module is not None and hasattr(module, "parameters"):
                for p in module.parameters():
                    p.requires_grad_(False)
                module.eval()

    def encode_video_batch(self, video_batch: list[bytes]):
        """
        Return universal hidden states [B, T, D_hidden].

        This placeholder intentionally raises until TRIBE internals are pinned.
        The baseline public API returns final predictions, not the required
        pre-readout X_univ.
        """
        raise NotImplementedError("Implement after TRIBE R0 hidden-state spike.")


def make_loader(shard_urls: Iterable[str], batch_size: int):
    import webdataset as wds

    return (
        wds.WebDataset(list(shard_urls), shardshuffle=True)
        .shuffle(512)
        .to_tuple("mp4", "y_target.npy", "cluster_ids.npy", "json")
        .batched(batch_size, partial=False)
    )


@app.cls(
    gpu="A100",
    image=image,
    volumes={"/mnt/mux": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 8,
)
class DemographicMuxTrainer:
    @modal.enter()
    def setup(self) -> None:
        import torch

        from salience.mux.tribe_mux import TribeDemographicMux, TribeMuxConfig

        self.device = torch.device("cuda")
        self.latent_extractor = TribeLatentExtractor(cache_folder="/mnt/mux/hf_cache")

        # TODO: set hidden_dim from R0 measurement, not a guessed constant.
        config = TribeMuxConfig(
            hidden_dim=4096,
            num_clusters=16,
            cluster_emb_dim=128,
            readout_hidden_dim=2048,
            n_vertices=20484,
            dropout=0.1,
        )
        self.mux = TribeDemographicMux(config).to(self.device)

        # Strict trainable surface: only the demographic embedding and readout.
        for p in self.mux.parameters():
            p.requires_grad_(True)

        self.scaler = torch.amp.GradScaler("cuda")

    @modal.method()
    def train(
        self,
        shard_urls: list[str],
        *,
        epochs: int = 5,
        batch_size: int = 1,
        lr: float = 3e-4,
        embedding_weight_decay: float = 0.2,
        readout_weight_decay: float = 0.01,
        grad_clip_norm: float = 1.0,
        checkpoint_every_steps: int = 500,
    ) -> str:
        import json
        import time

        import torch
        import torch.nn.functional as F

        loader = make_loader(shard_urls, batch_size=batch_size)

        optimizer = torch.optim.AdamW(
            self.mux.trainable_parameter_groups(
                embedding_weight_decay=embedding_weight_decay,
                readout_weight_decay=readout_weight_decay,
            ),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        global_step = 0
        losses: list[float] = []
        self.mux.train()

        for epoch in range(epochs):
            for video_bytes_batch, y_target_np, cluster_ids_np, meta in loader:
                optimizer.zero_grad(set_to_none=True)

                # Frozen TRIBE path. No gradients should flow into the encoder.
                with torch.no_grad():
                    x_univ = self.latent_extractor.encode_video_batch(video_bytes_batch)
                    x_univ = x_univ.to(self.device, dtype=torch.float32, non_blocking=True)

                y_target = torch.as_tensor(
                    y_target_np,
                    device=self.device,
                    dtype=torch.float32,
                )
                cluster_ids = torch.as_tensor(
                    cluster_ids_np,
                    device=self.device,
                    dtype=torch.long,
                )

                # Expected y_target: [B, K, T, V].
                # If WebDataset emits [K,T,V] for B=1, normalize it here.
                if y_target.ndim == 3:
                    y_target = y_target.unsqueeze(0)
                if cluster_ids.ndim == 1:
                    cluster_ids = cluster_ids.unsqueeze(0)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    y_pred = self.mux(x_univ, cluster_ids)
                    if y_pred.shape != y_target.shape:
                        raise RuntimeError(
                            f"prediction shape {tuple(y_pred.shape)} != "
                            f"target shape {tuple(y_target.shape)}"
                        )
                    loss = F.mse_loss(y_pred, y_target)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.mux.parameters(), grad_clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()

                global_step += 1
                loss_value = float(loss.detach().cpu())
                losses.append(loss_value)

                if global_step % checkpoint_every_steps == 0:
                    self._save_checkpoint(global_step, epoch, optimizer, losses)

        ckpt_path = self._save_checkpoint(global_step, epochs - 1, optimizer, losses)
        metrics_path = Path("/mnt/mux/checkpoints") / f"metrics_{int(time.time())}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "final_checkpoint": ckpt_path,
                    "global_step": global_step,
                    "epochs": epochs,
                    "loss_tail": losses[-100:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        volume.commit()
        return str(metrics_path)

    def _save_checkpoint(self, step: int, epoch: int, optimizer, losses: list[float]) -> str:
        import torch

        out_dir = Path("/mnt/mux/checkpoints")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"mux_step_{step:08d}.pt"
        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "mux_state_dict": self.mux.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_tail": losses[-100:],
            },
            path,
        )
        volume.commit()
        return str(path)
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
    return_format: str = "npz",
) -> dict:
    ...
```

### Input

```text
video_bytes: raw mp4 bytes
cluster_ids: selected demographic prototypes
k_chunk: max clusters per readout pass
```

### Output

```text
{
  "schema_version": 1,
  "mesh": "fsaverage5",
  "n_vertices": 20484,
  "fps": 1.0,
  "cluster_ids": [0, 1, 2, ...],
  "vertex_ts_uri": "scout_data/sessions/<id>/mux_vertex_ts.npz",
  "chunks": [
    {
      "cluster_start": 0,
      "cluster_end": 8,
      "shape": [8, T, 20484],
      "dtype": "float32"
    }
  ]
}
```

### Runtime Flow

```text
video_bytes
  -> TRIBE preprocessing / get_events_dataframe
  -> frozen latent extractor returns X_univ[T,D]
  -> for cluster_ids chunks:
       y_chunk = mux(X_univ, cluster_ids_chunk)
       persist y_chunk or stream to Yeo-7 mapper
  -> concatenate logical output as [K,T,V]
```

### Integration With Existing Yeo-7 Mapper

For each cluster `k`:

```python
parcel_ts, parcel_ids = parcel_timeseries(vertex_ts[k], vertex_parcel)
net_ts, net_ids = network_timeseries(parcel_ts, parcel_ids, parcel_to_net)
z_net = zscore_network(net_ts, network_norms, net_ids)
hits = evaluate_rules(ctx, rules_path, insight_catalog_path)
```

Add cluster-aware storage later:

```text
session_cluster_network_timeseries
  session_id
  cluster_id
  t_idx
  yeo_network_id
  value
  z_vs_norm
  norm_id

session_cluster_threshold_hit
  session_id
  cluster_id
  rule_id
  t_start
  t_end
  evidence_json
  confidence
```

Until then, write cluster outputs as file-backed arrays and generate one `analysis_bundle` section per cluster.

## Training Validation

### Unit Tests

`tests/test_tribe_mux.py`:

1. `x_univ[T,D] + cluster_ids[K] -> [K,T,V]`.
2. `x_univ[B,T,D] + cluster_ids[K] -> [B,K,T,V]`.
3. `x_univ[B,T,D] + cluster_ids[B,K] -> [B,K,T,V]`.
4. Backward pass produces gradients for `cluster_embedding` and `readout`.
5. No gradients are required for the frozen TRIBE adapter parameters.

### Training Checks

1. Train on a synthetic WebDataset shard where `Y_target` is generated by a known linear map from `X_univ` and cluster ids.
2. Verify loss decreases in <100 steps.
3. Verify `K=1` inference fits a single centroid without changing encoder weights.
4. Log GPU memory:

```python
torch.cuda.reset_peak_memory_stats()
...
peak_gb = torch.cuda.max_memory_allocated() / 1e9
```

5. Test `K in {4, 16, 32}` with `k_chunk in {4, 8, 16}`.

## Milestones

### M1: TRIBE Hidden-State Spike

- Pin TRIBE commit.
- Locate universal hidden-state tensor.
- Record `D_hidden`, `T` behavior, dtype, and frame alignment.
- Prove frozen encoder is called once per video.

### M2: Data Prep Dry Run

- Harmonize a small metadata sample.
- Build 2-4 synthetic clusters with `N >= 20` or explicit suppressed status.
- Align one small subject/video sample to `[T, 20484]`.
- Package one WebDataset shard.

### M3: Mux Training Smoke Test

- Train `TribeDemographicMux` with synthetic `X_univ`.
- Train through the Modal skeleton after `TribeLatentExtractor` is implemented.
- Save and reload mux-only checkpoints.

### M4: End-to-End Demographic Inference

- `infer_mux(video, cluster_ids)` emits `[K,T,V]`.
- Existing `scout_core` parcellation produces `network_ts[K,T,7]`.
- Analysis bundle contains per-cluster network traces and threshold hits.

### M5: Neural Barrier Layer

- Compare cluster pairs over network z-scores.
- Detect divergence windows.
- Link divergence windows to DOM/video timeline once Playwright traces are available.

## Risks And Guardrails

- Do not market clusters as measuring real demographic emotion. They are model-relative prototype predictions.
- Suppress clusters below `N=20`.
- Keep subject-level train/val/test splits leakage-free.
- Preserve dataset provenance and consent scope in every centroid artifact.
- Do not train or fine-tune DINOv2, V-JEPA2, LLaMA, or other TRIBE backbone components.
- Keep `fsaverage5` vertex count fixed at `20484` until the optional volumetric phase.
- Treat any surface-space conversion from volumetric data as lower-confidence unless validated against a known reference.

## Definition Of Done

The demographic multiplexer is ready for the application when:

1. One video produces `vertex_ts[K,T,20484]` for selected cluster ids.
2. The frozen TRIBE encoder runs once per video, not once per cluster.
3. Training optimizes only mux parameters.
4. Cluster centroids are generated from aligned subject tensors with documented metadata provenance.
5. The existing Yeo-7 analysis can run per cluster and generate per-demographic `analysis_bundle` entries.
6. The viewer can show side-by-side cluster network traces and highlight divergence windows over the ad timeline.
