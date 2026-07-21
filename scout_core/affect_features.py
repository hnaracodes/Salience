"""Unified cortical + subcortical feature extraction for affect decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from scout_core.neuroEmoCode.roi_features import load_roi_feature_spec, reduce_vertices_to_rois
from scout_core.subcortical.atlas import (
    N_SUBCORTICAL_VOXELS,
    build_region_index,
    roi_means_from_voxels,
    roi_stats_from_voxels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERTEX_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "parcellation_manifest.yaml"

FEATURE_SPEC_FUSED_V1 = "fused_schaefer400_subcortical_v1"
FEATURE_SPEC_FUSED_V2 = "fused_v2_real_subcortical"
FEATURE_SPEC_CORTICAL_MEAN400 = "cortical_mean400_v1"
FEATURE_SPEC_CORTICAL_MEAN_STD800 = "cortical_mean_std800_v1"
FEATURE_SPEC_SUBCORTICAL_FAKE8 = "subcortical_fake8_v1"
FEATURE_SPEC_SUBCORTICAL_REAL16 = "subcortical_real16_v1"
FEATURE_SPEC_SUBCORTICAL_PCA32 = "subcortical_pca32_v1"

ALL_FEATURE_SPECS = (
    FEATURE_SPEC_FUSED_V1,
    FEATURE_SPEC_FUSED_V2,
    FEATURE_SPEC_CORTICAL_MEAN400,
    FEATURE_SPEC_CORTICAL_MEAN_STD800,
    FEATURE_SPEC_SUBCORTICAL_FAKE8,
    FEATURE_SPEC_SUBCORTICAL_REAL16,
    FEATURE_SPEC_SUBCORTICAL_PCA32,
)


@dataclass(frozen=True)
class AffectFeatureSpec:
    name: str
    cortical_reducers: tuple[str, ...]
    include_subcortical: bool
    include_cortical: bool
    subcortical_mode: str  # none | fake8 | real8 | real16 | pca32
    window_trs: int
    n_cortical_features: int
    n_subcortical_features: int

    @property
    def n_features(self) -> int:
        cort = self.n_cortical_features if self.include_cortical else 0
        sub = self.n_subcortical_features if self.include_subcortical else 0
        return cort + sub


def _cortical_n_features(reducers: tuple[str, ...]) -> int:
    roi_spec = load_roi_feature_spec(
        DEFAULT_VERTEX_CSV,
        n_vertices=20484,
        reducers=reducers,
        manifest_path=DEFAULT_MANIFEST if DEFAULT_MANIFEST.is_file() else None,
        allow_legacy_atlas=True,
        require_surface_native=False,
    )
    return roi_spec.n_features


def resolve_feature_spec(name: str = FEATURE_SPEC_FUSED_V1) -> AffectFeatureSpec:
    if name == FEATURE_SPEC_CORTICAL_MEAN400:
        return AffectFeatureSpec(
            name=name,
            cortical_reducers=("mean",),
            include_subcortical=False,
            include_cortical=True,
            subcortical_mode="none",
            window_trs=1,
            n_cortical_features=_cortical_n_features(("mean",)),
            n_subcortical_features=0,
        )
    if name == FEATURE_SPEC_CORTICAL_MEAN_STD800:
        return AffectFeatureSpec(
            name=name,
            cortical_reducers=("mean", "std"),
            include_subcortical=False,
            include_cortical=True,
            subcortical_mode="none",
            window_trs=1,
            n_cortical_features=_cortical_n_features(("mean", "std")),
            n_subcortical_features=0,
        )
    if name == FEATURE_SPEC_SUBCORTICAL_FAKE8:
        return AffectFeatureSpec(
            name=name,
            cortical_reducers=(),
            include_subcortical=True,
            include_cortical=False,
            subcortical_mode="fake8",
            window_trs=1,
            n_cortical_features=0,
            n_subcortical_features=8,
        )
    if name == FEATURE_SPEC_SUBCORTICAL_REAL16:
        return AffectFeatureSpec(
            name=name,
            cortical_reducers=(),
            include_subcortical=True,
            include_cortical=False,
            subcortical_mode="real16",
            window_trs=1,
            n_cortical_features=0,
            n_subcortical_features=16,
        )
    if name == FEATURE_SPEC_SUBCORTICAL_PCA32:
        return AffectFeatureSpec(
            name=name,
            cortical_reducers=(),
            include_subcortical=True,
            include_cortical=False,
            subcortical_mode="pca32",
            window_trs=1,
            n_cortical_features=0,
            n_subcortical_features=32,
        )
    if name == FEATURE_SPEC_FUSED_V2:
        return AffectFeatureSpec(
            name=name,
            cortical_reducers=("mean", "std"),
            include_subcortical=True,
            include_cortical=True,
            subcortical_mode="real16",
            window_trs=1,
            n_cortical_features=_cortical_n_features(("mean", "std")),
            n_subcortical_features=16,
        )
    if name == FEATURE_SPEC_FUSED_V1:
        return AffectFeatureSpec(
            name=name,
            cortical_reducers=("mean", "std"),
            include_subcortical=True,
            include_cortical=True,
            subcortical_mode="fake8",
            window_trs=1,
            n_cortical_features=_cortical_n_features(("mean", "std")),
            n_subcortical_features=8,
        )
    raise ValueError(f"Unknown affect feature spec: {name}")


def slice_fused_features(X: np.ndarray, spec: AffectFeatureSpec | None = None) -> np.ndarray:
    """Slice a fused_v1 (808-dim) matrix to match a derived spec (for cortical-only ablations)."""
    spec = spec or resolve_feature_spec()
    fused = resolve_feature_spec(FEATURE_SPEC_FUSED_V1)
    if X.shape[1] != fused.n_features:
        return X
    n_cort = fused.n_cortical_features
    if spec.name == FEATURE_SPEC_CORTICAL_MEAN400:
        # mean reducers are first half of cortical block
        half = n_cort // 2
        return X[:, :half]
    if spec.name == FEATURE_SPEC_CORTICAL_MEAN_STD800:
        return X[:, :n_cort]
    if spec.name in (FEATURE_SPEC_SUBCORTICAL_FAKE8, FEATURE_SPEC_FUSED_V1):
        return X[:, n_cort:]
    raise ValueError(f"Cannot slice fused features for spec {spec.name}")


def _subcortical_block(
    subcortical_preds: np.ndarray,
    spec: AffectFeatureSpec,
    *,
    pca_model: PCA | None = None,
) -> np.ndarray:
    if spec.subcortical_mode == "fake8":
        return roi_means_from_voxels(subcortical_preds, use_fake_equal_split=True)
    if spec.subcortical_mode == "real8":
        return roi_means_from_voxels(subcortical_preds, use_fake_equal_split=False)
    if spec.subcortical_mode == "real16":
        return roi_stats_from_voxels(subcortical_preds, use_fake_equal_split=False)
    if spec.subcortical_mode == "pca32":
        arr = np.asarray(subcortical_preds, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if pca_model is None:
            raise ValueError("pca32 spec requires a fitted PCA model")
        return pca_model.transform(arr).astype(np.float32)
    raise ValueError(f"Unknown subcortical_mode: {spec.subcortical_mode}")


def build_affect_features(
    cortical_preds: np.ndarray,
    subcortical_preds: np.ndarray | None,
    *,
    spec: AffectFeatureSpec | None = None,
    window_trs: int | None = None,
    vertex_csv: Path | None = None,
    pca_model: PCA | None = None,
) -> np.ndarray:
    """Build per-TR feature matrix (T, F) from cortical and optional subcortical preds."""
    spec = spec or resolve_feature_spec()
    window = window_trs if window_trs is not None else spec.window_trs
    cortical_preds = np.asarray(cortical_preds, dtype=np.float32)

    cortical_feats: np.ndarray | None = None
    if spec.include_cortical:
        if cortical_preds.ndim != 2:
            raise ValueError(f"cortical_preds must be (T, V), got {cortical_preds.shape}")
        roi_spec = load_roi_feature_spec(
            vertex_csv or DEFAULT_VERTEX_CSV,
            n_vertices=cortical_preds.shape[1],
            reducers=spec.cortical_reducers,
            manifest_path=DEFAULT_MANIFEST if DEFAULT_MANIFEST.is_file() else None,
            allow_legacy_atlas=True,
            require_surface_native=False,
        )
        t_count = cortical_preds.shape[0]
        if window <= 1:
            cortical_feats = reduce_vertices_to_rois(cortical_preds, roi_spec)
        else:
            rows: list[np.ndarray] = []
            half = window // 2
            for t in range(t_count):
                lo = max(0, t - half)
                hi = min(t_count, t + half + 1)
                window_block = cortical_preds[lo:hi]
                flat = window_block.reshape(-1, cortical_preds.shape[1])
                row = reduce_vertices_to_rois(flat[np.newaxis, ...], roi_spec)[0]
                rows.append(row)
            cortical_feats = np.stack(rows, axis=0)

    if not spec.include_subcortical:
        assert cortical_feats is not None
        return cortical_feats.astype(np.float32)

    if subcortical_preds is None:
        sub_block = np.zeros((cortical_preds.shape[0] if cortical_feats is None else cortical_feats.shape[0], spec.n_subcortical_features), dtype=np.float32)
    else:
        sub_block = _subcortical_block(subcortical_preds, spec, pca_model=pca_model)

    if cortical_feats is None:
        return sub_block.astype(np.float32)
    return np.concatenate([cortical_feats, sub_block], axis=1).astype(np.float32)


def fit_subcortical_pca(
    subcortical_clips: np.ndarray,
    *,
    n_components: int = 32,
) -> PCA:
    """Fit PCA on clip-level pooled subcortical voxels (N, 8802)."""
    X = np.asarray(subcortical_clips, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != N_SUBCORTICAL_VOXELS:
        raise ValueError(f"expected (N, {N_SUBCORTICAL_VOXELS}), got {X.shape}")
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    pca.fit(X)
    return pca


def build_affect_features_batch(
    cortical_batch: np.ndarray,
    subcortical_batch: np.ndarray | None,
    *,
    spec: AffectFeatureSpec | None = None,
    pca_model: PCA | None = None,
) -> np.ndarray:
    """Build (N, F) features for clip-level rows."""
    spec = spec or resolve_feature_spec()
    cortical_batch = np.asarray(cortical_batch, dtype=np.float32)
    if cortical_batch.ndim == 2:
        return build_affect_features(cortical_batch, subcortical_batch, spec=spec, pca_model=pca_model)
    if cortical_batch.ndim != 3:
        raise ValueError(f"expected (N,T,V) or (T,V), got {cortical_batch.shape}")

    rows = []
    for i in range(cortical_batch.shape[0]):
        sub_i = None if subcortical_batch is None else subcortical_batch[i]
        feat = build_affect_features(cortical_batch[i], sub_i, spec=spec, pca_model=pca_model)
        rows.append(feat.mean(axis=0))
    return np.stack(rows, axis=0).astype(np.float32)
