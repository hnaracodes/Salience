"""Build clip-level features from TRIBE intermediates for Horikawa decoding."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from scout_core.affect_features import (
    FEATURE_SPEC_FUSED_V1,
    build_affect_features,
    fit_subcortical_pca,
    resolve_feature_spec,
)
from scout_core.horikawaCode.labels import (
    PROJECT_ROOT,
    build_y_from_label_dict,
    label_names_for_target,
    load_horikawa_ratings,
)


def _pool_clip_features(
    cortical: np.ndarray,
    subcortical: np.ndarray,
    *,
    feature_spec: str = FEATURE_SPEC_FUSED_V1,
    pca_model: PCA | None = None,
) -> np.ndarray:
    """Mean over TRs after optional onset/offset trim."""
    t_count = cortical.shape[0]
    lo, hi = 0, t_count
    if t_count > 4:
        lo = 1
        hi = t_count - 1
    cortical = cortical[lo:hi]
    subcortical = subcortical[lo:hi]
    spec = resolve_feature_spec(feature_spec)
    feat_tr = build_affect_features(cortical, subcortical, spec=spec, pca_model=pca_model)
    return feat_tr.mean(axis=0).astype(np.float32)


def load_both_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        cortical = np.asarray(data["preds"], dtype=np.float32)
        sub = np.asarray(data["preds_subcortical"], dtype=np.float32)
    return cortical, sub


def atlas_sha256() -> str:
    manifest = PROJECT_ROOT / "configs" / "parcellation_manifest.yaml"
    vertex_csv = PROJECT_ROOT / "configs" / "vertex_regions.csv"
    sub_csv = PROJECT_ROOT / "configs" / "subcortical_voxel_regions.csv"
    h = hashlib.sha256()
    for p in (manifest, vertex_csv, sub_csv):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def data_hash(X: np.ndarray, y: np.ndarray, stimulus_id: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    h.update("|".join(str(s) for s in stimulus_id).encode("utf-8"))
    return h.hexdigest()


def _normalize_target_name(target: str) -> str:
    if target == "dimensions":
        return "dimensions_14"
    if target == "dims_14":
        return "dimensions_14"
    if target == "product_8":
        return "product_8"
    return target


def build_train_npz_from_manifest(
    manifest_clips: list[dict],
    *,
    intermediates_dir: Path,
    labels_cache: Path,
    corpus: str,
    target: str = "product_8",
    feature_spec: str = FEATURE_SPEC_FUSED_V1,
    zscore_y: bool = False,
    pca_model: PCA | None = None,
    fit_pca_on_corpus: bool = False,
) -> dict[str, np.ndarray]:
    """Assemble training NPZ from per-clip TRIBE intermediates and figshare labels."""
    target = _normalize_target_name(target)
    ratings = load_horikawa_ratings(labels_cache)
    spec = resolve_feature_spec(feature_spec)

    if fit_pca_on_corpus and spec.subcortical_mode == "pca32":
        pooled_sub: list[np.ndarray] = []
        for entry in manifest_clips:
            sid = str(entry["stimulus_id"])
            npz_path = intermediates_dir / f"{sid}_both.npz"
            if not npz_path.is_file() and sid.isdigit():
                npz_path = intermediates_dir / f"{int(sid):04d}_both.npz"
            if not npz_path.is_file():
                continue
            _, sub = load_both_npz(npz_path)
            t_count = sub.shape[0]
            lo, hi = 0, t_count
            if t_count > 4:
                lo, hi = 1, t_count - 1
            pooled_sub.append(sub[lo:hi].mean(axis=0))
        pca_model = fit_subcortical_pca(np.stack(pooled_sub, axis=0))

    X_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    stimulus_ids: list[str] = []

    for entry in manifest_clips:
        sid = str(entry["stimulus_id"])
        npz_path = intermediates_dir / f"{sid}_both.npz"
        if not npz_path.is_file() and sid.isdigit():
            npz_path = intermediates_dir / f"{int(sid):04d}_both.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing intermediate NPZ for {sid}: {npz_path}")
        if sid not in ratings:
            raise KeyError(f"No ratings for stimulus_id={sid}")

        cortical, sub = load_both_npz(npz_path)
        if cortical.ndim != 2 or sub.ndim != 2:
            raise ValueError(f"{sid}: expected 2D preds, got {cortical.shape}, {sub.shape}")
        if not np.all(np.isfinite(cortical)) or not np.all(np.isfinite(sub)):
            raise ValueError(f"{sid}: non-finite preds")

        X_rows.append(
            _pool_clip_features(
                cortical, sub, feature_spec=feature_spec, pca_model=pca_model
            )
        )
        y_rows.append(build_y_from_label_dict(ratings[sid], target))
        stimulus_ids.append(sid)

    X = np.stack(X_rows, axis=0).astype(np.float32)
    y = np.stack(y_rows, axis=0).astype(np.float32)
    if zscore_y:
        from scout_core.horikawaCode.labels import zscore_targets

        y = zscore_targets(y)
    sid_arr = np.asarray(stimulus_ids, dtype=object)
    groups = np.arange(len(stimulus_ids), dtype=np.int32)

    if len(np.unique(sid_arr)) != len(sid_arr):
        raise ValueError("duplicate stimulus_id rows in training matrix")

    label_names = label_names_for_target(target)

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "stimulus_id": sid_arr,
        "label_names": np.asarray(label_names, dtype=object),
        "feature_spec": np.asarray(feature_spec),
        "target_type": np.asarray(target),
        "atlas_sha256": np.asarray(atlas_sha256()),
        "data_hash": np.asarray(data_hash(X, y, sid_arr)),
        "corpus": np.asarray(corpus),
    }
