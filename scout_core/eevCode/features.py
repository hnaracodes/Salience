"""FeatureContract v1: TRIBE preds -> EEV training/inference features."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.eevCode.constants import (
    DEFAULT_COHERENCE_WINDOW_TRS,
    DEFAULT_TR_HZ,
    EEV_FEATURE_CONTRACT_V1,
    FEATURE_NAMES_V1,
)
from scout_core.eevCode.networks import NetworkMasks, load_network_masks
from scout_core.parcellation import load_parcellation_manifest, load_vertex_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rolling_pearson(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Pairwise rolling Pearson r along time. NaN where window incomplete."""
    T = a.shape[0]
    out = np.full(T, np.nan, dtype=np.float32)
    if window < 2 or T < window:
        return out
    for t in range(window - 1, T):
        xa = a[t - window + 1 : t + 1]
        xb = b[t - window + 1 : t + 1]
        if np.std(xa) < 1e-8 or np.std(xb) < 1e-8:
            out[t] = 0.0
        else:
            out[t] = float(np.corrcoef(xa, xb)[0, 1])
    return out


def build_network_amplitudes(
    preds: np.ndarray,
    *,
    masks: NetworkMasks | None = None,
) -> np.ndarray:
    masks = masks or load_network_masks()
    return masks.amplitudes(preds)


def build_feature_matrix(
    preds: np.ndarray,
    *,
    masks: NetworkMasks | None = None,
    coherence_window_trs: int = DEFAULT_COHERENCE_WINDOW_TRS,
) -> np.ndarray:
    """Build FeatureContract v1 matrix. Shape (T, 15). Row 0 may contain NaN in coherence cols."""
    amp = build_network_amplitudes(preds, masks=masks)
    T, n_net = amp.shape
    if n_net != 3:
        raise ValueError(f"Expected 3 networks, got {n_net}")

    deriv = np.zeros_like(amp)
    if T > 1:
        deriv[1:] = amp[1:] - amp[:-1]

    early = np.zeros_like(amp)
    if T > 1:
        early[1:] = amp[:-1]
    late = amp.copy()

    coh_vs = _rolling_pearson(amp[:, 0], amp[:, 1], coherence_window_trs)
    coh_vd = _rolling_pearson(amp[:, 0], amp[:, 2], coherence_window_trs)
    coh_sd = _rolling_pearson(amp[:, 1], amp[:, 2], coherence_window_trs)

    X = np.column_stack(
        [
            amp,
            deriv,
            early,
            late,
            coh_vs,
            coh_vd,
            coh_sd,
        ]
    ).astype(np.float32)
    return X


def feature_provenance(*, vertex_csv: Path | None = None) -> dict[str, Any]:
    vpath = vertex_csv or (PROJECT_ROOT / "configs" / "vertex_regions.csv")
    mpath = PROJECT_ROOT / "configs" / "parcellation_manifest.yaml"
    manifest = load_parcellation_manifest(mpath)
    table = load_vertex_table(vpath)
    csv_bytes = vpath.read_bytes()
    return {
        "feature_contract": EEV_FEATURE_CONTRACT_V1,
        "feature_names": list(FEATURE_NAMES_V1),
        "n_features": len(FEATURE_NAMES_V1),
        "vertex_csv": str(vpath),
        "vertex_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "atlas_id": manifest.get("atlas_id"),
        "manifest_sha256": hashlib.sha256(mpath.read_bytes()).hexdigest(),
        "n_vertices": table.n_vertices,
        "vertex_order": "lh_then_rh_fsaverage5",
        "tribe_model": "facebook/tribev2",
    }


def pack_features_npz(
    *,
    video_id: str,
    preds: np.ndarray,
    masks: NetworkMasks | None = None,
    tr_hz: float = DEFAULT_TR_HZ,
    include_raw_preds: bool = False,
    vertex_csv: Path | None = None,
) -> dict[str, Any]:
    preds = np.asarray(preds, dtype=np.float32)
    T = preds.shape[0]
    t_sec = (np.arange(T, dtype=np.float32) * tr_hz).astype(np.float32)
    network_amp = build_network_amplitudes(preds, masks=masks)
    X_feat = build_feature_matrix(preds, masks=masks)
    masks = masks or load_network_masks(vertex_csv=vertex_csv)
    prov = feature_provenance(vertex_csv=vertex_csv)
    payload: dict[str, Any] = {
        "schema_version": EEV_FEATURE_CONTRACT_V1,
        "video_id": np.array(str(video_id)),
        "t_sec": t_sec,
        "tr_hz": np.float32(tr_hz),
        "network_names": np.array(list(masks.names)),
        "network_amp": network_amp,
        "X_feat": X_feat,
        "feature_names": np.array(list(FEATURE_NAMES_V1)),
        "atlas_manifest_sha256": np.array(prov["manifest_sha256"]),
        "vertex_order": np.array(prov["vertex_order"]),
        "tribe_model": np.array(prov["tribe_model"]),
    }
    if include_raw_preds:
        payload["preds"] = preds
    return payload


def save_features_npz(path: Path, **kwargs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pack_features_npz(**kwargs))
    return path


def load_features_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def preds_from_npz_bytes(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(io.BytesIO(data), allow_pickle=True) as z:
        meta = {k: z[k] for k in z.files if k != "preds"}
        preds = np.asarray(z["preds"], dtype=np.float32)
    return preds, meta
