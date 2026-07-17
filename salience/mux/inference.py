from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch

from salience.mux.checkpoints import load_mux_checkpoint
from salience.mux.tribe_mux import PredsFallbackMux, TribeDemographicMux, TribeMuxConfig


DISCLAIMER = (
    "Model-relative demographic prototypes trained on movie-watching fMRI; "
    "not measured demographic emotion. UI walkthrough inference is out-of-distribution."
)


def infer_mux_arrays(
    mux: TribeDemographicMux | PredsFallbackMux,
    *,
    x_univ: torch.Tensor | None = None,
    preds: torch.Tensor | None = None,
    cluster_ids: list[int],
    k_chunk: int = 8,
) -> np.ndarray:
    """Return parcel/network tensor [K, T, N_out] as float32 numpy."""
    ids = torch.tensor(cluster_ids, dtype=torch.long)
    chunks: list[torch.Tensor] = []

    for start in range(0, ids.shape[0], k_chunk):
        chunk_ids = ids[start : start + k_chunk]
        if isinstance(mux, PredsFallbackMux):
            if preds is None:
                raise ValueError("preds required for PredsFallbackMux")
            out = mux(preds, chunk_ids)
        else:
            if x_univ is None:
                raise ValueError("x_univ required for TribeDemographicMux")
            out = mux(x_univ, chunk_ids)
        chunks.append(out.detach().cpu())

    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def build_inference_payload(
    parcel_ts: np.ndarray,
    cluster_ids: list[int],
    *,
    checkpoint_path: str,
    m0_passed: bool,
    eval_gate_passed: bool,
    target_space: str = "parcel",
    fps: float = 1.0,
) -> dict[str, Any]:
    K, T, P = parcel_ts.shape
    return {
        "schema_version": 2,
        "mesh": "fsaverage5",
        "target_space": target_space,
        "n_parcels": int(P),
        "fps": fps,
        "cluster_ids": cluster_ids,
        "shape": [K, T, P],
        "provenance": {
            "mux_checkpoint": checkpoint_path,
            "m0_passed": m0_passed,
            "eval_gate_passed": eval_gate_passed,
            "disclaimer": DISCLAIMER,
        },
        "chunks": [
            {
                "cluster_start": 0,
                "cluster_end": K,
                "shape": [K, T, P],
                "dtype": "float32",
            }
        ],
    }


def save_parcel_ts_npz(path: Path, parcel_ts: np.ndarray, cluster_ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, parcel_ts=parcel_ts, cluster_ids=np.asarray(cluster_ids, dtype=np.int64))
    path.write_bytes(buf.getvalue())


def load_mux_for_inference(checkpoint_path: Path, device: str = "cpu") -> TribeDemographicMux:
    mux, _payload = load_mux_checkpoint(checkpoint_path, device=device)
    mux.eval()
    return mux
