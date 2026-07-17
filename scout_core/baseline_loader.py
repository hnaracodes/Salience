"""Load and aggregate TRIBE baseline prediction corpora for dual-track Z-scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_baseline_npz(path: Path) -> np.ndarray:
    raw = np.load(path)
    preds = np.asarray(raw["preds"], dtype=np.float32)
    if preds.ndim != 2:
        raise ValueError(f"Expected preds 2D, got shape {preds.shape} at {path}")
    return preds


def aggregate_baseline_corpus(
    npz_paths: list[Path],
    *,
    max_trs: int | None = None,
) -> np.ndarray:
    """Stack baseline TRs from multiple npz files along time axis."""
    chunks: list[np.ndarray] = []
    for path in npz_paths:
        if not path.is_file():
            continue
        chunks.append(load_baseline_npz(path))
    if not chunks:
        raise FileNotFoundError("No baseline npz files found to aggregate")
    stacked = np.concatenate(chunks, axis=0).astype(np.float32)
    if max_trs is not None and stacked.shape[0] > max_trs:
        stacked = stacked[:max_trs]
    return stacked


def load_baseline_from_config(
    baseline_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return (preds_baseline, metadata) from corpus manifest or single npz."""
    meta: dict[str, Any] = {"source": None, "corpus_id": None, "n_trs": 0}
    manifest_path = baseline_dir / "corpus_manifest.json"
    if manifest is None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest and manifest.get("files"):
        paths = [baseline_dir / f for f in manifest["files"]]
        preds = aggregate_baseline_corpus(paths, max_trs=manifest.get("max_trs"))
        meta["source"] = "corpus"
        meta["corpus_id"] = manifest.get("corpus_id", "baseline_corpus_v1")
        meta["n_trs"] = int(preds.shape[0])
        meta["files"] = [str(p.name) for p in paths if p.is_file()]
        return preds, meta

    single = baseline_dir / "preds_baseline.npz"
    if single.is_file():
        preds = load_baseline_npz(single)
        meta["source"] = "single_npz"
        meta["corpus_id"] = "preds_baseline"
        meta["n_trs"] = int(preds.shape[0])
        return preds, meta

    return None, meta
