"""Audit Horikawa training data before ridge/MLP fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.constants import DEFAULT_PRODUCT_CATEGORIES, HORIKAWA_14_DIMENSIONS
from scout_core.horikawaCode.labels import DEFAULT_LABEL_CACHE, load_horikawa_ratings
from scout_core.horikawaCode.prepare import build_train_npz_from_manifest


def _target_stats(y: np.ndarray, names: list[str]) -> list[dict]:
    rows: list[dict] = []
    for i, name in enumerate(names):
        col = y[:, i]
        std = float(col.std())
        rows.append({
            "name": name,
            "mean": float(col.mean()),
            "std": std,
            "min": float(col.min()),
            "max": float(col.max()),
            "near_constant": std < 1e-4,
        })
    return rows


def audit_manifest(
    manifest_path: Path,
    *,
    intermediates_dir: Path,
    labels_cache: Path,
) -> dict:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = payload.get("clips") or []
    corpus = payload.get("corpus", manifest_path.stem)

    product = build_train_npz_from_manifest(
        clips,
        intermediates_dir=intermediates_dir,
        labels_cache=labels_cache,
        corpus=corpus,
        target="product_8",
    )
    dims = build_train_npz_from_manifest(
        clips,
        intermediates_dir=intermediates_dir,
        labels_cache=labels_cache,
        corpus=corpus,
        target="dimensions_14",
    )

    X = np.asarray(product["X"], dtype=np.float32)
    y8 = np.asarray(product["y"], dtype=np.float32)
    y14 = np.asarray(dims["y"], dtype=np.float32)
    sids = np.asarray(product["stimulus_id"])

    const_cols = int(np.sum(X.std(axis=0) < 1e-8))
    issues: list[str] = []
    if not np.all(np.isfinite(X)):
        issues.append("non-finite values in X")
    if not np.all(np.isfinite(y8)) or not np.all(np.isfinite(y14)):
        issues.append("non-finite values in y")
    if len(np.unique(sids)) != len(sids):
        issues.append("duplicate stimulus_id rows")

    ratings = load_horikawa_ratings(labels_cache)
    missing_labels = [str(c["stimulus_id"]) for c in clips if str(c["stimulus_id"]) not in ratings]

    return {
        "manifest": str(manifest_path),
        "corpus": corpus,
        "n_clips": len(clips),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "constant_feature_columns": const_cols,
        "issues": issues,
        "missing_labels": missing_labels,
        "product_8_stats": _target_stats(y8, list(DEFAULT_PRODUCT_CATEGORIES)),
        "dimensions_14_stats": _target_stats(y14, list(HORIKAWA_14_DIMENSIONS)),
        "pass": len(issues) == 0 and len(missing_labels) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "pilot_275.json",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal",
    )
    parser.add_argument("--labels-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "reports" / "pilot_275_data_audit.json",
    )
    args = parser.parse_args()

    report = audit_manifest(
        args.manifest,
        intermediates_dir=args.intermediates_dir,
        labels_cache=args.labels_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")
    if not report["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
