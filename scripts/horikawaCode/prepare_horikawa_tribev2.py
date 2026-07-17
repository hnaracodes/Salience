"""Prepare Horikawa training NPZ from TRIBE predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.labels import DEFAULT_LABEL_CACHE, synthetic_horikawa_dataset
from scout_core.horikawaCode.prepare import build_train_npz_from_manifest


def _load_manifest_clips(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips") or []
    if clips and isinstance(clips[0], str):
        return [{"stimulus_id": c} for c in clips]
    return list(clips)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "horikawa_decoding.yaml",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal",
    )
    parser.add_argument("--labels-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--demo", action="store_true", help="Use synthetic bootstrap data")
    parser.add_argument(
        "--target",
        choices=("product_8", "dimensions_14", "raw_34", "va_2"),
        default=None,
        help="Override target type from config",
    )
    parser.add_argument(
        "--feature-spec",
        default=None,
        help="Override feature_spec from config",
    )
    parser.add_argument(
        "--zscore-y",
        action="store_true",
        help="Z-score target columns before writing the training NPZ",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    out_raw = cfg.get("data", {}).get("train_npz", "scout_data/horikawaCode/tribev2_fused/train.npz")
    out_path = PROJECT_ROOT / out_raw
    out_path.parent.mkdir(parents=True, exist_ok=True)

    target = args.target or cfg.get("target_type", "product_8")
    if target == "dimensions":
        target = "dimensions_14"
    feature_spec = args.feature_spec or cfg.get("feature_spec", "fused_schaefer400_subcortical_v1")

    if args.demo:
        data = synthetic_horikawa_dataset(n_samples=args.n_samples)
        np.savez_compressed(out_path, **data)
        print(f"Wrote demo training NPZ -> {out_path}  X={data['X'].shape}  y={data['y'].shape}")
        return

    manifest_path = args.manifest
    if manifest_path is None:
        default_manifest = PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "pilot_150.json"
        if default_manifest.is_file():
            manifest_path = default_manifest
        else:
            raise SystemExit(
                "No --manifest and no intermediates path. Pass --demo or provide --manifest with TRIBE NPZs."
            )

    clips = _load_manifest_clips(manifest_path)
    corpus = json.loads(manifest_path.read_text(encoding="utf-8")).get("corpus", "pilot_150")
    data = build_train_npz_from_manifest(
        clips,
        intermediates_dir=args.intermediates_dir,
        labels_cache=args.labels_cache,
        corpus=corpus,
        target=target,
        feature_spec=feature_spec,
        zscore_y=args.zscore_y,
        fit_pca_on_corpus=feature_spec == "subcortical_pca32_v1",
    )
    np.savez_compressed(out_path, **data)
    print(f"Wrote training NPZ -> {out_path}  X={data['X'].shape}  y={data['y'].shape}  corpus={corpus}")


if __name__ == "__main__":
    main()
