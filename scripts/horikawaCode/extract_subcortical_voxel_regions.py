"""Extract tribev2 Harvard-Oxford subcortical voxel->region map via Modal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_JSON = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run" / "subcortical_voxel_regions.json"
OUT_CSV = PROJECT_ROOT / "configs" / "subcortical_voxel_regions.csv"


def _save_from_payload(payload: dict) -> None:
    from scout_core.subcortical.atlas import save_voxel_regions_csv

    region_index = np.asarray(payload["region_index"], dtype=np.int32)
    save_voxel_regions_csv(region_index, OUT_CSV)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        k: v for k, v in payload.items() if k != "region_index"
    }
    summary["csv_path"] = str(OUT_CSV)
    summary["n_voxels"] = int(region_index.shape[0])
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_CSV} and {OUT_JSON}")


def main() -> None:
    try:
        import modal
        from tribe import TribeInference, app

        with modal.enable_output(), app.run():
            inference = TribeInference()
            payload = inference.extract_subcortical_voxel_regions.remote()
        _save_from_payload(payload)
        return
    except Exception as modal_exc:
        print(f"Modal extraction failed ({modal_exc}); trying local tribev2...", flush=True)

    try:
        from scout_core.subcortical.atlas import (
            SUBCORTICAL_REGION_LABELS,
            build_region_index_from_nilearn_ho,
            build_region_index_from_tribev2_ho,
            region_voxel_counts,
        )

        try:
            region_index = build_region_index_from_tribev2_ho()
            source = "local tribev2.plotting.subcortical"
        except ImportError:
            region_index = build_region_index_from_nilearn_ho()
            source = "local nilearn HO fallback"
        payload = {
            "n_voxels": int(region_index.shape[0]),
            "region_labels": list(SUBCORTICAL_REGION_LABELS),
            "region_voxel_counts": region_voxel_counts(region_index),
            "region_index": region_index.tolist(),
            "source": source,
        }
        _save_from_payload(payload)
    except Exception as local_exc:
        raise SystemExit(
            "Could not extract subcortical voxel regions. "
            "Deploy Modal tribe-v2-brain-sim or install tribev2 locally."
        ) from local_exc


if __name__ == "__main__":
    main()
