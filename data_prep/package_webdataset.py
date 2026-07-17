from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import webdataset as wds
except ImportError:  # optional until training phase
    wds = None  # type: ignore[assignment]


def package_shard(
    samples: Iterable[dict],
    output_tar: Path,
) -> None:
    if wds is None:
        raise ImportError("webdataset is required for packaging; pip install webdataset")

    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with wds.TarWriter(str(output_tar)) as sink:
        for sample in samples:
            meta = sample.get("json", {})
            sink.write(
                {
                    "__key__": sample["key"],
                    "mp4": sample["mp4"],
                    "y_target.npy": np.asarray(sample["y_target"], dtype=np.float32),
                    "cluster_ids.npy": np.asarray(sample["cluster_ids"], dtype=np.int64),
                    "json": json.dumps(meta).encode("utf-8"),
                }
            )


def centroid_to_sample(
    *,
    key: str,
    video_bytes: bytes,
    centroid_npz: Path,
    video_id: str,
) -> dict:
    data = np.load(centroid_npz)
    return {
        "key": key,
        "mp4": video_bytes,
        "y_target": data["y_target"],
        "cluster_ids": data["cluster_ids"],
        "json": {
            "video_id": video_id,
            "fps": 1.0,
            "mesh": "fsaverage5",
            "n_parcels": int(data["y_target"].shape[-1]) if data["y_target"].size else 400,
            "provenance": data.get("provenance_json", "{}"),
        },
    }
