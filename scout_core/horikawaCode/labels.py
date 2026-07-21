"""Horikawa label loading and mapping."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from scout_core.horikawaCode.constants import (
    CATEGORY_34_TO_PRODUCT_8,
    DEFAULT_PRODUCT_CATEGORIES,
    DEMO_CATEGORY_RATINGS,
    DIMENSION_LIKERT_MAX,
    DIMENSION_LIKERT_MIN,
    HORIKAWA_14_DIMENSIONS,
    HORIKAWA_34_CATEGORIES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_CACHE = PROJECT_ROOT / "scout_data" / "horikawaCode" / "labels"
RATINGS_CACHE_NPZ = "ratings_cache.npz"
LABEL_MANIFEST_JSON = "label_manifest.json"


def build_label_map(categories: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    cats = list(categories or DEFAULT_PRODUCT_CATEGORIES)
    return [
        {
            "class_id": i,
            "horikawa_name": name,
            "product_label": name,
            "bundle_column": name,
        }
        for i, name in enumerate(cats)
    ]


def label_map_to_names(label_map: list[dict[str, Any]]) -> list[str]:
    return [str(entry["product_label"]) for entry in label_map]


def load_label_map(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "classes" in data:
        return list(data["classes"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Invalid label_map.json at {path}")


def save_label_map(path: Path, label_map: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"classes": label_map}, indent=2),
        encoding="utf-8",
    )


def normalize_dimension_scores(dimensions_14: np.ndarray) -> np.ndarray:
    """Map 1–9 Likert dimension scores to [0, 1]."""
    arr = np.asarray(dimensions_14, dtype=np.float32)
    span = DIMENSION_LIKERT_MAX - DIMENSION_LIKERT_MIN
    if span <= 0:
        return arr
    return np.clip((arr - DIMENSION_LIKERT_MIN) / span, 0.0, 1.0)


def build_y_product_8(categories_34: np.ndarray) -> np.ndarray:
    """Aggregate 34-category scores into 8 product targets (max over mapped sources)."""
    src = np.asarray(categories_34, dtype=np.float32).reshape(-1)
    if src.shape[0] != len(HORIKAWA_34_CATEGORIES):
        raise ValueError(
            f"expected {len(HORIKAWA_34_CATEGORIES)} category scores, got {src.shape[0]}"
        )
    out = np.zeros(len(DEFAULT_PRODUCT_CATEGORIES), dtype=np.float32)
    for i, cat_name in enumerate(HORIKAWA_34_CATEGORIES):
        product = CATEGORY_34_TO_PRODUCT_8.get(cat_name)
        if product is None:
            continue
        j = DEFAULT_PRODUCT_CATEGORIES.index(product)
        out[j] = max(out[j], float(src[i]))
    return np.clip(out, 0.0, 1.0)


def build_y_dimensions_14(dimensions_14: np.ndarray) -> np.ndarray:
    return normalize_dimension_scores(dimensions_14)


def build_y_raw_34(categories_34: np.ndarray) -> np.ndarray:
    """All 34 category scores, clipped to [0, 1]."""
    arr = np.asarray(categories_34, dtype=np.float32).reshape(-1)
    if arr.shape[0] != len(HORIKAWA_34_CATEGORIES):
        raise ValueError(
            f"expected {len(HORIKAWA_34_CATEGORIES)} category scores, got {arr.shape[0]}"
        )
    return np.clip(arr, 0.0, 1.0)


def build_y_va_2(dimensions_14: np.ndarray) -> np.ndarray:
    """Valence + arousal only (normalized 0-1)."""
    dims = normalize_dimension_scores(dimensions_14)
    idx_v = HORIKAWA_14_DIMENSIONS.index("valence")
    idx_a = HORIKAWA_14_DIMENSIONS.index("arousal")
    return np.asarray([dims[idx_v], dims[idx_a]], dtype=np.float32)


def label_names_for_target(target: str) -> list[str]:
    if target in ("dimensions_14", "dims_14"):
        return list(HORIKAWA_14_DIMENSIONS)
    if target in ("raw_34",):
        return list(HORIKAWA_34_CATEGORIES)
    if target in ("va_2",):
        return ["valence", "arousal"]
    return list(DEFAULT_PRODUCT_CATEGORIES)


def build_y_from_label_dict(label: dict[str, np.ndarray], target: str) -> np.ndarray:
    if target in ("dimensions_14", "dims_14"):
        return build_y_dimensions_14(label["dimensions_14"])
    if target == "raw_34":
        return build_y_raw_34(label["categories_34"])
    if target == "va_2":
        return build_y_va_2(label["dimensions_14"])
    return build_y_product_8(label["categories_34"])


def zscore_targets(y: np.ndarray) -> np.ndarray:
    """Per-target z-score (for diagnostics; Pearson r is scale-invariant)."""
    arr = np.asarray(y, dtype=np.float32)
    out = np.zeros_like(arr)
    for c in range(arr.shape[1]):
        col = arr[:, c]
        std = float(col.std())
        if std < 1e-8:
            out[:, c] = col - float(col.mean())
        else:
            out[:, c] = (col - float(col.mean())) / std
    return out


def _stimulus_id_from_mat_path(path: str) -> str:
    stem = Path(path).stem
    if stem.isdigit():
        return str(int(stem))
    return stem


def _parse_mat_feat(raw: bytes) -> np.ndarray:
    from scipy.io import loadmat

    data = loadmat(io.BytesIO(raw))
    if "feat" not in data:
        raise ValueError("mat file missing 'feat' array")
    return np.asarray(data["feat"], dtype=np.float32).reshape(-1)


def parse_figshare_features_zip(zip_path: Path) -> dict[str, np.ndarray]:
    """Parse KamitaniLab features.zip into aligned stimulus ratings."""
    with zipfile.ZipFile(zip_path) as zf:
        cat_files = sorted(
            n for n in zf.namelist()
            if "/category/" in n and n.endswith(".mat")
        )
        dim_files = sorted(
            n for n in zf.namelist()
            if "/dimension/" in n and n.endswith(".mat")
        )
        cat_by_id = {_stimulus_id_from_mat_path(p): p for p in cat_files}
        dim_by_id = {_stimulus_id_from_mat_path(p): p for p in dim_files}
        stimulus_ids = sorted(set(cat_by_id) & set(dim_by_id), key=lambda s: int(s) if s.isdigit() else s)

        categories: list[np.ndarray] = []
        dimensions: list[np.ndarray] = []
        for sid in stimulus_ids:
            cat = _parse_mat_feat(zf.read(cat_by_id[sid]))
            dim = _parse_mat_feat(zf.read(dim_by_id[sid]))
            if cat.shape[0] != len(HORIKAWA_34_CATEGORIES):
                raise ValueError(f"{sid}: category length {cat.shape[0]} != 34")
            if dim.shape[0] != len(HORIKAWA_14_DIMENSIONS):
                raise ValueError(f"{sid}: dimension length {dim.shape[0]} != 14")
            categories.append(cat)
            dimensions.append(dim)

    return {
        "stimulus_id": np.asarray(stimulus_ids, dtype=object),
        "categories_34": np.stack(categories, axis=0).astype(np.float32),
        "dimensions_14": np.stack(dimensions, axis=0).astype(np.float32),
    }


def write_ratings_cache(cache_dir: Path, ratings: dict[str, np.ndarray]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / RATINGS_CACHE_NPZ
    np.savez_compressed(out_path, **ratings)
    return out_path


def load_ratings_cache(cache_dir: Path) -> dict[str, np.ndarray] | None:
    path = cache_dir / RATINGS_CACHE_NPZ
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def ratings_by_stimulus_from_cache(ratings: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Convert cache arrays to per-stimulus dict."""
    ids = np.asarray(ratings["stimulus_id"])
    cats = np.asarray(ratings["categories_34"], dtype=np.float32)
    dims = np.asarray(ratings["dimensions_14"], dtype=np.float32)
    out: dict[str, dict[str, np.ndarray]] = {}
    for i, sid in enumerate(ids):
        out[str(sid)] = {
            "categories_34": cats[i],
            "dimensions_14": dims[i],
        }
    return out


def load_horikawa_ratings(cache_dir: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load clip-level ratings keyed by stimulus_id."""
    cache = cache_dir or DEFAULT_LABEL_CACHE
    ratings = load_ratings_cache(cache)
    if ratings is None:
        raise FileNotFoundError(
            f"Missing {cache / RATINGS_CACHE_NPZ}. "
            "Run scripts/horikawaCode/download_horikawa_labels.py first."
        )
    return ratings_by_stimulus_from_cache(ratings)


def build_label_manifest(cache_dir: Path, ratings: dict[str, np.ndarray], *, source_sha256: str) -> dict[str, Any]:
    ids = np.asarray(ratings["stimulus_id"])
    return {
        "schema_version": 1,
        "n_stimuli": int(len(ids)),
        "category_names": list(HORIKAWA_34_CATEGORIES),
        "dimension_names": list(HORIKAWA_14_DIMENSIONS),
        "product_categories": list(DEFAULT_PRODUCT_CATEGORIES),
        "source_sha256": source_sha256,
        "cache_npz": RATINGS_CACHE_NPZ,
    }


def synthetic_horikawa_dataset(
    n_samples: int = 40,
    *,
    n_vertices: int = 20484,
    n_subcortical: int = 8802,
    categories: tuple[str, ...] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate reproducible demo training data for bootstrap decoder."""
    rng = np.random.default_rng(seed)
    cats = list(categories or DEFAULT_PRODUCT_CATEGORIES)
    base = np.array([DEMO_CATEGORY_RATINGS.get(c, 0.05) for c in cats], dtype=np.float32)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    groups: list[int] = []
    stimulus_ids: list[str] = []

    for idx in range(n_samples):
        cortical = rng.normal(size=(3, n_vertices)).astype(np.float32)
        subcortical = rng.normal(size=(3, n_subcortical)).astype(np.float32)
        from scout_core.affect_features import build_affect_features_batch

        feat = build_affect_features_batch(
            cortical[np.newaxis, ...],
            subcortical[np.newaxis, ...],
        )[0]
        noise = rng.normal(scale=0.02, size=len(cats)).astype(np.float32)
        y = np.clip(base + noise, 0.0, 1.0)
        X_list.append(feat)
        y_list.append(y)
        groups.append(idx)  # LOVO: one group per clip
        stimulus_ids.append(f"demo_clip_{idx}")

    return {
        "X": np.stack(X_list, axis=0).astype(np.float32),
        "y": np.stack(y_list, axis=0).astype(np.float32),
        "groups": np.asarray(groups, dtype=np.int32),
        "stimulus_id": np.asarray(stimulus_ids, dtype=object),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_video_path(videos_dir: Path, stimulus_id: str) -> Path | None:
    """Resolve stimulus_id to a local MP4 (supports zero-padded and plain names)."""
    candidates = [
        videos_dir / f"{stimulus_id}.mp4",
        videos_dir / f"{int(stimulus_id):04d}.mp4" if stimulus_id.isdigit() else None,
        videos_dir / f"video_{stimulus_id}.mp4",
        videos_dir / f"video_{int(stimulus_id):04d}.mp4" if stimulus_id.isdigit() else None,
    ]
    for cand in candidates:
        if cand is not None and cand.is_file():
            return cand
    return None
