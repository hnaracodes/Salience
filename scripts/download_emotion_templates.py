"""Download emotion templates for the Dual-Track engine, resample to fsaverage5, save as .npy.

Usage:
    python scripts/download_emotion_templates.py

Output:
    configs/emotion_templates/template_{name}.npy  — shape (20484,) float32, unit-norm

Sources (verified against live APIs, 2026):
    Kragel & LaBar (2015 SCAN) — NeuroVault collection #12383
        contentment, amusement, surprise, fear, anger, sadness, neutral

Re-run after a fresh checkout; .npy files are gitignored.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "configs" / "emotion_templates"
NEUROVAULT_BASE = "https://neurovault.org/api"
DEBUG_LOG_PATH = PROJECT_ROOT / "debug-4a9fad.log"

# Kragel & LaBar (2015) — NeuroVault collection 12383
# https://neurovault.org/collections/12383/
KRAGEL_COLLECTION_ID = "12383"
KRAGEL_NV_IMAGES: dict[str, tuple[str, str]] = {
    "contentment": ("779132", "Contentment"),
    "amusement": ("779131", "Amusement"),
    "surprise": ("779133", "Surprise"),
    "fear": ("779134", "Fear"),
    "anger": ("779136", "Anger"),
    "sadness": ("779135", "Sadness"),
    "neutral": ("779137", "Neutral"),
}

ALL_TEMPLATE_NAMES = list(KRAGEL_NV_IMAGES.keys())


# region agent log
def _dbg_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "4a9fad",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


# endregion


def _neurovault_get(url: str, timeout: int = 30) -> dict:
    r = requests.get(url, timeout=timeout)
    # region agent log
    _dbg_log("H3", "download_emotion_templates.py:_neurovault_get", "http_response", {
        "url": url, "status": r.status_code,
    })
    # endregion
    r.raise_for_status()
    return r.json()


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=65536):
        buf.write(chunk)
    return buf.getvalue()


def _resample_nifti_to_fsaverage5(nifti_bytes: bytes) -> np.ndarray:
    """Project volumetric NIfTI (MNI152) to fsaverage5 surface → shape (20484,) float32."""
    import nibabel as nib
    from nilearn import datasets
    from nilearn.surface import vol_to_surf

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(nifti_bytes)
        tmp_path = f.name

    try:
        img = nib.load(tmp_path)
        fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

        lh = vol_to_surf(
            img, fs.pial_left,
            interpolation="linear", radius=3.0,
        ).astype(np.float32)
        rh = vol_to_surf(
            img, fs.pial_right,
            interpolation="linear", radius=3.0,
        ).astype(np.float32)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    surface = np.concatenate([lh, rh])
    return np.nan_to_num(surface, nan=0.0)


def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("Template is all-zero after projection — cannot L2-normalise.")
    return (arr / norm).astype(np.float32)


def _neurovault_image_file_url(
    image_id: str,
    *,
    expected_name: str,
    expected_collection_id: str = KRAGEL_COLLECTION_ID,
) -> str:
    meta = _neurovault_get(f"{NEUROVAULT_BASE}/images/{image_id}/?format=json")
    actual_name = str(meta.get("name") or "")
    actual_collection_id = str(meta.get("collection_id") or "")
    if actual_collection_id != expected_collection_id:
        raise ValueError(
            f"NeuroVault image {image_id} belongs to collection {actual_collection_id}, "
            f"expected {expected_collection_id}"
        )
    if actual_name.lower() != expected_name.lower():
        raise ValueError(
            f"NeuroVault image {image_id} is named {actual_name!r}, expected {expected_name!r}"
        )

    file_url = meta.get("file")
    if not file_url:
        raise ValueError(f"NeuroVault image {image_id} has no downloadable file URL")
    return file_url


def _save_template_from_nifti_url(
    local_name: str,
    url: str,
    out_dir: Path,
    *,
    source: str,
) -> None:
    out_path = out_dir / f"template_{local_name}.npy"
    print(f"  [{local_name}] ({source}) downloading …", end=" ", flush=True)
    nifti_bytes = _download_bytes(url)
    print("projecting …", end=" ", flush=True)
    surface = _resample_nifti_to_fsaverage5(nifti_bytes)
    template = _l2_normalise(surface)
    np.save(out_path, template)
    # region agent log
    _dbg_log("H1", "download_emotion_templates.py:_save_template", "saved", {
        "name": local_name, "source": source, "shape": list(template.shape),
        "norm": float(np.linalg.norm(template)),
    })
    # endregion
    print(f"saved {out_path.name}  shape={template.shape}  norm={np.linalg.norm(template):.6f}")


def download_kragel_neurovault_templates(out_dir: Path, force: bool = False) -> None:
    """Download the seven native Kragel & LaBar labels from NeuroVault collection 12383."""
    print(f"Kragel & LaBar (2015) — NeuroVault collection #{KRAGEL_COLLECTION_ID}")
    # region agent log
    _dbg_log("H1", "download_emotion_templates.py:download_kragel", "start", {
        "collection_id": KRAGEL_COLLECTION_ID,
        "images": KRAGEL_NV_IMAGES,
    })
    # endregion

    for local_name, (image_id, expected_name) in KRAGEL_NV_IMAGES.items():
        out_path = out_dir / f"template_{local_name}.npy"
        if out_path.is_file() and not force:
            print(f"  [{local_name}] already exists — skipping.")
            continue
        try:
            file_url = _neurovault_image_file_url(image_id, expected_name=expected_name)
            _save_template_from_nifti_url(
                local_name,
                file_url,
                out_dir,
                source=f"neurovault:{KRAGEL_COLLECTION_ID}:{image_id}",
            )
        except Exception as exc:
            print(f"  ERROR [{local_name}]: {exc}", file=sys.stderr)
            # region agent log
            _dbg_log("H1", "download_emotion_templates.py:download_kragel", "error", {
                "name": local_name, "image_id": image_id, "error": str(exc),
            })
            # endregion


def verify_templates(out_dir: Path, template_names: list[str]) -> bool:
    all_ok = True
    for name in template_names:
        p = out_dir / f"template_{name}.npy"
        if not p.is_file():
            print(f"  MISSING: {p.name}", file=sys.stderr)
            all_ok = False
            continue
        arr = np.load(p).astype(np.float32)
        norm = float(np.linalg.norm(arr))
        ok = abs(norm - 1.0) < 1e-5 and arr.shape[0] == 20484
        status = "OK" if ok else f"FAIL (shape={arr.shape}, norm={norm:.6f})"
        print(f"  {p.name}: shape={arr.shape}  norm={norm:.6f}  [{status}]")
        if not ok:
            all_ok = False
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=TEMPLATE_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        print(f"Verifying templates in {args.out_dir} …")
        sys.exit(0 if verify_templates(args.out_dir, ALL_TEMPLATE_NAMES) else 1)

    print(f"Downloading emotion templates to {args.out_dir}")
    print("=" * 60)
    download_kragel_neurovault_templates(args.out_dir, force=args.force)
    print()
    print("Verification:")
    ok = verify_templates(args.out_dir, ALL_TEMPLATE_NAMES)
    # region agent log
    _dbg_log("H1", "download_emotion_templates.py:main", "complete", {"all_ok": ok})
    # endregion
    if ok:
        print("\nAll templates ready.")
    else:
        print("\nSome templates failed verification — check errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
