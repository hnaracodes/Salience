"""Download Kragel (2015) and PINES (2015) emotion templates from NeuroVault,
resample each to fsaverage5 surface space, L2-normalise, and save as .npy files.

Usage:
    python scripts/download_emotion_templates.py

Output:
    configs/emotion_templates/template_{name}.npy  — shape (20484,) float32, unit-norm

Sources:
    Kragel et al. (2015) — NeuroVault collection #503
        6 emotion contrast maps: anger, disgust, fear, happy, neutral, sad
    Chang et al. / PINES (2015) — NeuroVault image #10704
        Negative affect template

Re-run this script after a fresh checkout; the .npy files are gitignored.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = PROJECT_ROOT / "configs" / "emotion_templates"
NEUROVAULT_BASE = "https://neurovault.org/api"

# ---------------------------------------------------------------------------
# NeuroVault collection #503 — Kragel (2015) 6 emotions
# These image IDs are stable NeuroVault identifiers for the six contrast maps.
# Keys are the canonical emotion names used throughout this project.
# ---------------------------------------------------------------------------
KRAGEL_IMAGE_NAMES: dict[str, str] = {
    "anger":   "anger",
    "disgust": "disgust",
    "fear":    "fear",
    "happy":   "happiness",   # NeuroVault uses "happiness" label
    "neutral": "neutral",
    "sad":     "sadness",     # NeuroVault uses "sadness" label
}
KRAGEL_COLLECTION_ID = "503"

# PINES negative affect template — single image
PINES_IMAGE_ID = "10704"


def _neurovault_get(url: str, timeout: int = 30) -> dict:
    r = requests.get(url, timeout=timeout)
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

    surface = np.concatenate([lh, rh])  # (20484,) — matches TRIBE vertex order

    # Replace NaN (vertices outside the volume field-of-view) with zero
    surface = np.nan_to_num(surface, nan=0.0)
    return surface


def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("Template is all-zero after projection — cannot L2-normalise.")
    return (arr / norm).astype(np.float32)


def _fetch_collection_images(collection_id: str) -> list[dict]:
    """Paginate through all images in a NeuroVault collection."""
    url = f"{NEUROVAULT_BASE}/collections/{collection_id}/images/?format=json&limit=100"
    images: list[dict] = []
    while url:
        data = _neurovault_get(url)
        images.extend(data.get("results", []))
        url = data.get("next")
    return images


def _find_kragel_image(images: list[dict], emotion_name: str) -> str | None:
    """Return the download URL for the given emotion from the Kragel collection.

    Matches against the 'name' field (case-insensitive partial match).
    """
    search = emotion_name.lower()
    for img in images:
        name_field = str(img.get("name", "")).lower()
        if search in name_field:
            return img.get("file")
    return None


def download_kragel_templates(out_dir: Path, force: bool = False) -> None:
    """Download and preprocess all 6 Kragel (2015) emotion templates."""
    print(f"Fetching Kragel collection #{KRAGEL_COLLECTION_ID} image list …")
    images = _fetch_collection_images(KRAGEL_COLLECTION_ID)
    print(f"  Found {len(images)} images in collection.")

    for local_name, nv_name in KRAGEL_IMAGE_NAMES.items():
        out_path = out_dir / f"template_{local_name}.npy"
        if out_path.is_file() and not force:
            print(f"  [{local_name}] already exists — skipping. (use --force to re-download)")
            continue

        url = _find_kragel_image(images, nv_name)
        if url is None:
            print(
                f"  WARNING: Could not find '{nv_name}' in collection {KRAGEL_COLLECTION_ID}. "
                "Skipping. Check the NeuroVault collection manually.",
                file=sys.stderr,
            )
            continue

        print(f"  [{local_name}] downloading {url} …", end=" ", flush=True)
        nifti_bytes = _download_bytes(url)
        print("done. Projecting to fsaverage5 …", end=" ", flush=True)
        surface = _resample_nifti_to_fsaverage5(nifti_bytes)
        template = _l2_normalise(surface)
        np.save(out_path, template)
        print(f"saved {out_path.name}  shape={template.shape}  norm={np.linalg.norm(template):.6f}")


def download_pines_template(out_dir: Path, force: bool = False) -> None:
    """Download and preprocess the PINES (2015) negative affect template."""
    out_path = out_dir / "template_negative_affect.npy"
    if out_path.is_file() and not force:
        print("  [negative_affect] already exists — skipping. (use --force to re-download)")
        return

    url_meta = f"{NEUROVAULT_BASE}/images/{PINES_IMAGE_ID}/?format=json"
    print(f"Fetching PINES image #{PINES_IMAGE_ID} metadata …", end=" ", flush=True)
    meta = _neurovault_get(url_meta)
    file_url = meta.get("file")
    if not file_url:
        print(
            f"\n  WARNING: No file URL found for PINES image {PINES_IMAGE_ID}.",
            file=sys.stderr,
        )
        return

    print(f"downloading …", end=" ", flush=True)
    nifti_bytes = _download_bytes(file_url)
    print("done. Projecting to fsaverage5 …", end=" ", flush=True)
    surface = _resample_nifti_to_fsaverage5(nifti_bytes)
    template = _l2_normalise(surface)
    np.save(out_path, template)
    print(f"saved {out_path.name}  shape={template.shape}  norm={np.linalg.norm(template):.6f}")


def verify_templates(out_dir: Path, template_names: list[str]) -> bool:
    """Check that all templates exist and are unit-norm. Returns True if all pass."""
    all_ok = True
    for name in template_names:
        p = out_dir / f"template_{name}.npy"
        if not p.is_file():
            print(f"  MISSING: {p.name}", file=sys.stderr)
            all_ok = False
            continue
        arr = np.load(p).astype(np.float32)
        norm = float(np.linalg.norm(arr))
        ok = abs(norm - 1.0) < 1e-5
        status = "OK" if ok else f"FAIL (norm={norm:.6f})"
        print(f"  {p.name}: shape={arr.shape}  norm={norm:.6f}  [{status}]")
        if not ok:
            all_ok = False
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=TEMPLATE_DIR, help="Output directory for .npy files")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing templates, don't download")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_names = list(KRAGEL_IMAGE_NAMES.keys()) + ["negative_affect"]

    if args.verify_only:
        print(f"Verifying templates in {args.out_dir} …")
        ok = verify_templates(args.out_dir, all_names)
        sys.exit(0 if ok else 1)

    print(f"Downloading emotion templates to {args.out_dir}")
    print("=" * 60)
    download_kragel_templates(args.out_dir, force=args.force)
    print()
    download_pines_template(args.out_dir, force=args.force)
    print()
    print("Verification:")
    ok = verify_templates(args.out_dir, all_names)
    if ok:
        print("\nAll templates ready.")
    else:
        print("\nSome templates failed verification — check errors above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
