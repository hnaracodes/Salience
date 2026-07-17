"""Download and cache Horikawa figshare emotion labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.constants import FIGSHARE_FEATURES_MD5, FIGSHARE_FEATURES_URL
from scout_core.horikawaCode.labels import (
    DEFAULT_LABEL_CACHE,
    LABEL_MANIFEST_JSON,
    build_label_manifest,
    parse_figshare_features_zip,
    sha256_file,
    write_ratings_cache,
)

# figshare ndownloader size for features.zip (bytes); used to detect truncated downloads.
EXPECTED_ZIP_BYTES = 11_642_218


def _download(url: str, dest: Path, *, retries: int = 5) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.part")
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{retries})", flush=True)
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length") or EXPECTED_ZIP_BYTES)
                written = 0
                with tmp.open("wb") as out:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        out.write(chunk)
                        written += len(chunk)
                if written < total * 0.99:
                    raise OSError(f"incomplete download: {written}/{total} bytes")
            tmp.replace(dest)
            if dest.stat().st_size < EXPECTED_ZIP_BYTES * 0.99:
                raise OSError(f"zip too small: {dest.stat().st_size} bytes")
            # Validate zip integrity before returning.
            with zipfile.ZipFile(dest) as zf:
                zf.testzip()
            print(f"Download OK ({dest.stat().st_size} bytes)", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"Download failed: {exc}", flush=True)
            tmp.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(30, 5 * attempt))
    raise SystemExit(f"Download failed after {retries} attempts: {last_err}") from last_err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument("--force", action="store_true", help="Re-download and rebuild cache")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "features.zip"
    manifest_path = cache_dir / LABEL_MANIFEST_JSON
    ratings_path = cache_dir / "ratings_cache.npz"

    if ratings_path.is_file() and manifest_path.is_file() and not args.force:
        print(f"Labels cache already exists: {ratings_path}")
        return

    need_download = args.force or not zip_path.is_file()
    if not need_download:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.testzip()
            if zip_path.stat().st_size < EXPECTED_ZIP_BYTES * 0.99:
                need_download = True
        except (zipfile.BadZipFile, OSError):
            need_download = True
            zip_path.unlink(missing_ok=True)

    if need_download:
        _download(FIGSHARE_FEATURES_URL, zip_path)

    digest = sha256_file(zip_path)
    if digest != FIGSHARE_FEATURES_MD5:
        print(f"WARN: features.zip SHA256 {digest} != expected {FIGSHARE_FEATURES_MD5}")

    ratings = parse_figshare_features_zip(zip_path)
    out_npz = write_ratings_cache(cache_dir, ratings)
    manifest = build_label_manifest(cache_dir, ratings, source_sha256=digest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_npz}  n={manifest['n_stimuli']}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
