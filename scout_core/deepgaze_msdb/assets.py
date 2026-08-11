"""Download / cache DeepGaze MSDB centerbias and related local assets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import numpy as np

from scout_core.deepgaze_msdb.constants import (
    CENTERBIAS_URL,
    DEFAULT_CENTERBIAS_NAME,
    DEFAULT_CHECKPOINT_DIR,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_sha256: Optional[str] = None) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        if expected_sha256 is None or sha256_file(dest) == expected_sha256:
            return dest
        dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".partial")
    with urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    if expected_sha256 is not None:
        digest = sha256_file(tmp)
        if digest != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {url}: got {digest}, expected {expected_sha256}"
            )
    tmp.replace(dest)
    return dest


def ensure_centerbias(
    checkpoint_dir: Path | str | None = None,
    url: str = CENTERBIAS_URL,
    filename: str = DEFAULT_CENTERBIAS_NAME,
) -> tuple[np.ndarray, dict]:
    """Ensure MIT1003 centerbias exists locally and return (array, provenance)."""
    checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR)
    path = checkpoint_dir / filename
    download_file(url, path)
    arr = np.load(path)
    provenance = {
        "path": str(path.resolve()),
        "url": url,
        "sha256": sha256_file(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    return np.asarray(arr), provenance
