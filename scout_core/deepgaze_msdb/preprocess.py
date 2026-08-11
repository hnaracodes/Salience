"""Image + centerbias preparation for DeepGaze MSDB."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, Image.Image]


def load_rgb_image(image: ImageInput) -> np.ndarray:
    """Load an RGB uint8 array shaped (H, W, 3)."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 array, got shape {arr.shape}")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

    path = Path(image)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def maybe_resize_long_side(
    image: np.ndarray,
    max_long_side: int | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Optionally clamp the long side; return (resized, original_hw)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}")
    orig_hw = (int(image.shape[0]), int(image.shape[1]))
    if max_long_side is None:
        return image, orig_hw
    long_side = max(orig_hw)
    if long_side <= max_long_side:
        return image, orig_hw
    scale = max_long_side / float(long_side)
    new_h = max(1, int(round(orig_hw[0] * scale)))
    new_w = max(1, int(round(orig_hw[1] * scale)))
    resized = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    return resized, orig_hw


def prepare_centerbias(
    height: int,
    width: int,
    centerbias_template: np.ndarray | None = None,
    uniform: bool = False,
) -> np.ndarray:
    """Return log-density centerbias shaped (H, W), renormalized with logsumexp."""
    from scipy.ndimage import zoom
    from scipy.special import logsumexp

    if height < 1 or width < 1:
        raise ValueError(f"Invalid centerbias size: {(height, width)}")

    if uniform or centerbias_template is None:
        template = np.zeros((1024, 1024), dtype=np.float64)
    else:
        template = np.asarray(centerbias_template, dtype=np.float64)
        if template.ndim != 2:
            raise ValueError(f"centerbias_template must be 2D, got {template.shape}")

    zoom_factors = (
        height / float(template.shape[0]),
        width / float(template.shape[1]),
    )
    centerbias = zoom(template, zoom_factors, order=0, mode="nearest")
    if centerbias.shape != (height, width):
        # Guard against rounding drift from zoom.
        centerbias = np.asarray(
            Image.fromarray(centerbias.astype(np.float32)).resize(
                (width, height), Image.Resampling.NEAREST
            ),
            dtype=np.float64,
        )
    centerbias = centerbias.astype(np.float64, copy=False)
    centerbias -= logsumexp(centerbias)
    return centerbias


def density_from_log_density(log_density: np.ndarray) -> np.ndarray:
    """Convert log density to a finite non-negative probability density summing to 1."""
    log_density = np.asarray(log_density, dtype=np.float64)
    if log_density.ndim != 2:
        raise ValueError(f"Expected 2D log density, got {log_density.shape}")
    # Numerically stable softmax over pixels.
    shifted = log_density - np.max(log_density)
    density = np.exp(shifted)
    total = float(density.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Failed to normalize log density into a probability map")
    density = (density / total).astype(np.float32)
    return density


def resize_density(density: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a probability density and renormalize so it still sums to 1."""
    density = np.asarray(density, dtype=np.float32)
    if density.shape == (height, width):
        return density
    resized = np.asarray(
        Image.fromarray(density, mode="F").resize(
            (width, height), Image.Resampling.BILINEAR
        ),
        dtype=np.float64,
    )
    resized = np.clip(resized, 0.0, None)
    total = float(resized.sum())
    if total <= 0:
        raise ValueError("Resized density has non-positive mass")
    return (resized / total).astype(np.float32)
