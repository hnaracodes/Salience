"""Heatmap / overlay rendering for visual sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Compact turbo-like LUT (sampled) so we do not require matplotlib in CI/venv311.
_TURBO_STOPS = np.array(
    [
        [0.00, 48, 18, 59],
        [0.10, 70, 90, 167],
        [0.20, 40, 151, 191],
        [0.30, 27, 192, 140],
        [0.40, 87, 210, 67],
        [0.50, 190, 208, 41],
        [0.60, 246, 170, 39],
        [0.70, 250, 112, 40],
        [0.80, 228, 53, 59],
        [0.90, 174, 15, 52],
        [1.00, 122, 4, 3],
    ],
    dtype=np.float64,
)


def _turbo_lut(n: int = 256) -> np.ndarray:
    xs = _TURBO_STOPS[:, 0]
    channels = []
    sample_x = np.linspace(0.0, 1.0, n)
    for c in range(1, 4):
        channels.append(np.interp(sample_x, xs, _TURBO_STOPS[:, c]))
    return np.stack(channels, axis=1).astype(np.uint8)


_TURBO_LUT = _turbo_lut(256)


def density_to_heatmap_rgb(density: np.ndarray) -> np.ndarray:
    """Map a density to an RGB heatmap using a turbo-like colormap."""
    density = np.asarray(density, dtype=np.float64)
    if density.ndim != 2:
        raise ValueError(f"Expected 2D density, got {density.shape}")
    vmax = float(density.max()) if density.size else 1.0
    if vmax <= 0:
        normed = np.zeros_like(density, dtype=np.float64)
    else:
        # Percentile clip avoids a single spike washing out the map.
        clip = float(np.percentile(density, 99.5))
        clip = max(clip, vmax * 1e-6)
        normed = np.clip(density / clip, 0.0, 1.0)

    idx = np.clip((normed * 255.0).astype(np.int32), 0, 255)
    return _TURBO_LUT[idx]


def overlay_heatmap(
    image_rgb: np.ndarray,
    density: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend heatmap over the source RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    heat = density_to_heatmap_rgb(density)
    if heat.shape[:2] != image_rgb.shape[:2]:
        heat = np.asarray(
            Image.fromarray(heat).resize(
                (image_rgb.shape[1], image_rgb.shape[0]),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    blended = (
        (1.0 - alpha) * image_rgb.astype(np.float32)
        + alpha * heat.astype(np.float32)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_rgb_png(path: Path | str, rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
    return path
