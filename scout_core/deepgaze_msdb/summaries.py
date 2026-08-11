"""Label-free summary statistics for a saliency probability density."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def summarize_density(density: np.ndarray) -> dict[str, Any]:
    """Compute summaries that do not require ground-truth fixations.

    Intentionally omits AUC/sAUC/NSS/IG — those need fixation samples.
    """
    density = np.asarray(density, dtype=np.float64)
    if density.ndim != 2:
        raise ValueError(f"Expected 2D density, got {density.shape}")
    if not np.all(np.isfinite(density)):
        raise ValueError("Density contains non-finite values")
    if np.any(density < -1e-8):
        raise ValueError("Density contains negative values")

    flat = density.ravel()
    total = float(flat.sum())
    if total <= 0:
        raise ValueError("Density has non-positive total mass")
    # Tolerate tiny float drift from resize / IO.
    probs = flat / total

    # Shannon entropy in bits; normalized by log2(HW).
    positive = probs[probs > 0]
    entropy = float(-np.sum(positive * np.log2(positive)))
    max_entropy = math.log2(probs.size) if probs.size > 1 else 1.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    peak_idx = int(np.argmax(probs))
    peak_y, peak_x = np.unravel_index(peak_idx, density.shape)
    peak_probability = float(probs[peak_idx])

    order = np.argsort(probs)[::-1]
    n_pixels = probs.size
    top_1 = max(1, int(math.ceil(0.01 * n_pixels)))
    top_5 = max(1, int(math.ceil(0.05 * n_pixels)))
    top_1pct_mass = float(probs[order[:top_1]].sum())
    top_5pct_mass = float(probs[order[:top_5]].sum())

    h, w = density.shape
    cy0, cy1 = int(0.25 * h), int(0.75 * h)
    cx0, cx1 = int(0.25 * w), int(0.75 * w)
    center_mass = float(density[cy0:cy1, cx0:cx1].sum() / total)
    periphery_mass = float(1.0 - center_mass)

    return {
        "height": int(h),
        "width": int(w),
        "sum": total,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "peak_probability": peak_probability,
        "peak_y": int(peak_y),
        "peak_x": int(peak_x),
        "top_1pct_mass": top_1pct_mass,
        "top_5pct_mass": top_5pct_mass,
        "center_mass": center_mass,
        "periphery_mass": periphery_mass,
        "metrics_note": (
            "Label-free density summaries only. AUC/sAUC/NSS require "
            "ground-truth fixation samples and are not computed here."
        ),
    }
