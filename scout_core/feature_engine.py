"""ViT attention extraction helpers for the Feature Isolation pipeline.

Converts raw HuggingFace ``output_attentions`` tensors into a normalized
spatial heatmap at Playwright capture resolution.

Modal GPU path (inside tribe.py):
    frame_bytes → PIL → processor → DINOv2(output_attentions=True)
    → attentions[-1] → cls_attention_to_patch_grid()
    → upscale_patch_grid() → (capture_h, capture_w) float32 heatmap

Local CPU path (tests, offline analysis):
    All public functions accept numpy arrays — torch is NOT required locally.
    upscale_patch_grid() uses scipy.ndimage.zoom (available transitively via
    nilearn) with a numpy nearest-neighbour fallback if scipy is missing.

DINOv2-large patch geometry (224 × 224 input, 14 px patches):
    patch_grid_h = patch_grid_w = 224 // 14 = 16  →  256 patch tokens.
    seq_len passed to the attention matrices = 1 (CLS) + 256 = 257.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core attention helpers
# ---------------------------------------------------------------------------

def cls_attention_to_patch_grid(
    attention_weights: np.ndarray,
    patch_grid_h: int,
    patch_grid_w: int,
) -> np.ndarray:
    """Convert [CLS] → patch attention from a ViT block to a 2-D spatial grid.

    Args:
        attention_weights: Shape ``(num_heads, seq_len, seq_len)`` — a single
            Transformer block's full attention matrix. ``seq_len`` must equal
            ``1 + patch_grid_h * patch_grid_w`` (CLS token + patch tokens).
        patch_grid_h: Patch-grid height (e.g. 16 for 224 px ÷ 14 px patches).
        patch_grid_w: Patch-grid width  (e.g. 16 for 224 px ÷ 14 px patches).

    Returns:
        float32 array of shape ``(patch_grid_h, patch_grid_w)`` — mean-across-
        heads CLS token attention over spatial patch positions.

    Raises:
        ValueError: If the derived ``n_patches`` does not match the tensor.
    """
    attn = np.asarray(attention_weights, dtype=np.float32)  # (H, S, S)
    # CLS is token 0; its attention over patch tokens (1 onward)
    cls_attn = attn[:, 0, 1:]        # (num_heads, n_patches)
    mean_attn = cls_attn.mean(axis=0)   # (n_patches,)

    n_patches = patch_grid_h * patch_grid_w
    if mean_attn.shape[0] != n_patches:
        raise ValueError(
            f"Expected {n_patches} patch tokens "
            f"(grid {patch_grid_h}\u00d7{patch_grid_w}) "
            f"but attention tensor has {mean_attn.shape[0]} patch tokens. "
            "Verify ViT patch size matches patch_grid parameters."
        )
    return mean_attn.reshape(patch_grid_h, patch_grid_w)


def upscale_patch_grid(
    patch_grid: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Bilinear upscale a 2-D patch attention grid to capture resolution.

    Uses ``scipy.ndimage.zoom`` (order=1, bilinear) when scipy is available;
    falls back to a pure-numpy nearest-neighbour if scipy is missing.

    Args:
        patch_grid: Shape ``(grid_h, grid_w)`` — raw attention values ≥ 0.
        target_h:   Output height in pixels (e.g. 1080 for 1920×1080 capture).
        target_w:   Output width  in pixels (e.g. 1920 for 1920×1080 capture).

    Returns:
        float32 array of shape ``(target_h, target_w)``, values clipped ≥ 0.
    """
    patch_grid = np.asarray(patch_grid, dtype=np.float32)
    grid_h, grid_w = patch_grid.shape

    try:
        from scipy.ndimage import zoom as _zoom
        zoom_h = target_h / grid_h
        zoom_w = target_w / grid_w
        # mode='nearest' extrapolates boundary values rather than decaying to 0
        # (scipy's default 'constant' cval=0 produces artefacts at patch edges).
        upscaled = _zoom(patch_grid, (zoom_h, zoom_w), order=1, mode="nearest")
    except ImportError:
        # Numpy nearest-neighbour fallback (no scipy required)
        row_idx = np.round(np.linspace(0, grid_h - 1, target_h)).astype(np.intp)
        col_idx = np.round(np.linspace(0, grid_w - 1, target_w)).astype(np.intp)
        upscaled = patch_grid[np.ix_(row_idx, col_idx)]

    return np.clip(upscaled, 0.0, None).astype(np.float32)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Min-max normalize a heatmap to [0, 1]. Returns zeros if map is uniform.

    Args:
        heatmap: Any shape float array.

    Returns:
        Same shape float32 array, values in [0, 1].
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    vmin, vmax = float(heatmap.min()), float(heatmap.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(heatmap)
    return ((heatmap - vmin) / (vmax - vmin)).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: derive patch grid geometry from seq_len
# ---------------------------------------------------------------------------

def patch_grid_size_from_seq_len(seq_len: int) -> tuple[int, int]:
    """Return (grid_h, grid_w) assuming square patch grid.

    seq_len = 1 (CLS) + grid_h * grid_w.

    Args:
        seq_len: Full sequence length from the ViT attention tensor.

    Returns:
        (grid_h, grid_w) — both equal (square grid assumed).

    Raises:
        ValueError: If seq_len - 1 is not a perfect square.
    """
    n_patches = seq_len - 1
    grid_size = int(n_patches ** 0.5)
    if grid_size * grid_size != n_patches:
        raise ValueError(
            f"seq_len={seq_len} → n_patches={n_patches} is not a perfect square. "
            "Only square patch grids are supported."
        )
    return grid_size, grid_size
