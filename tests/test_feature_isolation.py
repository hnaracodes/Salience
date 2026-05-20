"""Unit tests for the Feature Isolation pipeline.

Covers:
    - scout_core/feature_engine.py  — CLS attention → patch grid → upscale
    - scout_core/dom_intersect.py   — attention density, scroll, occlusion, winner

No Modal GPU, no TRIBE model, no NeuroVault downloads, no file I/O required.
Run with: pytest tests/test_feature_isolation.py
"""

from __future__ import annotations

import numpy as np
import pytest

from scout_core.dom_intersect import (
    _clip_bbox,
    compute_attention_density,
    find_nearest_snapshot,
    ground_snapshot,
    select_winner,
)
from scout_core.feature_engine import (
    cls_attention_to_patch_grid,
    normalize_heatmap,
    patch_grid_size_from_seq_len,
    upscale_patch_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRID_H, GRID_W = 16, 16      # DINOv2-large: 224 / 14 = 16
N_PATCHES = GRID_H * GRID_W  # 256
N_HEADS = 12
SEQ_LEN = 1 + N_PATCHES      # 257 (CLS + patches)

CAP_H, CAP_W = 540, 960      # smaller than 1080p for faster tests


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture()
def uniform_heatmap() -> np.ndarray:
    return np.ones((CAP_H, CAP_W), dtype=np.float32)


@pytest.fixture()
def zero_heatmap() -> np.ndarray:
    return np.zeros((CAP_H, CAP_W), dtype=np.float32)


@pytest.fixture()
def attention_block(rng: np.random.Generator) -> np.ndarray:
    """Random attention block: (N_HEADS, SEQ_LEN, SEQ_LEN), softmax-like rows."""
    raw = rng.random((N_HEADS, SEQ_LEN, SEQ_LEN)).astype(np.float32)
    # Row-normalise so each row sums to 1 (mimics softmax output)
    raw /= raw.sum(axis=-1, keepdims=True)
    return raw


@pytest.fixture()
def patch_grid(rng: np.random.Generator) -> np.ndarray:
    return rng.random((GRID_H, GRID_W)).astype(np.float32)


# ---------------------------------------------------------------------------
# feature_engine — cls_attention_to_patch_grid
# ---------------------------------------------------------------------------

class TestClsAttentionToPatchGrid:
    def test_output_shape(self, attention_block):
        grid = cls_attention_to_patch_grid(attention_block, GRID_H, GRID_W)
        assert grid.shape == (GRID_H, GRID_W)

    def test_output_dtype(self, attention_block):
        grid = cls_attention_to_patch_grid(attention_block, GRID_H, GRID_W)
        assert grid.dtype == np.float32

    def test_values_are_non_negative(self, attention_block):
        # Softmax attention is non-negative; mean stays non-negative.
        grid = cls_attention_to_patch_grid(attention_block, GRID_H, GRID_W)
        assert grid.min() >= 0.0

    def test_wrong_n_patches_raises(self, attention_block):
        # Ask for a 15×15 grid (225 patches) but the block has 256 patch tokens.
        with pytest.raises(ValueError, match="patch tokens"):
            cls_attention_to_patch_grid(attention_block, 15, 15)

    def test_mean_across_heads(self):
        """Construct attention with known per-head CLS rows; verify mean is correct."""
        attn = np.zeros((N_HEADS, SEQ_LEN, SEQ_LEN), dtype=np.float32)
        # Set CLS (row 0) attention to patch 0 only, different value per head.
        for h in range(N_HEADS):
            attn[h, 0, 1] = float(h + 1)   # patch token index 1
        grid = cls_attention_to_patch_grid(attn, GRID_H, GRID_W)
        expected_mean_at_0 = float(np.mean([h + 1 for h in range(N_HEADS)]))
        # grid[0, 0] corresponds to patch index 0 (first patch token)
        assert abs(grid.flat[0] - expected_mean_at_0) < 1e-5, (
            f"Expected {expected_mean_at_0}, got {grid.flat[0]}"
        )

    def test_accepts_numpy_input(self, rng):
        attn_np = rng.random((N_HEADS, SEQ_LEN, SEQ_LEN)).astype(np.float32)
        grid = cls_attention_to_patch_grid(attn_np, GRID_H, GRID_W)
        assert grid.shape == (GRID_H, GRID_W)


# ---------------------------------------------------------------------------
# feature_engine — upscale_patch_grid
# ---------------------------------------------------------------------------

class TestUpscalePatchGrid:
    def test_output_shape(self, patch_grid):
        up = upscale_patch_grid(patch_grid, CAP_H, CAP_W)
        assert up.shape == (CAP_H, CAP_W)

    def test_output_dtype(self, patch_grid):
        up = upscale_patch_grid(patch_grid, CAP_H, CAP_W)
        assert up.dtype == np.float32

    def test_no_negative_values(self, rng):
        # Even with small negatives in input, output is clipped to ≥ 0.
        grid = rng.standard_normal((GRID_H, GRID_W)).astype(np.float32)
        up = upscale_patch_grid(grid, CAP_H, CAP_W)
        assert up.min() >= 0.0

    def test_uniform_input_stays_uniform(self):
        grid = np.ones((GRID_H, GRID_W), dtype=np.float32) * 0.5
        up = upscale_patch_grid(grid, CAP_H, CAP_W)
        assert np.allclose(up, 0.5, atol=1e-5), f"Expected 0.5 everywhere, got range [{up.min()}, {up.max()}]"

    def test_upscale_1x1_grid(self):
        grid = np.array([[3.0]], dtype=np.float32)
        up = upscale_patch_grid(grid, CAP_H, CAP_W)
        assert up.shape == (CAP_H, CAP_W)
        assert np.allclose(up, 3.0, atol=1e-5)


# ---------------------------------------------------------------------------
# feature_engine — normalize_heatmap
# ---------------------------------------------------------------------------

class TestNormalizeHeatmap:
    def test_range_is_zero_to_one(self, rng):
        h = rng.random((CAP_H, CAP_W)).astype(np.float32)
        n = normalize_heatmap(h)
        assert n.min() >= 0.0 - 1e-6
        assert n.max() <= 1.0 + 1e-6

    def test_uniform_returns_zeros(self):
        h = np.ones((10, 10), dtype=np.float32) * 7.0
        n = normalize_heatmap(h)
        assert np.all(n == 0.0)

    def test_min_max_pinned(self):
        h = np.array([[0.0, 5.0], [2.5, 10.0]], dtype=np.float32)
        n = normalize_heatmap(h)
        assert np.isclose(n.min(), 0.0, atol=1e-6)
        assert np.isclose(n.max(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# feature_engine — patch_grid_size_from_seq_len
# ---------------------------------------------------------------------------

class TestPatchGridSizeFromSeqLen:
    def test_dinov2_large(self):
        g_h, g_w = patch_grid_size_from_seq_len(257)   # 1 + 16*16
        assert (g_h, g_w) == (16, 16)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="perfect square"):
            patch_grid_size_from_seq_len(100)  # 99 patches — not a perfect square


# ---------------------------------------------------------------------------
# dom_intersect — _clip_bbox
# ---------------------------------------------------------------------------

class TestClipBbox:
    def test_fully_inside(self):
        result = _clip_bbox(10, 20, 100, 50, img_h=200, img_w=300)
        assert result == (10, 20, 110, 70)

    def test_fully_outside_left(self):
        result = _clip_bbox(-200, 0, 100, 50, img_h=200, img_w=300)
        assert result is None

    def test_fully_outside_right(self):
        result = _clip_bbox(350, 0, 100, 50, img_h=200, img_w=300)
        assert result is None

    def test_partially_clipped(self):
        result = _clip_bbox(250, 0, 100, 50, img_h=200, img_w=300)
        assert result == (250, 0, 300, 50)

    def test_scroll_offset(self):
        # Element at doc coords (100, 200), scroll_y=150 → viewport y=50
        result = _clip_bbox(100, 200, 80, 40, img_h=1080, img_w=1920, scroll_y=150, scroll_x=0)
        assert result == (100, 50, 180, 90)

    def test_scroll_scrolls_element_off_screen(self):
        result = _clip_bbox(0, 10, 100, 50, img_h=200, img_w=300, scroll_y=200)
        assert result is None


# ---------------------------------------------------------------------------
# dom_intersect — compute_attention_density
# ---------------------------------------------------------------------------

class TestComputeAttentionDensity:
    def test_full_heatmap_uniform(self, uniform_heatmap):
        density = compute_attention_density(uniform_heatmap, [0, 0, CAP_W, CAP_H])
        assert density is not None
        assert abs(density - 1.0) < 1e-5

    def test_bbox_outside_returns_none(self, uniform_heatmap):
        # Bbox fully off to the right
        density = compute_attention_density(uniform_heatmap, [CAP_W + 10, 0, 100, 100])
        assert density is None

    def test_zero_heatmap_gives_zero_density(self, zero_heatmap):
        density = compute_attention_density(zero_heatmap, [0, 0, 100, 100])
        assert density == 0.0

    def test_hotspot_in_bbox(self):
        h = np.zeros((100, 100), dtype=np.float32)
        h[10:20, 10:20] = 1.0           # hotspot in [10,10,10,10]
        density = compute_attention_density(h, [10, 10, 10, 10])
        assert density is not None
        assert density > 0.0, "Expected non-zero density for bbox over hotspot"

    def test_scroll_adjustment(self):
        h = np.zeros((100, 100), dtype=np.float32)
        h[0:10, 0:10] = 2.0             # top-left hotspot (viewport coords)
        # Element at document coords (0, 100), scroll_y=100 → viewport y=0
        density = compute_attention_density(h, [0, 100, 10, 10], scroll_y=100)
        assert density is not None
        assert density > 0.0

    def test_returns_float(self, uniform_heatmap):
        density = compute_attention_density(uniform_heatmap, [0, 0, 50, 50])
        assert isinstance(density, float)


# ---------------------------------------------------------------------------
# dom_intersect — select_winner
# ---------------------------------------------------------------------------

class TestSelectWinner:
    def _make_element(self, dom_id, x, y, w, h, is_intersecting=True):
        return {
            "dom_id": dom_id,
            "tag": "DIV",
            "bbox": [x, y, w, h],
            "is_intersecting_viewport": is_intersecting,
            "z_index": 1,
        }

    def test_winner_is_element_over_hotspot(self):
        h = np.zeros((100, 100), dtype=np.float32)
        h[50:60, 50:60] = 10.0          # hotspot at bottom-right
        elements = [
            self._make_element("#a", 0, 0, 20, 20),    # top-left — no hotspot
            self._make_element("#b", 50, 50, 10, 10),  # over hotspot
        ]
        winner = select_winner(h, elements)
        assert winner is not None
        assert winner["dom_id"] == "#b"

    def test_no_eligible_elements_returns_none(self, zero_heatmap):
        # All elements not intersecting viewport
        elements = [
            {**self._make_element("#x", 0, 0, 50, 50), "is_intersecting_viewport": False}
        ]
        winner = select_winner(zero_heatmap, elements)
        assert winner is None

    def test_empty_elements_returns_none(self, uniform_heatmap):
        assert select_winner(uniform_heatmap, []) is None

    def test_winner_dict_has_required_keys(self, uniform_heatmap):
        elements = [self._make_element("#nav", 0, 0, 100, 50)]
        winner = select_winner(uniform_heatmap, elements)
        assert winner is not None
        assert set(winner.keys()) >= {"dom_id", "tag", "bbox", "attention_density"}

    def test_off_viewport_element_skipped(self):
        h = np.ones((100, 100), dtype=np.float32)
        elements = [
            {**self._make_element("#hidden", 0, 0, 50, 50), "is_intersecting_viewport": False},
            self._make_element("#visible", 50, 50, 10, 10),
        ]
        winner = select_winner(h, elements)
        assert winner is not None
        assert winner["dom_id"] == "#visible"

    def test_missing_bbox_element_skipped(self, uniform_heatmap):
        elements = [
            {"dom_id": "#no-bbox", "tag": "DIV", "is_intersecting_viewport": True},
            {"dom_id": "#has-bbox", "tag": "BUTTON", "bbox": [0, 0, 50, 50], "is_intersecting_viewport": True},
        ]
        winner = select_winner(uniform_heatmap, elements)
        assert winner is not None
        assert winner["dom_id"] == "#has-bbox"

    def test_attention_density_is_positive(self, uniform_heatmap):
        elements = [{"dom_id": "#el", "tag": "A", "bbox": [0, 0, 100, 100], "is_intersecting_viewport": True}]
        winner = select_winner(uniform_heatmap, elements)
        assert winner is not None
        assert winner["attention_density"] > 0.0


# ---------------------------------------------------------------------------
# dom_intersect — ground_snapshot
# ---------------------------------------------------------------------------

class TestGroundSnapshot:
    def test_returns_winner_from_snapshot(self):
        h = np.zeros((100, 100), dtype=np.float32)
        h[10:30, 10:30] = 5.0
        snapshot = {
            "t_idx": 10,
            "scrollY": 0,
            "scrollX": 0,
            "elements": [
                {"dom_id": "#btn", "tag": "BUTTON", "bbox": [10, 10, 20, 20], "is_intersecting_viewport": True},
                {"dom_id": "#nav", "tag": "NAV",    "bbox": [60, 60, 30, 30], "is_intersecting_viewport": True},
            ],
        }
        winner = ground_snapshot(h, snapshot)
        assert winner is not None
        assert winner["dom_id"] == "#btn"

    def test_empty_snapshot_returns_none(self, uniform_heatmap):
        snapshot = {"t_idx": 0, "elements": []}
        assert ground_snapshot(uniform_heatmap, snapshot) is None

    def test_scroll_propagated(self):
        h = np.zeros((200, 200), dtype=np.float32)
        h[0:10, 0:10] = 3.0            # hotspot at top-left (viewport pixel coords)
        snapshot = {
            "scrollY": 100,             # doc coords → subtract 100 → viewport y=0
            "scrollX": 0,
            "elements": [
                {"dom_id": "#el", "tag": "DIV", "bbox": [0, 100, 10, 10], "is_intersecting_viewport": True},
            ],
        }
        winner = ground_snapshot(h, snapshot)
        assert winner is not None
        assert winner["dom_id"] == "#el"


# ---------------------------------------------------------------------------
# dom_intersect — find_nearest_snapshot
# ---------------------------------------------------------------------------

class TestFindNearestSnapshot:
    def test_exact_match(self):
        manifest = {"dom_snapshots": [{"t_idx": 5}, {"t_idx": 10}, {"t_idx": 15}]}
        snap = find_nearest_snapshot(manifest, 10)
        assert snap["t_idx"] == 10

    def test_nearest_when_no_exact(self):
        manifest = {"dom_snapshots": [{"t_idx": 0}, {"t_idx": 20}]}
        snap = find_nearest_snapshot(manifest, 7)
        assert snap["t_idx"] == 0   # 7 is closer to 0 than 20

    def test_empty_manifest_returns_none(self):
        assert find_nearest_snapshot({}, 5) is None

    def test_no_snapshots_key_returns_none(self):
        assert find_nearest_snapshot({"capture": {}}, 5) is None
