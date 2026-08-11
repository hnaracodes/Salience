"""Unit tests for DeepGaze MSDB preprocessing (no torch / weights required)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scout_core.deepgaze_msdb.preprocess import (
    density_from_log_density,
    load_rgb_image,
    maybe_resize_long_side,
    prepare_centerbias,
    resize_density,
)


def test_load_rgb_image_from_array_and_path(tmp_path):
    arr = np.zeros((32, 48, 3), dtype=np.uint8)
    arr[0, 0] = [10, 20, 30]
    path = tmp_path / "tiny.png"
    Image.fromarray(arr).save(path)

    loaded = load_rgb_image(path)
    assert loaded.shape == (32, 48, 3)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded[0, 0], [10, 20, 30])

    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    assert load_rgb_image(rgba).shape == (8, 8, 3)


def test_prepare_centerbias_uniform_normalized():
    cb = prepare_centerbias(40, 60, uniform=True)
    assert cb.shape == (40, 60)
    # log-density: exp(cb) should sum to ~1
    dens = np.exp(cb - cb.max())
    dens = dens / dens.sum()
    assert dens.sum() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_prepare_centerbias_from_template():
    # Broad central bump so nearest-neighbor zoom cannot drop the mode.
    yy, xx = np.mgrid[0:100, 0:100]
    template = -((yy - 50) ** 2 + (xx - 50) ** 2).astype(np.float64)
    cb = prepare_centerbias(50, 50, centerbias_template=template, uniform=False)
    assert cb.shape == (50, 50)
    dens = np.exp(cb - cb.max())
    dens /= dens.sum()
    peak = np.unravel_index(np.argmax(dens), dens.shape)
    assert 15 <= peak[0] <= 35
    assert 15 <= peak[1] <= 35
    assert dens.sum() == pytest.approx(1.0, rel=1e-5)


def test_density_from_log_density_and_resize():
    log_d = np.zeros((10, 10), dtype=np.float64)
    log_d[2, 3] = 3.0
    dens = density_from_log_density(log_d)
    assert dens.dtype == np.float32
    assert dens.shape == (10, 10)
    assert dens.sum() == pytest.approx(1.0, rel=1e-5)
    assert dens.min() >= 0
    assert np.argmax(dens) == np.ravel_multi_index((2, 3), dens.shape)

    resized = resize_density(dens, 20, 30)
    assert resized.shape == (20, 30)
    assert resized.sum() == pytest.approx(1.0, rel=1e-4)
    assert resized.min() >= 0


def test_maybe_resize_long_side():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    same, hw = maybe_resize_long_side(img, None)
    assert same.shape == (200, 400, 3)
    assert hw == (200, 400)

    small, hw = maybe_resize_long_side(img, 200)
    assert hw == (200, 400)
    assert max(small.shape[:2]) == 200
