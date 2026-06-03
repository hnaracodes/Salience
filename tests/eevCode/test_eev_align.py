"""Tests for EEV label alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scout_core.eevCode.align import (
    apply_lag_features,
    downsample_labels_to_grid,
    parse_eev_csv,
)
from scout_core.eevCode.constants import EEV_TARGET_LABELS


def test_downsample_labels_to_grid():
    df = pd.DataFrame(
        {
            "video_id": ["v1"] * 4,
            "t_sec": [0.0, 0.5, 1.0, 1.5],
            "interest": [0.0, 0.5, 1.0, 0.5],
            "awe": [0.1, 0.1, 0.9, 0.9],
        }
    )
    t_grid = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    Y = downsample_labels_to_grid(df, t_grid, label_names=("interest", "awe"))
    assert Y.shape == (3, 2)
    assert 0.0 <= Y.min() <= Y.max() <= 1.0
    assert Y[1, 0] > 0.4


def test_apply_lag_features():
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    Y = np.arange(12, dtype=np.float32).reshape(6, 2)
    X2, Y2 = apply_lag_features(X, Y, lag_s=2.0, tr_hz=1.0)
    assert X2.shape[0] == 4
    assert Y2.shape[0] == 4
    np.testing.assert_array_equal(X2[0], X[0])
    np.testing.assert_array_equal(Y2[0], Y[2])


def test_parse_eev_csv_youtube_id_milliseconds(tmp_path):
    csv_path = tmp_path / "eev.csv"
    csv_path.write_text(
        "YouTube ID,Timestamp (milliseconds),interest,awe\n"
        "abc123,0,0.1,0.2\n"
        "abc123,500,0.3,0.4\n",
        encoding="utf-8",
    )
    df = parse_eev_csv(csv_path)
    assert df["video_id"].iloc[0] == "abc123"
    assert df["t_sec"].iloc[1] == pytest.approx(0.5)


def test_parse_eev_csv_header(tmp_path):
    csv_path = tmp_path / "mini.csv"
    csv_path.write_text(
        "Video ID,Timestamp (microseconds),interest,awe\n"
        "abc123,0,0.1,0.2\n"
        "abc123,166666,0.3,0.4\n",
        encoding="utf-8",
    )
    df = parse_eev_csv(csv_path)
    assert "video_id" in df.columns
    assert "t_sec" in df.columns
    assert df["t_sec"].iloc[1] == pytest.approx(166666 / 1e6)
