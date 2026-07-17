"""Tests for Horikawa real TRIBE training pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tiny_horikawa"


@pytest.fixture
def tiny_horikawa(tmp_path: Path) -> dict[str, Path]:
    """Build 3-clip synthetic Horikawa fixture in tmp_path."""
    from scout_core.horikawaCode.constants import HORIKAWA_14_DIMENSIONS, HORIKAWA_34_CATEGORIES
    from scout_core.horikawaCode.labels import write_ratings_cache

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    ids = ["101", "102", "103"]
    rng = np.random.default_rng(0)
    ratings = {
        "stimulus_id": np.asarray(ids, dtype=object),
        "categories_34": rng.uniform(0, 0.5, (3, 34)).astype(np.float32),
        "dimensions_14": rng.uniform(2, 8, (3, 14)).astype(np.float32),
    }
    ratings["categories_34"][0, HORIKAWA_34_CATEGORIES.index("amusement")] = 0.9
    ratings["categories_34"][1, HORIKAWA_34_CATEGORIES.index("fear")] = 0.85
    ratings["categories_34"][2, HORIKAWA_34_CATEGORIES.index("sadness")] = 0.8
    write_ratings_cache(labels_dir, ratings)

    inter_dir = tmp_path / "intermediates"
    inter_dir.mkdir()
    for i, sid in enumerate(ids):
        t_count = 5 + i
        cortical = (rng.standard_normal((t_count, 20484), dtype=np.float32) * 0.05).astype(np.float32)
        sub = (rng.standard_normal((t_count, 8802), dtype=np.float32) * 0.05).astype(np.float32)
        np.savez_compressed(
            inter_dir / f"{sid}_both.npz",
            preds=cortical,
            preds_subcortical=sub,
            stimulus_id=np.asarray(sid),
        )

    manifest = {
        "corpus": "pilot_150",
        "clips": [{"stimulus_id": sid} for sid in ids],
    }
    manifest_path = tmp_path / "pilot_3.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    return {
        "root": tmp_path,
        "labels_dir": labels_dir,
        "inter_dir": inter_dir,
        "manifest": manifest_path,
    }


def test_bootstrap_train_and_load(tmp_path: Path):
    py = sys.executable
    train_npz = tmp_path / "demo_train.npz"
    cfg_path = tmp_path / "demo_cfg.yaml"
    cfg_path.write_text(
        f"""
feature_spec: fused_schaefer400_subcortical_v1
cv:
  scheme: leave_one_video_out
ridge:
  alphas: [0.1, 1.0, 10.0]
model_id: horikawa_ridge_demo_test_v1
target_type: product_8
categories:
  - amusement
  - awe
  - contentment
  - excitement
  - fear
  - anger
  - sadness
  - confusion
data:
  train_npz: {train_npz}
  reports_dir: {tmp_path / "reports"}
  cv_report_name: cv_report_demo_test.json
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "horikawaCode" / "prepare_horikawa_tribev2.py"),
            "--config",
            str(cfg_path),
            "--demo",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    subprocess.run(
        [
            py,
            str(PROJECT_ROOT / "scripts" / "horikawaCode" / "train_horikawa_ridge_decoder.py"),
            "--config",
            str(cfg_path),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    model_path = PROJECT_ROOT / "scout_models" / "horikawa_ridge_demo_test_v1" / "model.joblib"
    assert model_path.is_file()


def test_label_loader_schema(tiny_horikawa: dict[str, Path]):
    from scout_core.horikawaCode.constants import HORIKAWA_14_DIMENSIONS, HORIKAWA_34_CATEGORIES
    from scout_core.horikawaCode.labels import load_horikawa_ratings

    ratings = load_horikawa_ratings(tiny_horikawa["labels_dir"])
    assert len(ratings) == 3
    sample = ratings["101"]
    assert sample["categories_34"].shape == (34,)
    assert sample["dimensions_14"].shape == (14,)
    assert len(HORIKAWA_34_CATEGORIES) == 34
    assert len(HORIKAWA_14_DIMENSIONS) == 14


def test_prepare_clip_level_rows(tiny_horikawa: dict[str, Path], tmp_path: Path):
    from scout_core.horikawaCode.prepare import build_train_npz_from_manifest

    clips = json.loads(tiny_horikawa["manifest"].read_text())["clips"]
    data = build_train_npz_from_manifest(
        clips,
        intermediates_dir=tiny_horikawa["inter_dir"],
        labels_cache=tiny_horikawa["labels_dir"],
        corpus="pilot_150",
        target="product_8",
    )
    assert data["X"].shape == (3, 808)
    assert data["y"].shape == (3, 8)
    assert len(np.unique(data["stimulus_id"])) == 3
    assert np.all(np.isfinite(data["X"]))
    assert data["groups"].tolist() == [0, 1, 2]


def test_prepare_honors_feature_spec(tiny_horikawa: dict[str, Path]):
    from scout_core.horikawaCode.prepare import build_train_npz_from_manifest

    clips = json.loads(tiny_horikawa["manifest"].read_text())["clips"]
    data = build_train_npz_from_manifest(
        clips,
        intermediates_dir=tiny_horikawa["inter_dir"],
        labels_cache=tiny_horikawa["labels_dir"],
        corpus="pilot_150",
        target="product_8",
        feature_spec="cortical_mean400_v1",
    )
    assert data["X"].shape == (3, 400)


def test_lovo_no_duplicate_stimulus_leakage(tiny_horikawa: dict[str, Path]):
    from sklearn.model_selection import LeaveOneGroupOut

    from scout_core.horikawaCode.prepare import build_train_npz_from_manifest

    clips = json.loads(tiny_horikawa["manifest"].read_text())["clips"]
    data = build_train_npz_from_manifest(
        clips,
        intermediates_dir=tiny_horikawa["inter_dir"],
        labels_cache=tiny_horikawa["labels_dir"],
        corpus="pilot_150",
    )
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(data["X"], data["y"], data["groups"]):
        train_ids = set(data["stimulus_id"][train_idx].tolist())
        test_ids = set(data["stimulus_id"][test_idx].tolist())
        assert train_ids.isdisjoint(test_ids)


def test_train_exports_manifest(tiny_horikawa: dict[str, Path], tmp_path: Path):
    py = sys.executable
    train_npz = tmp_path / "train.npz"
    from scout_core.horikawaCode.prepare import build_train_npz_from_manifest

    clips = json.loads(tiny_horikawa["manifest"].read_text())["clips"]
    data = build_train_npz_from_manifest(
        clips,
        intermediates_dir=tiny_horikawa["inter_dir"],
        labels_cache=tiny_horikawa["labels_dir"],
        corpus="pilot_150",
    )
    np.savez_compressed(train_npz, **data)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        f"""
feature_spec: fused_schaefer400_subcortical_v1
cv:
  scheme: leave_one_video_out
ridge:
  alphas: [0.1, 1.0, 10.0]
model_id: horikawa_ridge_test_v1
target_type: product_8
categories:
  - amusement
  - awe
  - contentment
  - excitement
  - fear
  - anger
  - sadness
  - confusion
data:
  train_npz: {train_npz}
  reports_dir: {tmp_path / "reports"}
  cv_report_name: cv_report_test.json
""",
        encoding="utf-8",
    )
    subprocess.run(
        [py, str(PROJECT_ROOT / "scripts" / "horikawaCode" / "train_horikawa_ridge_decoder.py"), "--config", str(cfg_path)],
        check=True,
        cwd=PROJECT_ROOT,
    )
    bundle = PROJECT_ROOT / "scout_models" / "horikawa_ridge_test_v1"
    assert (bundle / "manifest.json").is_file()
    meta = json.loads((bundle / "meta.json").read_text())
    assert meta["status"] == "pilot"
    assert meta["cv_scheme"] == "leave_one_video_out"


def test_dims_bundle_prepare(tiny_horikawa: dict[str, Path]):
    from scout_core.horikawaCode.prepare import build_train_npz_from_manifest

    clips = json.loads(tiny_horikawa["manifest"].read_text())["clips"]
    data = build_train_npz_from_manifest(
        clips,
        intermediates_dir=tiny_horikawa["inter_dir"],
        labels_cache=tiny_horikawa["labels_dir"],
        corpus="pilot_150",
        target="dimensions_14",
    )
    assert data["y"].shape == (3, 14)


def test_build_y_product_8_mapping():
    from scout_core.horikawaCode.constants import HORIKAWA_34_CATEGORIES
    from scout_core.horikawaCode.labels import build_y_product_8

    cats = np.zeros(len(HORIKAWA_34_CATEGORIES), dtype=np.float32)
    cats[HORIKAWA_34_CATEGORIES.index("amusement")] = 0.7
    cats[HORIKAWA_34_CATEGORIES.index("horror")] = 0.4
    y = build_y_product_8(cats)
    assert y[0] == pytest.approx(0.7)  # amusement
    assert y[4] == pytest.approx(0.4)  # fear from horror map


def test_lovo_cv_nonzero_on_signal():
    from scout_core.horikawaCode.cv import lovo_mean_r

    rng = np.random.default_rng(1)
    n = 20
    X = rng.normal(size=(n, 8)).astype(np.float32)
    w = rng.normal(size=(8, 3)).astype(np.float32)
    y = (X @ w).astype(np.float32)
    groups = np.arange(n, dtype=np.int32)
    r = lovo_mean_r(X, y, groups, [0.1, 1.0, 10.0])
    assert r > 0.1


def test_metric_name_order_is_numeric():
    from scout_core.horikawaCode.ablation import _name_per_target
    from scout_core.horikawaCode.constants import HORIKAWA_14_DIMENSIONS

    metrics = {f"class_{i}": float(i) for i in range(14)}
    named = _name_per_target(metrics, list(HORIKAWA_14_DIMENSIONS))
    assert named["attention"] == pytest.approx(2.0)
    assert named["obstruction"] == pytest.approx(10.0)


def test_real_subcortical_map_fails_closed(tmp_path: Path):
    from scout_core.subcortical.atlas import build_region_index

    with pytest.raises(FileNotFoundError):
        build_region_index(voxel_regions_csv=tmp_path / "missing_subcortical_voxel_regions.csv")


def test_feature_parity_train_infer(tiny_horikawa: dict[str, Path]):
    from scout_core.affect_features import build_affect_features, resolve_feature_spec
    from scout_core.horikawaCode.prepare import load_both_npz, _pool_clip_features

    spec = resolve_feature_spec()
    npz_path = tiny_horikawa["inter_dir"] / "101_both.npz"
    cortical, sub = load_both_npz(npz_path)
    row = _pool_clip_features(cortical, sub)
    tr_feat = build_affect_features(cortical[1:-1], sub[1:-1], spec=spec).mean(axis=0)
    np.testing.assert_allclose(row, tr_feat, rtol=1e-5, atol=1e-5)
