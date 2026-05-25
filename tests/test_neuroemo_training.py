from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scout_core.parcellation import (
    CANONICAL_VERTEX_ORDER,
    VertexParcellationTable,
    load_vertex_table,
    validate_vertex_table,
)
from scripts.train_neuroemo_emotion_model import (
    TrainConfig,
    _load_training_data,
    _make_estimator,
    _predict_probability_matrix,
)


def _write_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def test_validate_vertex_table_rejects_mixed_hemisphere_boundary():
    n_vertices = 20484
    vertex_index = np.arange(n_vertices, dtype=np.int64)
    parcel_id = np.ones(n_vertices, dtype=np.int64)
    parcel_label = np.asarray(["parcel_1"] * n_vertices, dtype=object)
    yeo_network_id = np.zeros(n_vertices, dtype=np.int64)
    yeo_network_name = np.asarray([""] * n_vertices, dtype=object)
    hemisphere = np.asarray(["lh"] * 10242 + ["rh"] * 10242, dtype=object)
    hemisphere[100] = "rh"
    hemisphere[15000] = "lh"
    table = VertexParcellationTable(
        vertex_index=vertex_index,
        parcel_id=parcel_id,
        parcel_label=parcel_label,
        yeo_network_id=yeo_network_id,
        yeo_network_name=yeo_network_name,
        hemisphere=hemisphere,
    )
    with pytest.raises(ValueError, match="first 10242 vertices"):
        validate_vertex_table(table)


def test_validate_vertex_table_rejects_conflicting_labels():
    table = VertexParcellationTable(
        vertex_index=np.arange(4, dtype=np.int64),
        parcel_id=np.asarray([1, 1, 2, 2], dtype=np.int64),
        parcel_label=np.asarray(["a", "b", "c", "c"], dtype=object),
        yeo_network_id=np.asarray([0, 0, 0, 0], dtype=np.int64),
        yeo_network_name=np.asarray(["", "", "", ""], dtype=object),
        hemisphere=np.asarray(["unknown", "unknown", "unknown", "unknown"], dtype=object),
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        validate_vertex_table(table, n_vertices=4, require_hemispheres=False)


def test_load_vertex_table_legacy_ids_are_deterministic(tmp_path: Path):
    csv_path = tmp_path / "legacy_vertex_regions.csv"
    csv_path.write_text(
        "\n".join(
            [
                "vertex_index,region_name",
                "0,b_region",
                "1,a_region",
                "2,b_region",
            ]
        ),
        encoding="utf-8",
    )

    first = load_vertex_table(csv_path)
    second = load_vertex_table(csv_path)

    assert first.parcel_id.tolist() == second.parcel_id.tolist()
    assert first.parcel_id.tolist() == [2, 1, 2]


def test_load_training_data_rejects_rewindowed_npz_by_default(tmp_path: Path):
    npz_path = tmp_path / "windowed_train.npz"
    _write_npz(
        npz_path,
        X=np.ones((4, 2, 3), dtype=np.float32),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        t_idx=np.asarray([0, 1, 0, 1], dtype=np.int64),
        time_s=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(2, dtype=np.int64),
        source_sample_axis=np.asarray("prep_window"),
    )

    cfg = TrainConfig(
        train_npz=str(npz_path),
        feature_mode="vertices",
        temporal_window_trs=2,
        exclude_labels="",
    )
    with pytest.raises(ValueError, match="already windowed"):
        _load_training_data(npz_path, cfg)


def test_load_training_data_requires_real_time_arrays_for_temporal_aggregation(tmp_path: Path):
    npz_path = tmp_path / "missing_time.npz"
    _write_npz(
        npz_path,
        X=np.ones((4, 3), dtype=np.float32),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(1, dtype=np.int64),
        source_sample_axis=np.asarray("tr"),
    )

    cfg = TrainConfig(
        train_npz=str(npz_path),
        feature_mode="vertices",
        temporal_window_trs=2,
        exclude_labels="",
    )
    with pytest.raises(ValueError, match="source t_idx metadata"):
        _load_training_data(npz_path, cfg)


def test_load_training_data_records_source_contract(tmp_path: Path):
    npz_path = tmp_path / "train.npz"
    _write_npz(
        npz_path,
        X=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.1, 0.0],
            ],
            dtype=np.float32,
        ),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        t_idx=np.asarray([0, 1, 0, 1], dtype=np.int64),
        time_s=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        dataset_id=np.asarray("ds005700"),
        snapshot_version=np.asarray("1.2.0"),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(1, dtype=np.int64),
        source_sample_axis=np.asarray("tr"),
        prep_metadata_json=np.asarray('{"bold_lag_s": 6.0, "drop_transition_trs": 0}'),
    )

    cfg = TrainConfig(
        train_npz=str(npz_path),
        feature_mode="vertices",
        temporal_window_trs=2,
        exclude_labels="",
    )
    data = _load_training_data(npz_path, cfg)

    assert data.metadata["source_contract"]["vertex_order"] == CANONICAL_VERTEX_ORDER
    assert data.metadata["source_contract"]["source_window_trs"] == 1
    assert data.metadata["source_contract"]["prep_metadata"]["bold_lag_s"] == 6.0
    assert data.metadata["temporal_aggregation"]["enabled"] is True
    assert data.X.shape[0] == 2


def test_linear_svc_probability_matrix_is_finite():
    cfg = TrainConfig(model_type="linear_svc", feature_mode="vertices", exclude_labels="")
    estimator = _make_estimator(cfg)
    X = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 2, 1, 2, 0], dtype=np.int64)
    estimator.fit(X, y)
    proba, method = _predict_probability_matrix(estimator, X, n_classes=3)

    assert method == "decision_function_softmax"
    assert proba is not None
    assert proba.shape == (6, 3)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
