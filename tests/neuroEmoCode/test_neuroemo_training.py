from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NEUROEMO_SCRIPTS = PROJECT_ROOT / "scripts" / "neuroEmoCode"
if str(_NEUROEMO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_NEUROEMO_SCRIPTS))

from scout_core.parcellation import (
    CANONICAL_VERTEX_ORDER,
    VertexParcellationTable,
    file_sha256,
    load_vertex_table,
    validate_manifest_matches_table,
    validate_vertex_table,
)
from train_neuroemo_emotion_model import (
    TEMPORAL_REDUCER_COMPONENTS,
    TrainConfig,
    _load_training_data,
    _make_estimator,
    _predict_probability_matrix,
    _reduce_temporal_window,
)
from train_neuroemo_hierarchical_model import _hierarchy_index, _parse_hierarchy
from run_neuroemo_experiment_matrix import build_arg_parser, build_specs

POSTFIX_MATRIX_DIR = PROJECT_ROOT / "scout_data" / "neuroEmoCode" / "models" / "2026-05-25_postfix_matrix"


def _verified_vertex_equivalence(
    proof_status: str = "mesh_identity_verified",
    report_sha256: str = "abc123",
) -> dict[str, object]:
    return {
        "proof_status": proof_status,
        "mesh": "fsaverage5",
        "vertex_order": CANONICAL_VERTEX_ORDER,
        "reference_kind": "test_fixture",
        "reference_detail": "synthetic test metadata",
        "report_sha256": report_sha256,
    }


def _write_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays.setdefault(
        "vertex_equivalence_json",
        np.asarray(json.dumps(_verified_vertex_equivalence(), sort_keys=True)),
    )
    arrays.setdefault("vertex_equivalence_report_sha256", np.asarray("abc123"))
    np.savez(path, **arrays)


def _write_manifest(
    path: Path,
    *,
    vertex_order: str,
    vertex_csv_sha256: str,
    proof_status: str = "mesh_identity_verified",
    label_source_kind: str = "surface_annot_cbig",
) -> None:
    path.write_text(
        "\n".join(
            [
                "mesh: fsaverage5",
                f"tribe_vertex_order: {vertex_order}",
                "atlas_id: schaefer2018_400_7networks_fsaverage5",
                "n_vertices_expected: 20484",
                "n_parcels: 400",
                "hemisphere_order: lh_then_rh",
                "source:",
                f"  label_source_kind: {label_source_kind}",
                "  filled_unassigned_vertices: 0",
                "vertex_order_proof:",
                "  status: annot_matches_nilearn_projection",
                "  reorder_applied: false",
                "vertex_equivalence:",
                f"  proof_status: {proof_status}",
                "  mesh: fsaverage5",
                f"  vertex_order: {CANONICAL_VERTEX_ORDER}",
                "  reference_kind: test_fixture",
                "  reference_detail: synthetic test metadata",
                "  report_sha256: abc123",
                "artifact_sha256:",
                f"  vertex_regions_csv: {vertex_csv_sha256}",
                "  vertex_equivalence_report_json: abc123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _load_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_validate_vertex_table_rejects_conflicting_networks():
    table = VertexParcellationTable(
        vertex_index=np.arange(4, dtype=np.int64),
        parcel_id=np.asarray([1, 1, 2, 2], dtype=np.int64),
        parcel_label=np.asarray(["a", "a", "b", "b"], dtype=object),
        yeo_network_id=np.asarray([1, 2, 0, 0], dtype=np.int64),
        yeo_network_name=np.asarray(["Vis", "SomMot", "", ""], dtype=object),
        hemisphere=np.asarray(["unknown", "unknown", "unknown", "unknown"], dtype=object),
    )
    with pytest.raises(ValueError, match="conflicting networks"):
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


def test_load_training_data_rejects_missing_vertex_equivalence_by_default(tmp_path: Path):
    npz_path = tmp_path / "missing_vertex_equivalence.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        X=np.ones((4, 3), dtype=np.float32),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        t_idx=np.asarray([0, 1, 0, 1], dtype=np.int64),
        time_s=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(1, dtype=np.int64),
        source_sample_axis=np.asarray("tr"),
    )

    cfg = TrainConfig(train_npz=str(npz_path), feature_mode="vertices", exclude_labels="")
    with pytest.raises(ValueError, match="Missing vertex_equivalence"):
        _load_training_data(npz_path, cfg)


def test_load_training_data_allows_missing_vertex_equivalence_with_opt_in(tmp_path: Path):
    npz_path = tmp_path / "missing_vertex_equivalence_optin.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        X=np.ones((4, 3), dtype=np.float32),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        t_idx=np.asarray([0, 1, 0, 1], dtype=np.int64),
        time_s=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(1, dtype=np.int64),
        source_sample_axis=np.asarray("tr"),
    )

    cfg = TrainConfig(
        train_npz=str(npz_path),
        feature_mode="vertices",
        exclude_labels="",
        allow_unverified_vertex_equivalence=True,
    )
    data = _load_training_data(npz_path, cfg)
    assert data.metadata["source_contract"]["vertex_equivalence"]["proof_status"] == "contract_only"


def test_load_training_data_rejects_vertex_equivalence_report_hash_mismatch(tmp_path: Path):
    npz_path = tmp_path / "report_hash_mismatch.npz"
    _write_npz(
        npz_path,
        X=np.ones((4, 3), dtype=np.float32),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        t_idx=np.asarray([0, 1, 0, 1], dtype=np.int64),
        time_s=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(1, dtype=np.int64),
        source_sample_axis=np.asarray("tr"),
        vertex_equivalence_json=np.asarray(
            json.dumps(_verified_vertex_equivalence(report_sha256="mismatch"), sort_keys=True)
        ),
        vertex_equivalence_report_sha256=np.asarray("expected_hash"),
    )
    cfg = TrainConfig(train_npz=str(npz_path), feature_mode="vertices", exclude_labels="")
    with pytest.raises(ValueError, match="vertex_equivalence_report_sha256 mismatch"):
        _load_training_data(npz_path, cfg)


def test_validate_manifest_matches_table_rejects_unsupported_vertex_order():
    vertex_csv = PROJECT_ROOT / "configs" / "vertex_regions.csv"
    table = load_vertex_table(vertex_csv)
    manifest = {
        "mesh": "fsaverage5",
        "tribe_vertex_order": "rh_then_lh_fsaverage5",
        "atlas_id": "schaefer2018_400_7networks_fsaverage5",
        "n_vertices_expected": 20484,
        "n_parcels": 400,
        "hemisphere_order": "lh_then_rh",
        "source": {
            "label_source_kind": "surface_annot_cbig",
            "filled_unassigned_vertices": 0,
        },
        "vertex_order_proof": {
            "status": "annot_matches_nilearn_projection",
            "reorder_applied": False,
        },
        "artifact_sha256": {
            "vertex_regions_csv": file_sha256(vertex_csv),
        },
    }
    with pytest.raises(ValueError, match="Unsupported vertex order"):
        validate_manifest_matches_table(manifest, table, vertex_csv_path=vertex_csv)


def test_validate_manifest_matches_table_rejects_vertex_csv_hash_mismatch():
    vertex_csv = PROJECT_ROOT / "configs" / "vertex_regions.csv"
    table = load_vertex_table(vertex_csv)
    manifest = {
        "mesh": "fsaverage5",
        "tribe_vertex_order": "facebook/tribev2_lh_then_rh",
        "atlas_id": "schaefer2018_400_7networks_fsaverage5",
        "n_vertices_expected": 20484,
        "n_parcels": 400,
        "hemisphere_order": "lh_then_rh",
        "source": {
            "label_source_kind": "surface_annot_cbig",
            "filled_unassigned_vertices": 0,
        },
        "vertex_order_proof": {
            "status": "annot_matches_nilearn_projection",
            "reorder_applied": False,
        },
        "artifact_sha256": {
            "vertex_regions_csv": "0" * 64,
        },
    }
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_manifest_matches_table(manifest, table, vertex_csv_path=vertex_csv)


def test_load_training_data_rejects_roi_manifest_vertex_order_mismatch(tmp_path: Path):
    vertex_csv = PROJECT_ROOT / "configs" / "vertex_regions.csv"
    manifest_path = tmp_path / "parcellation_manifest.yaml"
    _write_manifest(
        manifest_path,
        vertex_order="rh_then_lh_fsaverage5",
        vertex_csv_sha256=file_sha256(vertex_csv),
    )

    npz_path = tmp_path / "train.npz"
    _write_npz(
        npz_path,
        X=np.zeros((4, 20484), dtype=np.float32),
        y=np.asarray([0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        t_idx=np.asarray([0, 1, 0, 1], dtype=np.int64),
        time_s=np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(1, dtype=np.int64),
        source_sample_axis=np.asarray("tr"),
    )

    cfg = TrainConfig(
        train_npz=str(npz_path),
        feature_mode="roi",
        vertex_csv=str(vertex_csv),
        atlas_manifest=str(manifest_path),
        roi_reducers="mean",
        temporal_window_trs=1,
        exclude_labels="",
    )
    with pytest.raises(ValueError, match="Unsupported vertex order"):
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


def test_dynamic_temporal_reducer_returns_named_components():
    window = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    reduced = _reduce_temporal_window(window, "static_dynamic")

    assert TEMPORAL_REDUCER_COMPONENTS["static_dynamic"] == (
        "mean",
        "early_mean",
        "late_mean",
        "late_minus_early",
        "slope",
        "within_window_std",
        "last_minus_first",
    )
    assert reduced.shape == (7, 3)
    assert np.allclose(reduced[0], window.mean(axis=0))
    assert np.allclose(reduced[1], window[:2].mean(axis=0))
    assert np.allclose(reduced[2], window[2:].mean(axis=0))
    assert np.allclose(reduced[3], reduced[2] - reduced[1])
    assert np.allclose(reduced[-1], window[-1] - window[0])


def test_load_training_data_records_dynamic_temporal_components(tmp_path: Path):
    npz_path = tmp_path / "dynamic_train.npz"
    _write_npz(
        npz_path,
        X=np.arange(8 * 3, dtype=np.float32).reshape(8, 3),
        y=np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64),
        subject=np.asarray(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"]),
        labels=np.asarray(["calm", "afraid"]),
        t_idx=np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        time_s=np.asarray([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        mesh=np.asarray("fsaverage5"),
        vertex_order=np.asarray(CANONICAL_VERTEX_ORDER),
        source_window_trs=np.asarray(1, dtype=np.int64),
        source_sample_axis=np.asarray("tr"),
    )

    cfg = TrainConfig(
        train_npz=str(npz_path),
        feature_mode="vertices",
        temporal_window_trs=2,
        temporal_reducer="dynamic_basic",
        exclude_labels="",
    )
    data = _load_training_data(npz_path, cfg)

    assert data.X.shape == (4, 12)
    temporal = data.metadata["temporal_aggregation"]
    assert temporal["reducer"] == "dynamic_basic"
    assert temporal["reducer_components"] == ["mean", "within_window_std", "last_minus_first", "slope"]
    assert temporal["output_sample_shape"] == [4, 3]


def test_load_training_data_allows_rewindowed_npz_with_opt_in(tmp_path: Path):
    npz_path = tmp_path / "windowed_train.npz"
    _write_npz(
        npz_path,
        X=np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3),
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
        temporal_allow_rewindow=True,
        exclude_labels="",
    )
    data = _load_training_data(npz_path, cfg)

    assert data.metadata["source_contract"]["vertex_order"] == CANONICAL_VERTEX_ORDER
    assert data.metadata["source_contract"]["source_window_trs"] == 2
    assert data.metadata["source_contract"]["input_ndim"] == 3
    assert data.metadata["temporal_aggregation"]["enabled"] is True
    assert data.metadata["temporal_aggregation"]["output_samples"] == 2
    assert data.X.shape == (2, 6)


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


def test_hierarchical_model_requires_complete_label_assignment():
    labels = np.asarray(["calm", "afraid", "depressed"])
    hierarchy = _parse_hierarchy("calm=calm;negative=afraid")

    with pytest.raises(ValueError, match="does not assign all labels"):
        _hierarchy_index(labels, hierarchy)


def test_experiment_matrix_surface_phase_includes_svm_and_valence_runs(tmp_path: Path):
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--phase",
            "surface-postfix",
            "--train-npz",
            "train.npz",
            "--output-dir",
            str(tmp_path / "models"),
        ]
    )

    specs = build_specs(args)
    names = {spec.name for spec in specs}

    assert "linear_svc_5class_10tr" in names
    assert "mlp_valence_binary_10tr" in names
    assert "specialist_sigmoid_5class_10tr" in names


def test_postfix_matrix_metrics_record_shared_training_contract():
    metric_paths = sorted(POSTFIX_MATRIX_DIR.glob("*.metrics.json"))
    if not metric_paths:
        pytest.skip("postfix matrix metrics fixture not present in this checkout")
    assert len(metric_paths) == 7

    for metrics_path in metric_paths:
        metrics = _load_metrics(metrics_path)
        roi_spec = metrics["roi_spec"]
        validation = roi_spec["validation_summary"]
        temporal = metrics["temporal_aggregation"]
        label_filter = metrics["label_filter"]
        cross_validation = metrics["cross_validation"]

        assert metrics["feature_mode"] == "roi"
        assert metrics["roi_reducers"] == ["mean", "std", "mean_abs"]
        assert roi_spec["mesh"] == "fsaverage5"
        assert roi_spec["vertex_order"] == CANONICAL_VERTEX_ORDER
        assert roi_spec["source_metadata"]["source_space"] == "MNI152_volume_projected_to_fsaverage5"
        assert validation["mesh"] == "fsaverage5"
        assert validation["source_vertex_order"] == "facebook/tribev2_lh_then_rh"
        assert validation["vertex_regions_csv_sha256"] == roi_spec["vertex_csv_sha256"]
        assert validation["hemisphere_counts"] == {"lh": 10242, "rh": 10242}
        assert temporal["enabled"] is True
        assert temporal["window_trs"] == 10
        assert temporal["contiguity"] == "contiguous"
        assert cross_validation["n_splits"] == 5

        if metrics_path.name == "mlp_valence_binary_10tr.metrics.json":
            assert label_filter["excluded_labels"] == ["calm", "neutral"]
        else:
            assert label_filter["excluded_labels"] == ["neutral"]


def test_postfix_matrix_improves_documented_mlp_baselines():
    if not POSTFIX_MATRIX_DIR.is_dir():
        pytest.skip("postfix matrix metrics fixture not present in this checkout")
    mlp_5class = _load_metrics(POSTFIX_MATRIX_DIR / "mlp_5class_10tr.metrics.json")
    mlp_valence = _load_metrics(POSTFIX_MATRIX_DIR / "mlp_valence_binary_10tr.metrics.json")
    logistic_saga = _load_metrics(POSTFIX_MATRIX_DIR / "logistic_saga_5class_10tr.metrics.json")

    assert mlp_5class["cross_validation"]["mean_accuracy"] > 0.40
    assert mlp_valence["cross_validation"]["mean_accuracy"] > 0.64
    assert logistic_saga["cross_validation"]["mean_accuracy"] > mlp_5class["cross_validation"]["mean_accuracy"]
