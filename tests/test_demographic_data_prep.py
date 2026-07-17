from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_prep.metadata_harmonize import assign_clusters, harmonize_metadata
from data_prep.viability_partition import age_to_band, m0_passed_for_training


def test_age_to_band_boundaries():
    assert age_to_band(18) == "young_adult"
    assert age_to_band(35) == "young_adult"
    assert age_to_band(36) == "middle_adult"
    assert age_to_band(56) == "older_adult"


def test_harmonize_and_segment(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(
        {
            "subject_id": [f"s{i}" for i in range(25)],
            "source_subject_id": [f"s{i}" for i in range(25)],
            "age": [25] * 12 + [40] * 13,
            "sex": ["female"] * 12 + ["male"] * 13,
            "site_id": ["hcp"] * 25,
        }
    ).to_csv(raw / "hcp_7t_subjects.csv", index=False)

    meta_out = tmp_path / "meta.parquet"
    df = harmonize_metadata(raw, meta_out)
    assert len(df) == 25
    assert "age_band" in df.columns

    clusters_yaml = Path(__file__).resolve().parents[1] / "configs" / "clusters.yaml"
    out_dir = tmp_path / "demographic"
    members = assign_clusters(df, clusters_yaml, out_dir)
    assert not members.empty
    assert (out_dir / "cluster_members.parquet").is_file()
    assert (out_dir / "cluster_demographics.json").is_file()


def test_m0_passed_for_training_rejects_synthetic(tmp_path: Path):
    from data_prep.schemas import M0Report, M0NetworkResult
    from data_prep.viability_partition import write_m0_report

    report = M0Report(
        passed=True,
        is_synthetic=True,
        data_source="synthetic_fixture",
        n_subjects=100,
        n_permutations=100,
        networks=[],
        go_criteria={},
    )
    path = tmp_path / "m0.json"
    write_m0_report(report, path)
    assert m0_passed_for_training(path) is False
