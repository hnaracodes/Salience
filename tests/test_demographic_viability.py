from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data_prep.viability_partition import (
    analyze_network,
    evaluate_go_criteria,
    generate_synthetic_viability_df,
    load_m0_report,
    network_columns,
    run_viability_analysis,
    write_m0_report,
)


def test_network_columns_detects_yeo_names():
    df = generate_synthetic_viability_df(n_subjects=10)
    cols = network_columns(df)
    assert "net_Vis" in cols
    assert len(cols) == 7


def test_synthetic_viability_pipeline_logic(tmp_path: Path):
    df = generate_synthetic_viability_df(n_subjects=150, random_state=0)
    report = run_viability_analysis(
        df,
        data_source="synthetic_fixture",
        is_synthetic=True,
        n_permutations=200,
        random_state=0,
    )
    # Pipeline logic should detect signal in synthetic data
    assert report.go_criteria["variance_in_two_priority_networks"]
    # But synthetic must never pass production gate
    assert report.passed is False
    assert any("Synthetic" in n for n in report.notes)

    out = tmp_path / "m0_report.json"
    write_m0_report(report, out)
    loaded = load_m0_report(out)
    assert loaded.n_subjects == 150


def test_permutation_null_random_labels_lower_effect():
    rng_df = generate_synthetic_viability_df(n_subjects=80, random_state=1)
    real = analyze_network(rng_df, "net_Vis", n_permutations=100, random_state=1)

    shuffled = rng_df.copy()
    shuffled["age_band"] = shuffled["age_band"].sample(frac=1, random_state=99).values
    shuffled["sex"] = shuffled["sex"].sample(frac=1, random_state=99).values
    broken = analyze_network(shuffled, "net_Vis", n_permutations=50, random_state=2)
    # Real structured data should have >= shuffled demographic R² on average
    assert real.r2_demographic_given_site >= broken.r2_demographic_given_site - 0.05


def test_single_site_hcp_pilot_does_not_crash():
    """HCP-only cloud runs have one site_id — site one-hot must not be empty."""
    df = generate_synthetic_viability_df(n_subjects=5, random_state=3)
    df["site_id"] = "HCP"
    result = analyze_network(df, "net_Vis", n_permutations=20, random_state=3)
    assert result.r2_site_only == 0.0
    assert result.n_subjects == 5


def test_evaluate_go_criteria_requires_two_networks():
    from data_prep.schemas import M0NetworkResult

    results = [
        M0NetworkResult(
            network_name="Vis",
            network_id=1,
            r2_demographic_given_site=0.02,
            r2_site_only=0.01,
            r2_demographic_only=0.015,
            permutation_p=0.03,
            permutation_null_95=0.01,
            n_subjects=100,
        ),
        M0NetworkResult(
            network_name="Default",
            network_id=7,
            r2_demographic_given_site=0.015,
            r2_site_only=0.005,
            r2_demographic_only=0.01,
            permutation_p=0.04,
            permutation_null_95=0.008,
            n_subjects=100,
        ),
    ]
    go = evaluate_go_criteria(results)
    assert go["variance_in_two_priority_networks"]
    assert go["permutation_significant_in_two_networks"]


def test_load_subjects_csv_missing_columns(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"subject_id": ["a"]}).to_csv(csv_path, index=False)
    from data_prep.viability_partition import load_subjects_csv

    with pytest.raises(ValueError, match="missing required columns"):
        load_subjects_csv(csv_path)


def test_m0_passed_for_training_rejects_single_site(tmp_path: Path):
    from data_prep.schemas import M0NetworkResult, M0Report
    from data_prep.viability_partition import m0_passed_for_training, write_m0_report

    report = M0Report(
        passed=True,
        is_synthetic=False,
        multi_site_validated=False,
        n_sites=1,
        data_source="hcp_only",
        n_subjects=100,
        n_permutations=100,
        networks=[
            M0NetworkResult(
                network_name="Vis",
                network_id=1,
                r2_demographic_given_site=0.02,
                r2_site_only=0.0,
                r2_demographic_only=0.02,
                permutation_p=0.01,
                permutation_null_95=0.005,
                n_subjects=100,
            )
        ],
        go_criteria={"variance_in_two_priority_networks": True},
        notes=["hcp_sex_only_probe", "not_production_gate"],
    )
    path = tmp_path / "m0_report.json"
    write_m0_report(report, path)
    assert m0_passed_for_training(path) is False


def test_m0_passed_for_training_accepts_multi_site(tmp_path: Path):
    from data_prep.schemas import M0NetworkResult, M0Report
    from data_prep.viability_partition import m0_passed_for_training, write_m0_report

    report = M0Report(
        passed=True,
        is_synthetic=False,
        multi_site_validated=True,
        n_sites=2,
        data_source="hcp_camcan",
        n_subjects=200,
        n_permutations=100,
        networks=[
            M0NetworkResult(
                network_name="Vis",
                network_id=1,
                r2_demographic_given_site=0.02,
                r2_site_only=0.01,
                r2_demographic_only=0.015,
                permutation_p=0.01,
                permutation_null_95=0.005,
                n_subjects=200,
            )
        ],
        go_criteria={
            "variance_in_two_priority_networks": True,
            "permutation_significant_in_two_networks": True,
            "demographic_beats_site_in_one_network": True,
        },
        notes=[],
    )
    path = tmp_path / "m0_report.json"
    write_m0_report(report, path)
    assert m0_passed_for_training(path) is True
