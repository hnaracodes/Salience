from __future__ import annotations

from pathlib import Path

import yaml

from data_prep.hcp_s3 import HcpManifest, subject_dtseries_key


def test_manifest_loads():
    manifest_path = Path(__file__).resolve().parents[1] / "configs/hcp_7t_movie_manifest.yaml"
    manifest = HcpManifest.from_yaml(manifest_path, prefix="HCP_1200")
    assert len(manifest.movie_runs) == 4
    assert manifest.movie_runs[0].folder == "tfMRI_MOVIE1_7T_AP"
    assert len(manifest.subject_ids) > 100


def test_dtseries_key_exact_ap_pa():
    manifest_path = Path(__file__).resolve().parents[1] / "configs/hcp_7t_movie_manifest.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = HcpManifest.from_yaml(manifest_path, prefix="HCP_1200")
    key1 = subject_dtseries_key(manifest, "126426", manifest.movie_runs[0])
    assert key1.endswith("tfMRI_MOVIE1_7T_AP_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries.nii")
    key2 = subject_dtseries_key(manifest, "126426", manifest.movie_runs[1])
    assert "tfMRI_MOVIE2_7T_PA" in key2
    assert raw["movie_runs"][1]["folder"] == "tfMRI_MOVIE2_7T_PA"


def test_prefix_candidates_present():
    manifest_path = Path(__file__).resolve().parents[1] / "configs/hcp_7t_movie_manifest.yaml"
    manifest = HcpManifest.from_yaml(manifest_path)
    assert "HCP_1200" in manifest.prefix_candidates
