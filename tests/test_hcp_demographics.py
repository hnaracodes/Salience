from __future__ import annotations

import pandas as pd

from data_prep.hcp_demographics import _normalize_hcp_behavioral


def test_hcp_age_in_yrs_maps_to_age_band():
    df = pd.DataFrame(
        {
            "Subject": ["100206", "100307"],
            "Gender": ["M", "F"],
            "Age_in_Yrs": [28.5, 32.0],
        }
    )
    out = _normalize_hcp_behavioral(df)
    assert out.loc[out["source_subject_id"] == "100206", "age_band"].iloc[0] == "young_adult"
    assert out.loc[out["source_subject_id"] == "100307", "sex"].iloc[0] == "female"


def test_hcp_gender_normalization():
    df = pd.DataFrame({"Subject": ["1"], "Gender": ["M"], "Age_in_Yrs": [30]})
    out = _normalize_hcp_behavioral(df)
    assert out.iloc[0]["sex"] == "male"


def test_open_demographics_sex_codes():
    df = pd.DataFrame({"participant_id": ["sub-100206"], "age": [27], "sex": [1]})
    out = _normalize_hcp_behavioral(df)
    assert out.iloc[0]["sex"] == "male"
    assert out.iloc[0]["source_subject_id"] == "100206"
