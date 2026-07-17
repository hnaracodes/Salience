from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

from data_prep.hcp_s3 import behavioral_csv_keys, download_s3_to_bytes
from data_prep.viability_partition import age_to_band

DEFAULT_OPEN_DEMOGRAPHICS_URL = (
    "https://raw.githubusercontent.com/predictive-clinical-neuroscience/"
    "PCNtoolkit-demo/main/data/HCP1200_age_gender.csv"
)


def _normalize_sex(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"f", "female", "2", "2.0"}:
        return "female"
    if s in {"m", "male", "1", "1.0"}:
        return "male"
    return "unknown"


def load_hcp_demographics_open_url(url: str = DEFAULT_OPEN_DEMOGRAPHICS_URL) -> pd.DataFrame:
    """Load unrestricted HCP-YA age/sex from a public mirror (PCNtoolkit-demo)."""
    with urlopen(url, timeout=120) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw))
    colmap = {
        "participant_id": "source_subject_id",
        "Subject": "source_subject_id",
        "sex": "sex_raw",
        "Gender": "sex_raw",
        "age": "age",
        "Age_in_Yrs": "age",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    if "source_subject_id" not in df.columns:
        raise ValueError("Open demographics CSV missing subject id column")
    df["source_subject_id"] = (
        df["source_subject_id"].astype(str).str.replace("^sub-", "", regex=True).str.strip()
    )
    return _normalize_hcp_behavioral(df)


def load_hcp_demographics_df(
    client: Any,
    *,
    bucket: str,
    manifest_prefix: str,
    behavioral_candidates: list[str],
    open_url: str | None = DEFAULT_OPEN_DEMOGRAPHICS_URL,
    bundled_csv_path: Path | None = None,
) -> pd.DataFrame:
    """Fetch HCP demographics: S3 CSV, bundled file, then optional open URL."""
    from data_prep.hcp_s3 import HcpManifest

    manifest = HcpManifest(
        bucket=bucket,
        region="us-east-1",
        prefix=manifest_prefix,
        prefix_candidates=[],
        movie_runs=[],
        behavioral_csv_candidates=behavioral_candidates,
        subject_ids=[],
        parcellation_artifact="",
        demographics_open_url=open_url,
        demographics_bundled_csv=str(bundled_csv_path.name) if bundled_csv_path else None,
    )
    last_err: Exception | None = None
    for key in behavioral_csv_keys(manifest):
        try:
            raw = download_s3_to_bytes(client, bucket, key)
            df = pd.read_csv(io.BytesIO(raw))
            return _normalize_hcp_behavioral(df)
        except Exception as exc:
            last_err = exc
            continue

    if bundled_csv_path and bundled_csv_path.is_file():
        return _normalize_hcp_behavioral(pd.read_csv(bundled_csv_path))

    url = open_url or os.environ.get("HCP_DEMOGRAPHICS_OPEN_URL") or DEFAULT_OPEN_DEMOGRAPHICS_URL
    try:
        return load_hcp_demographics_open_url(url)
    except Exception as open_exc:
        raise RuntimeError(
            f"Could not load HCP behavioral CSV from S3 ({last_err}), "
            f"bundled file ({bundled_csv_path}), or open URL ({open_exc})"
        ) from open_exc


def _normalize_hcp_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "participant_id": "source_subject_id",
        "Subject": "source_subject_id",
        "Subject ID": "source_subject_id",
        "Gender": "sex_raw",
        "Sex": "sex_raw",
        "sex": "sex_raw",
        "Age": "age",
        "Age_in_Yrs": "age",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    if "source_subject_id" not in df.columns:
        for c in df.columns:
            if "subject" in c.lower():
                df["source_subject_id"] = df[c].astype(str)
                break
    if "source_subject_id" not in df.columns:
        raise ValueError("HCP behavioral CSV missing subject id column")

    df["source_subject_id"] = df["source_subject_id"].astype(str).str.replace("^sub-", "", regex=True).str.strip()
    if "sex_raw" in df.columns:
        df["sex"] = df["sex_raw"].map(_normalize_sex)
    else:
        df["sex"] = "unknown"

    if "age" in df.columns and df["age"].notna().any():
        df["age_band"] = df["age"].astype(float).map(age_to_band)
    else:
        df["age_band"] = "young_adult"

    df["subject_id"] = df["source_subject_id"].map(lambda s: f"hcp_7t_{s}")
    df["dataset"] = "hcp_7t"
    df["site_id"] = "hcp_7t"
    return df


def subject_row_from_demographics(
    demographics: pd.DataFrame,
    subject_id: str,
    net_features: dict[str, float],
) -> dict[str, Any]:
    source_id = subject_id.removeprefix("hcp_7t_")
    sub = demographics[demographics["source_subject_id"].astype(str) == str(source_id)]
    if sub.empty:
        meta = {
            "subject_id": f"hcp_7t_{source_id}",
            "source_subject_id": source_id,
            "dataset": "hcp_7t",
            "age_band": "young_adult",
            "sex": "unknown",
            "site_id": "hcp_7t",
        }
    else:
        row = sub.iloc[0]
        meta = {
            "subject_id": str(row.get("subject_id", f"hcp_7t_{source_id}")),
            "source_subject_id": source_id,
            "dataset": "hcp_7t",
            "age_band": str(row["age_band"]),
            "sex": str(row["sex"]),
            "site_id": "hcp_7t",
        }
    meta.update(net_features)
    return meta
