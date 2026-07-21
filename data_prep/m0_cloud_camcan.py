from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd

from data_prep.viability_partition import NET_COLUMN_PREFIX, age_to_band
from scout_core.aggregate import network_timeseries, parcel_timeseries
from scout_core.constants import YEO7_NAMES
from scout_core.parcellation import load_vertex_table, parcel_to_network_map

OPENNEURO_SNAPSHOT = "1.0.0"
OPENNEURO_DS221 = (
    f"https://openneuro.org/crn/datasets/ds000221/snapshots/{OPENNEURO_SNAPSHOT}/files"
)
CAMCAN_DEMO_URL = f"{OPENNEURO_DS221}/participants.tsv"
# OpenNeuro ds000221 mirror ships rest fMRI only (no movie BOLD); use AP run-01 rest.
CAMCAN_REST_TASK = "task-rest_acq-AP_run-01"
CAMCAN_DEMO_URL_LEGACY = (
    "https://raw.githubusercontent.com/CamCAN/CC700/master/"
    "meta/subjects_data_metadata.csv"
)


def _openneuro_colon_url(*parts: str) -> str:
    """OpenNeuro CRN file endpoint encodes path segments with colons."""
    return f"{OPENNEURO_DS221}/{':'.join(parts)}"


def _openneuro_url(relpath: str) -> str:
    return f"{OPENNEURO_DS221}/{relpath.lstrip('/')}"


def _openneuro_file_exists(url: str, *, timeout: int = 30) -> bool:
    try:
        req = urlopen(url, timeout=timeout)
        with req:
            req.read(1)
        return True
    except Exception:
        return False


def _parse_camcan_age(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    if re.match(r"^\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return None


def fetch_camcan_demographics(*, bundled_path: Path | None = None) -> pd.DataFrame:
    """Load Cam-CAN participant age/sex from bundled TSV or OpenNeuro."""
    if bundled_path and bundled_path.is_file():
        df = pd.read_csv(bundled_path, sep="\t")
        return _normalize_camcan_demographics(df)

    last_err: Exception | None = None
    for url in (CAMCAN_DEMO_URL, CAMCAN_DEMO_URL_LEGACY):
        try:
            with urlopen(url, timeout=120) as resp:
                raw = resp.read()
            sep = "\t" if url.endswith(".tsv") else ","
            df = pd.read_csv(io.BytesIO(raw), sep=sep)
            return _normalize_camcan_demographics(df)
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Could not load Cam-CAN demographics: {last_err}")


def _normalize_camcan_demographics(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "participant": "source_subject_id",
        "participant_id": "source_subject_id",
        "subject": "source_subject_id",
        "ID": "source_subject_id",
        "age": "age",
        "Age": "age",
        "age (5-year bins)": "age",
        "sex": "sex",
        "Sex": "sex",
        "gender": "sex",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    if "source_subject_id" not in df.columns:
        for c in df.columns:
            if re.search(r"sub|id|participant", c, re.I):
                df["source_subject_id"] = df[c].astype(str)
                break
    if "source_subject_id" not in df.columns:
        raise ValueError("Cam-CAN demographics missing subject id column")
    df["source_subject_id"] = df["source_subject_id"].astype(str).str.replace("^sub-", "", regex=True).str.strip()
    if "age" not in df.columns:
        raise ValueError("Cam-CAN demographics missing age column")
    df["age"] = df["age"].map(_parse_camcan_age)
    df = df[df["age"].notna()].copy()
    df["age_band"] = df["age"].astype(float).map(age_to_band)
    df["sex"] = df.get("sex", "unknown").astype(str).str.lower()
    df.loc[df["sex"].isin(["f", "female"]), "sex"] = "female"
    df.loc[df["sex"].isin(["m", "male"]), "sex"] = "male"
    df.loc[~df["sex"].isin(["female", "male", "unknown"]), "sex"] = "unknown"
    df["subject_id"] = df["source_subject_id"].map(lambda s: f"camcan_{s}")
    df["dataset"] = "camcan"
    df["site_id"] = "camcan"
    return df


def discover_camcan_rest_bold_url(subject: str) -> str | None:
    """Return OpenNeuro colon-path URL for Cam-CAN rest BOLD (movie not on ds000221 mirror)."""
    sub = subject if subject.startswith("sub-") else f"sub-{subject}"
    for ses in ("ses-01", "ses-02"):
        fname = f"{sub}_{ses}_{CAMCAN_REST_TASK}_bold.nii.gz"
        url = _openneuro_colon_url(sub, ses, "func", fname)
        json_url = url.replace(".nii.gz", ".json")
        if _openneuro_file_exists(json_url):
            return url
    return None


def list_camcan_rest_subjects(demographics: pd.DataFrame, *, limit: int | None = None) -> list[str]:
    """Return OpenNeuro subject ids (sub-010*) with rest fMRI on ds000221."""
    subjects: list[str] = []
    for sid in demographics["source_subject_id"].astype(str):
        sub = f"sub-{sid}" if not str(sid).startswith("sub-") else str(sid)
        if discover_camcan_rest_bold_url(sub):
            subjects.append(sub)
        if limit and len(subjects) >= limit:
            break
    return subjects


def list_camcan_movie_subjects(demographics: pd.DataFrame, *, limit: int | None = None) -> list[str]:
    """Alias: OpenNeuro ds000221 has rest only; movie lives on Cam-CAN portal, not here."""
    return list_camcan_rest_subjects(demographics, limit=limit)


def discover_movie_bold_url(subject: str) -> str | None:
    """Alias for rest BOLD discovery on OpenNeuro ds000221."""
    return discover_camcan_rest_bold_url(subject)


def volume_ts_to_network_features(
    vol_ts: np.ndarray,
    vertex_csv: Path,
    *,
    affine: np.ndarray | None = None,
    tr: float = 2.47,
) -> dict[str, float]:
    """
  Project MNI volume timeseries to fsaverage5 vertices via per-TR mean pooling proxy.

  For M0 scalars we use nilearn vol_to_surf on temporal mean volume — sufficient for
  between-subject variance partition at network level.
  """
    from nilearn import datasets, image, surface

    vol_ts = np.asarray(vol_ts, dtype=np.float32)
    if vol_ts.ndim != 4:
        raise ValueError(f"Expected 4D volume [X,Y,Z,T], got {vol_ts.shape}")
    mean_vol = vol_ts.mean(axis=-1)
    import nibabel as nib

    mean_img = nib.Nifti1Image(mean_vol, affine if affine is not None else np.eye(4))
    mni_template = datasets.load_mni152_template(resolution=2)
    mean_mni = image.resample_to_img(mean_img, mni_template, interpolation="continuous")
    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    lh = surface.vol_to_surf(mean_mni, fsavg.pial_left)
    rh = surface.vol_to_surf(mean_mni, fsavg.pial_right)
    vertex_ts = np.concatenate([lh, rh], axis=0)[np.newaxis, :]

    table = load_vertex_table(vertex_csv)
    parcel_ts, parcel_ids = parcel_timeseries(vertex_ts, table.parcel_id, reducer="mean_abs")
    p2n = parcel_to_network_map(table)
    net_ts, net_ids = network_timeseries(parcel_ts, parcel_ids, p2n, reducer="mean_abs")
    id_to_name = {i + 1: name for i, name in enumerate(YEO7_NAMES)}
    out: dict[str, float] = {}
    for j, nid in enumerate(net_ids):
        name = id_to_name.get(int(nid))
        if not name:
            continue
        val = float(np.mean(np.abs(net_ts[:, j])))
        if not np.isfinite(val):
            return {}
        out[f"{NET_COLUMN_PREFIX}{name}"] = val
    if len(out) < len(YEO7_NAMES):
        return {}
    return out


def fetch_camcan_subject_features(
    subject: str,
    demographics: pd.DataFrame,
    *,
    vertex_csv: Path,
    bold_url: str | None = None,
) -> dict[str, Any] | None:
    import nibabel as nib

    import os
    import tempfile

    url = bold_url or discover_movie_bold_url(subject)
    if not url:
        return None
    with urlopen(url, timeout=300) as resp:
        blob = resp.read()
    fd, path = tempfile.mkstemp(suffix=".nii.gz")
    try:
        os.write(fd, blob)
        os.close(fd)
        img = nib.load(path)
        data = np.asarray(img.get_fdata(dtype=np.float32))
        feats = volume_ts_to_network_features(data, vertex_csv, affine=img.affine)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    if not feats:
        return None
    source_id = subject.removeprefix("sub-")
    sub = demographics[demographics["source_subject_id"].astype(str) == source_id]
    if sub.empty:
        return None
    row = sub.iloc[0]
    out = row.to_dict()
    out.update(feats)
    return out


def build_camcan_subject_rows(
    *,
    vertex_csv: Path,
    limit: int | None = None,
    demographics_path: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Build Cam-CAN subject rows for M0 merge. Returns (df, skipped)."""
    demo = fetch_camcan_demographics(bundled_path=demographics_path)
    subjects = list_camcan_rest_subjects(demo, limit=limit)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for sub in subjects:
        try:
            row = fetch_camcan_subject_features(sub, demo, vertex_csv=vertex_csv)
            if row:
                rows.append(row)
            else:
                skipped.append({"subject": sub, "reason": "no_bold_or_demographics"})
        except Exception as exc:
            skipped.append({"subject": sub, "reason": str(exc)})
    return pd.DataFrame(rows), skipped
