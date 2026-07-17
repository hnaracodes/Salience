from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data_prep.schemas import AgeBand, Sex, SubjectMetadataRow
from data_prep.viability_partition import age_to_band


def load_cluster_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def harmonize_metadata(raw_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Merge dataset-specific CSV/TSV files into unified metadata parquet.

    Expected raw_dir layout (any subset):
      hcp_7t_subjects.csv
      camcan_subjects.csv
      nndb_subjects.tsv
    """
    frames: list[pd.DataFrame] = []
    mapping = {
        "hcp_7t_subjects.csv": "hcp_7t",
        "camcan_subjects.csv": "camcan",
        "nndb_subjects.tsv": "nndb",
        "cneuromod_participants.tsv": "cneuromod",
        "forrest_subjects.tsv": "forrest",
    }
    for fname, dataset in mapping.items():
        fpath = raw_dir / fname
        if not fpath.is_file():
            continue
        sep = "\t" if fname.endswith(".tsv") else ","
        part = pd.read_csv(fpath, sep=sep)
        part = _normalize_dataset(part, dataset)
        frames.append(part)

    if not frames:
        raise FileNotFoundError(f"No metadata files found in {raw_dir}")

    df = pd.concat(frames, ignore_index=True)
    if df["subject_id"].duplicated().any():
        raise ValueError("subject_id must be unique after harmonization")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def _normalize_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    colmap = {
        "participant_id": "source_subject_id",
        "subject": "source_subject_id",
        "Subject": "source_subject_id",
        "Gender": "sex",
        "gender": "sex",
        "Sex": "sex",
        "Age": "age",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})
    if "source_subject_id" not in df.columns and "subject_id" in df.columns:
        df["source_subject_id"] = df["subject_id"]
    if "subject_id" not in df.columns:
        df["subject_id"] = df["source_subject_id"].astype(str).map(lambda s: f"{dataset}_{s}")

    if "age" in df.columns:
        df["age_band"] = df["age"].astype(float).map(age_to_band)
    elif "age_band" not in df.columns:
        raise ValueError(f"{dataset}: need age or age_band column")

    df["sex"] = df.get("sex", "unknown").astype(str).str.lower()
    df.loc[~df["sex"].isin(["female", "male", "unknown"]), "sex"] = "unknown"
    df["dataset"] = dataset
    df["site_id"] = df.get("site_id", dataset).astype(str)
    df["metadata_quality"] = df.get("metadata_quality", "harmonized")
    df["consent_scope"] = df.get("consent_scope", None)
    df["dua_version"] = df.get("dua_version", None)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            SubjectMetadataRow(
                subject_id=str(row["subject_id"]),
                source_subject_id=str(row["source_subject_id"]),
                dataset=dataset,  # type: ignore[arg-type]
                age=float(row["age"]) if "age" in row and pd.notna(row["age"]) else None,
                age_band=str(row["age_band"]),  # type: ignore[arg-type]
                sex=str(row["sex"]),  # type: ignore[arg-type]
                site_id=str(row["site_id"]),
                metadata_quality=str(row.get("metadata_quality", "harmonized")),
                consent_scope=row.get("consent_scope"),
                dua_version=row.get("dua_version"),
            ).model_dump()
        )
    return pd.DataFrame(rows)


def assign_clusters(
    metadata_df: pd.DataFrame,
    clusters_yaml: Path,
    output_dir: Path,
) -> pd.DataFrame:
    cfg = load_cluster_config(clusters_yaml)
    min_n = int(cfg.get("min_subjects_per_cluster", 20))
    members: list[dict[str, Any]] = []

    for cluster in cfg.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        label = cluster["label"]
        where = cluster["where"]
        mask = pd.Series(True, index=metadata_df.index)
        for key, val in where.items():
            mask &= metadata_df[key].astype(str) == str(val)
        sub = metadata_df[mask]
        suppressed = len(sub) < min_n
        for _, row in sub.iterrows():
            members.append(
                {
                    "subject_id": row["subject_id"],
                    "cluster_id": cluster_id,
                    "cluster_label": label,
                    "suppressed": suppressed,
                    "dataset": row["dataset"],
                    "site_id": row["site_id"],
                }
            )

    out = pd.DataFrame(members)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_dir / "cluster_members.parquet", index=False)
    demo = {
        c["cluster_id"]: {
            "label": c["label"],
            "n_subjects": int((out["cluster_id"] == c["cluster_id"]).sum()) if not out.empty else 0,
            "suppressed": bool(
                (out["cluster_id"] == c["cluster_id"]).sum() < min_n if not out.empty else True
            ),
        }
        for c in cfg.get("clusters", [])
    }
    (output_dir / "cluster_demographics.json").write_text(
        json.dumps(demo, indent=2), encoding="utf-8"
    )
    return out
