from __future__ import annotations

from pathlib import Path

from data_prep.metadata_harmonize import assign_clusters, harmonize_metadata


def segment_clusters(
    metadata_parquet: Path,
    clusters_yaml: Path,
    output_dir: Path,
):
    import pandas as pd

    df = pd.read_parquet(metadata_parquet)
    return assign_clusters(df, clusters_yaml, output_dir)
