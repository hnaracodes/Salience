#!/usr/bin/env python3
"""Offline neuro analysis: parcellation → norms → calibrated rules → SQLite + analysis_bundle.json."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activation_store import DB_PATH, PROJECT_ROOT  # noqa: E402
from scout_core.aggregate import network_timeseries, parcel_timeseries  # noqa: E402
from scout_core.constants import network_names_for_ids  # noqa: E402
from scout_core.norms import zscore_network, zscore_roi  # noqa: E402
from scout_core.parcellation import dense_parcel_labels, load_vertex_table, parcel_to_network_map  # noqa: E402
from scout_core.schemas import AnalysisBundle, ThresholdContext, analysis_bundle_path  # noqa: E402
from scout_core.storage_migrations import (  # noqa: E402
    clear_session_neuro_rows,
    ensure_neuro_schema,
    insert_network_timeseries_batch,
    insert_roi_timeseries_batch,
    insert_threshold_hits,
)
from scout_core.threshold_engine import evaluate_rules  # noqa: E402


def _load_norm_bundle(conn: sqlite3.Connection, norm_id: str) -> tuple[Path, Path]:
    row = conn.execute(
        "SELECT path_roi_parquet, path_network_parquet FROM norm_bundle WHERE norm_id = ?",
        (norm_id,),
    ).fetchone()
    if not row:
        raise SystemExit(
            f"norm_id {norm_id!r} not registered. Run: python scripts/compute_norms.py --norm-id {norm_id} ..."
        )
    return Path(row[0]), Path(row[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--norm-id", required=True)
    parser.add_argument(
        "--vertex-csv",
        type=Path,
        default=ROOT / "configs" / "vertex_regions.csv",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=ROOT / "configs" / "calibrated_rules.yaml",
    )
    parser.add_argument(
        "--insight-catalog",
        type=Path,
        default=ROOT / "configs" / "insight_catalog.yaml",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--reducer", choices=("mean", "mean_abs", "median"), default="mean")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    ensure_neuro_schema(conn)
    sess = conn.execute(
        "SELECT preds_npz_path FROM sessions WHERE id = ?",
        (args.session_id,),
    ).fetchone()
    if not sess:
        conn.close()
        raise SystemExit(f"Unknown session_id {args.session_id}")

    preds_path = PROJECT_ROOT / sess[0]
    raw = np.load(preds_path)
    preds = np.asarray(raw["preds"], dtype=np.float32)

    roi_parquet, net_parquet = _load_norm_bundle(conn, args.norm_id)
    roi_norms = pd.read_parquet(roi_parquet)
    net_norms = pd.read_parquet(net_parquet)

    table = load_vertex_table(args.vertex_csv)
    vp = dense_parcel_labels(table, preds.shape[1])
    pmap = parcel_to_network_map(table)

    parcel_ts, parcel_ids = parcel_timeseries(preds, vp, reducer=args.reducer)
    net_ts, net_ids = network_timeseries(parcel_ts, parcel_ids, pmap, reducer=args.reducer)

    z_p = zscore_roi(parcel_ts, roi_norms, parcel_ids)
    z_n = zscore_network(net_ts, net_norms, net_ids)
    names = network_names_for_ids(net_ids)

    clear_session_neuro_rows(conn, args.session_id, args.norm_id)
    insert_roi_timeseries_batch(
        conn,
        session_id=args.session_id,
        norm_id=args.norm_id,
        parcel_ts=parcel_ts,
        z_parcel=z_p,
        parcel_ids=parcel_ids,
    )
    insert_network_timeseries_batch(
        conn,
        session_id=args.session_id,
        norm_id=args.norm_id,
        net_ts=net_ts,
        z_net=z_n,
        network_ids=net_ids,
    )

    ctx = ThresholdContext(
        session_id=args.session_id,
        fps=args.fps,
        network_names=names,
        network_ts=net_ts.astype(float).tolist(),
        z_network=z_n.astype(float).tolist(),
    )
    hits = evaluate_rules(ctx, args.rules, args.insight_catalog)

    hit_rows = [
        {
            "session_id": h.session_id,
            "rule_id": h.rule_id,
            "t_start": h.t_start,
            "t_end": h.t_end,
            "networks_json": h.networks_json,
            "evidence_json": h.evidence_json,
            "confidence": h.confidence,
            "insight_key": h.insight_key,
            "insight_text": h.insight_text,
        }
        for h in hits
    ]
    insert_threshold_hits(conn, hit_rows)

    bundle = AnalysisBundle(
        session_id=args.session_id,
        norm_id=args.norm_id,
        fps=args.fps,
        parcel_ids=[int(x) for x in parcel_ids.tolist()],
        network_ids=[int(x) for x in net_ids],
        network_names=names,
        threshold_hits=hit_rows,
    )
    session_dir = PROJECT_ROOT / "scout_data" / "sessions" / args.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    out_json = analysis_bundle_path(session_dir)
    out_json.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    conn.commit()
    conn.close()
    print(f"Wrote neuro tables + {len(hit_rows)} threshold hits")
    print(out_json)


if __name__ == "__main__":
    main()
