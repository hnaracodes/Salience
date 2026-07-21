#!/usr/bin/env python3
"""Build ROI/network norm bundles from many preds.npz clips (internal empirical norms)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activation_store import DB_PATH, PROJECT_ROOT  # noqa: E402
from scout_core.aggregate import network_timeseries, parcel_timeseries  # noqa: E402
from scout_core.constants import network_names_for_ids  # noqa: E402
from scout_core.parcellation import dense_parcel_labels, load_vertex_table, parcel_to_network_map  # noqa: E402
from scout_core.storage_migrations import ensure_neuro_schema, register_norm_bundle  # noqa: E402


def _collect_parcel_values(
    npz_paths: list[Path],
    *,
    vertex_csv: Path,
    reducer: str,
):
    table = load_vertex_table(vertex_csv)
    parcel_ids_ref = None
    by_parcel: dict[int, list[float]] = defaultdict(list)

    for p in npz_paths:
        raw = np.load(p)
        preds = np.asarray(raw["preds"], dtype=np.float32)
        _, V = preds.shape
        vp = dense_parcel_labels(table, V)
        pts, pids = parcel_timeseries(preds, vp, reducer=reducer)
        if parcel_ids_ref is None:
            parcel_ids_ref = pids
        elif not np.array_equal(pids, parcel_ids_ref):
            raise ValueError(f"parcel id order mismatch in {p}")
        for j in range(pids.size):
            by_parcel[int(pids[j])].extend(pts[:, j].tolist())

    return by_parcel, parcel_ids_ref, table


def _roi_norms_df(by_parcel: dict[int, list[float]]) -> pd.DataFrame:
    rows = []
    for pid, vals in sorted(by_parcel.items()):
        a = np.asarray(vals, dtype=np.float64)
        std = float(a.std()) if a.size > 1 else 1e-8
        rows.append(
            {
                "parcel_id": pid,
                "mean": float(a.mean()),
                "std": max(std, 1e-8),
                "n_samples": int(a.size),
                "q05": float(np.quantile(a, 0.05)),
                "q50": float(np.quantile(a, 0.50)),
                "q95": float(np.quantile(a, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _network_norms_from_clips(
    npz_paths: list[Path],
    *,
    table,
    parcel_ids_ref: np.ndarray,
    parcel_to_net: dict[int, int],
    reducer: str,
) -> pd.DataFrame:
    by_net: dict[int, list[float]] = defaultdict(list)
    for p in npz_paths:
        raw = np.load(p)
        preds = np.asarray(raw["preds"], dtype=np.float32)
        vp = dense_parcel_labels(table, preds.shape[1])
        pts, pids = parcel_timeseries(preds, vp, reducer=reducer)
        if not np.array_equal(pids, parcel_ids_ref):
            raise ValueError(f"parcel id mismatch in {p}")
        nets, net_ids = network_timeseries(pts, pids, parcel_to_net, reducer=reducer)
        for j, nid in enumerate(net_ids):
            by_net[int(nid)].extend(nets[:, j].tolist())

    rows = []
    for nid, vals in sorted(by_net.items()):
        a = np.asarray(vals, dtype=np.float64)
        std = float(a.std()) if a.size > 1 else 1e-8
        rows.append(
            {
                "yeo_network_id": nid,
                "mean": float(a.mean()),
                "std": max(std, 1e-8),
                "n_samples": int(a.size),
                "q05": float(np.quantile(a, 0.05)),
                "q50": float(np.quantile(a, 0.50)),
                "q95": float(np.quantile(a, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-id", required=True)
    parser.add_argument(
        "--vertex-csv",
        type=Path,
        default=ROOT / "configs" / "vertex_regions.csv",
    )
    parser.add_argument("--glob", type=str, default="scout_data/sessions/*/preds.npz")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Session directory name to exclude (repeatable)",
    )
    parser.add_argument("--reducer", choices=("mean", "mean_abs", "median"), default="mean")
    parser.add_argument("--no-register-db", action="store_true")
    args = parser.parse_args()

    excluded = set(args.exclude or [])
    paths = sorted(p for p in ROOT.glob(args.glob) if p.parent.name not in excluded)
    if not paths:
        raise SystemExit(f"No preds.npz matched glob {args.glob} after exclusions")

    by_parcel, parcel_ids_ref, table = _collect_parcel_values(
        paths,
        vertex_csv=args.vertex_csv,
        reducer=args.reducer,
    )
    roi_df = _roi_norms_df(by_parcel)
    pmap = parcel_to_network_map(table)
    net_df = _network_norms_from_clips(
        paths,
        table=table,
        parcel_ids_ref=parcel_ids_ref,
        parcel_to_net=pmap,
        reducer=args.reducer,
    )

    out_dir = ROOT / "scout_norms" / args.norm_id
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_path = out_dir / "roi_norms.parquet"
    net_path = out_dir / "network_norms.parquet"
    roi_df.to_parquet(roi_path, index=False)
    net_df.to_parquet(net_path, index=False)

    meta = {
        "norm_id": args.norm_id,
        "n_clips": len(paths),
        "reducer": args.reducer,
        "network_names_preview": network_names_for_ids(net_df["yeo_network_id"].tolist()),
        "parcellation_csv": str(args.vertex_csv.resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.no_register_db:
        import sqlite3

        conn = sqlite3.connect(DB_PATH)
        ensure_neuro_schema(conn)
        register_norm_bundle(
            conn,
            norm_id=args.norm_id,
            roi_parquet=roi_path,
            network_parquet=net_path,
            meta=meta,
        )
        conn.commit()
        conn.close()

    print(f"Wrote {roi_path} ({len(roi_df)} parcels)")
    print(f"Wrote {net_path} ({len(net_df)} networks)")
    print(f"Meta {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
