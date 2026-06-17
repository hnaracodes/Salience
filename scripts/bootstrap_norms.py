#!/usr/bin/env python3
"""Generate synthetic bootstrap norm bundles for pipeline testing.

Produces roi_norms.parquet and network_norms.parquet derived from
Gaussian-drawn preds[T, V] (zero-mean unit-variance per vertex).
Use ONLY for pipeline validation; replace with real empirical norms
once modal run tribe.py::record has produced held-out preds.npz clips.

Usage:
    python scripts/bootstrap_norms.py --norm-id synthetic_bootstrap_v1
    python scripts/bootstrap_norms.py --norm-id synthetic_bootstrap_v1 --n-clips 5 --t-steps 50
"""

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

from activation_store import DB_PATH  # noqa: E402
from scout_core.aggregate import network_timeseries, parcel_timeseries  # noqa: E402
from scout_core.constants import network_names_for_ids  # noqa: E402
from scout_core.parcellation import dense_parcel_labels, load_vertex_table, parcel_to_network_map  # noqa: E402
from scout_core.storage_migrations import ensure_neuro_schema, register_norm_bundle  # noqa: E402


def _build_synthetic_parquet(
    vertex_csv: Path,
    n_clips: int,
    t_steps: int,
    reducer: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    table = load_vertex_table(vertex_csv)
    V = table.n_vertices
    pmap = parcel_to_network_map(table)

    from collections import defaultdict
    by_parcel: dict[int, list[float]] = defaultdict(list)
    by_net: dict[int, list[float]] = defaultdict(list)
    parcel_ids_ref = None

    for _ in range(n_clips):
        preds = rng.standard_normal((t_steps, V)).astype(np.float32)
        vp = dense_parcel_labels(table, V)
        pts, pids = parcel_timeseries(preds, vp, reducer=reducer)
        if parcel_ids_ref is None:
            parcel_ids_ref = pids
        nets, nids = network_timeseries(pts, pids, pmap, reducer=reducer)
        for j in range(pids.size):
            by_parcel[int(pids[j])].extend(pts[:, j].tolist())
        for j, nid in enumerate(nids):
            by_net[int(nid)].extend(nets[:, j].tolist())

    roi_rows = []
    for pid, vals in sorted(by_parcel.items()):
        a = np.asarray(vals, dtype=np.float64)
        std = float(a.std()) if a.size > 1 else 1e-8
        roi_rows.append({
            "parcel_id": pid,
            "mean": float(a.mean()),
            "std": max(std, 1e-8),
            "n_samples": int(a.size),
            "q05": float(np.quantile(a, 0.05)),
            "q50": float(np.quantile(a, 0.50)),
            "q95": float(np.quantile(a, 0.95)),
        })

    net_rows = []
    for nid, vals in sorted(by_net.items()):
        a = np.asarray(vals, dtype=np.float64)
        std = float(a.std()) if a.size > 1 else 1e-8
        net_rows.append({
            "yeo_network_id": nid,
            "mean": float(a.mean()),
            "std": max(std, 1e-8),
            "n_samples": int(a.size),
            "q05": float(np.quantile(a, 0.05)),
            "q50": float(np.quantile(a, 0.50)),
            "q95": float(np.quantile(a, 0.95)),
        })

    return pd.DataFrame(roi_rows), pd.DataFrame(net_rows), n_clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-id", required=True)
    parser.add_argument(
        "--vertex-csv",
        type=Path,
        default=ROOT / "configs" / "vertex_regions.csv",
    )
    parser.add_argument("--n-clips", type=int, default=10,
                        help="Number of synthetic preds[T,V] draws")
    parser.add_argument("--t-steps", type=int, default=30,
                        help="Timesteps per synthetic clip")
    parser.add_argument("--reducer", choices=("mean", "mean_abs", "median"), default="mean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-register-db", action="store_true")
    args = parser.parse_args()

    if not args.vertex_csv.is_file():
        raise SystemExit(
            f"vertex_regions.csv not found at {args.vertex_csv}. "
            "Run: python scripts/build_vertex_regions_csv.py demo --n-vertices <V>"
        )

    rng = np.random.default_rng(args.seed)
    roi_df, net_df, n_clips = _build_synthetic_parquet(
        args.vertex_csv,
        n_clips=args.n_clips,
        t_steps=args.t_steps,
        reducer=args.reducer,
        rng=rng,
    )

    out_dir = ROOT / "scout_norms" / args.norm_id
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_path = out_dir / "roi_norms.parquet"
    net_path = out_dir / "network_norms.parquet"
    roi_df.to_parquet(roi_path, index=False)
    net_df.to_parquet(net_path, index=False)

    meta = {
        "norm_id": args.norm_id,
        "source": "synthetic_bootstrap",
        "n_clips": n_clips,
        "t_steps_per_clip": args.t_steps,
        "reducer": args.reducer,
        "seed": args.seed,
        "network_names": network_names_for_ids(net_df["yeo_network_id"].tolist()),
        "parcellation_csv": args.vertex_csv.resolve().relative_to(ROOT).as_posix()
        if args.vertex_csv.resolve().is_relative_to(ROOT)
        else str(args.vertex_csv.resolve()),
        "leakage_note": (
            "SYNTHETIC BOOTSTRAP — replace with empirical norms from held-out "
            "naturalistic clips before production use."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if not args.no_register_db:
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
        print(f"Registered norm_id={args.norm_id!r} in {DB_PATH}")

    print(f"Wrote {roi_path} ({len(roi_df)} parcels)")
    print(f"Wrote {net_path} ({len(net_df)} networks)")
    print(f"Meta: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
