#!/usr/bin/env python3
"""Apply demographic mux to a session using preds fallback (local smoke path)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

torch = __import__("torch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demographic mux on session preds")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cluster-ids", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    from activation_store import SESSIONS_DIR
    from data_prep.viability_partition import load_m0_report, m0_passed_for_training
    from salience.mux.inference import build_inference_payload, save_parcel_ts_npz
    from salience.mux.tribe_mux import PredsFallbackMux
    from salience.mux.checkpoints import load_mux_checkpoint

    m0_path = PROJECT_ROOT / "scout_data/demographic/viability/m0_report.json"
    if not m0_passed_for_training(m0_path):
        raise SystemExit(
            "M0 gate not passed — demographic mux disabled. "
            f"See {m0_path}"
        )

    session_dir = SESSIONS_DIR / args.session_id
    preds_path = session_dir / "preds.npz"
    if not preds_path.is_file():
        raise SystemExit(f"Missing preds.npz for session {args.session_id}")

    preds = np.load(preds_path)["preds"].astype(np.float32)
    mux, payload_meta = load_mux_checkpoint(args.checkpoint, device="cpu")
    fallback = PredsFallbackMux(mux.config)
    fallback.mux.load_state_dict(mux.state_dict())
    fallback.eval()

    with torch.no_grad():
        out = fallback(torch.from_numpy(preds), torch.tensor(args.cluster_ids, dtype=torch.long))
    parcel_ts = out.numpy().astype(np.float32)

    out_path = session_dir / "mux_parcel_ts.npz"
    save_parcel_ts_npz(out_path, parcel_ts, args.cluster_ids)

    m0_report = load_m0_report(m0_path) if m0_path.is_file() else None
    meta = build_inference_payload(
        parcel_ts,
        args.cluster_ids,
        checkpoint_path=str(args.checkpoint),
        m0_passed=bool(m0_report and m0_report.passed),
        eval_gate_passed=bool(payload_meta.get("metrics", {}).get("eval_gate_passed", False)),
    )
    (session_dir / "mux_inference.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
