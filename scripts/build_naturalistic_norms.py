#!/usr/bin/env python3
"""Build naturalistic_v1 norm bundle from session preds + engagement samples.

Usage:
    python scripts/build_naturalistic_norms.py
    python scripts/build_naturalistic_norms.py --norm-id naturalistic_v1 --min-clips 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activation_store import SESSIONS_DIR


def _collect_engagement_means(session_dirs: list[Path]) -> list[float]:
    means: list[float] = []
    for d in session_dirs:
        bundle_path = d / "analysis_bundle.json"
        if not bundle_path.is_file():
            continue
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        scores = (bundle.get("engagement_track") or {}).get("scores") or []
        numeric = [float(s) for s in scores if s is not None]
        if not numeric:
            act = (bundle.get("activation_track") or {}).get("raw_scores") or []
            numeric = [float(s) for s in act if s is not None]
        if numeric:
            means.append(float(np.mean(numeric)))
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--norm-id", default="naturalistic_v1")
    parser.add_argument("--glob", default="scout_data/sessions/*/preds.npz")
    parser.add_argument("--min-clips", type=int, default=15)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()

    paths = sorted(ROOT.glob(args.glob))
    paths = [p for p in paths if p.parent.name not in set(args.exclude)]
    if not paths:
        raise SystemExit("No preds.npz files found. Capture sessions and run Modal TRIBE first.")
    if len(paths) < args.min_clips:
        raise SystemExit(
            f"Need at least {args.min_clips} preds.npz after exclusions; found {len(paths)}. "
            "Capture more sessions or lower --min-clips only for non-production experiments."
        )

    compute_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "compute_norms.py"),
        "--norm-id",
        args.norm_id,
        "--glob",
        args.glob,
    ]
    for sid in args.exclude:
        compute_cmd.extend(["--exclude", sid])
    subprocess.run(compute_cmd, cwd=ROOT, check=True)

    session_dirs = [p.parent for p in paths]
    engagement_means = _collect_engagement_means(session_dirs)
    out_dir = ROOT / "scout_norms" / args.norm_id
    out_dir.mkdir(parents=True, exist_ok=True)
    eng_path = out_dir / "engagement_samples.json"
    eng_path.write_text(
        json.dumps(
            {
                "norm_id": args.norm_id,
                "n_sessions": len(engagement_means),
                "session_mean_engagement": engagement_means,
                "source": "dual_track_or_activation_fallback",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    meta_path = out_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["engagement_samples_path"] = str(eng_path.name)
        meta["n_engagement_sessions"] = len(engagement_means)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote engagement reference: {eng_path} ({len(engagement_means)} sessions)")
    print(f"Norm bundle: {out_dir}")


if __name__ == "__main__":
    main()
