#!/usr/bin/env python3
"""Aggregate multiple baseline preds.npz clips into scout_data/baseline/ corpus.

Usage:
    python scripts/build_baseline_corpus.py --inputs clip1.npz clip2.npz --corpus-id neutral_browsing_v1
    python scripts/build_baseline_corpus.py --write preds_baseline.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scout_core.baseline_loader import aggregate_baseline_corpus

BASELINE_DIR = ROOT / "scout_data" / "baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[],
        help="Baseline npz files to stack (default: all *.npz in scout_data/baseline/corpus/)",
    )
    parser.add_argument("--corpus-id", default="neutral_browsing_v1")
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Also write merged preds to this path (e.g. scout_data/baseline/preds_baseline.npz)",
    )
    parser.add_argument("--max-trs", type=int, default=None)
    args = parser.parse_args()

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = BASELINE_DIR / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    paths = list(args.inputs)
    if not paths:
        paths = sorted(corpus_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(
            "No baseline npz inputs. Add clips under scout_data/baseline/corpus/ or pass --inputs."
        )

    merged = aggregate_baseline_corpus(paths, max_trs=args.max_trs)
    manifest = {
        "corpus_id": args.corpus_id,
        "files": [p.name if p.parent == corpus_dir else str(p) for p in paths],
        "n_trs": int(merged.shape[0]),
        "n_vertices": int(merged.shape[1]),
        "max_trs": args.max_trs,
    }
    manifest_path = BASELINE_DIR / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path} ({merged.shape[0]} TRs)")

    if args.write:
        out = args.write if args.write.is_absolute() else ROOT / args.write
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, preds=merged.astype(np.float32))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
