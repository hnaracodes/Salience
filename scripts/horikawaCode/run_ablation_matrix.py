"""Run feature_set x label_framing LOVO ablation matrix on ready corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.ablation import ABLATION_FEATURE_SETS, LABEL_FRAMINGS, run_ablation_matrix
from scout_core.horikawaCode.labels import DEFAULT_LABEL_CACHE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "ready_all.json",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal",
    )
    parser.add_argument("--labels-cache", type=Path, default=DEFAULT_LABEL_CACHE)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "horikawa_decoding.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "reports" / "ablation_matrix.json",
    )
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=ABLATION_FEATURE_SETS,
        help="Feature spec to include; repeat to run a subset",
    )
    parser.add_argument(
        "--label-framing",
        action="append",
        choices=LABEL_FRAMINGS,
        help="Label framing to include; repeat to run a subset",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    alphas = [float(a) for a in cfg.get("ridge", {}).get("alphas", [0.1, 1.0, 10.0])]
    cv_cfg = cfg.get("cv") or {}
    n_perm = args.n_permutations or int(cv_cfg.get("n_permutations", 100))
    n_bootstrap = args.n_bootstrap or int(cv_cfg.get("n_bootstrap", 500))

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = list(payload.get("clips") or [])
    corpus = payload.get("corpus", args.manifest.stem)

    matrix = run_ablation_matrix(
        clips,
        intermediates_dir=args.intermediates_dir,
        labels_cache=args.labels_cache,
        corpus=corpus,
        alphas=alphas,
        n_permutations=n_perm,
        n_bootstrap=n_bootstrap,
        feature_sets=tuple(args.feature_set or ABLATION_FEATURE_SETS),
        label_framings=tuple(args.label_framing or LABEL_FRAMINGS),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(json.dumps({"best_cell": matrix["best_cell"], "n_significant_p005": matrix["n_significant_p005"]}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
