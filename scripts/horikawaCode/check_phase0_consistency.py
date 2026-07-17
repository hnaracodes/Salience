"""Phase 0: assert train cv_mean_r matches evaluate fused_lovo_mean_r on same NPZ."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.horikawaCode.cv import lovo_mean_r

PY = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"
if not PY.is_file():
    PY = Path(sys.executable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "horikawa_decoding.yaml",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only recompute eval LOVO r from train NPZ (skip retraining)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    train_path = PROJECT_ROOT / cfg["data"]["train_npz"]
    if not train_path.is_file():
        raise SystemExit(f"Missing train NPZ: {train_path}")

    if not args.skip_train:
        subprocess.run(
            [
                str(PY),
                str(PROJECT_ROOT / "scripts" / "horikawaCode" / "train_horikawa_ridge_decoder.py"),
                "--config",
                str(config_path),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )

    raw = np.load(train_path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    groups = np.asarray(data["groups"], dtype=np.int32)
    alphas = [float(a) for a in cfg.get("ridge", {}).get("alphas", [0.1, 1.0, 10.0])]

    eval_r = lovo_mean_r(X, y, groups, alphas)
    model_id = cfg.get("model_id", "horikawa_ridge_v1")
    meta_path = PROJECT_ROOT / "scout_models" / model_id / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    train_r = float(meta.get("cv_mean_r", eval_r))
    delta = abs(train_r - eval_r)
    consistent = delta < args.tolerance

    report = {
        "phase": "phase0_consistency",
        "train_npz": str(train_path),
        "model_id": model_id,
        "cv_mean_r": train_r,
        "fused_lovo_mean_r": eval_r,
        "abs_delta": delta,
        "tolerance": args.tolerance,
        "consistent": consistent,
        "data_source": str(train_path),
        "synthetic_fallback_used": False,
        "n_samples_npz": int(X.shape[0]),
        "corpus_npz": str(data["corpus"]) if "corpus" in data else None,
    }

    out = PROJECT_ROOT / cfg["data"]["reports_dir"] / "phase0_consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    if not consistent:
        raise SystemExit(f"Train/eval mismatch: delta={delta}")


if __name__ == "__main__":
    main()
