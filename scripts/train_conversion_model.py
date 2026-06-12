#!/usr/bin/env python3
"""Train conversion intent model from labeled sessions.

Usage:
    python scripts/train_conversion_model.py --labels scout_data/validation/conversion_labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activation_store import SESSIONS_DIR
from scout_core.conversion_model import LogisticConversionModel, extract_session_features, load_training_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "scout_data" / "models" / "conversion_v1.npz")
    parser.add_argument("--calibration-id", default="conversion_v1")
    args = parser.parse_args()

    labels = load_training_labels(args.labels)
    if len(labels) < 2:
        raise SystemExit("Need at least 2 labeled sessions to train.")

    import numpy as np

    X_rows = []
    y_rows = []
    for sid, label in labels.items():
        bundle_path = SESSIONS_DIR / sid / "analysis_bundle.json"
        if not bundle_path.is_file():
            print(f"Skip {sid}: no bundle", file=sys.stderr)
            continue
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        X_rows.append(extract_session_features(bundle))
        y_rows.append(float(label))

    if len(X_rows) < 2:
        raise SystemExit("Insufficient bundles for training.")

    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=np.float64)
    model = LogisticConversionModel()
    model.fit(X, y, calibration_id=args.calibration_id)
    model.save(args.output)
    print(f"Saved model: {args.output} (n={len(y_rows)})")


if __name__ == "__main__":
    main()
