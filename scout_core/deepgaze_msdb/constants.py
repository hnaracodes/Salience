"""Paths and upstream provenance for the DeepGaze MSDB evaluation baseline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ID = "deepgaze_msdb_v1"
UPSTREAM_REPO = "https://github.com/matthias-k/DeepGaze"
UPSTREAM_COMMIT = "c87b106e8698497c59998b469c45770e993baca3"
WEIGHTS_RELEASE = "v1.2.0"
WEIGHTS_URL = (
    "https://github.com/matthias-k/DeepGaze/releases/download/"
    f"{WEIGHTS_RELEASE}/deepgazemsdb.pth"
)
CENTERBIAS_URL = (
    "https://github.com/matthias-k/DeepGaze/releases/download/"
    "v1.0.0/centerbias_mit1003.npy"
)

DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "scout_models" / MODEL_ID
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "deepgaze_msdb.yaml"
DEFAULT_PIXELS_PER_DEGREE = 35.0
DEFAULT_CENTERBIAS_NAME = "centerbias_mit1003.npy"

# Label-free summaries only. AUC/sAUC/NSS require ground-truth fixations.
SUMMARY_METRIC_NAMES = (
    "entropy",
    "normalized_entropy",
    "peak_probability",
    "peak_y",
    "peak_x",
    "top_1pct_mass",
    "top_5pct_mass",
    "center_mass",
    "periphery_mass",
)
