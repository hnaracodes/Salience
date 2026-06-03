"""EEV research-track constants."""

from __future__ import annotations

EEV_FEATURE_CONTRACT_V1 = "eev_features_v1"
EEV_SCHEMA_VERSION = "eev_emotion_v1"

# Coarse Yeo-7 network keys used for feature extraction (Schaefer prefix match).
EEV_NETWORK_NAMES: tuple[str, ...] = ("Vis", "SalVentAttn", "DorsAttn")

# 5 UX targets for SVR (subset of EEV 15 expressions).
EEV_TARGET_LABELS: tuple[str, ...] = (
    "confusion",
    "concentration",
    "interest",
    "awe",
    "contentment",
)

# Full EEV CSV expression columns (reference).
EEV_ALL_EXPRESSION_LABELS: tuple[str, ...] = (
    "amusement",
    "anger",
    "awe",
    "concentration",
    "confusion",
    "contempt",
    "contentment",
    "disappointment",
    "doubt",
    "elation",
    "interest",
    "pain",
    "sadness",
    "surprise",
    "triumph",
)

FEATURE_NAMES_V1: tuple[str, ...] = (
    "amp_vis",
    "amp_sal",
    "amp_dors",
    "d_amp_vis",
    "d_amp_sal",
    "d_amp_dors",
    "early_vis",
    "early_sal",
    "early_dors",
    "late_vis",
    "late_sal",
    "late_dors",
    "coh_vis_sal",
    "coh_vis_dors",
    "coh_sal_dors",
)

DEFAULT_TR_HZ = 1.0
DEFAULT_COHERENCE_WINDOW_TRS = 3
