"""Zero-Shot Dual-Track Engine — engagement Z-score + emotion cosine similarity.

Both tracks operate on raw TRIBE v2 preds[T, 20484] with no trained model.

Track 1 — Visual Engagement:
    Z-scores VAN and DMN network means against a plain-text reading baseline, then
    computes engagement_score = Z(VAN) - Z(DMN) per timestep.

Track 2 — Discrete Emotion:
    Computes cosine similarity between each normalised timestep vector and 7
    pre-downloaded NeuroVault templates (Kragel 2015 × 6 + PINES 2015).

See configs/dual_track.yaml for thresholds and template paths.
See scripts/download_emotion_templates.py to fetch and preprocess templates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERTEX_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"
DEFAULT_TEMPLATE_DIR = PROJECT_ROOT / "configs" / "emotion_templates"
TEMPLATE_NAMES = ["anger", "disgust", "fear", "happy", "neutral", "sad", "negative_affect"]


# ---------------------------------------------------------------------------
# Network index helpers
# ---------------------------------------------------------------------------

def load_network_indices(
    vertex_csv: Path | None = None,
    van_name: str = "VentralAttention",
    dmn_name: str = "DefaultMode",
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Return (van_idx, dmn_idx, error_message).

    Returns (None, None, reason) when the CSV is missing or lacks yeo_network_name.
    Caller should check the error_message and suppress Track 1 accordingly.
    """
    path = vertex_csv or DEFAULT_VERTEX_CSV
    if not path.is_file():
        return None, None, f"vertex_regions.csv not found at {path}"

    try:
        from scout_core.parcellation import load_vertex_table
        table = load_vertex_table(path)
    except Exception as exc:
        return None, None, f"Failed to load vertex table: {exc}"

    van_mask = table.yeo_network_name == van_name
    dmn_mask = table.yeo_network_name == dmn_name

    if not np.any(van_mask):
        return None, None, f"No vertices found with yeo_network_name='{van_name}'"
    if not np.any(dmn_mask):
        return None, None, f"No vertices found with yeo_network_name='{dmn_name}'"

    return (
        table.vertex_index[van_mask].astype(np.intp),
        table.vertex_index[dmn_mask].astype(np.intp),
        None,
    )


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def load_templates(
    template_dir: Path | None = None,
    template_names: list[str] | None = None,
) -> tuple[np.ndarray | None, list[str] | None, str | None]:
    """Return (templates[n_emotions, V], names, error_message).

    Returns (None, None, reason) when any template file is missing.
    """
    tdir = template_dir or DEFAULT_TEMPLATE_DIR
    names = template_names or TEMPLATE_NAMES
    arrays: list[np.ndarray] = []
    for name in names:
        p = tdir / f"template_{name}.npy"
        if not p.is_file():
            return (
                None,
                None,
                f"Template missing: {p}. Run `python scripts/download_emotion_templates.py`.",
            )
        arr = np.load(p).astype(np.float32)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-8:
            return None, None, f"Template {name} is all-zero after loading."
        arrays.append(arr / norm)  # enforce unit-norm even if file drifted
    return np.stack(arrays, axis=0), names, None


# ---------------------------------------------------------------------------
# Track 1 — Visual Engagement
# ---------------------------------------------------------------------------

def compute_engagement_track(
    preds: np.ndarray,
    preds_baseline: np.ndarray | None,
    van_idx: np.ndarray,
    dmn_idx: np.ndarray,
    *,
    threshold_high: float = 1.5,
    threshold_low: float = -1.5,
    min_baseline_trs: int = 30,
) -> dict[str, Any]:
    """Compute VAN/DMN engagement Z-score per timestep.

    Args:
        preds:           Shape (T, V) — live session predictions.
        preds_baseline:  Shape (T_base, V) — baseline reading predictions, or None.
        van_idx:         Integer vertex indices for Ventral Attention Network.
        dmn_idx:         Integer vertex indices for Default Mode Network.
        threshold_high:  Z-score above which label is "engaging".
        threshold_low:   Z-score below which label is "boring".
        min_baseline_trs: Minimum baseline TRs needed to emit labels.

    Returns dict with keys: scores, labels, baseline_trs, baseline_flag, thresholds.
    """
    preds = np.asarray(preds, dtype=np.float32)
    T = preds.shape[0]

    # Baseline validation
    if preds_baseline is None:
        return {
            "scores": [None] * T,
            "labels": [None] * T,
            "baseline_trs": 0,
            "baseline_flag": "no_baseline_provided",
            "thresholds": {"high": threshold_high, "low": threshold_low},
        }

    preds_baseline = np.asarray(preds_baseline, dtype=np.float32)
    T_base = preds_baseline.shape[0]

    if T_base < min_baseline_trs:
        return {
            "scores": [None] * T,
            "labels": [None] * T,
            "baseline_trs": int(T_base),
            "baseline_flag": "insufficient_baseline",
            "thresholds": {"high": threshold_high, "low": threshold_low},
        }

    # Baseline statistics (estimated once; never updated with live data)
    van_base = preds_baseline[:, van_idx].mean(axis=1)   # (T_base,)
    dmn_base = preds_baseline[:, dmn_idx].mean(axis=1)   # (T_base,)
    mu_van, sigma_van = float(van_base.mean()), float(van_base.std())
    mu_dmn, sigma_dmn = float(dmn_base.mean()), float(dmn_base.std())

    # Live scoring
    van_live = preds[:, van_idx].mean(axis=1)   # (T,)
    dmn_live = preds[:, dmn_idx].mean(axis=1)   # (T,)
    z_van = (van_live - mu_van) / (sigma_van + 1e-8)
    z_dmn = (dmn_live - mu_dmn) / (sigma_dmn + 1e-8)
    scores = (z_van - z_dmn).astype(np.float32)   # (T,)

    labels: list[str | None] = []
    for s in scores.tolist():
        if s > threshold_high:
            labels.append("engaging")
        elif s < threshold_low:
            labels.append("boring")
        else:
            labels.append(None)

    return {
        "scores": scores.tolist(),
        "labels": labels,
        "baseline_trs": int(T_base),
        "baseline_flag": None,
        "thresholds": {"high": threshold_high, "low": threshold_low},
    }


# ---------------------------------------------------------------------------
# Track 2 — Discrete Emotion (Template Matching)
# ---------------------------------------------------------------------------

def compute_emotion_track(
    preds: np.ndarray,
    templates: np.ndarray,
    template_names: list[str],
) -> dict[str, Any]:
    """Compute cosine similarity between each timestep and all emotion templates.

    Args:
        preds:          Shape (T, V) — live session predictions.
        templates:      Shape (n_emotions, V) — L2-normalised template vectors.
        template_names: List of emotion labels, length n_emotions.

    Returns dict with keys: template_names, cosine_scores, template_source.
    All cosine_scores values are in [-1, 1] by construction.
    """
    preds = np.asarray(preds, dtype=np.float32)
    templates = np.asarray(templates, dtype=np.float32)

    # Normalise each timestep vector: (T, V) → (T, V) unit-norm rows
    norms = np.linalg.norm(preds, axis=1, keepdims=True)   # (T, 1)
    preds_norm = preds / (norms + 1e-8)

    # Vectorised cosine similarity: (T, V) @ (V, n_emotions) → (T, n_emotions)
    cosine_scores = (preds_norm @ templates.T).astype(np.float32)

    # Clamp to [-1, 1] to guard against any floating-point drift
    cosine_scores = np.clip(cosine_scores, -1.0, 1.0)

    return {
        "template_names": list(template_names),
        "cosine_scores": cosine_scores.tolist(),
        "template_source": "neurovault",
        "template_collection_ids": ["503", "10704"],
    }


# ---------------------------------------------------------------------------
# Grounding trigger detection
# ---------------------------------------------------------------------------

def find_grounding_triggers(
    engagement_result: dict[str, Any] | None,
    emotion_result: dict[str, Any] | None,
    engagement_trigger: float = 2.0,
    emotion_trigger: float = 0.85,
) -> list[dict[str, Any]]:
    """Return a list of timestep-indexed grounding trigger events.

    Each event: {t_idx, trigger_type, value, channel}.
    """
    triggers: list[dict[str, Any]] = []

    if engagement_result:
        scores = engagement_result.get("scores") or []
        for t, s in enumerate(scores):
            if s is not None and abs(s) >= engagement_trigger:
                triggers.append({"t_idx": t, "trigger_type": "engagement", "value": float(s), "channel": "engagement"})

    if emotion_result:
        names = emotion_result.get("template_names", [])
        cosine_scores = emotion_result.get("cosine_scores", [])
        for t, row in enumerate(cosine_scores):
            for i, v in enumerate(row):
                if v >= emotion_trigger:
                    channel = names[i] if i < len(names) else f"channel_{i}"
                    triggers.append({"t_idx": t, "trigger_type": "emotion", "value": float(v), "channel": channel})

    return triggers
