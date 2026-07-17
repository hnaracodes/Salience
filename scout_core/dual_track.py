"""Zero-Shot Dual-Track Engine — engagement, activation, and emotion scoring.

All tracks operate on raw TRIBE v2 preds[T, 20484] with no trained model.

Track 1 — Visual Engagement:
    Z-scores VAN and DMN network means against a gray-background baseline, then
    computes engagement_score = Z(VAN) - Z(DMN) per timestep.

Track 2 — Discrete Emotion:
    Computes cosine similarity between each normalised timestep vector and 7
    pre-downloaded Kragel & LaBar (2015) NeuroVault templates, then session-relative
    Z-scores per emotion channel for grounding triggers.

Track 3 — Mean Vertex Activation (attention proxy):
    ViralAnalyser-style mean(|preds[t]|) per TR, with baseline-relative Z when
    preds_baseline is available, otherwise session-relative Z.

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
TEMPLATE_NAMES = ["contentment", "amusement", "surprise", "fear", "anger", "sadness", "neutral"]


# ---------------------------------------------------------------------------
# Network index helpers
# ---------------------------------------------------------------------------

def _network_name_mask(names: np.ndarray, target: str) -> np.ndarray:
    """Match Schaefer subnetwork names (e.g. Default_PFC) to coarse Yeo-7 keys."""
    target = target.strip()
    if not target:
        return np.zeros(len(names), dtype=bool)
    as_str = np.asarray(names).astype(str)
    exact = as_str == target
    if np.any(exact):
        return exact
    # Schaefer CSV stores subnetworks like SalVentAttn_ParOper, Default_PFC.
    prefix = f"{target}_"
    return np.char.startswith(as_str, prefix)


def load_network_indices(
    vertex_csv: Path | None = None,
    van_name: str = "SalVentAttn",
    dmn_name: str = "Default",
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Return (van_idx, dmn_idx, error_message).

    Uses configs/vertex_regions.csv (Schaefer 400 @ fsaverage5). Network names may
    be coarse Yeo-7 keys (SalVentAttn, Default) or exact subnetwork labels.

    Returns (None, None, reason) when the CSV is missing or lacks matching vertices.
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

    names = np.asarray(table.yeo_network_name).astype(str)
    van_mask = _network_name_mask(names, van_name)
    dmn_mask = _network_name_mask(names, dmn_name)

    if not np.any(van_mask):
        return None, None, f"No vertices found matching network '{van_name}'"
    if not np.any(dmn_mask):
        return None, None, f"No vertices found matching network '{dmn_name}'"

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

def _robust_vertex_stats(
    baseline: np.ndarray,
    *,
    min_scale_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex median and MAD-based scale from baseline TRs, shape (V,)."""
    med = np.median(baseline, axis=0).astype(np.float32)
    mad = np.median(np.abs(baseline - med), axis=0).astype(np.float32)
    scale = np.maximum(mad * 1.4826, min_scale_eps).astype(np.float32)
    return med, scale


def _network_engagement_from_vertex_z(
    preds: np.ndarray,
    van_idx: np.ndarray,
    dmn_idx: np.ndarray,
    van_center: np.ndarray,
    van_scale: np.ndarray,
    dmn_center: np.ndarray,
    dmn_scale: np.ndarray,
) -> np.ndarray:
    """Z-score each vertex vs baseline stats, then mean within network and subtract."""
    z_van = (preds[:, van_idx] - van_center[van_idx]) / van_scale[van_idx]
    z_dmn = (preds[:, dmn_idx] - dmn_center[dmn_idx]) / dmn_scale[dmn_idx]
    return (z_van.mean(axis=1) - z_dmn.mean(axis=1)).astype(np.float32)


def compute_engagement_track(
    preds: np.ndarray,
    preds_baseline: np.ndarray | None,
    van_idx: np.ndarray,
    dmn_idx: np.ndarray,
    *,
    threshold_high: float = 1.5,
    threshold_low: float = -1.5,
    min_baseline_trs: int = 30,
    normalization_method: str = "robust_mad",
    min_scale_eps: float = 1e-6,
    baseline_corpus_id: str | None = None,
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

    norm_method = (normalization_method or "robust_mad").strip().lower()
    if norm_method == "robust_mad":
        van_med, van_scale = _robust_vertex_stats(preds_baseline[:, van_idx], min_scale_eps=min_scale_eps)
        dmn_med, dmn_scale = _robust_vertex_stats(preds_baseline[:, dmn_idx], min_scale_eps=min_scale_eps)
        # Expand per-network vertex stats to full vertex axis for helper
        V = preds.shape[1]
        van_center = np.zeros(V, dtype=np.float32)
        van_scale_full = np.full(V, min_scale_eps, dtype=np.float32)
        dmn_center = np.zeros(V, dtype=np.float32)
        dmn_scale_full = np.full(V, min_scale_eps, dtype=np.float32)
        van_center[van_idx] = van_med
        van_scale_full[van_idx] = van_scale
        dmn_center[dmn_idx] = dmn_med
        dmn_scale_full[dmn_idx] = dmn_scale
        scores = _network_engagement_from_vertex_z(
            preds, van_idx, dmn_idx,
            van_center, van_scale_full, dmn_center, dmn_scale_full,
        )
    else:
        # Legacy mean-then-Z path
        van_base = preds_baseline[:, van_idx].mean(axis=1)
        dmn_base = preds_baseline[:, dmn_idx].mean(axis=1)
        mu_van, sigma_van = float(van_base.mean()), float(van_base.std())
        mu_dmn, sigma_dmn = float(dmn_base.mean()), float(dmn_base.std())
        van_live = preds[:, van_idx].mean(axis=1)
        dmn_live = preds[:, dmn_idx].mean(axis=1)
        z_van = (van_live - mu_van) / (sigma_van + 1e-8)
        z_dmn = (dmn_live - mu_dmn) / (sigma_dmn + 1e-8)
        scores = (z_van - z_dmn).astype(np.float32)

    labels: list[str | None] = []
    for s in scores.tolist():
        if s > threshold_high:
            labels.append("engaging")
        elif s < threshold_low:
            labels.append("boring")
        else:
            labels.append(None)

    result: dict[str, Any] = {
        "scores": scores.tolist(),
        "labels": labels,
        "baseline_trs": int(T_base),
        "baseline_flag": None,
        "thresholds": {"high": threshold_high, "low": threshold_low},
        "normalization_method": norm_method,
        "source": "van_dmn_baseline_z",
    }
    if baseline_corpus_id:
        result["baseline_corpus_id"] = baseline_corpus_id
    return result


# ---------------------------------------------------------------------------
# Track 3 — Mean Vertex Activation (attention proxy)
# ---------------------------------------------------------------------------

def compute_mean_activation(preds: np.ndarray) -> np.ndarray:
    """ViralAnalyser-style scalar activation per TR: mean(|preds[t]|) across vertices."""
    preds = np.asarray(preds, dtype=np.float32)
    return np.mean(np.abs(preds), axis=1).astype(np.float32)


def compute_activation_track(
    preds: np.ndarray,
    preds_baseline: np.ndarray | None,
    *,
    min_baseline_trs: int = 30,
    threshold_high: float = 1.0,
    threshold_low: float = -1.0,
) -> dict[str, Any]:
    """Compute mean vertex activation with baseline- or session-relative Z scores.

    ``scores`` is the primary comparison metric:
    - baseline-relative Z when preds_baseline has enough TRs
    - otherwise session-relative Z (always available via ``session_z``)

    ``raw_scores`` holds unsigned mean(|preds|) for display layers (marketing, viewer).
    """
    preds = np.asarray(preds, dtype=np.float32)
    T = preds.shape[0]
    raw = compute_mean_activation(preds)
    mu_session = float(raw.mean())
    sigma_session = float(raw.std())
    session_z = ((raw - mu_session) / (sigma_session + 1e-8)).astype(np.float32)

    result: dict[str, Any] = {
        "raw_scores": raw.tolist(),
        "session_z": session_z.tolist(),
        "scores": session_z.tolist(),
        "labels": [None] * T,
        "reducer": "mean_abs",
        "source": "preds",
        "comparison_mode": "session_relative",
        "baseline_trs": 0,
        "baseline_flag": "no_baseline_provided",
        "thresholds": {"high": threshold_high, "low": threshold_low},
        "session_stats": {"mu": mu_session, "sigma": sigma_session},
    }

    if preds_baseline is None:
        result["labels"] = _activation_labels(session_z, threshold_high, threshold_low)
        return result

    preds_baseline = np.asarray(preds_baseline, dtype=np.float32)
    base_raw = compute_mean_activation(preds_baseline)
    T_base = int(base_raw.shape[0])
    result["baseline_trs"] = T_base

    if T_base < min_baseline_trs:
        result["baseline_flag"] = "insufficient_baseline"
        result["labels"] = _activation_labels(session_z, threshold_high, threshold_low)
        return result

    mu_base = float(base_raw.mean())
    sigma_base = float(base_raw.std())
    baseline_z = ((raw - mu_base) / (sigma_base + 1e-8)).astype(np.float32)
    result["baseline_z"] = baseline_z.tolist()
    result["scores"] = baseline_z.tolist()
    result["comparison_mode"] = "baseline_relative"
    result["baseline_flag"] = None
    result["baseline_stats"] = {"mu": mu_base, "sigma": sigma_base}
    result["labels"] = _activation_labels(baseline_z, threshold_high, threshold_low)
    return result


def _activation_labels(
    z_scores: np.ndarray,
    threshold_high: float,
    threshold_low: float,
) -> list[str | None]:
    labels: list[str | None] = []
    for s in np.asarray(z_scores, dtype=np.float32).tolist():
        if s > threshold_high:
            labels.append("high_attention")
        elif s < threshold_low:
            labels.append("low_attention")
        else:
            labels.append(None)
    return labels


# ---------------------------------------------------------------------------
# Track 2 — Discrete Emotion (Template Matching)
# ---------------------------------------------------------------------------

def apply_session_z_scores(
    cosine_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Session-relative Z-score per emotion channel over timesteps.

    For each channel i: Z_{t,i} = (X_{t,i} - mu_i) / (sigma_i + 1e-8)
    where mu_i and sigma_i are the session mean and std of raw cosine scores.
    """
    cosine_scores = np.asarray(cosine_scores, dtype=np.float32)
    mu = cosine_scores.mean(axis=0, keepdims=True)     # (1, n_emotions)
    sigma = cosine_scores.std(axis=0, keepdims=True)   # (1, n_emotions)
    # Channels with no session variance → Z≈0 (avoids blow-up when sigma≈0)
    z_scores = np.where(
        sigma < 1e-7,
        0.0,
        (cosine_scores - mu) / (sigma + 1e-8),
    ).astype(np.float32)
    mu = mu.ravel()
    sigma = sigma.ravel()
    session_stats = {
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
    }
    return z_scores, session_stats


def dominant_emotion_at_timestep(
    cosine_row: list[float] | np.ndarray | None,
    z_row: list[float] | np.ndarray | None,
    template_names: list[str],
    *,
    prob_row: list[float] | np.ndarray | None = None,
    class_names: list[str] | None = None,
    mode: str = "template",
) -> dict[str, Any]:
    """Return dominant emotion at one timestep (template Z or decoder probability)."""
    if mode == "decoder" and prob_row is not None:
        names = class_names or template_names
        prob_arr = np.asarray(prob_row, dtype=np.float32)
        idx = int(np.argmax(prob_arr))
        name = names[idx] if idx < len(names) else f"class_{idx}"
        return {
            "dominant": name,
            "probability": float(prob_arr[idx]),
            "score": float(prob_arr[idx]),
            "mode": "decoder",
        }

    z_arr = np.asarray(z_row, dtype=np.float32)
    cos_arr = np.asarray(cosine_row, dtype=np.float32)
    idx = int(np.argmax(z_arr))
    name = template_names[idx] if idx < len(template_names) else f"channel_{idx}"
    return {
        "dominant": name,
        "raw_cosine": float(cos_arr[idx]),
        "z_score": float(z_arr[idx]),
        "mode": "template",
    }


def compute_emotion_track(
    preds: np.ndarray,
    templates: np.ndarray,
    template_names: list[str],
    *,
    grounding_z_threshold: float = 2.0,
) -> dict[str, Any]:
    """Compute cosine similarity and session-relative Z-scores per emotion.

    Args:
        preds:          Shape (T, V) — live session predictions (all 20484 vertices).
        templates:      Shape (n_emotions, V) — L2-normalised template vectors.
        template_names: List of emotion labels, length n_emotions.
        grounding_z_threshold: Z threshold documented in bundle for UI/grounding.

    Returns dict with cosine_scores, z_scores, session_stats, and metadata.
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

    z_scores, session_stats = apply_session_z_scores(cosine_scores)

    return {
        "template_names": list(template_names),
        "cosine_scores": cosine_scores.tolist(),
        "z_scores": z_scores.tolist(),
        "session_stats": session_stats,
        "z_scoring": "session_relative",
        "grounding_z_threshold": float(grounding_z_threshold),
        "template_source": "neurovault",
        "template_collection_ids": ["12383"],
        "mode": "template",
    }


def compute_emotion_track_decoder(
    cortical_preds: np.ndarray,
    subcortical_preds: np.ndarray | None,
    *,
    model_id: str = "horikawa_ridge_v1",
    window_trs: int = 1,
    feature_spec: str | None = None,
    prob_grounding_threshold: float = 0.5,
) -> dict[str, Any]:
    """Supervised Horikawa ridge decoder track (requires scout_models bundle)."""
    from scout_core.mvpa_engine import predict_emotion_track

    result = predict_emotion_track(
        cortical_preds,
        subcortical_preds,
        model_id=model_id,
        window_trs=window_trs,
        feature_spec_name=feature_spec,
    )
    result["grounding_prob_threshold"] = float(prob_grounding_threshold)
    result["template_source"] = f"decoder:{model_id}"
    return result


def compute_emotion_track_auto(
    cortical_preds: np.ndarray,
    *,
    mode: str = "template",
    templates: np.ndarray | None = None,
    template_names: list[str] | None = None,
    subcortical_preds: np.ndarray | None = None,
    model_id: str = "horikawa_ridge_v1",
    window_trs: int = 1,
    feature_spec: str | None = None,
    require_subcortical: bool = True,
    grounding_z_threshold: float = 2.0,
    prob_grounding_threshold: float = 0.5,
) -> tuple[dict[str, Any] | None, str]:
    """Return (emotion_track, effective_mode). Falls back to template when decoder unavailable."""
    mode = (mode or "template").strip().lower()
    if mode == "decoder":
        if require_subcortical and subcortical_preds is None:
            mode = "template"
        else:
            try:
                return (
                    compute_emotion_track_decoder(
                        cortical_preds,
                        subcortical_preds,
                        model_id=model_id,
                        window_trs=window_trs,
                        feature_spec=feature_spec,
                        prob_grounding_threshold=prob_grounding_threshold,
                    ),
                    "decoder",
                )
            except (FileNotFoundError, ValueError, OSError):
                mode = "template"

    if templates is None or template_names is None:
        return None, mode
    return (
        compute_emotion_track(
            cortical_preds,
            templates,
            template_names,
            grounding_z_threshold=grounding_z_threshold,
        ),
        "template",
    )


# ---------------------------------------------------------------------------
# Grounding trigger detection
# ---------------------------------------------------------------------------

def find_grounding_triggers(
    engagement_result: dict[str, Any] | None,
    emotion_result: dict[str, Any] | None,
    engagement_trigger: float = 2.0,
    emotion_z_trigger: float = 2.0,
    *,
    emotion_prob_trigger: float | None = None,
) -> list[dict[str, Any]]:
    """Return a list of timestep-indexed grounding trigger events.

    Engagement uses baseline-relative Z; template emotion uses session-relative Z;
    decoder emotion uses class probability thresholds.
    """
    triggers: list[dict[str, Any]] = []

    if engagement_result:
        scores = engagement_result.get("scores") or []
        for t, s in enumerate(scores):
            if s is not None and abs(s) >= engagement_trigger:
                triggers.append({"t_idx": t, "trigger_type": "engagement", "value": float(s), "channel": "engagement"})

    if emotion_result:
        emo_mode = emotion_result.get("mode", "template")
        if emo_mode == "decoder":
            names = emotion_result.get("class_names", [])
            probabilities = emotion_result.get("probabilities") or []
            prob_thr = emotion_prob_trigger
            if prob_thr is None:
                prob_thr = float(emotion_result.get("grounding_prob_threshold", 0.5))
            for t, prob_row in enumerate(probabilities):
                for i, p in enumerate(prob_row):
                    if float(p) >= prob_thr:
                        channel = names[i] if i < len(names) else f"class_{i}"
                        triggers.append({
                            "t_idx": t,
                            "trigger_type": "emotion",
                            "channel": channel,
                            "value": float(p),
                            "probability": float(p),
                            "mode": "decoder",
                        })
        else:
            names = emotion_result.get("template_names", [])
            cosine_scores = emotion_result.get("cosine_scores", [])
            z_scores = emotion_result.get("z_scores", [])
            for t, (cos_row, z_row) in enumerate(zip(cosine_scores, z_scores)):
                for i, z in enumerate(z_row):
                    if z > emotion_z_trigger:
                        channel = names[i] if i < len(names) else f"channel_{i}"
                        raw = float(cos_row[i]) if i < len(cos_row) else 0.0
                        triggers.append({
                            "t_idx": t,
                            "trigger_type": "emotion",
                            "channel": channel,
                            "value": float(z),
                            "raw_cosine": raw,
                            "z_score": float(z),
                            "mode": "template",
                        })

    return triggers
