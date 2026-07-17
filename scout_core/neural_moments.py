"""Per-TR neural decomposition from parcellated TRIBE predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

from scout_core.dual_track import (
    TEMPLATE_NAMES,
    dominant_emotion_at_timestep,
)


def _network_index(names: list[str], target: str) -> int | None:
    target = target.strip()
    for i, name in enumerate(names):
        if name == target or str(name).startswith(f"{target}_"):
            return i
    return None


def engagement_scores_from_network_z(
    z_network: np.ndarray,
    network_names: list[str],
    *,
    van_name: str = "SalVentAttn",
    dmn_name: str = "Default",
    threshold_high: float = 1.5,
    threshold_low: float = -1.5,
) -> dict[str, Any]:
    """Engagement = Z(SalVentAttn) - Z(Default) from norm-referenced network traces."""
    z = np.asarray(z_network, dtype=np.float32)
    T = z.shape[0]
    van_i = _network_index(network_names, van_name)
    dmn_i = _network_index(network_names, dmn_name)
    if van_i is None or dmn_i is None:
        return {
            "scores": [None] * T,
            "labels": [None] * T,
            "baseline_flag": "network_indices_missing",
            "thresholds": {"high": threshold_high, "low": threshold_low},
            "source": "parcel_network_norms",
            "van_network": van_name,
            "dmn_network": dmn_name,
        }

    scores = (z[:, van_i] - z[:, dmn_i]).astype(np.float32)
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
        "baseline_flag": None,
        "baseline_trs": 0,
        "thresholds": {"high": threshold_high, "low": threshold_low},
        "source": "parcel_network_norms",
        "comparison_mode": "norm_referenced",
        "van_network": van_name,
        "dmn_network": dmn_name,
        "van_index": van_i,
        "dmn_index": dmn_i,
    }


def _top_k_by_abs(values: np.ndarray, labels: list[Any], k: int) -> list[dict[str, Any]]:
    if values.size == 0:
        return []
    order = np.argsort(-np.abs(values))[:k]
    out: list[dict[str, Any]] = []
    for idx in order:
        i = int(idx)
        out.append({"index": i, "label": labels[i], "z": round(float(values[i]), 4)})
    return out


def moment_at_timestep(
    t: int,
    *,
    z_parcel: np.ndarray,
    parcel_ids: np.ndarray,
    parcel_labels: dict[int, str],
    z_network: np.ndarray,
    network_names: list[str],
    emotion_track: dict[str, Any] | None = None,
    top_k_parcels: int = 5,
    top_k_networks: int = 3,
) -> dict[str, Any]:
    """Compact neural snapshot for one TR."""
    zp = np.asarray(z_parcel[t], dtype=np.float32) if t < z_parcel.shape[0] else np.zeros(z_parcel.shape[1])
    zn = np.asarray(z_network[t], dtype=np.float32) if t < z_network.shape[0] else np.zeros(z_network.shape[1])

    if zp.size and len(parcel_ids) == zp.size:
        order_p = np.argsort(-np.abs(zp))[:top_k_parcels]
        top_parcels = [
            {
                "parcel_id": int(parcel_ids[i]),
                "label": parcel_labels.get(int(parcel_ids[i]), f"parcel_{parcel_ids[i]}"),
                "z": round(float(zp[i]), 4),
            }
            for i in order_p
        ]
    else:
        top_parcels = []

    top_networks = _top_k_by_abs(zn, list(network_names), top_k_networks)
    for row in top_networks:
        row["network"] = row.pop("label")

    moment: dict[str, Any] = {
        "t_idx": t,
        "top_parcels": top_parcels,
        "top_networks": top_networks,
    }

    if emotion_track:
        emo_mode = emotion_track.get("mode", "template")
        if emo_mode == "decoder":
            prob_rows = emotion_track.get("probabilities") or []
            class_names = emotion_track.get("class_names") or []
            if t < len(prob_rows):
                dom = dominant_emotion_at_timestep(
                    None,
                    None,
                    class_names,
                    prob_row=prob_rows[t],
                    class_names=class_names,
                    mode="decoder",
                )
                moment["dominant_emotion"] = dom
        else:
            cos_rows = emotion_track.get("cosine_scores") or []
            z_rows = emotion_track.get("z_scores") or []
            names = emotion_track.get("template_names") or TEMPLATE_NAMES
            if t < len(cos_rows) and t < len(z_rows):
                dom = dominant_emotion_at_timestep(cos_rows[t], z_rows[t], names)
                moment["dominant_emotion"] = dom

    van_i = _network_index(network_names, "SalVentAttn")
    dmn_i = _network_index(network_names, "Default")
    if van_i is not None and dmn_i is not None:
        moment["engagement_z"] = round(float(zn[van_i] - zn[dmn_i]), 4)

    return moment


def build_neural_moments_by_t(
    z_parcel: np.ndarray,
    parcel_ids: np.ndarray,
    parcel_labels: dict[int, str],
    z_network: np.ndarray,
    network_names: list[str],
    *,
    emotion_track: dict[str, Any] | None = None,
    t_indices: list[int] | None = None,
    top_k_parcels: int = 5,
    top_k_networks: int = 3,
) -> dict[str, dict[str, Any]]:
    """Map TR index (str) -> neural moment dict."""
    T = z_network.shape[0]
    indices = t_indices if t_indices is not None else list(range(T))
    out: dict[str, dict[str, Any]] = {}
    for t in indices:
        if 0 <= t < T:
            out[str(t)] = moment_at_timestep(
                t,
                z_parcel=z_parcel,
                parcel_ids=parcel_ids,
                parcel_labels=parcel_labels,
                z_network=z_network,
                network_names=network_names,
                emotion_track=emotion_track,
                top_k_parcels=top_k_parcels,
                top_k_networks=top_k_networks,
            )
    return out


def neural_moment_strength(moment: dict[str, Any] | None) -> float:
    """Scalar strength from top network |Z| for attribution weighting."""
    if not moment:
        return 0.0
    nets = moment.get("top_networks") or []
    if not nets:
        return 0.0
    return float(max(abs(float(n.get("z", 0.0))) for n in nets))


def network_masked_emotion_cosine(
    preds_row: np.ndarray,
    template: np.ndarray,
    network_vertex_idx: np.ndarray,
) -> float:
    """Cosine similarity within a network mask (preserves spatial pattern subset)."""
    if network_vertex_idx.size == 0:
        return 0.0
    p = np.asarray(preds_row[network_vertex_idx], dtype=np.float32)
    t = np.asarray(template[network_vertex_idx], dtype=np.float32)
    pn = float(np.linalg.norm(p))
    tn = float(np.linalg.norm(t))
    if pn < 1e-8 or tn < 1e-8:
        return 0.0
    return float(np.dot(p / pn, t / tn))
