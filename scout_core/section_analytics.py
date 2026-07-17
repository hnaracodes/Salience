"""Section-level marketing analytics — assign TRs to UI sections and aggregate neural traces.

Tier 1 (CPU): uses full-session ``emotion_track``, ``engagement_track``, and
``activation_track`` (mean |preds| over full brain) from dual_track.
Section IDs come from a hybrid model: URL + DOM landmarks + optional ``site_sections.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml

from scout_core.dual_track import dominant_emotion_at_timestep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_SECTIONS = PROJECT_ROOT / "configs" / "site_sections.yaml"

LANDMARK_TAGS = frozenset({"header", "nav", "main", "footer", "section", "article", "aside"})
LANDMARK_ROLES = frozenset({"banner", "navigation", "main", "contentinfo", "region", "complementary"})

AROUSAL_CHANNELS = frozenset({"fear", "anger", "surprise"})
POSITIVE_CHANNELS = frozenset({"contentment", "amusement"})


def _emotion_series(emotion: dict[str, Any]) -> tuple[list[str], list[list[float]], str]:
    """Return (names, per-TR values, mode) for template Z or decoder probabilities."""
    if emotion.get("mode") == "decoder":
        return (
            list(emotion.get("class_names") or []),
            list(emotion.get("probabilities") or []),
            "decoder",
        )
    return (
        list(emotion.get("template_names") or []),
        list(emotion.get("z_scores") or []),
        "template",
    )


@dataclass
class SectionAssignment:
    t_idx: int
    section_id: str
    method: str
    url: str = ""
    landmark_dom_id: str = ""
    section_root_bbox: list[int] | None = None


def load_site_sections_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_SITE_SECTIONS
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_url_path(url: str) -> str:
    if not url:
        return "/"
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


def _hostname(url: str) -> str:
    if not url:
        return "unknown"
    host = urlparse(url).netloc or urlparse(f"http://{url}").netloc
    return host.split(":")[0] or "unknown"


def _viewport_rect(snapshot: dict[str, Any], capture: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return document-space viewport bounds (x1, y1, x2, y2)."""
    scroll_x = int(snapshot.get("scrollX", 0))
    scroll_y = int(snapshot.get("scrollY", 0))
    vw = int(capture.get("width", snapshot.get("viewport_width", 1920)))
    vh = int(capture.get("height", snapshot.get("viewport_height", 1080)))
    return scroll_x, scroll_y, scroll_x + vw, scroll_y + vh


def _viewport_center(snapshot: dict[str, Any], capture: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = _viewport_rect(snapshot, capture)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_intersection_area(bbox: list[int], vx1: int, vy1: int, vx2: int, vy2: int) -> float:
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(x, vx1)
    y1 = max(y, vy1)
    x2 = min(x + w, vx2)
    y2 = min(y + h, vy2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def _bbox_contains_point(bbox: list[int], px: float, py: float) -> bool:
    x, y, w, h = [int(v) for v in bbox]
    return x <= px < x + w and y <= py < y + h


def _is_landmark(el: dict[str, Any]) -> bool:
    tag = str(el.get("tag", "")).lower()
    role = str(el.get("role", "")).lower()
    if tag in LANDMARK_TAGS:
        return True
    if role in LANDMARK_ROLES:
        return True
    dom_id = str(el.get("dom_id", "")).lower()
    if dom_id.startswith("#") and any(k in dom_id for k in ("hero", "header", "footer", "main", "nav", "section")):
        return True
    return False


def _document_height(elements: list[dict[str, Any]], default: int = 3000) -> int:
    max_y = 0
    for el in elements:
        bbox = el.get("bbox")
        if bbox and len(bbox) == 4:
            max_y = max(max_y, int(bbox[1]) + int(bbox[3]))
    return max(max_y, default)


def _scroll_band(url_path: str, scroll_y: int, doc_height: int, vh: int) -> str:
    if doc_height <= vh:
        return f"{url_path}#viewport"
    third = max((doc_height - vh) // 3, 1)
    if scroll_y < third:
        band = "top"
    elif scroll_y < 2 * third:
        band = "middle"
    else:
        band = "bottom"
    return f"{url_path}#scroll_{band}"


def _match_manual_section(
    elements: list[dict[str, Any]],
    manual_sections: list[dict[str, Any]],
    cx: float,
    cy: float,
) -> tuple[str, list[int] | None] | None:
    """Return (section_id, root_bbox) if viewport center hits a manual selector."""
    for sec in manual_sections:
        sec_id = str(sec.get("id", ""))
        selectors = sec.get("selectors") or []
        for el in elements:
            dom_id = str(el.get("dom_id", ""))
            for sel in selectors:
                sel = str(sel).strip()
                if not sel:
                    continue
                if sel.startswith("#") and dom_id == sel:
                    bbox = el.get("bbox")
                    if bbox and _bbox_contains_point(bbox, cx, cy):
                        return f"manual/{sec_id}", [int(b) for b in bbox]
                elif sel in dom_id or dom_id.endswith(sel.replace(".", "")):
                    bbox = el.get("bbox")
                    if bbox and _bbox_contains_point(bbox, cx, cy):
                        return f"manual/{sec_id}", [int(b) for b in bbox]
    return None


def _pick_landmark_section(
    elements: list[dict[str, Any]],
    url_path: str,
    cx: float,
    cy: float,
    vx1: int,
    vy1: int,
    vx2: int,
    vy2: int,
) -> tuple[str, str, list[int] | None]:
    """Pick landmark with largest viewport intersection; fallback scroll band."""
    landmarks = [el for el in elements if _is_landmark(el) and el.get("bbox")]
    best_area = -1.0
    best: dict[str, Any] | None = None
    for el in landmarks:
        area = _bbox_intersection_area(el["bbox"], vx1, vy1, vx2, vy2)
        if area > best_area:
            best_area = area
            best = el
    if best is not None and best_area > 0:
        dom_id = str(best.get("dom_id", "landmark"))
        sid = f"{url_path}#{dom_id.lstrip('#')}"
        return sid, dom_id, [int(b) for b in best["bbox"]]
    return "", "", None


def assign_section_for_snapshot(
    snapshot: dict[str, Any],
    capture: dict[str, Any],
    manual_sections: list[dict[str, Any]] | None = None,
) -> tuple[str, str, list[int] | None]:
    """Assign one section_id for a single DOM snapshot."""
    url = str(snapshot.get("url", ""))
    url_path = _normalize_url_path(url)
    elements = snapshot.get("elements") or []
    cx, cy = _viewport_center(snapshot, capture)
    vx1, vy1, vx2, vy2 = _viewport_rect(snapshot, capture)

    if manual_sections:
        hit = _match_manual_section(elements, manual_sections, cx, cy)
        if hit:
            return hit[0], "manual", hit[1]

    sid, dom_id, bbox = _pick_landmark_section(elements, url_path, cx, cy, vx1, vy1, vx2, vy2)
    if sid:
        return sid, "landmark", bbox

    doc_h = _document_height(elements)
    vh = int(capture.get("height", 1080))
    scroll_y = int(snapshot.get("scrollY", 0))
    band_id = _scroll_band(url_path, scroll_y, doc_h, vh)
    return band_id, "scroll_band", None


def _nearest_snapshot_index(snapshots: list[dict[str, Any]], t: int) -> int:
    if not snapshots:
        return -1
    return int(min(range(len(snapshots)), key=lambda i: abs(int(snapshots[i].get("t_idx", 0)) - t)))


def assign_timesteps_to_sections(
    n_timesteps: int,
    manifest: dict[str, Any] | None,
    site_sections_cfg: dict[str, Any] | None = None,
    *,
    default_url: str = "",
) -> list[SectionAssignment]:
    """Map each ``t_idx`` to a hybrid section_id."""
    assignments: list[SectionAssignment] = []
    capture = (manifest or {}).get("capture") or {}
    snapshots = (manifest or {}).get("dom_snapshots") or []

    if not snapshots:
        for t in range(n_timesteps):
            band = "top" if t < n_timesteps / 3 else ("middle" if t < 2 * n_timesteps / 3 else "bottom")
            assignments.append(
                SectionAssignment(
                    t_idx=t,
                    section_id=f"temporal/{band}",
                    method="temporal_fallback",
                    url=default_url,
                )
            )
        return assignments

    for t in range(n_timesteps):
        si = _nearest_snapshot_index(snapshots, t)
        snap = snapshots[si]
        url = str(snap.get("url", default_url))
        host = _hostname(url)
        sites = (site_sections_cfg or {}).get("sites") or {}
        manual = sites.get(host, {}).get("sections") or sites.get(host.replace("www.", ""), {}).get("sections") or []

        section_id, method, root_bbox = assign_section_for_snapshot(snap, capture, manual)
        landmark_dom = ""
        if method == "landmark" and "#" in section_id:
            landmark_dom = section_id.split("#", 1)[-1]

        assignments.append(
            SectionAssignment(
                t_idx=t,
                section_id=section_id,
                method=method,
                url=url,
                landmark_dom_id=landmark_dom,
                section_root_bbox=root_bbox,
            )
        )
    return assignments


def aggregate_section_metrics(
    bundle: dict[str, Any],
    assignments: list[SectionAssignment],
    *,
    tr_duration_sec: float = 1.0,
    high_arousal_z: float = 1.0,
    positive_valence_z: float = 1.0,
) -> list[dict[str, Any]]:
    """Aggregate engagement, activation, and emotion traces per section (all TRs, not spike-only)."""
    engagement = bundle.get("engagement_track") or {}
    activation = bundle.get("activation_track") or {}
    emotion = bundle.get("emotion_track") or {}
    eng_scores = engagement.get("scores") or []
    eng_labels = engagement.get("labels") or []
    act_raw = activation.get("raw_scores") or []
    act_z = activation.get("scores") or []
    act_labels = activation.get("labels") or []
    template_names, z_scores, emo_mode = _emotion_series(emotion)
    cosine_scores = emotion.get("cosine_scores") or []

    by_section: dict[str, list[int]] = {}
    meta: dict[str, SectionAssignment] = {}
    for a in assignments:
        by_section.setdefault(a.section_id, []).append(a.t_idx)
        if a.section_id not in meta:
            meta[a.section_id] = a

    reports: list[dict[str, Any]] = []
    for section_id, t_indices in sorted(by_section.items()):
        t_indices = sorted(t_indices)
        dwell_trs = len(t_indices)
        dwell_sec = dwell_trs * tr_duration_sec

        valid_eng = [
            float(eng_scores[t])
            for t in t_indices
            if t < len(eng_scores) and eng_scores[t] is not None
        ]
        labels_slice = [
            eng_labels[t]
            for t in t_indices
            if t < len(eng_labels)
        ]
        n_labeled = len(labels_slice)
        pct_engaging = (
            sum(1 for l in labels_slice if l == "engaging") / n_labeled if n_labeled else None
        )
        pct_boring = (
            sum(1 for l in labels_slice if l == "boring") / n_labeled if n_labeled else None
        )

        valid_act_raw = [
            float(act_raw[t])
            for t in t_indices
            if t < len(act_raw)
        ]
        valid_act_z = [
            float(act_z[t])
            for t in t_indices
            if t < len(act_z)
        ]
        act_labels_slice = [
            act_labels[t]
            for t in t_indices
            if t < len(act_labels)
        ]
        n_act_labeled = len(act_labels_slice)
        pct_high_attention = (
            sum(1 for l in act_labels_slice if l == "high_attention") / n_act_labeled
            if n_act_labeled else None
        )
        pct_low_attention = (
            sum(1 for l in act_labels_slice if l == "low_attention") / n_act_labeled
            if n_act_labeled else None
        )

        mean_z: dict[str, float] = {}
        peak_channel = ""
        peak_z = -999.0
        peak_t = t_indices[0] if t_indices else 0

        if z_scores and template_names:
            z_arr = np.array(
                [z_scores[t] for t in t_indices if t < len(z_scores)],
                dtype=np.float32,
            )
            if z_arr.size:
                for i, name in enumerate(template_names):
                    if i < z_arr.shape[1]:
                        mean_z[name] = round(float(z_arr[:, i].mean()), 4)
                for t in t_indices:
                    if t >= len(z_scores):
                        continue
                    row = z_scores[t]
                    if not row:
                        continue
                    idx = int(np.argmax(row))
                    z_val = float(row[idx])
                    if z_val > peak_z:
                        peak_z = z_val
                        peak_channel = template_names[idx] if idx < len(template_names) else f"ch_{idx}"
                        peak_t = t

        flags: list[str] = []
        for ch in AROUSAL_CHANNELS:
            if mean_z.get(ch, 0) >= high_arousal_z:
                flags.append("high_arousal")
                break
        for ch in POSITIVE_CHANNELS:
            if mean_z.get(ch, 0) >= positive_valence_z:
                flags.append("positive_valence")
                break

        peak_emotion: dict[str, Any] | None = None
        if peak_channel and z_scores:
            if emo_mode == "decoder":
                peak_emotion = {
                    "channel": peak_channel,
                    "peak_prob": round(peak_z, 4),
                    "t_idx": peak_t,
                    **dominant_emotion_at_timestep(
                        None,
                        None,
                        template_names,
                        prob_row=z_scores[peak_t] if peak_t < len(z_scores) else None,
                        class_names=template_names,
                        mode="decoder",
                    ),
                }
            elif cosine_scores and peak_t < len(cosine_scores):
                peak_emotion = {
                    "channel": peak_channel,
                    "z_score": round(peak_z, 4),
                    "t_idx": peak_t,
                    **dominant_emotion_at_timestep(
                        cosine_scores[peak_t],
                        z_scores[peak_t],
                        template_names,
                    ),
                }

        emotion_block: dict[str, Any] = {
            "mean_z": mean_z,
            "peak": peak_emotion,
            "mode": emo_mode,
        }
        if emo_mode == "decoder":
            emotion_block["mean_prob"] = mean_z
            emotion_block["peak_prob"] = peak_emotion.get("peak_prob") if peak_emotion else None

        m = meta.get(section_id)
        reports.append({
            "section_id": section_id,
            "method": m.method if m else "unknown",
            "url": m.url if m else "",
            "landmark_dom_id": m.landmark_dom_id if m else "",
            "section_root_bbox": m.section_root_bbox if m else None,
            "dwell_trs": dwell_trs,
            "dwell_sec": round(dwell_sec, 2),
            "t_indices": t_indices,
            "engagement": {
                "mean": round(float(np.mean(valid_eng)), 4) if valid_eng else None,
                "median": round(float(np.median(valid_eng)), 4) if valid_eng else None,
                "pct_engaging": round(pct_engaging, 4) if pct_engaging is not None else None,
                "pct_boring": round(pct_boring, 4) if pct_boring is not None else None,
                "n_scored": len(valid_eng),
            },
            "activation": {
                "mean_raw": round(float(np.mean(valid_act_raw)), 6) if valid_act_raw else None,
                "median_raw": round(float(np.median(valid_act_raw)), 6) if valid_act_raw else None,
                "mean_z": round(float(np.mean(valid_act_z)), 4) if valid_act_z else None,
                "pct_high_attention": round(pct_high_attention, 4) if pct_high_attention is not None else None,
                "pct_low_attention": round(pct_low_attention, 4) if pct_low_attention is not None else None,
                "comparison_mode": activation.get("comparison_mode"),
                "n_scored": len(valid_act_raw),
            },
            "emotion": emotion_block,
            "flags": flags,
            "top_elements": [],
            "sample_t_indices": [],
            "recommendations": [],
        })

    return reports


def build_section_report(
    bundle: dict[str, Any],
    manifest: dict[str, Any] | None,
    *,
    site_sections_path: Path | None = None,
    tr_duration_sec: float = 1.0,
    high_arousal_z: float = 1.0,
    positive_valence_z: float = 1.0,
) -> list[dict[str, Any]]:
    """Full Tier-1 pipeline: assign TRs then aggregate metrics."""
    emotion = bundle.get("emotion_track") or {}
    z_scores = emotion.get("z_scores") or []
    n_t = len(z_scores)
    if n_t == 0:
        eng = bundle.get("engagement_track") or {}
        n_t = len(eng.get("scores") or [])

    site_cfg = load_site_sections_config(site_sections_path)
    default_url = str((manifest or {}).get("initial_url", ""))
    assignments = assign_timesteps_to_sections(n_t, manifest, site_cfg, default_url=default_url)
    return aggregate_section_metrics(
        bundle,
        assignments,
        tr_duration_sec=tr_duration_sec,
        high_arousal_z=high_arousal_z,
        positive_valence_z=positive_valence_z,
    )
