"""Stratified TR sampling per section for sparse ViT heatmap extraction."""

from __future__ import annotations

from typing import Any


def attach_sample_timesteps(
    section_report: list[dict[str, Any]],
    bundle: dict[str, Any],
    *,
    max_per_section: int = 3,
    max_total: int = 20,
) -> list[int]:
    """Annotate each section with ``sample_t_indices``; return global unique list."""
    engagement = bundle.get("engagement_track") or {}
    emotion = bundle.get("emotion_track") or {}
    eng_scores = engagement.get("scores") or []
    z_scores = emotion.get("z_scores") or []

    global_seen: set[int] = set()
    global_list: list[int] = []

    for sec in section_report:
        t_indices = list(sec.get("t_indices") or [])
        if not t_indices:
            sec["sample_t_indices"] = []
            continue

        picks: list[int] = []
        picks.append(t_indices[0])

        peak_t = t_indices[0]
        peak_val = -1.0
        for t in t_indices:
            z_peak = 0.0
            if t < len(z_scores) and z_scores[t]:
                z_peak = max(abs(float(v)) for v in z_scores[t])
            eng_val = 0.0
            if t < len(eng_scores) and eng_scores[t] is not None:
                eng_val = abs(float(eng_scores[t]))
            combined = max(z_peak, eng_val)
            if combined > peak_val:
                peak_val = combined
                peak_t = t
        if peak_t not in picks:
            picks.append(peak_t)
        if t_indices[-1] not in picks:
            picks.append(t_indices[-1])

        sec_samples = picks[:max_per_section]
        sec["sample_t_indices"] = sec_samples

        for t in sec_samples:
            if t not in global_seen and len(global_list) < max_total:
                global_seen.add(t)
                global_list.append(t)

    return sorted(global_list)
