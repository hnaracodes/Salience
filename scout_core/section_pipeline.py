"""Orchestrate Tier-1 section metrics + Tier-2 heatmap element rollup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scout_core.attention_attribution import (
    clarity_host_warning,
    enrich_sections_with_attribution,
    enrich_sections_with_clarity,
    load_clarity_rows,
    resolve_clarity_csv,
)
from scout_core.dom_intersect import (
    filter_elements_to_section,
    find_nearest_snapshot,
    rollup_section_elements,
    score_all_elements,
)
from scout_core.heatmap_extract import (
    load_heatmap_for_dom_scoring,
    read_heatmaps_manifest,
)
from scout_core.section_analytics import build_section_report
from scout_core.section_recommendations import apply_recommendations
from scout_core.section_sampling import attach_sample_timesteps
from scout_core.session_align import is_timestep_in_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECTION_CFG = PROJECT_ROOT / "configs" / "section_analytics.yaml"


def load_section_config(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_SECTION_CFG
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def enrich_section_report_with_attention(
    session_dir: Path,
    manifest: dict[str, Any],
    section_report: list[dict[str, Any]],
    *,
    heatmaps_subdir: str = "heatmaps",
    min_element_area: int = 400,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Attach ``top_elements`` per section from cached heatmaps at ``sample_t_indices``.

    Timesteps beyond manifest snapshot coverage are skipped rather than silently
    clamped to the nearest snapshot, which would produce misleading DOM state.
    Heatmap provenance (placeholder vs real) is propagated to each section via
    a ``heatmap_source`` field on top element entries.
    """
    heatmaps_dir = session_dir / heatmaps_subdir
    hm_manifest = read_heatmaps_manifest(heatmaps_dir)

    for sec in section_report:
        samples: list[tuple[int, list[dict[str, Any]]]] = []
        root_bbox = sec.get("section_root_bbox")
        placeholder_count = 0
        real_count = 0

        for t in sec.get("sample_t_indices") or []:
            # Skip timesteps that lie outside manifest snapshot coverage.
            if not is_timestep_in_manifest(t, manifest, tolerance=1):
                continue

            heatmap_path = heatmaps_dir / f"t_{t}.npy"
            if not heatmap_path.is_file():
                continue

            hm_meta = hm_manifest.get(t, {})
            is_placeholder = bool(hm_meta.get("placeholder", False)) or \
                             hm_meta.get("source") == "uniform_placeholder"
            heatmap_source = hm_meta.get("source") or ("uniform_placeholder" if is_placeholder else "unknown")

            if is_placeholder:
                placeholder_count += 1
            else:
                real_count += 1

            heatmap = load_heatmap_for_dom_scoring(heatmap_path, hm_meta)
            snapshot = find_nearest_snapshot(manifest, t)
            if snapshot is None:
                continue
            elements = snapshot.get("elements") or []
            elements = filter_elements_to_section(elements, root_bbox)
            scroll_y = int(snapshot.get("scrollY", 0))
            scroll_x = int(snapshot.get("scrollX", 0))
            scored = score_all_elements(
                heatmap, elements, scroll_y, scroll_x, min_area=min_element_area,
            )
            # Tag each scored element with the heatmap source.
            for el in scored:
                el["heatmap_source"] = heatmap_source
            samples.append((t, scored))

        sec["top_elements"] = rollup_section_elements(samples, top_k=top_k)
        sec["heatmap_stats"] = {
            "real": real_count,
            "placeholder": placeholder_count,
            "low_confidence": placeholder_count > 0 and real_count == 0,
        }

    return section_report



def run_section_analytics(
    session_dir: Path,
    bundle: dict[str, Any],
    *,
    site_sections_path: Path | None = None,
    section_cfg_path: Path | None = None,
    attach_heatmaps: bool = True,
    clarity_csv: Path | None = None,
    pipeline_extras: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build full ``section_report`` with optional heatmap enrichment and recommendations."""
    cfg = load_section_config(section_cfg_path)
    tr_duration = float(cfg.get("tr_duration_sec", 1.0))
    high_arousal_z = float(cfg.get("high_arousal_z", 1.0))
    positive_valence_z = float(cfg.get("positive_valence_z", 1.0))
    max_per = int(cfg.get("max_samples_per_section", 3))
    max_total = int(cfg.get("max_heatmaps_per_session", 20))
    min_area = int(cfg.get("min_element_area", 400))
    top_k = int(cfg.get("top_elements_per_section", 5))
    attribution_cfg = dict(cfg.get("attribution") or {})
    calibrated_path = PROJECT_ROOT / "configs" / "attribution_calibrated.yaml"
    if calibrated_path.is_file():
        with calibrated_path.open(encoding="utf-8") as f:
            calibrated = yaml.safe_load(f) or {}
        for key in ("attention_weight", "click_weight", "engagement_activation_blend"):
            if key in calibrated:
                attribution_cfg[key] = calibrated[key]
    clarity_cfg = cfg.get("clarity") or {}

    manifest: dict[str, Any] = {}
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    report = build_section_report(
        bundle,
        manifest,
        site_sections_path=site_sections_path,
        tr_duration_sec=tr_duration,
        high_arousal_z=high_arousal_z,
        positive_valence_z=positive_valence_z,
    )

    attach_sample_timesteps(
        report, bundle, max_per_section=max_per, max_total=max_total,
    )

    if attach_heatmaps and manifest.get("dom_snapshots"):
        enrich_section_report_with_attention(
            session_dir,
            manifest,
            report,
            min_element_area=min_area,
            top_k=top_k,
        )

    capture = manifest.get("capture") or {}
    enrich_sections_with_attribution(
        report,
        bundle,
        attention_weight=float(attribution_cfg.get("attention_weight", 0.68)),
        click_weight=float(attribution_cfg.get("click_weight", 0.32)),
        viewport_h=int(capture.get("height", 1080)),
        mode=str(attribution_cfg.get("mode", "per_tr_sum")),
        session_dir=session_dir,
        manifest=manifest,
        min_element_area=min_area,
        engagement_blend=float(attribution_cfg.get("engagement_activation_blend", 0.72)),
        max_tr_per_section=int(attribution_cfg.get("max_tr_per_section", 30)),
        top_k=top_k,
    )

    clarity_path = resolve_clarity_csv(session_dir, clarity_csv)
    if clarity_csv is not None and clarity_path is None:
        import sys

        print(
            f"WARNING: --clarity-csv not found: {clarity_csv}",
            file=sys.stderr,
        )
    if clarity_path is not None:
        clarity_rows = load_clarity_rows(clarity_path)
        clarity_meta = enrich_sections_with_clarity(
            report,
            clarity_rows,
            blend_weight=float(clarity_cfg.get("blend_weight", 0.5)),
            attention_weight=float(attribution_cfg.get("attention_weight", 0.68)),
            click_weight=float(attribution_cfg.get("click_weight", 0.32)),
        )
        clarity_meta["path"] = str(clarity_path)
        warn = clarity_host_warning(manifest, clarity_rows)
        if warn:
            clarity_meta["host_warning"] = warn
        if pipeline_extras is not None:
            pipeline_extras["clarity_attribution"] = clarity_meta

    apply_recommendations(report)
    return report
