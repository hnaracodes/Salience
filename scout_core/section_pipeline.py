"""Orchestrate Tier-1 section metrics + Tier-2 heatmap element rollup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scout_core.dom_intersect import (
    filter_elements_to_section,
    find_nearest_snapshot,
    rollup_section_elements,
    score_all_elements,
)
from scout_core.heatmap_extract import read_heatmaps_manifest
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

            heatmap = np.load(heatmap_path).astype(np.float32)
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

    apply_recommendations(report)
    return report
