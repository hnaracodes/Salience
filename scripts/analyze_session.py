#!/usr/bin/env python3
"""Offline neuro analysis: parcellation → norms → calibrated rules → SQLite + analysis_bundle.json.

Optional --ground flag extends the pipeline with Spatial Credit Assignment
(feature isolation): detects dual-track spikes, loads per-spike heatmaps,
runs DOM intersection, and appends grounding events to analysis_bundle.json.

Workflow (full pipeline):
    1. modal run tribe.py                               → preds.npz + MP4
    2. python scripts/run_dual_track.py --session-id X  → dual-track scores
    3. python scripts/analyze_session.py --session-id X --norm-id Y --ground
       → parcellation + threshold rules + grounding events in one bundle

The --ground step reads isolation thresholds from configs/isolation_thresholds.yaml
and loads pre-saved heatmaps from scout_data/sessions/<id>/heatmaps/t_<N>.npy.
Heatmaps are produced by tribe.py::extract_frame_attention (Modal GPU).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activation_store import DB_PATH, PROJECT_ROOT  # noqa: E402
from scout_core.aggregate import network_timeseries, parcel_timeseries  # noqa: E402
from scout_core.constants import network_names_for_ids  # noqa: E402
from scout_core.dom_intersect import find_nearest_snapshot, ground_snapshot  # noqa: E402
from scout_core.norms import zscore_network, zscore_roi  # noqa: E402
from scout_core.parcellation import dense_parcel_labels, load_vertex_table, parcel_to_network_map  # noqa: E402
from scout_core.schemas import (  # noqa: E402
    AnalysisBundle,
    GroundingEvent,
    GroundingResult,
    HeatmapProvenance,
    SessionCaptureMeta,
    ThresholdContext,
    analysis_bundle_path,
)
from scout_core.heatmap_extract import read_heatmaps_manifest  # noqa: E402
from scout_core.session_align import (  # noqa: E402
    load_manifest,
    tr_duration_from_manifest,
    validate_session_dir,
)
from scout_core.storage_migrations import (  # noqa: E402
    clear_session_neuro_rows,
    ensure_neuro_schema,
    insert_network_timeseries_batch,
    insert_roi_timeseries_batch,
    insert_threshold_hits,
)
from scout_core.threshold_engine import evaluate_rules  # noqa: E402

_ISO_CONFIG_PATH = ROOT / "configs" / "isolation_thresholds.yaml"

# Fields written by run_dual_track.py that we must preserve across bundle writes.
_DUAL_TRACK_PASSTHROUGH_KEYS = (
    "engagement_track",
    "emotion_track",
    "grounding_triggers",
    "dual_track_schema_version",
    "section_report",
)


# ---------------------------------------------------------------------------
# Feature Isolation — grounding helpers
# ---------------------------------------------------------------------------

def _load_iso_config() -> dict[str, Any]:
    if _ISO_CONFIG_PATH.is_file():
        with _ISO_CONFIG_PATH.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _select_spikes(
    bundle: dict[str, Any],
    eng_min: float,
    emotion_z_min: float,
    combine: str,
    cooldown_trs: int,
    max_spikes: int,
) -> list[dict[str, Any]]:
    """Filter dual-track trigger events against isolation thresholds.

    Track 1 (engagement): any trigger with channel=="engagement" and
    |value| >= eng_min passes.

    Track 2 (emotion): ANY emotion channel (not just anger) with
    value >= emotion_z_min passes. The highest-scoring channel at each
    timestep is recorded as ``emotion_channel`` + ``emotion_z`` in the
    trigger payload.  For backward compatibility, if the winning channel is
    "anger", ``kragel_anger_z`` is also written.

    Applies cooldown (minimum gap between consecutive spikes) and optional
    max_spikes cap. Returns [{t_idx, triggers}].
    """
    raw_triggers = bundle.get("grounding_triggers", [])
    if not raw_triggers:
        return []

    # Group triggers by timestep so we can evaluate both conditions at once.
    by_t: dict[int, dict[str, float]] = {}
    for tr in raw_triggers:
        t = int(tr["t_idx"])
        channel = tr.get("channel", "")
        value = float(tr.get("value", 0.0))
        if t not in by_t:
            by_t[t] = {}
        by_t[t][channel] = value

    spikes: list[dict[str, Any]] = []
    last_t = -cooldown_trs - 1

    for t in sorted(by_t.keys()):
        vals = by_t[t]
        eng_val = vals.get("engagement", None)

        # Find best emotion channel across all non-engagement channels.
        emotion_channels = {k: v for k, v in vals.items() if k != "engagement"}
        best_channel: str | None = None
        best_emotion_z: float = 0.0
        if emotion_channels:
            best_channel = max(emotion_channels, key=lambda k: emotion_channels[k])
            best_emotion_z = emotion_channels[best_channel]

        eng_ok = eng_val is not None and abs(eng_val) >= eng_min
        emotion_ok = best_channel is not None and best_emotion_z >= emotion_z_min

        if combine == "and":
            passes = eng_ok and emotion_ok
        else:
            passes = eng_ok or emotion_ok

        if not passes:
            continue
        if t - last_t < cooldown_trs:
            continue

        trigger_payload: dict[str, Any] = {}
        if eng_val is not None:
            trigger_payload["engagement_score"] = round(eng_val, 4)
        if best_channel is not None and best_emotion_z >= emotion_z_min:
            trigger_payload["emotion_channel"] = best_channel
            trigger_payload["emotion_z"] = round(best_emotion_z, 4)
            # Backward compat: keep kragel_anger_z when anger is the winner.
            if best_channel == "anger":
                trigger_payload["kragel_anger_z"] = round(best_emotion_z, 4)

        spikes.append({"t_idx": t, "triggers": trigger_payload})
        last_t = t

        if max_spikes and len(spikes) >= max_spikes:
            break

    return spikes


def _run_grounding_step(
    session_dir: Path,
    existing_bundle: dict[str, Any],
    iso_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Detect spikes, load saved heatmaps, run DOM intersection, return events.

    Heatmaps must be pre-saved at session_dir/heatmaps/t_<N>.npy by a prior
    call to tribe.py::extract_frame_attention. Missing heatmaps produce an
    event with grounding=None and a grounding_skip_reason.

    Args:
        session_dir:     Path to scout_data/sessions/<id>/.
        existing_bundle: Current analysis_bundle dict (may contain dual-track data).
        iso_cfg:         Parsed isolation_thresholds.yaml.

    Returns:
        List of GroundingEvent-compatible dicts for events[].
    """
    trigger_cfg = iso_cfg.get("trigger", {})
    eng_min = float(trigger_cfg.get("engagement_score_min", 2.0))

    # emotion_z_min accepts the new key; fall back to deprecated kragel_anger_z_min.
    if "emotion_z_min" in trigger_cfg:
        emotion_z_min = float(trigger_cfg["emotion_z_min"])
    elif "kragel_anger_z_min" in trigger_cfg:
        print(
            "WARNING: kragel_anger_z_min is deprecated; use emotion_z_min in isolation_thresholds.yaml.",
            file=sys.stderr,
        )
        emotion_z_min = float(trigger_cfg["kragel_anger_z_min"])
    elif "kragel_anger_min" in trigger_cfg:
        print(
            "WARNING: kragel_anger_min is deprecated (absolute cosine); "
            "use emotion_z_min in isolation_thresholds.yaml.",
            file=sys.stderr,
        )
        emotion_z_min = 2.0
    else:
        emotion_z_min = 2.0
    combine = str(trigger_cfg.get("combine", "or"))

    spike_policy = iso_cfg.get("spike_policy", {})
    max_spikes = int(spike_policy.get("max_spikes_per_session", 20))
    cooldown_trs = int(spike_policy.get("cooldown_trs", 3))

    heatmaps_subdir = str(iso_cfg.get("heatmaps_dir", "heatmaps"))
    heatmaps_dir = session_dir / heatmaps_subdir

    # Load manifest (optional — Playwright recorder populates this in P0)
    manifest: dict[str, Any] = {}
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: session_manifest.json is malformed — DOM grounding skipped.", file=sys.stderr)

    spikes = _select_spikes(existing_bundle, eng_min, emotion_z_min, combine, cooldown_trs, max_spikes)
    print(f"  Grounding: {len(spikes)} spike(s) selected after cooldown / cap filter.")

    # Load heatmap manifest for provenance (source / placeholder fields).
    hm_manifest = read_heatmaps_manifest(heatmaps_dir)

    events: list[dict[str, Any]] = []
    for spike in spikes:
        t = spike["t_idx"]
        triggers = spike["triggers"]

        heatmap_path = heatmaps_dir / f"t_{t}.npy"
        hm_meta = hm_manifest.get(t, {})
        provenance = HeatmapProvenance(
            heatmap_path=str(heatmap_path.relative_to(session_dir)) if heatmap_path.is_file() else None,
            source=hm_meta.get("source") or ("uniform_placeholder" if hm_meta.get("placeholder") else None),
            placeholder=bool(hm_meta.get("placeholder", False)),
            frame_path=hm_meta.get("frame_path"),
            sha256=hm_meta.get("sha256"),
        )

        if not heatmap_path.is_file():
            print(f"  t={t}: heatmap not found at {heatmap_path} — recording ungrounded event.")
            events.append(
                GroundingEvent(
                    t_spike=t,
                    triggers=triggers,
                    grounding=None,
                    grounding_skip_reason="heatmap_not_found",
                    heatmap_provenance=provenance,
                ).model_dump()
            )
            continue

        heatmap = np.load(heatmap_path).astype(np.float32)

        snapshot = find_nearest_snapshot(manifest, t)
        if snapshot is None:
            print(f"  t={t}: no DOM snapshot in manifest — recording ungrounded event.")
            events.append(
                GroundingEvent(
                    t_spike=t,
                    triggers=triggers,
                    grounding=None,
                    grounding_skip_reason="no_manifest_snapshot",
                    heatmap_provenance=provenance,
                ).model_dump()
            )
            continue

        winner = ground_snapshot(heatmap, snapshot)
        if winner is None:
            print(f"  t={t}: dom_intersect found no eligible element.")
            events.append(
                GroundingEvent(
                    t_spike=t,
                    triggers=triggers,
                    grounding=None,
                    grounding_skip_reason="no_eligible_dom_element",
                    heatmap_provenance=provenance,
                ).model_dump()
            )
        else:
            print(f"  t={t}: winner → {winner['dom_id']} ({winner['tag']}) density={winner['attention_density']:.4f}")
            events.append(
                GroundingEvent(
                    t_spike=t,
                    triggers=triggers,
                    grounding=GroundingResult(**winner),
                    heatmap_provenance=provenance,
                ).model_dump()
            )

    return events


# ---------------------------------------------------------------------------
# Norm bundle loader
# ---------------------------------------------------------------------------

def _load_norm_bundle(conn: sqlite3.Connection, norm_id: str) -> tuple[Path, Path]:
    row = conn.execute(
        "SELECT path_roi_parquet, path_network_parquet FROM norm_bundle WHERE norm_id = ?",
        (norm_id,),
    ).fetchone()
    if not row:
        raise SystemExit(
            f"norm_id {norm_id!r} not registered. Run: python scripts/compute_norms.py --norm-id {norm_id} ..."
        )
    return Path(row[0]), Path(row[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--norm-id", required=True)
    parser.add_argument(
        "--vertex-csv",
        type=Path,
        default=ROOT / "configs" / "vertex_regions.csv",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=ROOT / "configs" / "calibrated_rules.yaml",
    )
    parser.add_argument(
        "--insight-catalog",
        type=Path,
        default=ROOT / "configs" / "insight_catalog.yaml",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--reducer", choices=("mean", "mean_abs", "median"), default="mean")
    parser.add_argument(
        "--ground",
        action="store_true",
        default=False,
        help=(
            "Run Spatial Credit Assignment after parcellation: detect dual-track spikes, "
            "load heatmaps from session_dir/heatmaps/t_<N>.npy, intersect with DOM bboxes, "
            "and append grounding events[] to analysis_bundle.json (schema v2). "
            "Requires run_dual_track.py to have been run first."
        ),
    )
    parser.add_argument(
        "--sections",
        action="store_true",
        default=False,
        help=(
            "Build section_report[] from full-session emotion/engagement tracks and "
            "session_manifest.json (hybrid URL/landmark/manual sections). "
            "Uses cached heatmaps for top_elements when present. "
            "Requires run_dual_track.py; manifest from record_session_manifest.py."
        ),
    )
    parser.add_argument(
        "--website",
        action="store_true",
        default=False,
        help=(
            "Website session mode: enables --sections, writes session_capture metadata, "
            "sets bundle schema_version=3. Requires session_manifest.json v2 + walkthrough video."
        ),
    )
    parser.add_argument(
        "--with-heatmaps",
        action="store_true",
        default=False,
        help="If heatmaps/ is missing, print hint to run extract_section_heatmaps.py --modal",
    )
    args = parser.parse_args()

    if args.website:
        args.sections = True

    conn = sqlite3.connect(DB_PATH)
    ensure_neuro_schema(conn)
    sess = conn.execute(
        "SELECT preds_npz_path FROM sessions WHERE id = ?",
        (args.session_id,),
    ).fetchone()
    if not sess:
        conn.close()
        raise SystemExit(f"Unknown session_id {args.session_id}")

    preds_path = PROJECT_ROOT / sess[0]
    raw = np.load(preds_path)
    preds = np.asarray(raw["preds"], dtype=np.float32)

    roi_parquet, net_parquet = _load_norm_bundle(conn, args.norm_id)
    roi_norms = pd.read_parquet(roi_parquet)
    net_norms = pd.read_parquet(net_parquet)

    table = load_vertex_table(args.vertex_csv)
    vp = dense_parcel_labels(table, preds.shape[1])
    pmap = parcel_to_network_map(table)

    parcel_ts, parcel_ids = parcel_timeseries(preds, vp, reducer=args.reducer)
    net_ts, net_ids = network_timeseries(parcel_ts, parcel_ids, pmap, reducer=args.reducer)

    z_p = zscore_roi(parcel_ts, roi_norms, parcel_ids)
    z_n = zscore_network(net_ts, net_norms, net_ids)
    names = network_names_for_ids(net_ids)

    clear_session_neuro_rows(conn, args.session_id, args.norm_id)
    insert_roi_timeseries_batch(
        conn,
        session_id=args.session_id,
        norm_id=args.norm_id,
        parcel_ts=parcel_ts,
        z_parcel=z_p,
        parcel_ids=parcel_ids,
    )
    insert_network_timeseries_batch(
        conn,
        session_id=args.session_id,
        norm_id=args.norm_id,
        net_ts=net_ts,
        z_net=z_n,
        network_ids=net_ids,
    )

    ctx = ThresholdContext(
        session_id=args.session_id,
        fps=args.fps,
        network_names=names,
        network_ts=net_ts.astype(float).tolist(),
        z_network=z_n.astype(float).tolist(),
    )
    hits = evaluate_rules(ctx, args.rules, args.insight_catalog)

    hit_rows = [
        {
            "session_id": h.session_id,
            "rule_id": h.rule_id,
            "t_start": h.t_start,
            "t_end": h.t_end,
            "networks_json": h.networks_json,
            "evidence_json": h.evidence_json,
            "confidence": h.confidence,
            "insight_key": h.insight_key,
            "insight_text": h.insight_text,
        }
        for h in hits
    ]
    insert_threshold_hits(conn, hit_rows)

    session_dir = PROJECT_ROOT / "scout_data" / "sessions" / args.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    out_json = analysis_bundle_path(session_dir)

    # Preserve dual-track fields that may have been written by run_dual_track.py.
    existing_bundle: dict[str, Any] = {}
    if out_json.is_file():
        try:
            existing_bundle = json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("WARNING: existing analysis_bundle.json is malformed — will overwrite.", file=sys.stderr)

    bundle = AnalysisBundle(
        session_id=args.session_id,
        norm_id=args.norm_id,
        fps=args.fps,
        parcel_ids=[int(x) for x in parcel_ids.tolist()],
        network_ids=[int(x) for x in net_ids],
        network_names=names,
        threshold_hits=hit_rows,
    )

    # Start from the new bundle dict, then layer in passthrough and grounding data.
    bundle_dict: dict[str, Any] = bundle.model_dump()

    # Restore dual-track fields written by run_dual_track.py (non-destructive merge).
    for key in _DUAL_TRACK_PASSTHROUGH_KEYS:
        if key in existing_bundle:
            bundle_dict[key] = existing_bundle[key]

    # -----------------------------------------------------------------------
    # Website session capture metadata (--website)
    # -----------------------------------------------------------------------
    if args.website:
        manifest = load_manifest(session_dir)
        if manifest is None:
            print(
                "WARNING: session_manifest.json missing — website mode expects v2 manifest.",
                file=sys.stderr,
            )
        else:
            align = validate_session_dir(session_dir)
            video_path = (manifest.get("video") or {}).get("path", "walkthrough.mp4")
            bundle_dict["session_capture"] = SessionCaptureMeta(
                video_path=video_path,
                manifest_schema_version=int(manifest.get("schema_version", 1)),
                tr_duration_sec=tr_duration_from_manifest(manifest),
                alignment_ok=align.get("ok") if not align.get("skipped") else None,
                alignment_message=align.get("message"),
            ).model_dump()
            bundle_dict["schema_version"] = 3
        heatmaps_dir = session_dir / "heatmaps"
        if args.with_heatmaps and not any(heatmaps_dir.glob("t_*.npy")):
            print(
                "  No heatmaps in session — run: "
                "python scripts/extract_section_heatmaps.py --session-id",
                args.session_id,
                "--modal --refresh-sections",
                file=sys.stderr,
            )

    # -----------------------------------------------------------------------
    # Section-level marketing analytics (--sections)
    # -----------------------------------------------------------------------
    if args.sections:
        print("\nSection analytics (--sections) …")
        if not bundle_dict.get("emotion_track"):
            print(
                "  No emotion_track in bundle — run run_dual_track.py first.",
                file=sys.stderr,
            )
        else:
            from scout_core.section_pipeline import run_section_analytics

            section_report = run_section_analytics(session_dir, bundle_dict)
            bundle_dict["section_report"] = section_report
            print(f"  {len(section_report)} section(s) in section_report[].")

    # -----------------------------------------------------------------------
    # Feature Isolation — Spatial Credit Assignment (--ground)
    # -----------------------------------------------------------------------
    if args.ground:
        print("\nSpatial Credit Assignment (--ground) …")
        iso_cfg = _load_iso_config()
        if not iso_cfg:
            print(
                f"  WARNING: {_ISO_CONFIG_PATH} not found — grounding skipped.",
                file=sys.stderr,
            )
        elif not bundle_dict.get("grounding_triggers"):
            print(
                "  No grounding_triggers in bundle — run run_dual_track.py first.",
                file=sys.stderr,
            )
        else:
            events = _run_grounding_step(session_dir, bundle_dict, iso_cfg)
            bundle_dict["events"] = events
            if bundle_dict.get("schema_version", 1) < 2:
                bundle_dict["schema_version"] = 2
            print(f"  {len(events)} grounding event(s) written to events[].")

    if args.website and bundle_dict.get("section_report") and bundle_dict.get("schema_version", 1) < 3:
        bundle_dict["schema_version"] = 3

    out_json.write_text(json.dumps(bundle_dict, indent=2), encoding="utf-8")

    conn.commit()
    conn.close()
    print(f"\nWrote neuro tables + {len(hit_rows)} threshold hits")
    print(out_json)


if __name__ == "__main__":
    main()
