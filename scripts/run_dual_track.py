"""Run the Zero-Shot Dual-Track Engine for a saved session.

Loads preds.npz from scout_data/sessions/<id>/, runs engagement Z-score scoring
(Track 1) and emotion template cosine similarity (Track 2), persists results to
SQLite, and writes/merges analysis_bundle.json in the session directory.

Usage:
    # Both tracks (baseline required for Track 1):
    python scripts/run_dual_track.py --session-id <id> --baseline-session-id <baseline_id>

    # Skip Track 1 (no baseline available), run Track 2 only:
    python scripts/run_dual_track.py --session-id <id>

    # Override baseline path directly:
    python scripts/run_dual_track.py --session-id <id> --baseline-preds path/to/baseline.npz

    # Use a direct preds path (custom location):
    python scripts/run_dual_track.py --session-id <id> --preds path/to/preds.npz

Prerequisites:
    1. Download emotion templates: python scripts/download_emotion_templates.py
    2. Provide configs/vertex_regions.csv with yeo_network_name column (for Track 1)
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from activation_store import DATA_DIR, DB_PATH, SESSIONS_DIR
from scout_core.dual_track import (
    TEMPLATE_NAMES,
    compute_emotion_track,
    compute_engagement_track,
    find_grounding_triggers,
    load_network_indices,
    load_templates,
)
from scout_core.storage_migrations import (
    ensure_dual_track_schema,
    insert_emotion_trace,
    insert_engagement_trace,
    upsert_dual_track_meta,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "dual_track.yaml"


def _load_config() -> dict:
    if CONFIG_PATH.is_file():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_preds(session_id: str, override_path: Path | None) -> np.ndarray:
    path = override_path or (SESSIONS_DIR / session_id / "preds.npz")
    if not path.is_file():
        raise FileNotFoundError(f"preds.npz not found: {path}")
    return np.load(path)["preds"].astype(np.float32)


def _load_baseline(
    baseline_session_id: str | None,
    baseline_path: Path | None,
) -> tuple[np.ndarray | None, str | None]:
    """Return (preds_baseline, source_session_id)."""
    if baseline_path is not None:
        if not baseline_path.is_file():
            print(f"WARNING: baseline-preds path not found: {baseline_path}", file=sys.stderr)
            return None, None
        return np.load(baseline_path)["preds"].astype(np.float32), None

    if baseline_session_id is not None:
        p = SESSIONS_DIR / baseline_session_id / "preds.npz"
        if not p.is_file():
            print(f"WARNING: baseline session preds.npz not found: {p}", file=sys.stderr)
            return None, None
        return np.load(p)["preds"].astype(np.float32), baseline_session_id

    return None, None


def _merge_analysis_bundle(session_dir: Path, bundle_filename: str, updates: dict) -> dict:
    """Load existing analysis_bundle.json (if any), merge updates, return merged dict."""
    bundle_path = session_dir / bundle_filename
    existing: dict = {}
    if bundle_path.is_file():
        try:
            existing = json.loads(bundle_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"WARNING: existing {bundle_filename} is malformed — overwriting.", file=sys.stderr)
    existing.update(updates)
    return existing


def _db_connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_dual_track_schema(conn)
    conn.commit()
    return conn


def _session_exists_in_db(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return row is not None


def run(
    session_id: str,
    *,
    preds_override: Path | None = None,
    baseline_session_id: str | None = None,
    baseline_preds: Path | None = None,
    config: dict | None = None,
) -> dict:
    """Execute both dual-track scoring steps for a session.

    Returns the merged analysis_bundle dict that was written to disk.
    """
    cfg = config or _load_config()
    eng_cfg = cfg.get("engagement", {})
    emo_cfg = cfg.get("emotion", {})
    out_cfg = cfg.get("output", {})

    threshold_high = float(eng_cfg.get("threshold_high", 1.5))
    threshold_low = float(eng_cfg.get("threshold_low", -1.5))
    grounding_eng = float(eng_cfg.get("grounding_trigger", 2.0))
    min_base_trs = int(eng_cfg.get("min_baseline_trs", 30))
    van_name = str(eng_cfg.get("van_network_name", "VentralAttention"))
    dmn_name = str(eng_cfg.get("dmn_network_name", "DefaultMode"))

    template_dir_raw = emo_cfg.get("template_dir", "configs/emotion_templates")
    template_dir = PROJECT_ROOT / template_dir_raw
    template_names = emo_cfg.get("template_names") or TEMPLATE_NAMES
    grounding_emo_z = float(emo_cfg.get("grounding_z_trigger", 2.0))

    bundle_filename = str(out_cfg.get("analysis_bundle_filename", "analysis_bundle.json"))
    merge_existing = bool(out_cfg.get("merge_existing", True))

    session_dir = SESSIONS_DIR / session_id
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print(f"Loading preds for session {session_id} …")
    preds = _load_preds(session_id, preds_override)
    T, V = preds.shape
    print(f"  preds shape: {T} timesteps × {V} vertices")

    preds_baseline, baseline_src = _load_baseline(baseline_session_id, baseline_preds)
    if preds_baseline is not None:
        print(f"  baseline: {preds_baseline.shape[0]} TRs from {'path override' if baseline_preds else baseline_src}")
    else:
        print("  No baseline provided — Track 1 will skip label emission.")

    # -----------------------------------------------------------------------
    # Track 1 — Visual Engagement
    # -----------------------------------------------------------------------
    print("\nTrack 1 — Visual Engagement …")
    van_idx, dmn_idx, net_err = load_network_indices(van_name=van_name, dmn_name=dmn_name)
    if net_err:
        print(f"  WARNING: {net_err}")
        engagement_result = {
            "scores": [None] * T,
            "labels": [None] * T,
            "baseline_trs": 0,
            "baseline_flag": f"no_vertex_map: {net_err}",
            "thresholds": {"high": threshold_high, "low": threshold_low},
        }
    else:
        print(f"  VAN vertices: {len(van_idx)}  DMN vertices: {len(dmn_idx)}")
        engagement_result = compute_engagement_track(
            preds, preds_baseline, van_idx, dmn_idx,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            min_baseline_trs=min_base_trs,
        )
        flag = engagement_result.get("baseline_flag")
        if flag:
            print(f"  baseline_flag: {flag}")
        else:
            engaging = engagement_result["labels"].count("engaging")
            boring = engagement_result["labels"].count("boring")
            print(f"  Labelled: {engaging} engaging, {boring} boring, {T - engaging - boring} unlabelled")

    # -----------------------------------------------------------------------
    # Track 2 — Discrete Emotion
    # -----------------------------------------------------------------------
    print("\nTrack 2 — Discrete Emotion (Template Matching) …")
    templates, t_names, tmpl_err = load_templates(template_dir, template_names)
    if tmpl_err:
        print(f"  WARNING: {tmpl_err}")
        emotion_result = None
    else:
        print(f"  Loaded {len(t_names)} templates from {template_dir.name}/")
        emotion_result = compute_emotion_track(
            preds, templates, t_names, grounding_z_threshold=grounding_emo_z,
        )
        scores_arr = np.array(emotion_result["cosine_scores"])
        z_arr = np.array(emotion_result["z_scores"])
        peak_means = scores_arr.mean(axis=0)
        peak_z = z_arr.max(axis=0)
        summary = "  Mean cosine: " + ", ".join(
            f"{n}={v:.3f}" for n, v in zip(t_names, peak_means.tolist())
        )
        print(summary)
        z_summary = "  Peak Z:      " + ", ".join(
            f"{n}={v:.2f}" for n, v in zip(t_names, peak_z.tolist())
        )
        print(z_summary)

    # -----------------------------------------------------------------------
    # Grounding triggers
    # -----------------------------------------------------------------------
    triggers = find_grounding_triggers(
        engagement_result, emotion_result,
        engagement_trigger=grounding_eng,
        emotion_z_trigger=grounding_emo_z,
    )
    print(f"\nGrounding triggers: {len(triggers)}")

    # -----------------------------------------------------------------------
    # Persist to SQLite
    # -----------------------------------------------------------------------
    print("\nPersisting to SQLite …")
    conn = _db_connect()
    try:
        if not _session_exists_in_db(conn, session_id):
            print(
                f"  WARNING: session {session_id} not found in sessions table. "
                "Foreign key constraints are enabled — inserting dual-track rows without a sessions row "
                "will fail. Run `modal run tribe.py` first.",
                file=sys.stderr,
            )

        thresholds_json = json.dumps({"high": threshold_high, "low": threshold_low})
        upsert_dual_track_meta(
            conn,
            session_id=session_id,
            baseline_session_id=baseline_src,
            baseline_trs=engagement_result.get("baseline_trs"),
            baseline_flag=engagement_result.get("baseline_flag"),
            template_source="neurovault:collection:12383" if emotion_result else None,
            thresholds_json=thresholds_json,
        )

        if _session_exists_in_db(conn, session_id):
            insert_engagement_trace(
                conn,
                session_id=session_id,
                scores=engagement_result["scores"],
                labels=engagement_result["labels"],
            )
            if emotion_result:
                insert_emotion_trace(
                    conn,
                    session_id=session_id,
                    cosine_scores=emotion_result["cosine_scores"],
                    template_names=emotion_result["template_names"],
                )
        conn.commit()
        print("  Done.")
    finally:
        conn.close()

    # -----------------------------------------------------------------------
    # Write analysis_bundle.json
    # -----------------------------------------------------------------------
    bundle_updates = {
        "session_id": session_id,
        "engagement_track": engagement_result,
        "emotion_track": emotion_result,
        "grounding_triggers": triggers,
        "dual_track_schema_version": 2,
    }

    if merge_existing:
        bundle = _merge_analysis_bundle(session_dir, bundle_filename, bundle_updates)
    else:
        bundle = bundle_updates

    bundle_path = session_dir / bundle_filename
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nanalysis_bundle.json → {bundle_path}")

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--session-id", required=True, help="Session ID (hex string from tribe.py output)")
    parser.add_argument("--preds", type=Path, default=None, help="Override path to preds.npz")
    parser.add_argument(
        "--baseline-session-id", default=None,
        help="Session ID of the 60-second baseline recording (loads its preds.npz)",
    )
    parser.add_argument(
        "--baseline-preds", type=Path, default=None,
        help="Direct path to baseline preds.npz (overrides --baseline-session-id)",
    )
    args = parser.parse_args()

    bundle = run(
        args.session_id,
        preds_override=args.preds,
        baseline_session_id=args.baseline_session_id,
        baseline_preds=args.baseline_preds,
    )

    print("\nSummary")
    print("-------")
    et = bundle.get("engagement_track", {})
    print(f"  Engagement baseline_flag : {et.get('baseline_flag')}")
    if et.get("scores"):
        valid = [s for s in et["scores"] if s is not None]
        if valid:
            arr = np.array(valid)
            print(f"  Engagement scores        : min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}")
    emo = bundle.get("emotion_track")
    if emo:
        print(f"  Emotion templates        : {', '.join(emo.get('template_names', []))}")
    triggers = bundle.get("grounding_triggers", [])
    print(f"  Grounding triggers       : {len(triggers)}")
    print(f"\nRun inspect: python scripts/inspect_session.py --session-id {args.session_id}")


if __name__ == "__main__":
    main()
