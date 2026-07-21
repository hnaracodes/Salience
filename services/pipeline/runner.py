"""Portable pipeline runner: chains all TribeV2 scan stages in order.

Stages: capture → tribe → [demographic_mux] → dual_track → heatmaps → analyze →
        copy_signals → narrative → export_viewer

Each stage calls on_stage(stage_name, status) with "running" before
execution and "done" after, or "failed" if the stage errors out.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from activation_store import SESSIONS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

PRODUCTION_EXPLORE_CONFIG = PROJECT_ROOT / "configs" / "explore_production.yaml"
DEFAULT_NORM_ID = os.environ.get("PIPELINE_NORM_ID", "synthetic_bootstrap_v1")

StatusCallback = Callable[[str, str], None]


def _fake_tribe_mode() -> bool:
    return os.environ.get("FAKE_TRIBE", "0") == "1"


def _demographic_mux_enabled() -> bool:
    return os.environ.get("DEMOGRAPHIC_MUX", "0") == "1"


def _run_subprocess(args: list[str], stage: str) -> None:
    """Run a subprocess and raise RuntimeError if it exits non-zero."""
    logger.debug("Stage %s: running %s", stage, " ".join(str(a) for a in args))
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "Stage %s failed (exit %d):\nstdout: %s\nstderr: %s",
            stage,
            result.returncode,
            result.stdout[-2000:],
            result.stderr[-2000:],
        )
        raise RuntimeError(f"Stage {stage} failed (exit {result.returncode})")
    if result.stdout:
        logger.debug("Stage %s stdout: %s", stage, result.stdout[-1000:])


def _stage(
    name: str,
    on_stage: StatusCallback,
    fn: Callable[[], None],
) -> None:
    """Wrap a stage callable with status callbacks and error handling."""
    on_stage(name, "running")
    try:
        fn()
    except Exception as exc:
        on_stage(name, "failed")
        raise RuntimeError(f"Stage {name} failed: {exc}") from exc
    on_stage(name, "done")


# ---------------------------------------------------------------------------
# Individual stage implementations
# ---------------------------------------------------------------------------


def _stage_capture(session_id: str, url: str, config: dict[str, Any]) -> None:
    # Re-validate at worker time to close DNS-rebinding window between API intake and goto.
    from services.api.ssrf import SSRFViolation, validate_url  # noqa: PLC0415

    try:
        validate_url(url)
    except SSRFViolation as exc:
        raise RuntimeError(f"capture blocked: {exc.reason}") from exc

    script_path = config.get("explore_script", str(PRODUCTION_EXPLORE_CONFIG))
    python = sys.executable
    args = [
        python,
        str(PROJECT_ROOT / "scripts" / "record_website_session.py"),
        "--session-id", session_id,
        "--url", url,
        "--script", script_path,
    ]
    _run_subprocess(args, "capture")


def _sync_sqlite_session(session_id: str) -> None:
    """Register preds in activations.sqlite so dual_track FK constraints succeed."""
    import numpy as np

    from activation_store import save_cortical_timeseries, save_subcortical_timeseries

    session_dir = SESSIONS_DIR / session_id
    npz_path = session_dir / "preds.npz"
    if not npz_path.is_file():
        raise RuntimeError(f"Missing preds.npz for session {session_id}")

    preds = np.asarray(np.load(npz_path)["preds"], dtype=np.float32)
    source_video = "walkthrough.webm"
    for candidate in ("walkthrough.webm", "walkthrough.mp4"):
        if (session_dir / candidate).is_file():
            source_video = candidate
            break

    save_cortical_timeseries(
        preds,
        session_id=session_id,
        source_video=source_video,
        notes="pipeline runner",
    )

    sub_path = session_dir / "preds_subcortical.npz"
    if sub_path.is_file():
        sub_preds = np.asarray(np.load(sub_path)["preds"], dtype=np.float32)
        save_subcortical_timeseries(sub_preds, session_id=session_id)


def _stage_tribe(session_id: str) -> None:
    session_dir = SESSIONS_DIR / session_id

    if _fake_tribe_mode():
        # Generate synthetic preds: shape (T, 20484) float32 zeros.
        import numpy as np

        from scout_core.subcortical.io import save_subcortical_npz
        from scout_core.subcortical.tribev2_adapter import fake_subcortical_preds

        manifest_path = session_dir / "session_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            n_snapshots = len(manifest.get("dom_snapshots") or [])
            t_count = max(1, n_snapshots)
        else:
            t_count = 1

        preds = np.zeros((t_count, 20484), dtype=np.float32)
        out_path = session_dir / "preds.npz"
        np.savez(out_path, preds=preds)
        sub_preds = fake_subcortical_preds(t_count)
        save_subcortical_npz(session_dir, sub_preds)
        logger.info(
            "FAKE_TRIBE: wrote synthetic preds shape %s and subcortical %s → %s",
            preds.shape,
            sub_preds.shape,
            session_dir,
        )
        _sync_sqlite_session(session_id)
        return

    # Production: call Modal SDK.
    import modal  # type: ignore[import]

    cls = modal.Cls.lookup("tribe-v2-brain-sim", "TribeInference")
    video_path = session_dir / "walkthrough.webm"
    if not video_path.is_file():
        # Fall back to mp4 variant
        video_path = session_dir / "walkthrough.mp4"
    video_bytes = video_path.read_bytes()
    cortical_bytes, subcortical_bytes = cls().predict_brain_both_npz.remote(video_bytes)
    (session_dir / "preds.npz").write_bytes(cortical_bytes)
    (session_dir / "preds_subcortical.npz").write_bytes(subcortical_bytes)
    logger.info(
        "tribe: wrote preds.npz (%d bytes) and preds_subcortical.npz (%d bytes)",
        len(cortical_bytes),
        len(subcortical_bytes),
    )
    _sync_sqlite_session(session_id)


def _stage_demographic_mux(session_id: str) -> None:
    """Optional stage: preds-fallback mux when M0 passed and checkpoint configured."""
    from data_prep.viability_partition import m0_passed_for_training

    m0_path = PROJECT_ROOT / "scout_data" / "demographic" / "viability" / "m0_report.json"
    if not m0_passed_for_training(m0_path):
        logger.warning(
            "demographic_mux skipped: M0 gate not passed (%s)", m0_path
        )
        return

    ckpt = os.environ.get("DEMOGRAPHIC_MUX_CHECKPOINT", "")
    if not ckpt:
        logger.warning("demographic_mux skipped: DEMOGRAPHIC_MUX_CHECKPOINT not set")
        return

    cluster_ids = os.environ.get("DEMOGRAPHIC_MUX_CLUSTER_IDS", "0,1")
    ids_arg = cluster_ids.split(",")

    python = sys.executable
    args = [
        python,
        str(PROJECT_ROOT / "scripts" / "run_demographic_mux_session.py"),
        "--session-id",
        session_id,
        "--checkpoint",
        ckpt,
        "--cluster-ids",
        *ids_arg,
    ]
    _run_subprocess(args, "demographic_mux")


def _stage_dual_track(session_id: str) -> None:
    python = sys.executable
    args = [
        python,
        str(PROJECT_ROOT / "scripts" / "run_dual_track.py"),
        "--session-id", session_id,
    ]
    _run_subprocess(args, "dual_track")


def _stage_heatmaps(session_id: str) -> None:
    python = sys.executable
    script = str(PROJECT_ROOT / "scripts" / "extract_section_heatmaps.py")
    if _fake_tribe_mode():
        args = [python, script, "--session-id", session_id, "--uniform-heatmap"]
    else:
        args = [python, script, "--session-id", session_id, "--modal"]
    _run_subprocess(args, "heatmaps")


def _stage_analyze(session_id: str) -> None:
    python = sys.executable
    args = [
        python,
        str(PROJECT_ROOT / "scripts" / "analyze_session.py"),
        "--session-id", session_id,
        "--norm-id", DEFAULT_NORM_ID,
        "--website",
        "--ground",
        "--with-heatmaps",
    ]
    _run_subprocess(args, "analyze")


def _stage_copy_signals(session_id: str, site_goal: str | None) -> None:
    from scout_core.copy_signals import build_copy_signals

    build_copy_signals(session_id, site_goal=site_goal)


def _stage_narrative(session_id: str) -> None:
    python = sys.executable
    args = [
        python,
        str(PROJECT_ROOT / "scripts" / "generate_session_narrative.py"),
        "--session-id", session_id,
    ]
    _run_subprocess(args, "narrative")


def _stage_export_viewer(session_id: str, scan_id: str) -> str:
    python = sys.executable
    args = [
        python,
        str(PROJECT_ROOT / "scripts" / "export_ux_viewer.py"),
        "--session-id", session_id,
    ]
    _run_subprocess(args, "export_viewer")

    from services.pipeline.artifacts import upload_viewer

    viewer_url = upload_viewer(session_id, scan_id, sanitize=False)
    logger.info("export_viewer: viewer available at %s", viewer_url)
    return viewer_url


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_scan(
    session_id: str,
    url: str,
    site_goal: str | None,
    config: dict[str, Any],
    on_stage: StatusCallback,
    *,
    scan_id: str | None = None,
) -> dict[str, Any]:
    """Run the full TribeV2 scan pipeline for one session.

    Parameters
    ----------
    session_id:
        Unique session identifier (used for all file paths).
    url:
        Target URL to crawl.
    site_goal:
        Optional plain-language site goal used for copy-signal scoring.
    config:
        Pipeline config dict. Keys:
          - ``explore_script``: path to explore YAML (default: configs/explore_production.yaml)
    on_stage:
        Callback ``(stage_name, status)`` where status is "running" | "done" | "failed".
    scan_id:
        Optional scan identifier for artifact storage keys.
        Defaults to session_id if not provided.

    Returns
    -------
    dict with keys: ``session_id``, ``scan_id``, ``viewer_url``, ``stages_completed``.
    """
    effective_scan_id = scan_id or session_id
    stages_completed: list[str] = []
    viewer_url: str | None = None

    # Gate before any filesystem writes or subprocess spawns.
    from services.api.ssrf import SSRFViolation, validate_url  # noqa: PLC0415

    try:
        validate_url(url)
    except SSRFViolation as exc:
        on_stage("capture", "failed")
        raise RuntimeError(f"Scan blocked: {exc.reason}") from exc

    def _wrap(name: str, fn: Callable[[], None]) -> None:
        _stage(name, on_stage, fn)
        stages_completed.append(name)

    _wrap("capture", lambda: _stage_capture(session_id, url, config))
    _wrap("tribe", lambda: _stage_tribe(session_id))
    if _demographic_mux_enabled():
        _wrap("demographic_mux", lambda: _stage_demographic_mux(session_id))
    _wrap("dual_track", lambda: _stage_dual_track(session_id))
    _wrap("heatmaps", lambda: _stage_heatmaps(session_id))
    _wrap("analyze", lambda: _stage_analyze(session_id))
    _wrap("copy_signals", lambda: _stage_copy_signals(session_id, site_goal))
    _wrap("narrative", lambda: _stage_narrative(session_id))

    # export_viewer also returns the viewer URL
    on_stage("export_viewer", "running")
    try:
        viewer_url = _stage_export_viewer(session_id, effective_scan_id)
    except Exception as exc:
        on_stage("export_viewer", "failed")
        raise RuntimeError(f"Stage export_viewer failed: {exc}") from exc
    on_stage("export_viewer", "done")
    stages_completed.append("export_viewer")

    return {
        "session_id": session_id,
        "scan_id": effective_scan_id,
        "viewer_url": viewer_url,
        "stages_completed": stages_completed,
    }
