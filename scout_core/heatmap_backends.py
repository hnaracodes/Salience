"""Heatmap backend selection and subprocess command construction."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Mapping

HEATMAP_BACKENDS = (
    "deepgaze_msdb",
    "modal",
    "visual_saliency",
    "uniform",
)
DEFAULT_HEATMAP_BACKEND = "deepgaze_msdb"


def resolve_heatmap_backend(
    requested: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve a backend from an explicit value, environment, or product default."""
    env = os.environ if environ is None else environ
    backend = (requested or env.get("PIPELINE_HEATMAP_BACKEND") or DEFAULT_HEATMAP_BACKEND).strip().lower()
    if backend not in HEATMAP_BACKENDS:
        choices = "|".join(HEATMAP_BACKENDS)
        raise ValueError(f"Unsupported heatmap backend '{backend}'. Use {choices}.")
    return backend


def resolve_deepgaze_python(
    project_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Locate the isolated DeepGaze interpreter and fail with setup guidance."""
    env = os.environ if environ is None else environ
    override = env.get("DEEPGAZE_MSDB_PYTHON") or env.get("PIPELINE_DEEPGAZE_PYTHON")
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
    elif sys.platform == "win32":
        candidate = project_root / ".venv-deepgaze-msdb" / "Scripts" / "python.exe"
    else:
        candidate = project_root / ".venv-deepgaze-msdb" / "bin" / "python"

    candidate = candidate.resolve()
    if not candidate.is_file():
        raise RuntimeError(
            "DeepGaze MSDB backend selected, but its isolated Python interpreter "
            f"was not found at {candidate}. Create .venv-deepgaze-msdb from "
            "requirements-deepgaze-msdb.txt or set DEEPGAZE_MSDB_PYTHON."
        )
    return candidate


def build_heatmap_stage_command(
    project_root: Path,
    session_id: str,
    *,
    backend: str | None = None,
    current_python: str | Path | None = None,
    all_frames: bool = True,
    force: bool = False,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Build one per-session heatmap command using the backend's environment."""
    selected = resolve_heatmap_backend(backend, environ=environ)
    if selected == "deepgaze_msdb":
        python = resolve_deepgaze_python(project_root, environ=environ)
    else:
        python = Path(current_python or sys.executable)

    command = [
        str(python),
        str(project_root / "scripts" / "extract_section_heatmaps.py"),
        "--session-id",
        session_id,
        "--backend",
        selected,
    ]
    if all_frames:
        command.append("--all-frames")
    if force:
        command.append("--force")
    return command
