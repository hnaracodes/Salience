"""Focused coverage for production heatmap backend selection and adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scout_core.heatmap_backends import (
    build_heatmap_stage_command,
    resolve_heatmap_backend,
)
from scout_core.heatmap_extract import adapt_heatmap_for_dom_scoring


def test_backend_selection_defaults_to_deepgaze_and_honors_environment():
    assert resolve_heatmap_backend(environ={}) == "deepgaze_msdb"
    assert resolve_heatmap_backend(environ={"PIPELINE_HEATMAP_BACKEND": "modal"}) == "modal"
    assert resolve_heatmap_backend("visual_saliency", environ={}) == "visual_saliency"
    with pytest.raises(ValueError, match="Unsupported heatmap backend"):
        resolve_heatmap_backend("unknown", environ={})


def test_deepgaze_command_uses_isolated_interpreter_and_every_frame(tmp_path):
    project_root = tmp_path / "repo"
    interpreter = tmp_path / "deepgaze-env" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()

    command = build_heatmap_stage_command(
        project_root,
        "session-1",
        backend="deepgaze_msdb",
        current_python="wrong-python",
        environ={"DEEPGAZE_MSDB_PYTHON": str(interpreter)},
    )

    assert Path(command[0]) == interpreter.resolve()
    assert Path(command[1]).name == "extract_section_heatmaps.py"
    assert Path(command[1]).parent.name == "scripts"
    assert command[command.index("--backend") + 1] == "deepgaze_msdb"
    assert "--all-frames" in command


def test_modal_rollback_command_uses_current_interpreter(tmp_path):
    command = build_heatmap_stage_command(
        tmp_path,
        "session-2",
        backend="modal",
        current_python="website-python",
        environ={},
    )
    assert command[0] == "website-python"
    assert command[command.index("--backend") + 1] == "modal"


def test_service_runner_uses_deepgaze_command_and_fake_uniform(tmp_path, monkeypatch):
    from services.pipeline import runner

    interpreter = tmp_path / "deepgaze-python.exe"
    interpreter.touch()
    monkeypatch.setenv("DEEPGAZE_MSDB_PYTHON", str(interpreter))
    monkeypatch.delenv("FAKE_TRIBE", raising=False)

    command = runner._heatmap_stage_command("service-session")
    assert Path(command[0]) == interpreter.resolve()
    assert command[command.index("--backend") + 1] == "deepgaze_msdb"
    assert "--all-frames" in command

    monkeypatch.setenv("FAKE_TRIBE", "1")
    fake_command = runner._heatmap_stage_command("fake-session")
    assert fake_command[fake_command.index("--backend") + 1] == "uniform"


def test_deepgaze_command_fails_clearly_when_interpreter_missing(tmp_path):
    with pytest.raises(RuntimeError, match="DEEPGAZE_MSDB_PYTHON"):
        build_heatmap_stage_command(
            tmp_path,
            "session-3",
            backend="deepgaze_msdb",
            environ={"DEEPGAZE_MSDB_PYTHON": str(tmp_path / "missing-python")},
        )


def test_probability_density_adapter_preserves_saved_values():
    density = np.array([[0.05, 0.15], [0.20, 0.60]], dtype=np.float32)
    original = density.copy()
    adapted = adapt_heatmap_for_dom_scoring(
        density,
        {"value_semantics": "fixation_probability_density"},
    )

    assert density == pytest.approx(original)
    assert float(density.sum()) == pytest.approx(1.0)
    assert float(adapted.max()) == pytest.approx(1.0)
    assert adapted[0, 0] == pytest.approx(density[0, 0] / density.max())


def test_every_captured_frame_uses_one_deepgaze_model(tmp_path, monkeypatch):
    from scripts import extract_section_heatmaps as script

    session_id = "deepgaze-every-frame"
    session_dir = tmp_path / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True)
    manifest = {
        "capture": {"height": 2, "width": 3},
        "tr_mapping": {"tr_duration_sec": 1.0},
        "dom_snapshots": [{"t_idx": i, "elements": []} for i in range(3)],
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for t in range(3):
        (frames_dir / f"t_{t}.jpg").write_bytes(b"fake-jpeg")

    model = object()
    load_calls: list[str] = []
    inference_models: list[object] = []

    def fake_load(device: str):
        load_calls.append(device)
        return model, "cpu", {"source": "fake"}

    def fake_extract(frame_path, capture_h, capture_w, *, model, **_kwargs):
        inference_models.append(model)
        density = np.full((capture_h, capture_w), 1.0 / (capture_h * capture_w), dtype=np.float32)
        return density, {
            "value_semantics": "fixation_probability_density",
            "normalization": "sum_1",
            "density_sum": 1.0,
            "dom_scoring_adapter": "max_normalize",
            "license_status": "no_active_license_upstream",
            "provenance": {
                "model_id": "fake-msdb",
                "dataset_mode": None,
                "pixels_per_degree": 35.0,
                "centerbias": {"mode": "uniform"},
                "device": "cpu",
            },
            "timings": {"model_seconds": 0.0},
            "warnings": [],
        }

    monkeypatch.setattr(script, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(script, "find_walkthrough_video", lambda _session_dir: None)
    monkeypatch.setattr(script, "_load_deepgaze_model", fake_load)
    monkeypatch.setattr(script, "extract_heatmap_deepgaze", fake_extract)

    script.main([
        "--session-id",
        session_id,
        "--backend",
        "deepgaze_msdb",
        "--all-frames",
    ])

    assert load_calls == ["auto"]
    assert inference_models == [model, model, model]
    entries = json.loads(
        (session_dir / "heatmaps" / "manifest.json").read_text(encoding="utf-8")
    )["entries"]
    assert [entry["t_idx"] for entry in entries] == [0, 1, 2]
    assert all(entry["source"] == "deepgaze_msdb" for entry in entries)
    for t in range(3):
        saved = np.load(session_dir / "heatmaps" / f"t_{t}.npy")
        assert saved.dtype == np.float32
        assert float(saved.sum()) == pytest.approx(1.0)
