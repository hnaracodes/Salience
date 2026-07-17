from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from salience.mux.tribe_mux import TribeDemographicMux, TribeMuxConfig


def save_mux_checkpoint(
    path: Path,
    mux: TribeDemographicMux,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "config": mux.config.__dict__,
        "mux_state_dict": mux.state_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_mux_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[TribeDemographicMux, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config_dict = payload.get("config", {})
    config = TribeMuxConfig(**config_dict)
    mux = TribeDemographicMux(config).to(device)
    mux.load_state_dict(payload["mux_state_dict"])
    return mux, payload


def write_metrics_json(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
