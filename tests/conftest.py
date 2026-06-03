"""Pytest hooks for TribeV2."""

from __future__ import annotations

from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # noqa: ARG001
    """Skip archived research-track tests in the default suite."""
    parts = collection_path.parts
    args = [str(a).replace("\\", "/") for a in config.args]
    if "neuroEmoCode" in parts:
        if any("neuroEmoCode" in a for a in args):
            return False
        return True
    if "eevCode" in parts:
        if any("eevCode" in a for a in args):
            return False
        return True
    return False
