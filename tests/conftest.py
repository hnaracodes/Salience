"""Pytest hooks for TribeV2."""

from __future__ import annotations

from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # noqa: ARG001
    """Skip archived NeuroEmo training tests in the default suite."""
    return "neuroEmoCode" in collection_path.parts
