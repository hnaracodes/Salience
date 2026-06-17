from __future__ import annotations

from pathlib import Path

from scout_core.storage_migrations import (
    PROJECT_ROOT,
    _portable_storage_path,
    resolve_storage_path,
)


def test_resolve_storage_path_handles_windows_absolute_norm_path(tmp_path, monkeypatch):
    norm_dir = tmp_path / "scout_norms" / "synthetic_bootstrap_v1"
    norm_dir.mkdir(parents=True)
    roi_file = norm_dir / "roi_norms.parquet"
    roi_file.write_bytes(b"parquet")

    monkeypatch.setattr(
        "scout_core.storage_migrations.PROJECT_ROOT",
        tmp_path,
    )

    stored = (
        r"C:\Users\dev\TribeV2\scout_norms\synthetic_bootstrap_v1\roi_norms.parquet"
    )
    resolved = resolve_storage_path(stored)
    assert resolved == roi_file
    assert resolved.is_file()


def test_resolve_storage_path_handles_backslash_relative_norm_path(tmp_path, monkeypatch):
    norm_dir = tmp_path / "scout_norms" / "synthetic_bootstrap_v1"
    norm_dir.mkdir(parents=True)
    roi_file = norm_dir / "roi_norms.parquet"
    roi_file.write_bytes(b"parquet")

    monkeypatch.setattr(
        "scout_core.storage_migrations.PROJECT_ROOT",
        tmp_path,
    )

    stored = r"scout_norms\synthetic_bootstrap_v1\roi_norms.parquet"
    resolved = resolve_storage_path(stored)
    assert resolved == roi_file
    assert resolved.is_file()


def test_portable_storage_path_uses_posix_slashes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scout_core.storage_migrations.PROJECT_ROOT",
        tmp_path,
    )
    target = tmp_path / "scout_norms" / "demo" / "roi_norms.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert _portable_storage_path(target) == "scout_norms/demo/roi_norms.parquet"
