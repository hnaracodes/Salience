from __future__ import annotations

import pytest

from services.infra.config import (
    load_object_storage_config,
    load_redis_dsn,
    normalize_redis_dsn,
    normalize_s3_endpoint,
)


def test_normalize_s3_endpoint_fixes_https_typo():
    assert (
        normalize_s3_endpoint("https:s3.us-east-005.backblazeb2.com")
        == "https://s3.us-east-005.backblazeb2.com"
    )


def test_normalize_redis_dsn_upgrades_upstash_to_tls():
    url = "redis://default:secret@united-midge-113206.upstash.io:6379"
    assert normalize_redis_dsn(url) == (
        "rediss://default:secret@united-midge-113206.upstash.io:6379"
    )


def test_normalize_redis_dsn_leaves_localhost_untouched():
    assert normalize_redis_dsn("redis://localhost:6379") == "redis://localhost:6379"


def test_load_object_storage_config_prefers_s3_env(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "key-id")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("S3_BUCKET", "salience-prod")
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)

    cfg = load_object_storage_config()
    assert cfg.endpoint_url == "https://s3.us-east-005.backblazeb2.com"
    assert cfg.bucket == "salience-prod"
    assert cfg.configured


def test_load_object_storage_config_falls_back_to_legacy_r2_env(monkeypatch):
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("R2_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "minioadmin")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("R2_BUCKET", "scout")

    cfg = load_object_storage_config()
    assert cfg.endpoint_url == "http://minio:9000"
    assert cfg.bucket == "scout"


def test_load_redis_dsn_from_env(monkeypatch):
    monkeypatch.setenv(
        "REDIS_URL",
        "redis://default:pass@united-midge-113206.upstash.io:6379",
    )
    assert load_redis_dsn().startswith("rediss://")
