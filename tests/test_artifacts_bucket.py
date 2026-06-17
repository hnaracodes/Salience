from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from services.infra.config import load_object_storage_config
from services.pipeline import artifacts


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "HeadBucket")


def test_ensure_bucket_creates_missing_minio_bucket(monkeypatch):
    client = MagicMock()
    client.head_bucket.side_effect = [_client_error("404")]

    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")

    cfg = load_object_storage_config()
    artifacts._ensure_bucket(client, "scout", endpoint=cfg.endpoint_url)

    client.create_bucket.assert_called_once_with(Bucket="scout")


def test_ensure_bucket_skips_create_for_production_b2(monkeypatch):
    client = MagicMock()
    client.head_bucket.side_effect = [_client_error("NoSuchBucket")]

    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")

    with pytest.raises(RuntimeError, match="does not exist"):
        cfg = load_object_storage_config()
        artifacts._ensure_bucket(client, "scout", endpoint=cfg.endpoint_url)

    client.create_bucket.assert_not_called()


def test_sanitize_html_comments_script_tags():
    out = artifacts._sanitize_html('<html><!-- script src="x"></script>')
    assert "<!-- script" in out
    assert "<script" not in out.replace("<!-- script", "")


def test_presign_uses_browser_endpoint_for_local_minio(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.delenv("S3_PUBLIC_ENDPOINT_URL", raising=False)
    cfg = load_object_storage_config()
    assert artifacts._browser_endpoint_url(cfg) == "http://127.0.0.1:9000"


def test_presign_respects_explicit_public_endpoint(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("S3_PUBLIC_ENDPOINT_URL", "http://localhost:9000")
    cfg = load_object_storage_config()
    assert artifacts._browser_endpoint_url(cfg) == "http://localhost:9000"
