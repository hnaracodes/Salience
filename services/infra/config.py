"""Environment-backed config for Upstash Redis and S3-compatible object storage (Backblaze B2 in prod)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from arq.connections import RedisSettings


def _env(*names: str, default: str = "") -> str:
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default


def normalize_s3_endpoint(url: str) -> str:
    """Normalize S3-compatible endpoint URLs (fixes common https: typo)."""
    url = url.strip()
    if not url:
        return ""
    if url.startswith("https:") and not url.startswith("https://"):
        url = "https://" + url[len("https:") :].lstrip("/")
    if url.startswith("http:") and not url.startswith("http://"):
        url = "http://" + url[len("http:") :].lstrip("/")
    return url.rstrip("/")


def normalize_redis_dsn(url: str) -> str:
    """Use TLS for Upstash when only an redis:// URL was provided."""
    url = (url.strip() or "redis://localhost:6379").strip()
    if "upstash.io" in url and url.startswith("redis://"):
        return "rediss://" + url[len("redis://") :]
    return url


@dataclass(frozen=True)
class ObjectStorageConfig:
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    public_url: str = ""
    public_endpoint_url: str = ""

    @property
    def configured(self) -> bool:
        return bool(
            self.endpoint_url
            and self.access_key_id
            and self.secret_access_key
            and self.bucket
        )


def load_object_storage_config() -> ObjectStorageConfig:
    """Load B2/S3 settings. Accepts S3_*, B2_*, or legacy R2_* (local MinIO compose)."""
    endpoint = normalize_s3_endpoint(
        _env("S3_ENDPOINT_URL", "B2_ENDPOINT_URL", "R2_ENDPOINT_URL")
    )
    return ObjectStorageConfig(
        endpoint_url=endpoint,
        access_key_id=_env("S3_ACCESS_KEY_ID", "B2_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
        secret_access_key=_env(
            "S3_SECRET_ACCESS_KEY", "B2_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"
        ),
        bucket=_env("S3_BUCKET", "B2_BUCKET", "R2_BUCKET", default="scout"),
        public_url=_env("S3_PUBLIC_URL", "B2_PUBLIC_URL", "R2_PUBLIC_URL"),
        public_endpoint_url=_env("S3_PUBLIC_ENDPOINT_URL", "R2_PUBLIC_ENDPOINT_URL"),
    )


def load_redis_dsn() -> str:
    return normalize_redis_dsn(_env("REDIS_URL", default="redis://localhost:6379"))


def load_redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(load_redis_dsn())
