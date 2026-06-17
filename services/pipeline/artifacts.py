"""S3-compatible artifact storage for scan results (Backblaze B2 in prod, MinIO locally)."""

from __future__ import annotations

import logging
import mimetypes
import warnings
from pathlib import Path

from activation_store import SESSIONS_DIR
from services.infra.config import load_object_storage_config

logger = logging.getLogger(__name__)


def _is_local_dev_object_store(endpoint: str) -> bool:
    lowered = endpoint.lower()
    return "minio" in lowered or "localhost:9000" in lowered or "127.0.0.1:9000" in lowered


def _browser_endpoint_url(cfg) -> str:
    """S3 endpoint hostname reachable from the user's browser (not Docker-internal)."""
    if cfg.public_endpoint_url:
        return cfg.public_endpoint_url
    if _is_local_dev_object_store(cfg.endpoint_url):
        return "http://127.0.0.1:9000"
    return cfg.endpoint_url


def _client(cfg, *, endpoint_url: str | None = None):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        config=Config(signature_version="s3v4"),
    )


def _ensure_bucket(client, bucket: str, *, endpoint: str) -> None:
    """Create the dev MinIO bucket when missing; production buckets must pre-exist."""
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

    if not _is_local_dev_object_store(endpoint):
        raise RuntimeError(
            f"Object storage bucket {bucket!r} does not exist at {endpoint}. "
            "Create it in Backblaze B2 (or your S3 provider) first."
        )

    client.create_bucket(Bucket=bucket)
    logger.info("Created missing dev bucket %r at %s", bucket, endpoint)


def _sanitize_html(content: str) -> str:
    """Basic defense-in-depth: comment out any inline script tags."""
    if "<script" in content.lower():
        warnings.warn(
            "upload_viewer: <script> tag found in index.html; commenting out for upload safety.",
            stacklevel=3,
        )
        import re

        content = re.sub(r"<script", "<!-- script", content, flags=re.IGNORECASE)
        logger.warning("upload_viewer: <script> tags neutralized in index.html before upload.")
    return content


def upload_viewer(
    session_id: str,
    scan_id: str,
    *,
    sanitize: bool = False,
) -> str:
    """Upload ux_viewer/ directory to object storage under scans/<scan_id>/ux_viewer/.

    sanitize=False by default: exported index.html is a trusted repo template and
    must keep its script tags for the viewer to boot.
    """
    viewer_dir = SESSIONS_DIR / session_id / "ux_viewer"
    cfg = load_object_storage_config()

    if not cfg.configured:
        index = viewer_dir / "index.html"
        local_url = index.as_uri() if index.is_file() else viewer_dir.as_uri()
        logger.info("upload_viewer: object storage not configured; returning local path %s", local_url)
        return local_url

    upload_client = _client(cfg)
    _ensure_bucket(upload_client, cfg.bucket, endpoint=cfg.endpoint_url)

    for file_path in sorted(viewer_dir.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(viewer_dir)
        key = f"scans/{scan_id}/ux_viewer/{relative.as_posix()}"

        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"

        if sanitize and file_path.name == "index.html":
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            body: bytes = _sanitize_html(raw).encode("utf-8")
        else:
            body = file_path.read_bytes()

        upload_client.put_object(
            Bucket=cfg.bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
        logger.debug("upload_viewer: uploaded %s → s3://%s/%s", file_path.name, cfg.bucket, key)

    if cfg.public_url:
        viewer_url = f"{cfg.public_url.rstrip('/')}/scans/{scan_id}/ux_viewer/index.html"
    else:
        presign_client = _client(cfg, endpoint_url=_browser_endpoint_url(cfg))
        viewer_url = presign_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": cfg.bucket, "Key": f"scans/{scan_id}/ux_viewer/index.html"},
            ExpiresIn=604800,
        )

    return viewer_url
