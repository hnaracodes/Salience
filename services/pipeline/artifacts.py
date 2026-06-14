"""R2 / S3-compatible artifact storage for scan results."""

from __future__ import annotations

import logging
import mimetypes
import os
import warnings
from pathlib import Path

from activation_store import SESSIONS_DIR

logger = logging.getLogger(__name__)


def _client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
    )


def _sanitize_html(content: str) -> str:
    """Basic defense-in-depth: comment out any inline script tags."""
    if "<script" in content.lower():
        warnings.warn(
            "upload_viewer: <script> tag found in index.html; commenting out for upload safety.",
            stacklevel=3,
        )
        # Replace case-insensitively while preserving original casing of surrounding content.
        import re
        content = re.sub(r"<script", "<!-- script", content, flags=re.IGNORECASE)
        logger.warning("upload_viewer: <script> tags neutralized in index.html before upload.")
    return content


def upload_viewer(
    session_id: str,
    scan_id: str,
    *,
    sanitize: bool = True,
) -> str:
    """Upload ux_viewer/ directory to R2 under scans/<scan_id>/ux_viewer/.

    Parameters
    ----------
    session_id:
        Local session directory name under SESSIONS_DIR.
    scan_id:
        Key prefix for R2 storage: ``scans/<scan_id>/ux_viewer/``.
    sanitize:
        If True (default), comments out ``<script`` tags in index.html
        before upload as a lightweight defense-in-depth measure.

    Returns
    -------
    Public URL to the uploaded index.html, or a local file:// path
    when R2 is not configured (dev mode).
    """
    viewer_dir = SESSIONS_DIR / session_id / "ux_viewer"

    # Local dev fallback: no R2 configured.
    if not os.environ.get("R2_ENDPOINT_URL"):
        index = viewer_dir / "index.html"
        local_url = index.as_uri() if index.is_file() else viewer_dir.as_uri()
        logger.info("upload_viewer: R2 not configured; returning local path %s", local_url)
        return local_url

    bucket = os.environ["R2_BUCKET"]
    client = _client()

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

        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
        logger.debug("upload_viewer: uploaded %s → s3://%s/%s", file_path.name, bucket, key)

    public_url_base = os.environ.get("R2_PUBLIC_URL", "")
    if public_url_base:
        return f"{public_url_base.rstrip('/')}/scans/{scan_id}/ux_viewer/index.html"

    # Generate a presigned URL when no public base URL is set.
    presigned = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": f"scans/{scan_id}/ux_viewer/index.html"},
        ExpiresIn=604800,  # 7 days
    )
    return presigned
