from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml

DEFAULT_BUCKET = "hcp-openaccess"
DEFAULT_REGION = "us-east-1"
HCP_7T_SESSION_SUMMARY_ZIP = "HCP_Resources/CSV/sessionSummaryCSV_7T.zip"


@dataclass(frozen=True)
class HcpMovieRunSpec:
    folder: str
    dtseries_primary: str
    dtseries_fallback: str
    dtseries_legacy: str | None = None

    def dtseries_filenames(self) -> list[str]:
        names = [self.dtseries_primary, self.dtseries_fallback]
        if self.dtseries_legacy:
            names.append(self.dtseries_legacy)
        out: list[str] = []
        for name in names:
            if name not in out:
                out.append(name)
        return out


@dataclass(frozen=True)
class HcpManifest:
    bucket: str
    region: str
    prefix: str
    prefix_candidates: list[str]
    movie_runs: list[HcpMovieRunSpec]
    behavioral_csv_candidates: list[str]
    subject_ids: list[str]
    parcellation_artifact: str
    demographics_open_url: str | None = None
    demographics_bundled_csv: str | None = None
    session_summary_7t_zip: str = HCP_7T_SESSION_SUMMARY_ZIP

    @classmethod
    def from_yaml(cls, path: Path, *, prefix: str | None = None) -> HcpManifest:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        runs = [
            HcpMovieRunSpec(
                folder=str(r["folder"]),
                dtseries_primary=str(r["dtseries_primary"]),
                dtseries_fallback=str(r["dtseries_fallback"]),
                dtseries_legacy=str(r["dtseries_legacy"]) if r.get("dtseries_legacy") else None,
            )
            for r in raw["movie_runs"]
        ]
        candidates = [str(p) for p in raw.get("prefix_candidates", [])]
        resolved_prefix = prefix or (candidates[0] if candidates else "HCP_1200")
        return cls(
            bucket=str(raw.get("bucket", DEFAULT_BUCKET)),
            region=str(raw.get("region", DEFAULT_REGION)),
            prefix=resolved_prefix,
            prefix_candidates=candidates,
            movie_runs=runs,
            behavioral_csv_candidates=[str(p) for p in raw.get("behavioral_csv_candidates", [])],
            subject_ids=[str(s) for s in raw["subject_ids"]],
            parcellation_artifact=str(raw.get("parcellation_artifact", "configs/hcp_fslr_schaefer400_yeo7.npz")),
            demographics_open_url=str(raw["demographics_open_url"]) if raw.get("demographics_open_url") else None,
            demographics_bundled_csv=str(raw["demographics_bundled_csv"]) if raw.get("demographics_bundled_csv") else None,
            session_summary_7t_zip=str(raw.get("session_summary_7t_zip", HCP_7T_SESSION_SUMMARY_ZIP)),
        )


def make_s3_client(
    *,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = DEFAULT_REGION,
):
    import boto3

    kwargs: dict[str, Any] = {"region_name": region}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def s3_client_from_env() -> Any:
    access = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("HCP_ACCESS_KEY")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("HCP_SECRET_ACCESS_KEY")
    return make_s3_client(
        access_key=access,
        secret_key=secret,
        region=os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION),
    )


def subject_dtseries_key(
    manifest: HcpManifest,
    subject_id: str,
    run: HcpMovieRunSpec,
    *,
    use_fallback: bool = False,
) -> str:
    fname = run.dtseries_fallback if use_fallback else run.dtseries_primary
    return (
        f"{manifest.prefix}/{subject_id}/MNINonLinear/Results/"
        f"{run.folder}/{fname}"
    )


def behavioral_csv_keys(manifest: HcpManifest) -> list[str]:
    keys: list[str] = []
    for name in manifest.behavioral_csv_candidates:
        keys.append(f"{manifest.prefix}/{name}")
        keys.append(name)
    return keys


def head_object_exists(client: Any, bucket: str, key: str) -> bool:
    return head_object_error(client, bucket, key) is None


def head_object_error(client: Any, bucket: str, key: str) -> str | None:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return None
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        return str(code or type(exc).__name__)


def subject_dtseries_keys(
    manifest: HcpManifest,
    subject_id: str,
    run: HcpMovieRunSpec,
) -> list[str]:
    base = (
        f"{manifest.prefix}/{subject_id}/MNINonLinear/Results/"
        f"{run.folder}/"
    )
    return [f"{base}{fname}" for fname in run.dtseries_filenames()]


def list_prefix_exists(client: Any, bucket: str, prefix: str) -> bool:
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/", MaxKeys=1)
        return bool(resp.get("Contents") or resp.get("KeyCount", 0) > 0)
    except Exception:
        return False


def discover_prefix(
    client: Any,
    manifest: HcpManifest,
    *,
    probe_subject: str | None = None,
) -> str | None:
    probe_subjects = _probe_subject_ids(client, manifest, probe_subject)
    if not probe_subjects or not manifest.movie_runs:
        return None
    run = manifest.movie_runs[0]
    errors: set[str] = set()
    for prefix in manifest.prefix_candidates:
        trial = HcpManifest(
            bucket=manifest.bucket,
            region=manifest.region,
            prefix=prefix,
            prefix_candidates=manifest.prefix_candidates,
            movie_runs=manifest.movie_runs,
            behavioral_csv_candidates=manifest.behavioral_csv_candidates,
            subject_ids=manifest.subject_ids,
            parcellation_artifact=manifest.parcellation_artifact,
            demographics_open_url=manifest.demographics_open_url,
            session_summary_7t_zip=manifest.session_summary_7t_zip,
        )
        for subject in probe_subjects:
            for key in subject_dtseries_keys(trial, subject, run):
                err = head_object_error(client, manifest.bucket, key)
                if err is None:
                    return prefix
                if err:
                    errors.add(err)
    if errors.intersection({"403", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}):
        return None
    return None


def _probe_subject_ids(
    client: Any,
    manifest: HcpManifest,
    probe_subject: str | None,
) -> list[str]:
    if probe_subject:
        return [probe_subject]
    cohort = load_hcp_7t_subject_ids_from_s3(client, manifest)
    if cohort:
        return cohort[:20]
    return manifest.subject_ids[:50]


def load_hcp_7t_subject_ids_from_s3(client: Any, manifest: HcpManifest) -> list[str]:
    """Return subject IDs listed in the HCP 7T session summary zip on S3."""
    try:
        raw = download_s3_to_bytes(client, manifest.bucket, manifest.session_summary_7t_zip)
    except Exception:
        return []
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        subjects: list[str] = []
        for name in archive.namelist():
            stem = Path(name).name
            if not stem.endswith("_all.csv"):
                continue
            subjects.append(stem.removesuffix("_all.csv"))
        return sorted(subjects)


def subject_has_movie_data(
    client: Any,
    manifest: HcpManifest,
    subject_id: str,
) -> bool:
    for run in manifest.movie_runs:
        if resolve_dtseries_key(client, manifest, subject_id, run):
            return True
    return False


def filter_subjects_with_7t_movie(
    client: Any,
    manifest: HcpManifest,
    subject_ids: list[str],
) -> list[str]:
    cohort = set(load_hcp_7t_subject_ids_from_s3(client, manifest))
    candidates = [sid for sid in subject_ids if not cohort or sid in cohort]
    return [sid for sid in candidates if subject_has_movie_data(client, manifest, sid)]


def discover_prefix_diagnostics(
    client: Any,
    manifest: HcpManifest,
    *,
    probe_subject: str | None = None,
) -> dict[str, Any]:
    probe_subjects = _probe_subject_ids(client, manifest, probe_subject)
    subject = probe_subjects[0] if probe_subjects else None
    run = manifest.movie_runs[0] if manifest.movie_runs else None
    probes: list[dict[str, Any]] = []
    prefix_listings: dict[str, bool] = {}
    errors: set[str] = set()
    for prefix in manifest.prefix_candidates:
        prefix_listings[prefix] = list_prefix_exists(client, manifest.bucket, prefix)
        if not subject or not run:
            continue
        trial = HcpManifest(
            bucket=manifest.bucket,
            region=manifest.region,
            prefix=prefix,
            prefix_candidates=manifest.prefix_candidates,
            movie_runs=manifest.movie_runs,
            behavioral_csv_candidates=manifest.behavioral_csv_candidates,
            subject_ids=manifest.subject_ids,
            parcellation_artifact=manifest.parcellation_artifact,
        )
        for key in subject_dtseries_keys(trial, subject, run):
            err = head_object_error(client, manifest.bucket, key)
            probes.append({"prefix": prefix, "key": key, "error": err})
            if err:
                errors.add(err)
    return {
        "probe_subject": subject,
        "probe_subjects_n": len(probe_subjects),
        "seven_t_cohort_n": len(load_hcp_7t_subject_ids_from_s3(client, manifest)),
        "prefix_listings": prefix_listings,
        "probes": probes[:12],
        "errors": sorted(errors),
        "auth_failure": bool(
            errors.intersection(
                {"403", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}
            )
        ),
    }


def download_s3_to_bytes(client: Any, bucket: str, key: str) -> bytes:
    buf = BytesIO()
    client.download_fileobj(bucket, key, buf)
    return buf.getvalue()


def resolve_dtseries_key(
    client: Any,
    manifest: HcpManifest,
    subject_id: str,
    run: HcpMovieRunSpec,
) -> str | None:
    for key in subject_dtseries_keys(manifest, subject_id, run):
        if head_object_exists(client, manifest.bucket, key):
            return key
    return None
