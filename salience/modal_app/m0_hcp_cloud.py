from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import modal
import pandas as pd

from data_prep.viability_partition import run_viability_analysis, write_m0_report
from salience.modal_app.image_defs import (
    M0_PREFLIGHT_PATH,
    M0_REPORT_PATH,
    M0_SUBJECTS_CSV,
    app,
    m0_image,
    volume,
)

MANIFEST_PATH = Path("/root/configs/hcp_7t_movie_manifest.yaml")
PARC_PATH = Path("/root/configs/hcp_fslr_schaefer400_yeo7.npz")
VERTEX_CSV = Path("/root/configs/vertex_regions.csv")
CAMCAN_DEMO_PATH = Path("/root/configs/camcan_participants.tsv")
CONFIGS_DIR = Path("/root/configs")


def _demographics_bundled_path(manifest: Any) -> Path | None:
    name = getattr(manifest, "demographics_bundled_csv", None)
    if not name:
        return None
    return CONFIGS_DIR / name


def _load_demographics(client: Any, manifest: Any):
    from data_prep.hcp_demographics import load_hcp_demographics_df

    return load_hcp_demographics_df(
        client,
        bucket=manifest.bucket,
        manifest_prefix=manifest.prefix,
        behavioral_candidates=manifest.behavioral_csv_candidates,
        open_url=manifest.demographics_open_url,
        bundled_csv_path=_demographics_bundled_path(manifest),
    )


@app.function(
    image=m0_image,
    secrets=[modal.Secret.from_name("hcp-aws-secret")],
    volumes={"/mnt/mux": volume},
    timeout=60 * 10,
    memory=2048,
)
def preflight_s3() -> dict[str, Any]:
    """Verify S3 prefix, behavioral CSV, and dtseries coverage for probe subjects."""
    from data_prep.hcp_s3 import (
        HcpManifest,
        behavioral_csv_keys,
        discover_prefix,
        discover_prefix_diagnostics,
        head_object_exists,
        load_hcp_7t_subject_ids_from_s3,
        resolve_dtseries_key,
        s3_client_from_env,
    )

    manifest = HcpManifest.from_yaml(MANIFEST_PATH)
    client = s3_client_from_env()
    prefix = discover_prefix(client, manifest)
    if not prefix:
        diag = discover_prefix_diagnostics(client, manifest)
        out = {
            "ok": False,
            "error": "Could not discover S3 prefix",
            "tried": manifest.prefix_candidates,
            "diagnostics": diag,
        }
        _write_json(M0_PREFLIGHT_PATH, out)
        volume.commit()
        return out

    manifest = HcpManifest.from_yaml(MANIFEST_PATH, prefix=prefix)
    probes = load_hcp_7t_subject_ids_from_s3(client, manifest)[:3]
    if not probes:
        probes = manifest.subject_ids[:3]
    run_coverage: dict[str, dict[str, bool]] = {}
    for sid in probes:
        run_coverage[sid] = {}
        for run in manifest.movie_runs:
            key = resolve_dtseries_key(client, manifest, sid, run)
            run_coverage[sid][run.folder] = key is not None

    behavioral_key = None
    for key in behavioral_csv_keys(manifest):
        if head_object_exists(client, manifest.bucket, key):
            behavioral_key = key
            break

    demo_ok = False
    demo_source = None
    try:
        _load_demographics(client, manifest)
        demo_ok = True
        demo_source = behavioral_key or str(_demographics_bundled_path(manifest) or manifest.demographics_open_url)
    except Exception:
        demo_ok = False

    ok = all(any(r.values()) for r in run_coverage.values()) and demo_ok
    out = {
        "ok": ok,
        "bucket": manifest.bucket,
        "prefix": manifest.prefix,
        "behavioral_key": behavioral_key,
        "demographics_ok": demo_ok,
        "demographics_source": demo_source,
        "seven_t_cohort_n": len(load_hcp_7t_subject_ids_from_s3(client, manifest)),
        "run_coverage": run_coverage,
        "n_subjects_manifest": len(manifest.subject_ids),
    }
    _write_json(M0_PREFLIGHT_PATH, out)
    volume.commit()
    return out


@app.function(
    image=m0_image,
    secrets=[modal.Secret.from_name("hcp-aws-secret")],
    timeout=60 * 15,
    memory=4096,
)
def extract_hcp_subject_features(subject_id: str, prefix: str):
    """Stream one subject's 7T movie runs from S3; return net_* features only."""
    from data_prep.hcp_cifti_network import (
        aggregate_run_features,
        dtseries_array_to_network_features,
        load_dtseries_from_bytes,
        load_parcellation_artifact,
    )
    from data_prep.hcp_s3 import HcpManifest, download_s3_to_bytes, resolve_dtseries_key, s3_client_from_env

    manifest = HcpManifest.from_yaml(MANIFEST_PATH, prefix=prefix)
    client = s3_client_from_env()
    artifact = load_parcellation_artifact(PARC_PATH)

    run_feats: list[dict[str, float]] = []
    for run in manifest.movie_runs:
        key = resolve_dtseries_key(client, manifest, subject_id, run)
        if not key:
            continue
        blob = download_s3_to_bytes(client, manifest.bucket, key)
        ts = load_dtseries_from_bytes(blob)
        run_feats.append(dtseries_array_to_network_features(ts, artifact))

    return aggregate_run_features(run_feats) or None


@app.function(
    image=m0_image,
    secrets=[modal.Secret.from_name("hcp-aws-secret")],
    timeout=60 * 5,
    memory=2048,
)
def load_hcp_demographics_remote(prefix: str):
    """Fetch HCP demographics once per pipeline run."""
    from data_prep.hcp_s3 import HcpManifest, s3_client_from_env

    manifest = HcpManifest.from_yaml(MANIFEST_PATH, prefix=prefix)
    client = s3_client_from_env()
    df = _load_demographics(client, manifest)
    return df.to_dict(orient="records")


@app.function(
    image=m0_image,
    secrets=[modal.Secret.from_name("hcp-aws-secret")],
    volumes={"/mnt/mux": volume},
    timeout=60 * 60 * 6,
    memory=8192,
)
def run_m0_pipeline(
    *,
    phase: str = "hcp",
    max_subjects: int = 0,
    n_permutations: int = 1000,
) -> dict[str, Any]:
    """Extract features, merge Cam-CAN if phase=hcp_camcan, run M0, write report."""
    from data_prep.hcp_s3 import HcpManifest, discover_prefix, filter_subjects_with_7t_movie, s3_client_from_env
    from data_prep.m0_cloud_camcan import build_camcan_subject_rows

    manifest = HcpManifest.from_yaml(MANIFEST_PATH)
    client = s3_client_from_env()
    prefix = discover_prefix(client, manifest)
    if not prefix:
        raise RuntimeError("S3 prefix discovery failed — run preflight first")
    manifest = HcpManifest.from_yaml(MANIFEST_PATH, prefix=prefix)
    available = filter_subjects_with_7t_movie(client, manifest, manifest.subject_ids)
    if not available:
        raise RuntimeError("No HCP 7T movie subjects found on S3 for discovered prefix")
    subject_ids = available[:max_subjects] if max_subjects > 0 else available

    demo_records = load_hcp_demographics_remote.remote(prefix)
    demographics = pd.DataFrame(demo_records)

    feature_results = list(
        extract_hcp_subject_features.map(subject_ids, [prefix] * len(subject_ids))
    )
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    notes_extra: list[str] = []
    from data_prep.hcp_demographics import subject_row_from_demographics

    for sid, net_feats in zip(subject_ids, feature_results, strict=True):
        if net_feats:
            rows.append(subject_row_from_demographics(demographics, sid, net_feats))
        else:
            skipped.append({"subject_id": sid, "reason": "no_runs"})

    if phase == "hcp_camcan":
        camcan_df, camcan_skipped = build_camcan_subject_rows(
            vertex_csv=VERTEX_CSV,
            limit=max_subjects if max_subjects > 0 else None,
            demographics_path=CAMCAN_DEMO_PATH,
        )
        skipped.extend(camcan_skipped)
        if not camcan_df.empty:
            rows.extend(camcan_df.to_dict(orient="records"))
            notes_extra.append(
                "camcan_openneuro_rest — ds000221 mirror has rest fMRI only (HCP uses 7T movie)"
            )

    if not rows:
        raise RuntimeError("No subject rows extracted")

    df = pd.DataFrame(rows)
    from scout_core.constants import YEO7_NAMES

    yeo7_cols = [f"net_{name}" for name in YEO7_NAMES]
    present = [c for c in yeo7_cols if c in df.columns]
    if present:
        df = df.dropna(subset=present, how="any")
    if df.empty:
        raise RuntimeError("No subject rows with finite network features")
    Path(M0_SUBJECTS_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(M0_SUBJECTS_CSV, index=False)

    is_hcp_only = phase == "hcp"
    if is_hcp_only:
        notes_extra = ["hcp_sex_only_probe", "single_site", "not_production_gate"]

    report = run_viability_analysis(
        df,
        data_source=f"cloud_s3:{phase}:{prefix}",
        is_synthetic=False,
        multi_site_validated=(phase == "hcp_camcan"),
        n_permutations=n_permutations,
    )
    report = report.model_copy(update={"notes": report.notes + notes_extra})
    if is_hcp_only:
        report = report.model_copy(update={"passed": False, "multi_site_validated": False})

    write_m0_report(report, Path(M0_REPORT_PATH))
    _write_json("/mnt/mux/viability/skipped_subjects.json", {"skipped": skipped})
    volume.commit()

    return {
        "passed": report.passed,
        "multi_site_validated": report.multi_site_validated,
        "n_subjects": report.n_subjects,
        "n_sites": report.n_sites,
        "subjects_csv": M0_SUBJECTS_CSV,
        "report_path": M0_REPORT_PATH,
        "skipped_n": len(skipped),
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.local_entrypoint()
def main(
    cmd: str = "pipeline",
    phase: str = "hcp",
    max_subjects: int = 0,
    n_permutations: int = 1000,
) -> None:
    if cmd == "preflight":
        out = preflight_s3.remote()
        print(json.dumps(out, indent=2))
        if not out.get("ok"):
            raise SystemExit(1)
        return
    if cmd == "pipeline":
        out = run_m0_pipeline.remote(
            phase=phase,
            max_subjects=max_subjects,
            n_permutations=n_permutations,
        )
        print(json.dumps(out, indent=2))
        if not out.get("passed") and phase == "hcp_camcan":
            print(
                "Note: production gate not passed (expected until go criteria met at scale).",
                file=sys.stderr,
            )
        return
    raise SystemExit(f"Unknown cmd: {cmd}")
