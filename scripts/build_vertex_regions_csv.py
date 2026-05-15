#!/usr/bin/env python3
"""Build configs/vertex_regions.csv from label GIFTI (lh/rh) or synthetic demo rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "parcellation_manifest.yaml"

YEO7_NAMES = (
    "Vis",
    "SomMot",
    "DorsAttn",
    "SalVentAttn",
    "Limbic",
    "Cont",
    "Default",
)


def _try_import_nibabel():
    try:
        import nibabel as nib  # noqa: F401

        return nib
    except ImportError as e:
        raise SystemExit(
            "Reading label GIFTI requires nibabel. Install with: pip install nibabel"
        ) from e


def load_label_vector(path: Path) -> np.ndarray:
    nib = _try_import_nibabel()
    img = nib.load(str(path))
    arr = np.asarray(img.agg_data(), dtype=np.int64).ravel()
    return arr


def build_from_gifti(lh_path: Path, rh_path: Path, output_csv: Path) -> int:
    lh = load_label_vector(lh_path)
    rh = load_label_vector(rh_path)
    n_vertices = lh.size + rh.size
    rows = []
    for i, lab in enumerate(lh):
        rows.append((i, int(lab), f"Lparcel_{lab}", None, "", "lh"))
    offset = lh.size
    for j, lab in enumerate(rh):
        i = offset + j
        rows.append((i, int(lab), f"Rparcel_{lab}", None, "", "rh"))

    _validate_vertices(rows, n_vertices)
    yeo_ids = _parcel_to_yeo(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "vertex_index",
                "parcel_id",
                "parcel_label",
                "yeo_network_id",
                "yeo_network_name",
                "hemisphere",
            ]
        )
        for row, yn_id in zip(rows, yeo_ids, strict=True):
            vi, pid, plabel, _, _, hemi = row
            yn_name = YEO7_NAMES[yn_id - 1] if yn_id else ""
            w.writerow([vi, pid, plabel, yn_id or "", yn_name, hemi])

    return n_vertices


def _parcel_to_yeo(rows: list) -> list[int | None]:
    """Stable pseudo-Yeo bucket from parcel_id until real atlas mapping CSV exists."""
    out = []
    for row in rows:
        pid = row[1]
        yn = (abs(pid) % 7) + 1
        out.append(int(yn))
    return out


def _validate_vertices(rows: list[tuple], n_vertices: int) -> None:
    indices = [r[0] for r in rows]
    if sorted(indices) != list(range(n_vertices)):
        raise ValueError(
            f"vertex_index must be contiguous 0..{n_vertices - 1}, got mismatched set"
        )


def build_demo(n_vertices: int, n_parcels: int, output_csv: Path) -> None:
    if n_parcels < 1 or n_vertices < n_parcels:
        raise ValueError("n_vertices must be >= n_parcels >= 1")
    verts_per = n_vertices // n_parcels
    remainder = n_vertices % n_parcels

    rows = []
    vidx = 0
    for p in range(n_parcels):
        count = verts_per + (1 if p < remainder else 0)
        yn_id = (p % 7) + 1
        yn_name = YEO7_NAMES[yn_id - 1]
        for _ in range(count):
            hemi = "lh" if vidx < n_vertices // 2 else "rh"
            rows.append(
                (
                    vidx,
                    parcel_id,
                    f"demo_parcel_{parcel_id}",
                    yn_id,
                    yn_name,
                    hemi,
                )
            )
            vidx += 1

    _validate_vertices(rows, n_vertices)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "vertex_index",
                "parcel_id",
                "parcel_label",
                "yeo_network_id",
                "yeo_network_name",
                "hemisphere",
            ]
        )
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(
    manifest_path: Path,
    *,
    n_vertices: int,
    lh_gii: str | None,
    rh_gii: str | None,
    atlas_id: str,
) -> None:
    try:
        import yaml  # type: ignore
    except ImportError:
        manifest_path.write_text(
            json.dumps(
                {
                    "mesh": "fsaverage5",
                    "tribe_vertex_order": "facebook/tribev2",
                    "atlas_id": atlas_id,
                    "n_vertices_expected": n_vertices,
                    "artifact_sha256": {},
                    "inputs": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    data: dict = {}
    if manifest_path.is_file():
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    data["mesh"] = "fsaverage5"
    data["tribe_vertex_order"] = "facebook/tribev2"
    data["atlas_id"] = atlas_id
    data["n_vertices_expected"] = n_vertices
    sha_block = data.setdefault("artifact_sha256", {})
    inp_block = data.setdefault("inputs", {})
    if lh_gii:
        p = Path(lh_gii)
        sha_block["labels_gii_lh"] = sha256_file(p)
        inp_block["labels_gii_lh"] = str(p.resolve())
    if rh_gii:
        p = Path(rh_gii)
        sha_block["labels_gii_rh"] = sha256_file(p)
        inp_block["labels_gii_rh"] = str(p.resolve())
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Synthetic contiguous parcels + pseudo-Yeo7.")
    p_demo.add_argument("--n-vertices", type=int, required=True)
    p_demo.add_argument("--n-parcels", type=int, default=400)
    p_demo.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "vertex_regions.csv",
    )
    p_demo.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    p_gii = sub.add_parser("from-gifti", help="Concatenate lh+rh label GIFTI vertices.")
    p_gii.add_argument("--lh-label", type=Path, required=True)
    p_gii.add_argument("--rh-label", type=Path, required=True)
    p_gii.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "configs" / "vertex_regions.csv",
    )
    p_gii.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    args = parser.parse_args()
    if args.cmd == "demo":
        build_demo(args.n_vertices, args.n_parcels, args.output)
        update_manifest(
            args.manifest,
            n_vertices=args.n_vertices,
            lh_gii=None,
            rh_gii=None,
            atlas_id="synthetic_demo_v1",
        )
        print(f"Wrote {args.output} ({args.n_vertices} vertices)")
        print(f"Updated {args.manifest}")
        return

    nv = build_from_gifti(args.lh_label, args.rh_label, args.output)
    update_manifest(
        args.manifest,
        n_vertices=nv,
        lh_gii=str(args.lh_label),
        rh_gii=str(args.rh_label),
        atlas_id="gifti_labels_user_supplied",
    )
    print(f"Wrote {args.output} ({nv} vertices)")
    print(f"Updated {args.manifest}")


if __name__ == "__main__":
    main()
