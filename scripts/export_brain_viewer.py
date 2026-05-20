"""Export preds.npz + fsaverage5 mesh for the local Three.js brain viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = PROJECT_ROOT / "scout_data" / "sessions"
VIEWER_DIR = PROJECT_ROOT / "viewer"


def _load_fsaverage5_mesh() -> tuple[np.ndarray, np.ndarray]:
    from nilearn import datasets
    from nilearn.surface import load_surf_mesh

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    coords_list = []
    faces_list = []
    offset = 0
    for hemi in ("left", "right"):
        coords, faces = load_surf_mesh(fs[f"pial_{hemi}"])
        coords = np.asarray(coords, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        coords_list.append(coords)
        faces_list.append(faces + offset)
        offset += len(coords)
    return np.vstack(coords_list), np.vstack(faces_list)


def export_session_viewer(session_id: str, *, preds_path: Path | None = None) -> Path:
    session_dir = SESSIONS_DIR / session_id
    npz_path = preds_path or (session_dir / "preds.npz")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    preds = np.asarray(data["preds"], dtype=np.float32)
    n_t, n_v = preds.shape

    out_dir = session_dir / "viewer"
    out_dir.mkdir(parents=True, exist_ok=True)

    coords, faces = _load_fsaverage5_mesh()
    if len(coords) != n_v:
        raise ValueError(
            f"Mesh vertex count {len(coords)} != preds n_vertices {n_v}. "
            "TRIBE vertex order may differ from nilearn pial mesh — verify parcellation manifest."
        )

    coords.astype(np.float32).tofile(out_dir / "coords.bin")
    faces.astype(np.int32).tofile(out_dir / "faces.bin")
    preds.astype(np.float32).tofile(out_dir / "preds.bin")

    vmin = float(np.percentile(preds, 2))
    vmax = float(np.percentile(preds, 98))
    if vmax <= vmin:
        vmin, vmax = float(preds.min()), float(preds.max())

    manifest = {
        "session_id": session_id,
        "n_timesteps": int(n_t),
        "n_vertices": int(n_v),
        "n_faces": int(len(faces)),
        "vmin": vmin,
        "vmax": vmax,
        "coords_dtype": "float32",
        "faces_dtype": "int32",
        "preds_dtype": "float32",
        "preds_layout": "row_major_timestep_vertex",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    html_src = VIEWER_DIR / "brain_viewer.html"
    if html_src.is_file():
        (out_dir / "index.html").write_text(html_src.read_text(encoding="utf-8"), encoding="utf-8")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--preds", type=Path, default=None, help="Override preds.npz path")
    args = parser.parse_args()

    out = export_session_viewer(args.session_id, preds_path=args.preds)
    print(f"Viewer bundle: {out}")
    print(f"Open: file:///{out / 'index.html'.as_posix()}")


if __name__ == "__main__":
    main()
