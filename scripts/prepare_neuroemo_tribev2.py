#!/usr/bin/env python3
"""Download NeuroEmo and format emotion-task fMRI for TribeV2-compatible training.

This script prepares OpenNeuro ds005700 (NeuroEmo) as supervised training data
whose feature tensor matches the public TribeV2 output shape:

    X_subject: (T, 20484) float32  # fsaverage5 vertices, lh then rh

It downloads the emotion-task BOLD runs, projects each 4D NIfTI onto the
fsaverage5 pial surfaces with nilearn.surface.vol_to_surf, assigns per-TR
emotion labels from the published task schedule, and writes a combined training
NPZ that can be consumed by a downstream classifier.

Important scientific note:
    The OpenNeuro files are raw BIDS fMRI. Direct projection to fsaverage5 is
    only valid if the volume is already aligned closely enough to the fsaverage
    coordinate space for your purpose. For publishable / serious model work,
    run motion correction, slice timing, coregistration, normalization to MNI,
    and smoothing first, then point --raw-dir at those normalized task-fe files
    and run with --skip-download.

Usage:
    python scripts/prepare_neuroemo_tribev2.py --subjects 1-40
    python scripts/prepare_neuroemo_tribev2.py --subjects 1,2,3 --bold-lag-s 6
    python scripts/prepare_neuroemo_tribev2.py --skip-download --raw-dir path/to/bids

Outputs:
    scout_data/neuroemo/raw/...
        Downloaded BIDS emotion-task files.

    scout_data/neuroemo/tribev2_surface/subjects/sub-XX_task-fe_fsaverage5.npz
        Per-subject surface timeseries, labels, and metadata.

    scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz
        Combined ML-ready samples:
            X:        (N, 20484) or (N, window_trs, 20484) float32
            y:        (N,) int64
            subject:  (N,) unicode subject id
            t_idx:    (N,) int64 source TR index
            time_s:   (N,) float32 source TR onset time
            labels:   class-name array ordered by y id
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "scout_data" / "neuroemo" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "scout_data" / "neuroemo" / "tribev2_surface"

DATASET_ID = "ds005700"
SNAPSHOT_VERSION = "1.2.0"
OPENNEURO_CRN_BASE = "https://openneuro.org/crn/datasets"
EXPECTED_FSAVERAGE5_VERTICES = 20484
EXPECTED_HEMI_VERTICES = EXPECTED_FSAVERAGE5_VERTICES // 2

EMOTION_CLASSES = ["calm", "afraid", "delighted", "depressed", "excited"]
WHITE_NOISE_CLASS = "white_noise"
NEUTRAL_CLASS = "neutral"
UNLABELED = "unlabeled"

# Published NeuroEmo emotion task: 30 s emotional clips interleaved with 30 s
# white-noise segments, total 600 s. Emotion labels are intentionally lower-case
# so they are stable for ML class ids and file metadata.
TASK_EVENTS: list[tuple[float, float, str]] = [
    (0.0, 30.0, "calm"),
    (30.0, 30.0, WHITE_NOISE_CLASS),
    (60.0, 30.0, "afraid"),
    (90.0, 30.0, WHITE_NOISE_CLASS),
    (120.0, 30.0, "delighted"),
    (150.0, 30.0, WHITE_NOISE_CLASS),
    (180.0, 30.0, "depressed"),
    (210.0, 30.0, WHITE_NOISE_CLASS),
    (240.0, 30.0, "excited"),
    (270.0, 30.0, WHITE_NOISE_CLASS),
    (300.0, 30.0, "delighted"),
    (330.0, 30.0, WHITE_NOISE_CLASS),
    (360.0, 30.0, "depressed"),
    (390.0, 30.0, WHITE_NOISE_CLASS),
    (420.0, 30.0, "calm"),
    (450.0, 30.0, WHITE_NOISE_CLASS),
    (480.0, 30.0, "excited"),
    (510.0, 30.0, WHITE_NOISE_CLASS),
    (540.0, 30.0, "afraid"),
    (570.0, 30.0, WHITE_NOISE_CLASS),
]


@dataclass(frozen=True)
class SubjectArtifact:
    subject: str
    n_trs: int
    n_vertices: int
    source_nifti: str
    source_json: str
    output_npz: str
    tr_seconds: float
    bold_lag_s: float
    n_labeled_trs: int
    class_counts: dict[str, int]


def _parse_subjects(raw: str) -> list[str]:
    """Parse '1-3,7,sub-10' into ['sub-01', 'sub-02', 'sub-03', 'sub-07', 'sub-10']."""
    subjects: list[str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item and not item.startswith("sub-"):
            lo_raw, hi_raw = item.split("-", maxsplit=1)
            lo, hi = int(lo_raw), int(hi_raw)
            subjects.extend(f"sub-{i:02d}" for i in range(lo, hi + 1))
        else:
            if item.startswith("sub-"):
                sid = item
            else:
                sid = f"sub-{int(item):02d}"
            subjects.append(sid)

    deduped = list(dict.fromkeys(subjects))
    if not deduped:
        raise ValueError("No subjects parsed. Use e.g. --subjects 1-40")
    return deduped


def _bids_relpaths(subject: str) -> tuple[Path, Path]:
    return (
        Path(subject) / "func" / f"{subject}_task-fe_bold.nii.gz",
        Path(subject) / "func" / f"{subject}_task-fe_bold.json",
    )


def _openneuro_crn_url(relpath: Path) -> str:
    # OpenNeuro's file endpoint encodes path separators as colons.
    colon_path = ":".join(relpath.parts)
    return f"{OPENNEURO_CRN_BASE}/{DATASET_ID}/snapshots/{SNAPSHOT_VERSION}/files/{colon_path}"


def _download_file(url: str, out_path: Path, *, force: bool = False, timeout: int = 60) -> None:
    if out_path.is_file() and not force:
        print(f"  exists: {out_path.relative_to(PROJECT_ROOT)}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    print(f"  downloading: {url}")
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        last_report = time.monotonic()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if total and now - last_report > 5:
                    pct = 100.0 * downloaded / total
                    print(f"    {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)")
                    last_report = now
    tmp_path.replace(out_path)


def download_subject(subject: str, raw_dir: Path, *, force: bool = False) -> tuple[Path, Path]:
    bold_rel, json_rel = _bids_relpaths(subject)
    bold_path = raw_dir / bold_rel
    json_path = raw_dir / json_rel
    print(f"\n{subject} download")
    _download_file(_openneuro_crn_url(json_rel), json_path, force=force)
    _download_file(_openneuro_crn_url(bold_rel), bold_path, force=force, timeout=300)
    return bold_path, json_path


def _load_repetition_time(json_path: Path, fallback: float = 3.0) -> float:
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return float(meta.get("RepetitionTime") or fallback)


def _surface_to_time_by_vertex(surface: np.ndarray, *, expected_vertices: int) -> np.ndarray:
    """Convert nilearn vol_to_surf output into (T, V_hemi)."""
    arr = np.asarray(surface, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] == expected_vertices:
        arr = arr.T
    elif arr.shape[1] == expected_vertices:
        pass
    else:
        raise ValueError(
            f"Unexpected surface shape {arr.shape}; expected one axis to be {expected_vertices}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def project_bold_to_fsaverage5(
    bold_path: Path,
    *,
    radius: float = 3.0,
    interpolation: str = "linear",
) -> np.ndarray:
    """Project a 4D BOLD NIfTI to TribeV2-style fsaverage5 surface shape (T, 20484)."""
    import nibabel as nib
    from nilearn import datasets
    from nilearn.surface import vol_to_surf

    img = nib.load(str(bold_path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D BOLD image, got shape {img.shape}: {bold_path}")

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    print("  projecting left hemisphere ...")
    lh = vol_to_surf(img, fs.pial_left, radius=radius, interpolation=interpolation)
    print("  projecting right hemisphere ...")
    rh = vol_to_surf(img, fs.pial_right, radius=radius, interpolation=interpolation)

    lh_tv = _surface_to_time_by_vertex(lh, expected_vertices=EXPECTED_HEMI_VERTICES)
    rh_tv = _surface_to_time_by_vertex(rh, expected_vertices=EXPECTED_HEMI_VERTICES)
    if lh_tv.shape[0] != rh_tv.shape[0]:
        raise ValueError(f"Left/right hemisphere TR mismatch: {lh_tv.shape} vs {rh_tv.shape}")

    surface = np.concatenate([lh_tv, rh_tv], axis=1)
    if surface.shape[1] != EXPECTED_FSAVERAGE5_VERTICES:
        raise ValueError(
            f"Projected surface has V={surface.shape[1]}, expected {EXPECTED_FSAVERAGE5_VERTICES}"
        )
    return surface.astype(np.float32)


def standardize_surface(surface: np.ndarray, mode: str) -> np.ndarray:
    """Apply simple subject-local scaling to reduce scanner-scale effects."""
    arr = np.asarray(surface, dtype=np.float32)
    if mode == "none":
        return arr
    if mode == "per_subject_vertex":
        mu = arr.mean(axis=0, keepdims=True)
        sigma = arr.std(axis=0, keepdims=True)
        return np.where(sigma < 1e-7, 0.0, (arr - mu) / (sigma + 1e-8)).astype(np.float32)
    if mode == "per_subject_global":
        mu = float(arr.mean())
        sigma = float(arr.std())
        if sigma < 1e-7:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - mu) / (sigma + 1e-8)).astype(np.float32)
    raise ValueError(f"Unknown standardization mode: {mode}")


def _event_label_at_time(stim_time_s: float, *, drop_transition_s: float) -> str:
    for onset, duration, label in TASK_EVENTS:
        start = onset + drop_transition_s
        end = onset + duration - drop_transition_s
        if start <= stim_time_s < end:
            return label
    return UNLABELED


def build_tr_labels(
    n_trs: int,
    tr_seconds: float,
    *,
    bold_lag_s: float,
    include_white_noise: bool,
    drop_transition_trs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, time_s) for each acquired TR."""
    times = np.arange(n_trs, dtype=np.float32) * float(tr_seconds)
    drop_transition_s = float(drop_transition_trs) * float(tr_seconds)
    labels: list[str] = []
    for t in times.tolist():
        stim_time = float(t) - float(bold_lag_s)
        label = _event_label_at_time(stim_time, drop_transition_s=drop_transition_s)
        if label == WHITE_NOISE_CLASS and not include_white_noise:
            label = UNLABELED
        labels.append(label)
    return np.asarray(labels, dtype=object), times


def _class_names(include_white_noise: bool, include_neutral: bool = False) -> list[str]:
    names = list(EMOTION_CLASSES)
    if include_neutral:
        names.append(NEUTRAL_CLASS)
    if include_white_noise:
        names.append(WHITE_NOISE_CLASS)
    return names


def _with_balanced_neutral(labels: np.ndarray) -> np.ndarray:
    """Convert a balanced subset of white-noise TRs to neutral labels.

    The NeuroEmo task has more white-noise TRs than each emotion class. To keep
    class priors balanced, this selects the same number of white-noise TRs as
    the smallest emotion class count and leaves the rest unlabeled.
    """
    out = np.asarray(labels, dtype=object).copy()
    emotion_counts = [int(np.sum(out == name)) for name in EMOTION_CLASSES]
    neutral_count = min(emotion_counts) if emotion_counts else 0
    white_noise_idx = np.flatnonzero(out == WHITE_NOISE_CLASS)
    out[white_noise_idx] = UNLABELED
    if neutral_count <= 0 or white_noise_idx.size == 0:
        return out

    if white_noise_idx.size <= neutral_count:
        selected = white_noise_idx
    else:
        positions = np.linspace(0, white_noise_idx.size - 1, neutral_count)
        selected = white_noise_idx[np.rint(positions).astype(np.int64)]
    out[selected] = NEUTRAL_CLASS
    return out


def _label_ids(labels: np.ndarray, class_names: list[str]) -> np.ndarray:
    name_to_id = {name: i for i, name in enumerate(class_names)}
    return np.asarray([name_to_id.get(str(label), -1) for label in labels], dtype=np.int64)


def _windowed_samples(
    surface: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
    times: np.ndarray,
    *,
    subject: str,
    window_trs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_t = np.flatnonzero(label_ids >= 0)
    if window_trs < 1:
        raise ValueError("--window-trs must be >= 1")

    half_left = window_trs - 1
    samples: list[np.ndarray] = []
    ys: list[int] = []
    subjects: list[str] = []
    t_indices: list[int] = []
    sample_times: list[float] = []

    for t in valid_t.tolist():
        start = t - half_left
        end = t + 1
        if start < 0:
            continue
        # Only keep windows whose member TRs have the same supervised label.
        if not np.all(labels[start:end] == labels[t]):
            continue
        window = surface[start:end]
        samples.append(window[0] if window_trs == 1 else window)
        ys.append(int(label_ids[t]))
        subjects.append(subject)
        t_indices.append(int(t))
        sample_times.append(float(times[t]))

    if not samples:
        shape = (0, surface.shape[1]) if window_trs == 1 else (0, window_trs, surface.shape[1])
        return (
            np.empty(shape, dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=f"<U{len(subject)}"),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.stack(samples).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(subjects),
        np.asarray(t_indices, dtype=np.int64),
        np.asarray(sample_times, dtype=np.float32),
    )


def _save_npz(path: Path, *, compress: bool, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_subject(
    subject: str,
    *,
    raw_dir: Path,
    out_dir: Path,
    skip_download: bool,
    force_download: bool,
    radius: float,
    interpolation: str,
    standardize: str,
    bold_lag_s: float,
    include_white_noise: bool,
    include_neutral: bool,
    drop_transition_trs: int,
    window_trs: int,
    compress: bool,
    write_subject_files: bool,
) -> tuple[SubjectArtifact, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if skip_download:
        bold_rel, json_rel = _bids_relpaths(subject)
        bold_path = raw_dir / bold_rel
        json_path = raw_dir / json_rel
        if not bold_path.is_file():
            raise FileNotFoundError(f"Missing BOLD file for {subject}: {bold_path}")
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing sidecar JSON for {subject}: {json_path}")
    else:
        bold_path, json_path = download_subject(subject, raw_dir, force=force_download)

    print(f"\n{subject} format")
    tr_seconds = _load_repetition_time(json_path)
    surface = project_bold_to_fsaverage5(
        bold_path,
        radius=radius,
        interpolation=interpolation,
    )
    surface = standardize_surface(surface, standardize)
    labels, times = build_tr_labels(
        surface.shape[0],
        tr_seconds,
        bold_lag_s=bold_lag_s,
        include_white_noise=include_white_noise or include_neutral,
        drop_transition_trs=drop_transition_trs,
    )
    if include_neutral:
        labels = _with_balanced_neutral(labels)
    class_names = _class_names(include_white_noise, include_neutral)
    label_ids = _label_ids(labels, class_names)

    subject_npz = out_dir / "subjects" / f"{subject}_task-fe_fsaverage5.npz"
    if write_subject_files:
        _save_npz(
            subject_npz,
            compress=compress,
            surface=surface,
            labels=labels.astype("U32"),
            label_ids=label_ids,
            time_s=times,
            class_names=np.asarray(class_names),
        )

    class_counts = {
        name: int(np.sum(labels == name))
        for name in class_names
    }
    artifact = SubjectArtifact(
        subject=subject,
        n_trs=int(surface.shape[0]),
        n_vertices=int(surface.shape[1]),
        source_nifti=str(bold_path),
        source_json=str(json_path),
        output_npz=str(subject_npz) if write_subject_files else "",
        tr_seconds=float(tr_seconds),
        bold_lag_s=float(bold_lag_s),
        n_labeled_trs=int(np.sum(label_ids >= 0)),
        class_counts=class_counts,
    )
    print(f"  surface shape: {surface.shape}; labeled TRs: {artifact.n_labeled_trs}")
    print("  class counts:", ", ".join(f"{k}={v}" for k, v in class_counts.items()))

    samples = _windowed_samples(
        surface,
        labels,
        label_ids,
        times,
        subject=subject,
        window_trs=window_trs,
    )
    return artifact, samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subjects", default="1-40", help="Subjects to process, e.g. 1-40 or 1,2,sub-03")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="BIDS root for downloaded/raw NeuroEmo files")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for surface + train artifacts")
    parser.add_argument("--skip-download", action="store_true", help="Use files already present under --raw-dir")
    parser.add_argument("--force-download", action="store_true", help="Re-download files even if present")
    parser.add_argument("--include-white-noise", action="store_true", help="Include all white-noise blocks as an extra class")
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Include a balanced subset of white-noise TRs as a neutral class",
    )
    parser.add_argument("--bold-lag-s", type=float, default=6.0, help="Shift labels by this hemodynamic lag in seconds")
    parser.add_argument(
        "--drop-transition-trs",
        type=int,
        default=0,
        help="Drop this many TRs at the start and end of each stimulus block after lag correction",
    )
    parser.add_argument(
        "--window-trs",
        type=int,
        default=1,
        help="Number of same-label TRs per sample. 1 gives X shape (N, 20484); >1 gives (N, window_trs, 20484)",
    )
    parser.add_argument(
        "--standardize",
        choices=("per_subject_vertex", "per_subject_global", "none"),
        default="per_subject_vertex",
        help="Feature scaling applied before writing samples",
    )
    parser.add_argument("--radius", type=float, default=3.0, help="nilearn vol_to_surf sampling radius")
    parser.add_argument(
        "--interpolation",
        choices=("linear", "nearest", "nearest_most_frequent"),
        default="linear",
        help="nilearn vol_to_surf interpolation",
    )
    parser.add_argument("--compress", action="store_true", help="Use compressed NPZ output; smaller but slower")
    parser.add_argument("--no-subject-files", action="store_true", help="Only write combined training NPZ")
    parser.add_argument("--no-combined", action="store_true", help="Only write per-subject surface NPZ files")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    subjects = _parse_subjects(args.subjects)
    class_names = _class_names(args.include_white_noise, args.include_neutral)

    print("NeuroEmo -> TribeV2-compatible surface formatter")
    print(f"  dataset: {DATASET_ID} snapshot {SNAPSHOT_VERSION}")
    print(f"  subjects: {', '.join(subjects)}")
    print(f"  raw_dir: {args.raw_dir}")
    print(f"  out_dir: {args.out_dir}")
    print(f"  classes: {', '.join(class_names)}")

    artifacts: list[SubjectArtifact] = []
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_subject: list[np.ndarray] = []
    all_t_idx: list[np.ndarray] = []
    all_time_s: list[np.ndarray] = []
    label_rows: list[dict[str, Any]] = []

    for subject in subjects:
        artifact, samples = prepare_subject(
            subject,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            skip_download=args.skip_download,
            force_download=args.force_download,
            radius=args.radius,
            interpolation=args.interpolation,
            standardize=args.standardize,
            bold_lag_s=args.bold_lag_s,
            include_white_noise=args.include_white_noise,
            include_neutral=args.include_neutral,
            drop_transition_trs=args.drop_transition_trs,
            window_trs=args.window_trs,
            compress=args.compress,
            write_subject_files=not args.no_subject_files,
        )
        artifacts.append(artifact)

        X, y, subject_arr, t_idx, time_s = samples
        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)
            all_subject.append(subject_arr)
            all_t_idx.append(t_idx)
            all_time_s.append(time_s)
            id_to_label = {i: name for i, name in enumerate(class_names)}
            for i in range(y.shape[0]):
                label_rows.append(
                    {
                        "sample_idx": len(label_rows),
                        "subject": str(subject_arr[i]),
                        "t_idx": int(t_idx[i]),
                        "time_s": float(time_s[i]),
                        "label_id": int(y[i]),
                        "label": id_to_label[int(y[i])],
                    }
                )

    if not args.no_combined:
        if not all_X:
            raise SystemExit("No labeled samples were produced; check label lag / transition settings.")

        X_combined = np.concatenate(all_X, axis=0).astype(np.float32)
        y_combined = np.concatenate(all_y, axis=0).astype(np.int64)
        subject_combined = np.concatenate(all_subject, axis=0)
        t_idx_combined = np.concatenate(all_t_idx, axis=0).astype(np.int64)
        time_s_combined = np.concatenate(all_time_s, axis=0).astype(np.float32)

        train_npz = args.out_dir / "neuroemo_tribev2_train.npz"
        _save_npz(
            train_npz,
            compress=args.compress,
            X=X_combined,
            y=y_combined,
            subject=subject_combined,
            t_idx=t_idx_combined,
            time_s=time_s_combined,
            labels=np.asarray(class_names),
            dataset_id=np.asarray(DATASET_ID),
            snapshot_version=np.asarray(SNAPSHOT_VERSION),
        )
        _write_rows_csv(args.out_dir / "neuroemo_tribev2_labels.csv", label_rows)
        print(f"\nCombined train NPZ: {train_npz}")
        print(f"  X shape: {X_combined.shape}")
        print(f"  y shape: {y_combined.shape}")

    metadata = {
        "dataset_id": DATASET_ID,
        "snapshot_version": SNAPSHOT_VERSION,
        "created_at_unix_ms": int(time.time() * 1000),
        "mesh": "fsaverage5",
        "vertex_order": "lh_then_rh_nilearn_fsaverage5",
        "tribev2_expected_shape": ["T", EXPECTED_FSAVERAGE5_VERTICES],
        "class_names": class_names,
        "task_events": [
            {"onset": onset, "duration": duration, "label": label}
            for onset, duration, label in TASK_EVENTS
        ],
        "preprocessing": {
            "projection": "nilearn.surface.vol_to_surf",
            "surface": "pial",
            "radius": args.radius,
            "interpolation": args.interpolation,
            "standardize": args.standardize,
            "bold_lag_s": args.bold_lag_s,
            "include_neutral": args.include_neutral,
            "neutral_source": "balanced_subset_of_white_noise_trs" if args.include_neutral else None,
            "drop_transition_trs": args.drop_transition_trs,
            "window_trs": args.window_trs,
        },
        "warning": (
            "OpenNeuro NeuroEmo files are raw BIDS fMRI. Direct fsaverage5 projection "
            "is a format bridge for model development, not a substitute for subject-level "
            "fMRI preprocessing and normalization."
        ),
        "subjects": [asdict(a) for a in artifacts],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
