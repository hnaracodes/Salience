#!/usr/bin/env python3
"""Download EEV stimulus videos and build pilot manifests."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eevCode._config import load_eev_config, path_cfg  # noqa: E402
from scout_core.eevCode.align import parse_eev_csv  # noqa: E402


def _video_durations(df: pd.DataFrame) -> pd.Series:
    return df.groupby("video_id")["t_sec"].max()


def create_pilot_manifest(
    csv_path: Path,
    out_path: Path,
    *,
    n_videos: int = 50,
    min_duration_s: float = 30.0,
) -> list[str]:
    df = parse_eev_csv(csv_path)
    durations = _video_durations(df)
    candidates = durations[durations >= min_duration_s].sort_index()
    if candidates.empty:
        raise SystemExit(f"No videos >= {min_duration_s}s in {csv_path}")

    # Stratify by duration quartile.
    qs = candidates.quantile([0.25, 0.5, 0.75])
    bins = [0.0, qs.iloc[0], qs.iloc[1], qs.iloc[2], float("inf")]
    labels = ["q1", "q2", "q3", "q4"]
    bucket = pd.cut(candidates, bins=bins, labels=labels, include_lowest=True)

    per_bucket = max(1, n_videos // len(labels))
    selected: list[str] = []
    for label in labels:
        ids = candidates.index[bucket == label].tolist()
        selected.extend(ids[:per_bucket])
    if len(selected) < n_videos:
        remaining = [v for v in candidates.index.tolist() if v not in selected]
        selected.extend(remaining[: n_videos - len(selected)])
    selected = selected[:n_videos]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video_ids": selected, "n": len(selected), "source_csv": str(csv_path)}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Pilot manifest: {out_path} ({len(selected)} videos)")
    return selected


def _resolve_video_path(out_dir: Path, video_id: str) -> Path | None:
    """Return canonical {video_id}.mp4, coalescing yt-dlp fragment outputs when needed."""
    target = out_dir / f"{video_id}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return target
    candidates = [
        p
        for p in sorted(out_dir.glob(f"{video_id}.*"))
        if p.suffix.lower() in {".mp4", ".webm", ".mkv"} and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p.stat().st_size)
    if best != target:
        best.replace(target)
    return target if target.is_file() else None


def download_video(
    video_id: str,
    out_dir: Path,
    *,
    sleep_s: float = 1.0,
    ytdlp_format: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.mp4"
    existing = _resolve_video_path(out_dir, video_id)
    if existing is not None:
        return {"video_id": video_id, "status": "skipped", "path": str(existing)}

    url = f"https://www.youtube.com/watch?v={video_id}"
    ytdlp_bin = "yt-dlp"
    cmd = [
        ytdlp_bin,
        "-f",
        ytdlp_format or "best[ext=mp4]/best",
        "-o",
        str(out_path),
        "--no-playlist",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        resolved = _resolve_video_path(out_dir, video_id)
        status = "ok" if resolved is not None else "missing"
        out_path = resolved or out_path
    except FileNotFoundError:
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            ytdlp_format or "best[ext=mp4]/best",
            "-o",
            str(out_path),
            "--no-playlist",
            url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            resolved = _resolve_video_path(out_dir, video_id)
            status = "ok" if resolved is not None else "missing"
            out_path = resolved or out_path
        except FileNotFoundError:
            return {"video_id": video_id, "status": "error", "error": "yt-dlp not found on PATH or as python -m yt_dlp"}
        except subprocess.CalledProcessError as exc:
            return {
                "video_id": video_id,
                "status": "error",
                "error": (exc.stderr or exc.stdout or str(exc))[:500],
            }
    except subprocess.CalledProcessError as exc:
        return {
            "video_id": video_id,
            "status": "error",
            "error": (exc.stderr or exc.stdout or str(exc))[:500],
        }
    time.sleep(sleep_s)
    return {"video_id": video_id, "status": status, "path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--create-pilot", action="store_true")
    parser.add_argument("--create-full", action="store_true", help="Write all video IDs from train+val CSV")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--video-ids-file", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="EEV train.csv path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_eev_config(args.config)
    csv_dir = path_cfg(cfg, "csv_dir")
    videos_dir = path_cfg(cfg, "videos_dir")
    manifests_dir = path_cfg(cfg, "manifests_dir")
    pilot_cfg = cfg.get("pilot") or {}

    train_csv = args.csv or (csv_dir / (cfg.get("eev") or {}).get("csv_files", {}).get("train", "train.csv"))
    pilot_path = args.video_ids_file or (manifests_dir / "pilot_50.json")

    if args.create_pilot:
        if not train_csv.is_file():
            raise SystemExit(f"Missing EEV train CSV: {train_csv}")
        create_pilot_manifest(
            train_csv,
            pilot_path,
            n_videos=int(pilot_cfg.get("n_videos", 50)),
            min_duration_s=float(pilot_cfg.get("min_duration_s", 30)),
        )

    if args.create_full:
        eev_files = (cfg.get("eev") or {}).get("csv_files") or {}
        all_ids: set[str] = set()
        for split_name in ("train", "val"):
            split_csv = csv_dir / eev_files.get(split_name, f"{split_name}.csv")
            if split_csv.is_file():
                df = parse_eev_csv(split_csv)
                all_ids.update(df["video_id"].astype(str).tolist())
        if not all_ids:
            raise SystemExit("No video IDs found in train/val CSVs")
        full_path = manifests_dir / "full_split.json"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(
            json.dumps({"video_ids": sorted(all_ids), "n": len(all_ids)}, indent=2),
            encoding="utf-8",
        )
        print(f"Full manifest: {full_path} ({len(all_ids)} videos)")

    if args.download:
        if pilot_path.is_file():
            manifest = json.loads(pilot_path.read_text(encoding="utf-8"))
            video_ids = manifest.get("video_ids") or []
        elif train_csv.is_file():
            df = parse_eev_csv(train_csv)
            video_ids = sorted(df["video_id"].unique().tolist())
        else:
            raise SystemExit("Provide --video-ids-file or EEV train.csv")

        if args.limit:
            video_ids = video_ids[: args.limit]

        dl_cfg = cfg.get("download") or {}
        rows = []
        for vid in video_ids:
            print(f"Downloading {vid} …")
            rows.append(
                download_video(
                    vid,
                    videos_dir,
                    sleep_s=float(dl_cfg.get("rate_limit_sleep_s", 1.0)),
                    ytdlp_format=dl_cfg.get("ytdlp_format"),
                )
            )
        status_path = manifests_dir / "download_status.json"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        ok = sum(1 for r in rows if r.get("status") in ("ok", "skipped"))
        print(f"Download status: {status_path} ({ok}/{len(rows)} ok/skipped)")


if __name__ == "__main__":
    main()
