#!/usr/bin/env python3
"""Run TRIBE v2 on EEV videos and cache network feature NPZs."""

from __future__ import annotations

import argparse
import json
import msvcrt
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eevCode._config import load_eev_config, path_cfg  # noqa: E402
from scout_core.eevCode.features import load_features_npz, save_features_npz  # noqa: E402

DEFAULT_LOCK = PROJECT_ROOT / "scout_data" / "eevCode" / "manifests" / "generate_features.lock"


def _load_video_ids(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("video_ids") or payload.get("ids") or [])
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@contextmanager
def _exclusive_process_lock(lock_path: Path):
    """Prevent duplicate generate_features Modal workers (duplicate workers cancel Modal RPCs)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise SystemExit(
            f"Another generate_features.py process holds {lock_path}. "
            "Only one Modal worker may run at a time or in-flight inputs get canceled."
        ) from exc
    handle.write(str(os.getpid()))
    handle.flush()
    try:
        yield
    finally:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        handle.close()
        lock_path.unlink(missing_ok=True)


class TribeModalBatch:
    """One Modal app.run() for many videos — avoids reconnect churn and duplicate sessions."""

    def __init__(self) -> None:
        self._stack: list[object] = []
        self._inference = None

    def __enter__(self) -> TribeModalBatch:
        import modal

        from tribe import TribeInference, app

        enable = modal.enable_output()
        enable.__enter__()
        self._stack.append(enable)
        run = app.run()
        run.__enter__()
        self._stack.append(run)
        self._inference = TribeInference()
        return self

    def predict_from_video(self, video_path: Path) -> np.ndarray:
        if self._inference is None:
            raise RuntimeError("TribeModalBatch not entered")
        video_bytes = video_path.read_bytes()
        if hasattr(self._inference, "predict_brain_npz"):
            npz_bytes = self._inference.predict_brain_npz.remote(video_bytes)
            from scout_core.eevCode.features import preds_from_npz_bytes

            preds, _meta = preds_from_npz_bytes(npz_bytes)
            return preds
        preds_list = self._inference.predict_brain.remote(video_bytes)
        return np.asarray(preds_list, dtype=np.float32)

    def __exit__(self, exc_type, exc, tb) -> None:
        while self._stack:
            ctx = self._stack.pop()
            ctx.__exit__(exc_type, exc, tb)  # type: ignore[union-attr]


def process_video(
    video_id: str,
    *,
    videos_dir: Path,
    out_dir: Path,
    execute_tribe: bool,
    skip_existing: bool,
    include_raw_preds: bool,
    tr_hz: float,
    synthetic: bool = False,
    tribe_batch: TribeModalBatch | None = None,
) -> dict:
    out_path = out_dir / f"{video_id}_features.npz"
    if skip_existing and out_path.is_file():
        return {"video_id": video_id, "status": "skipped", "path": str(out_path)}

    if synthetic:
        rng = np.random.default_rng(abs(hash(video_id)) % (2**32))
        T = int(rng.integers(30, 120))
        preds = (rng.standard_normal((T, 20484), dtype=np.float32) * 0.05).astype(np.float32)
        save_features_npz(
            out_path,
            video_id=video_id,
            preds=preds,
            tr_hz=tr_hz,
            include_raw_preds=include_raw_preds,
        )
        return {"video_id": video_id, "status": "synthetic", "path": str(out_path), "T": T}

    video_path = videos_dir / f"{video_id}.mp4"
    if not video_path.is_file():
        return {"video_id": video_id, "status": "error", "error": f"missing video {video_path}"}

    if not execute_tribe:
        return {
            "video_id": video_id,
            "status": "error",
            "error": "pass --execute-tribe to run Modal inference",
        }

    if tribe_batch is None:
        return {
            "video_id": video_id,
            "status": "error",
            "error": "Modal batch session missing (internal error)",
        }

    try:
        preds = tribe_batch.predict_from_video(video_path)
    except Exception as exc:  # noqa: BLE001
        return {"video_id": video_id, "status": "error", "error": str(exc)[:500]}

    save_features_npz(
        out_path,
        video_id=video_id,
        preds=preds,
        tr_hz=tr_hz,
        include_raw_preds=include_raw_preds,
    )
    loaded = load_features_npz(out_path)
    return {
        "video_id": video_id,
        "status": "ok",
        "path": str(out_path),
        "T": int(loaded["X_feat"].shape[0]),
        "n_features": int(loaded["X_feat"].shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--video-ids-file", type=Path, required=True)
    parser.add_argument("--execute-tribe", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-raw-preds", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Build feature NPZ from random preds (no Modal/video)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override intermediates output directory")
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK, help="Exclusive lock path for Modal runs")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_eev_config(args.config)
    videos_dir = path_cfg(cfg, "videos_dir")
    out_dir = args.output_dir or path_cfg(cfg, "intermediates_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    tr_hz = float((cfg.get("features") or {}).get("tr_hz", 1.0))

    video_ids = _load_video_ids(args.video_ids_file)
    if args.limit:
        video_ids = video_ids[: args.limit]

    rows: list[dict] = []

    def _run_batch(tribe_batch: TribeModalBatch | None) -> None:
        for vid in video_ids:
            print(f"[features] {vid}", flush=True)
            rows.append(
                process_video(
                    vid,
                    videos_dir=videos_dir,
                    out_dir=out_dir,
                    execute_tribe=args.execute_tribe,
                    skip_existing=args.skip_existing,
                    include_raw_preds=args.include_raw_preds,
                    tr_hz=tr_hz,
                    synthetic=args.synthetic,
                    tribe_batch=tribe_batch,
                )
            )

    if args.execute_tribe and not args.synthetic:
        with _exclusive_process_lock(args.lock_file):
            print("[features] Opening single Modal session for batch (do not start a second worker)", flush=True)
            with TribeModalBatch() as tribe_batch:
                _run_batch(tribe_batch)
    else:
        _run_batch(None)

    ok = sum(1 for r in rows if r.get("status") in ("ok", "skipped", "synthetic"))
    errors = [r for r in rows if r.get("status") == "error"]
    print(f"Done: {ok}/{len(rows)} ok/skipped → {out_dir}", flush=True)
    if errors:
        print(f"Errors: {len(errors)}", flush=True)
        for row in errors[:5]:
            print(f"  {row.get('video_id')}: {row.get('error')}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
