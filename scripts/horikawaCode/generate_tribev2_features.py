"""Run TRIBE cortical+subcortical inference on Horikawa clips via Modal."""

from __future__ import annotations

import argparse
import io
import json
import msvcrt
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LOCK = PROJECT_ROOT / "scout_data" / "horikawaCode" / "generate_tribev2_features.lock"
MODAL_APP_NAME = "tribe-v2-brain-sim"
FATAL_MODAL_MARKERS = (
    "APP_STATE_STOPPED",
    "local client disconnected",
    "Function call was cancelled",
    "Connection lost",
)


def _is_fatal_modal_error(message: str) -> bool:
    lower = message.lower()
    return any(marker.lower() in lower for marker in FATAL_MODAL_MARKERS)


@contextmanager
def _exclusive_process_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise SystemExit(
            f"Another generate_tribev2_features process holds {lock_path}. "
            "Only one Modal worker may run at a time."
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


class TribeModalBothBatch:
    """Modal session for predict_brain_both_npz (deployed class by default)."""

    def __init__(self, *, use_deployed: bool = True) -> None:
        self._stack: list[object] = []
        self._inference = None
        self._use_deployed = use_deployed

    def __enter__(self) -> TribeModalBothBatch:
        import modal

        enable = modal.enable_output()
        enable.__enter__()
        self._stack.append(enable)
        if self._use_deployed:
            tribe_cls = modal.Cls.from_name(MODAL_APP_NAME, "TribeInference")
            self._inference = tribe_cls()
            print("[horikawa] Using deployed Modal class (no local heartbeat)", flush=True)
        else:
            from tribe import TribeInference, app

            run = app.run()
            run.__enter__()
            self._stack.append(run)
            self._inference = TribeInference()
            print("[horikawa] Using ephemeral app.run() session", flush=True)
        return self

    def predict_both(self, video_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        if self._inference is None:
            raise RuntimeError("TribeModalBothBatch not entered")
        video_bytes = video_path.read_bytes()
        cortical_bytes, sub_bytes = self._inference.predict_brain_both_npz.remote(video_bytes)
        cortical = np.load(io.BytesIO(cortical_bytes))["preds"].astype(np.float32)
        sub_npz = np.load(io.BytesIO(sub_bytes))
        sub = sub_npz["preds"].astype(np.float32)
        meta = {
            "tribe_checkpoint": str(sub_npz.get("tribe_checkpoint", "")),
            "prediction_source": str(sub_npz.get("prediction_source", "unknown")),
        }
        return cortical, sub, meta

    def __exit__(self, exc_type, exc, tb) -> None:
        while self._stack:
            ctx = self._stack.pop()
            ctx.__exit__(exc_type, exc, tb)  # type: ignore[union-attr]


def _load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips") or []
    if clips and isinstance(clips[0], str):
        return [{"stimulus_id": c} for c in clips]
    return list(clips)


def _save_both_npz(
    out_path: Path,
    *,
    stimulus_id: str,
    cortical: np.ndarray,
    subcortical: np.ndarray,
    duration_s: float | None = None,
    tr_hz: float = 2.0,
    prediction_source: str | None = None,
    tribe_checkpoint: str | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "preds": cortical.astype(np.float32),
        "preds_subcortical": subcortical.astype(np.float32),
        "stimulus_id": np.asarray(stimulus_id),
        "duration_s": np.float32(duration_s or 0.0),
        "tr_hz": np.float32(tr_hz),
    }
    if prediction_source:
        payload["prediction_source"] = np.asarray(prediction_source)
    if tribe_checkpoint:
        payload["tribe_checkpoint"] = np.asarray(tribe_checkpoint)
    np.savez_compressed(out_path, **payload)


def _is_valid_subcortical_artifact(path: Path) -> bool:
    """True when NPZ already has real subcortical TRIBE output (not legacy proxy)."""
    try:
        with np.load(path, allow_pickle=True) as data:
            if "prediction_source" not in data:
                return False
            src = str(np.asarray(data["prediction_source"]))
            return src in {
                "tribev2_subcortical_checkpoint",
                "tribev2_subcortical_checkpoint_regions",
            }
    except OSError:
        return False


def process_clip(
    clip: dict,
    *,
    videos_dir: Path,
    out_dir: Path,
    execute_tribe: bool,
    skip_existing: bool,
    synthetic: bool,
    tr_hz: float,
    batch: TribeModalBothBatch | None,
) -> dict:
    sid = str(clip["stimulus_id"])
    out_path = out_dir / f"{sid}_both.npz"
    if skip_existing and out_path.is_file() and _is_valid_subcortical_artifact(out_path):
        return {"stimulus_id": sid, "status": "skipped", "path": str(out_path)}

    video_path = clip.get("video_path")
    if video_path:
        vp = Path(video_path)
    else:
        from scout_core.horikawaCode.labels import resolve_video_path

        vp = resolve_video_path(videos_dir, sid)

    if synthetic:
        rng = np.random.default_rng(abs(hash(sid)) % (2**32))
        t_count = int(rng.integers(4, 12))
        cortical = (rng.standard_normal((t_count, 20484), dtype=np.float32) * 0.05).astype(np.float32)
        subcortical = (rng.standard_normal((t_count, 8802), dtype=np.float32) * 0.05).astype(np.float32)
        _save_both_npz(out_path, stimulus_id=sid, cortical=cortical, subcortical=subcortical, tr_hz=tr_hz)
        return {"stimulus_id": sid, "status": "synthetic", "path": str(out_path), "T": t_count}

    if vp is None or not Path(vp).is_file():
        return {"stimulus_id": sid, "status": "error", "error": f"missing video for {sid}"}

    if not execute_tribe:
        return {"stimulus_id": sid, "status": "error", "error": "pass --execute-tribe for Modal inference"}

    if batch is None:
        return {"stimulus_id": sid, "status": "error", "error": "Modal batch missing"}

    t0 = time.perf_counter()
    try:
        cortical, sub, sub_meta = batch.predict_both(Path(vp))
    except Exception as exc:  # noqa: BLE001
        return {"stimulus_id": sid, "status": "error", "error": str(exc)[:500]}
    elapsed_s = time.perf_counter() - t0

    if cortical.shape[1] != 20484 or sub.shape[1] != 8802:
        return {
            "stimulus_id": sid,
            "status": "error",
            "error": f"bad shapes cortical={cortical.shape} sub={sub.shape}",
        }

    _save_both_npz(
        out_path,
        stimulus_id=sid,
        cortical=cortical,
        subcortical=sub,
        tr_hz=tr_hz,
        prediction_source=sub_meta.get("prediction_source"),
        tribe_checkpoint=sub_meta.get("tribe_checkpoint"),
    )
    return {
        "stimulus_id": sid,
        "status": "ok",
        "path": str(out_path),
        "T": int(cortical.shape[0]),
        "elapsed_s": round(elapsed_s, 2),
        "prediction_source": sub_meta.get("prediction_source"),
        "tribe_checkpoint": sub_meta.get("tribe_checkpoint"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "manifests" / "pilot_150.json",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "scout_data" / "horikawaCode" / "intermediates_modal",
    )
    parser.add_argument("--execute-tribe", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Random preds for offline tests")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--tr-hz", type=float, default=2.0)
    parser.add_argument(
        "--ephemeral-modal",
        action="store_true",
        help="Use app.run() instead of deployed TribeInference (not for long batches)",
    )
    args = parser.parse_args()

    clips = _load_manifest(args.manifest)
    if args.limit:
        clips = clips[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def _run(batch: TribeModalBothBatch | None) -> None:
        for clip in clips:
            sid = clip.get("stimulus_id")
            print(f"[horikawa] {sid}", flush=True)
            row = process_clip(
                clip,
                videos_dir=args.videos_dir,
                out_dir=args.output_dir,
                execute_tribe=args.execute_tribe,
                skip_existing=args.skip_existing,
                synthetic=args.synthetic,
                tr_hz=args.tr_hz,
                batch=batch,
            )
            rows.append(row)
            if row.get("status") == "error":
                err = str(row.get("error") or "")
                print(f"[horikawa] ERROR {sid}: {err[:200]}", flush=True)
                if _is_fatal_modal_error(err):
                    print(
                        "[horikawa] FATAL: Modal session lost — stopping batch "
                        "(re-run with --skip-existing to resume)",
                        flush=True,
                    )
                    break

    if args.execute_tribe and not args.synthetic:
        with _exclusive_process_lock(args.lock_file):
            print("[horikawa] Opening Modal session", flush=True)
            with TribeModalBothBatch(use_deployed=not args.ephemeral_modal) as batch:
                _run(batch)
    else:
        _run(None)

    report = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "results": rows,
        "ok": sum(1 for r in rows if r.get("status") in ("ok", "skipped", "synthetic")),
        "errors": [r for r in rows if r.get("status") == "error"],
    }
    report_path = args.output_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Done: {report['ok']}/{len(rows)} -> {args.output_dir}")
    if report["errors"]:
        for row in report["errors"][:5]:
            print(f"  {row['stimulus_id']}: {row.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
