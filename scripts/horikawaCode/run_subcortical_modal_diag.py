"""Background-friendly Modal subcortical diagnostics (uses .venv311)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "scout_data" / "horikawaCode" / "pilot_run"
sys.path.insert(0, str(PROJECT_ROOT))


def _log(msg: str, log_path: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_checkpoint_health(log_path: Path) -> dict:
    import modal

    from tribe import TribeInference, app

    _log("Opening Modal session for verify_subcortical_checkpoint", log_path)
    with modal.enable_output(), app.run():
        inference = TribeInference()
        t0 = time.perf_counter()
        report = inference.verify_subcortical_checkpoint.remote()
        report["elapsed_s"] = round(time.perf_counter() - t0, 2)
    _log(f"Health check: {json.dumps(report)}", log_path)
    return report


def run_one_clip(stimulus_id: str, log_path: Path) -> dict:
    _log(f"Starting one-clip verify stimulus_id={stimulus_id}", log_path)
    from scripts.horikawaCode.verify_subcortical_modal import main as verify_main

    argv = ["verify_subcortical_modal.py", "--stimulus-id", stimulus_id]
    old_argv = sys.argv
    try:
        sys.argv = argv
        verify_main()
    finally:
        sys.argv = old_argv

    report_path = PROJECT_ROOT / "scout_data" / "horikawaCode" / "reports" / "subcortical_verify.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _log(f"Clip verify done: pass={report.get('analysis', {}).get('pass')}", log_path)
        return report
    return {"error": "subcortical_verify.json not written"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("health", "clip", "both"), default="both")
    parser.add_argument("--stimulus-id", default="1")
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_DIR / "subcortical_modal_diag.log",
    )
    args = parser.parse_args()

    _log(f"=== subcortical modal diag start mode={args.mode} ===", args.log)
    summary: dict = {"mode": args.mode, "steps": []}

    if args.mode in ("health", "both"):
        try:
            health = run_checkpoint_health(args.log)
            summary["steps"].append({"step": "health", **health})
            if not health.get("ok"):
                _log("STOP: subcortical checkpoint health failed", args.log)
                out = LOG_DIR / "subcortical_modal_diag.json"
                out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                sys.exit(1)
        except Exception as exc:
            _log(f"STOP: health check exception: {exc}", args.log)
            summary["steps"].append({"step": "health", "ok": False, "error": str(exc)})
            (LOG_DIR / "subcortical_modal_diag.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            sys.exit(1)

    if args.mode in ("clip", "both"):
        try:
            clip_report = run_one_clip(args.stimulus_id, args.log)
            summary["steps"].append({"step": "clip", **clip_report})
            if not clip_report.get("analysis", {}).get("pass"):
                _log("STOP: clip verify failed (not real subcortical or Modal error)", args.log)
                (LOG_DIR / "subcortical_modal_diag.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                sys.exit(1)
        except SystemExit as exc:
            _log(f"STOP: clip verify exited code={exc.code}", args.log)
            summary["steps"].append({"step": "clip", "ok": False, "exit_code": exc.code})
            (LOG_DIR / "subcortical_modal_diag.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            raise
        except Exception as exc:
            _log(f"STOP: clip verify exception: {exc}", args.log)
            summary["steps"].append({"step": "clip", "ok": False, "error": str(exc)})
            (LOG_DIR / "subcortical_modal_diag.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            sys.exit(1)

    summary["ok"] = True
    (LOG_DIR / "subcortical_modal_diag.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _log("=== subcortical modal diag PASS ===", args.log)


if __name__ == "__main__":
    main()
