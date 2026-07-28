#!/usr/bin/env python3
"""CLI: DeepGaze MSDB saliency for webpage screenshots (evaluation-only).

Example:
  .\\.venv-deepgaze-msdb\\Scripts\\python.exe scripts/deepgazeCode/run_frame_saliency.py \\
      --input scout_data/sessions/validation_govuk_20260722T180329Z_r1/frames/t_0.jpg \\
      --out-dir scout_data/deepgaze_msdb_eval/manual \\
      --device cuda --pixels-per-degree 35 --allow-default-ppd
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scout_core.deepgaze_msdb.inference import (  # noqa: E402
    load_model,
    predict_saliency_path,
)
from scout_core.deepgaze_msdb.io import write_prediction_artifacts  # noqa: E402
from scout_core.deepgaze_msdb.license_notice import license_banner  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path}")
        return [path]
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No images found under {path}")
        return files
    raise FileNotFoundError(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepGaze MSDB screenshot saliency (eval-only)")
    p.add_argument("--input", required=True, type=Path, help="Image file or directory")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--pixels-per-degree",
        type=float,
        default=None,
        help="Display pixels per degree of visual angle (required for honest geometry)",
    )
    p.add_argument(
        "--allow-default-ppd",
        action="store_true",
        help="Allow MIT1003 default ppd=35 when --pixels-per-degree is omitted",
    )
    p.add_argument(
        "--centerbias",
        default="mit1003",
        choices=["mit1003", "uniform"],
    )
    p.add_argument("--max-long-side", type=int, default=None)
    p.add_argument("--overlay-alpha", type=float, default=0.45)
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Extra warm inference repeats on the first image (timing only)",
    )
    p.add_argument(
        "--quiet-license",
        action="store_true",
        help="Suppress the license banner (still recorded in metadata)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.quiet_license:
        print(license_banner(), file=sys.stderr)

    if args.pixels_per_degree is None and not args.allow_default_ppd:
        print(
            "ERROR: pass --pixels-per-degree <float> or --allow-default-ppd "
            "(screenshots lack physical viewing geometry).",
            file=sys.stderr,
        )
        return 2

    inputs = _collect_inputs(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Cold load
    t_cold = time.perf_counter()
    model, device, model_meta = load_model(device=args.device, pretrained=True)
    cold_load_s = time.perf_counter() - t_cold

    run_rows: list[dict] = []
    input_is_dir = args.input.is_dir()
    for idx, image_path in enumerate(inputs):
        stem = image_path.stem
        # Disambiguate nested directory batches without mangling single-file stems.
        if input_is_dir and image_path.parent.resolve() != args.input.resolve():
            stem = f"{image_path.parent.name}__{stem}"

        result = predict_saliency_path(
            image_path,
            pixels_per_degree=args.pixels_per_degree,
            allow_default_ppd=args.allow_default_ppd,
            device=device,
            centerbias=args.centerbias,
            max_long_side=args.max_long_side,
            model=model,
            image_id=stem,
        )

        warm_times: list[float] = []
        if idx == 0 and args.warmup > 0:
            for _ in range(args.warmup):
                warm = predict_saliency_path(
                    image_path,
                    pixels_per_degree=args.pixels_per_degree,
                    allow_default_ppd=True,
                    device=device,
                    centerbias=args.centerbias,
                    max_long_side=args.max_long_side,
                    model=model,
                    image_id=stem,
                )
                warm_times.append(float(warm.timings["model_seconds"]))

        paths = write_prediction_artifacts(
            result,
            args.out_dir,
            stem=stem,
            source_image=image_path,
            write_overlay=not args.no_overlay,
            overlay_alpha=args.overlay_alpha,
        )
        row = {
            "image": str(image_path.resolve()),
            "stem": stem,
            "artifacts": paths,
            "summary": result.summary,
            "timings": result.timings,
            "warnings": result.warnings,
            "warm_model_seconds": warm_times,
        }
        run_rows.append(row)
        print(
            f"[{idx + 1}/{len(inputs)}] {stem}  "
            f"model={result.timings['model_seconds']:.3f}s  "
            f"total={result.timings['total_seconds']:.3f}s  "
            f"peak=({result.summary['peak_x']},{result.summary['peak_y']})"
        )

    manifest = {
        "model_meta": model_meta,
        "cold_load_seconds": cold_load_s,
        "device": device,
        "n_images": len(run_rows),
        "pixels_per_degree": args.pixels_per_degree,
        "centerbias": args.centerbias,
        "results": run_rows,
    }
    manifest_path = args.out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
