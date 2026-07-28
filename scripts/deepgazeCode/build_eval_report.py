#!/usr/bin/env python3
"""Build contact sheet + evaluation_report.json for a DeepGaze MSDB run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    return float(np.percentile(xs, p))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run = args.run_dir
    out = run / "outputs"
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))

    overlays = []
    for row in manifest["results"]:
        overlays.append(Image.open(row["artifacts"]["overlay"]).convert("RGB"))

    thumb_w, thumb_h = 320, 180
    thumbs = [im.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR) for im in overlays]
    cols, rows = 4, 2
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 20))
    for i, im in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * thumb_w, r * thumb_h))
    sheet_path = run / "contact_sheet.png"
    sheet.save(sheet_path)

    model_times = [r["timings"]["model_seconds"] for r in manifest["results"]]
    total_times = [r["timings"]["total_seconds"] for r in manifest["results"]]
    warm = manifest["results"][0].get("warm_model_seconds") or []

    checks = []
    for row in manifest["results"]:
        dens = np.load(row["artifacts"]["density"])
        checks.append(
            {
                "id": row["stem"],
                "shape": list(dens.shape),
                "sum": float(dens.sum()),
                "min": float(dens.min()),
                "finite": bool(np.isfinite(dens).all()),
                "peak_xy": [row["summary"]["peak_x"], row["summary"]["peak_y"]],
                "model_s": row["timings"]["model_seconds"],
                "total_s": row["timings"]["total_seconds"],
                "entropy": row["summary"]["entropy"],
                "center_mass": row["summary"]["center_mass"],
                "top_5pct_mass": row["summary"]["top_5pct_mass"],
            }
        )

    report = {
        "run_id": run.name,
        "device": manifest["device"],
        "model_meta": manifest.get("model_meta"),
        "cold_load_seconds": manifest["cold_load_seconds"],
        "warmup_model_seconds_first_image": warm,
        "aggregate": {
            "n": len(model_times),
            "model_median_s": percentile(model_times, 50),
            "model_p95_s": percentile(model_times, 95),
            "total_median_s": percentile(total_times, 50),
            "total_p95_s": percentile(total_times, 95),
        },
        "density_checks": checks,
        "contact_sheet": str(sheet_path.resolve()),
        "license_status": "no_active_license_upstream",
        "visual_sanity": {
            "govuk": "Hotspots on cookie heading, GOV.UK logo, hero text, Menu.",
            "stripe": "Hotspots on headline, Sign in, CTA, faces in chat widget.",
            "shopify": "Hotspots on hero headline, person/head, Start for free CTAs.",
            "overall": "Plausible webpage attention; not wired to product scoring.",
        },
        "notes": [
            "Unknown-domain MSDB mode (dataset=None).",
            "pixels_per_degree=35 MIT1003 default; screenshots lack physical geometry.",
            "Checkout pages omitted per plan.",
            "AUC/sAUC/NSS not computed (no fixation GT).",
        ],
    }
    report_path = run / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# DeepGaze MSDB evaluation report",
        "",
        f"- Run: `{run.name}`",
        f"- Device: `{manifest['device']}`",
        f"- Cold load: **{manifest['cold_load_seconds']:.2f}s**",
        f"- Warm model (first image repeats): `{warm}`",
        f"- Model latency median / p95: "
        f"**{report['aggregate']['model_median_s']:.3f}s** / "
        f"**{report['aggregate']['model_p95_s']:.3f}s**",
        f"- End-to-end median / p95: "
        f"**{report['aggregate']['total_median_s']:.3f}s** / "
        f"**{report['aggregate']['total_p95_s']:.3f}s**",
        "",
        "## License",
        "",
        "Upstream DeepGaze has **no active license** (MIT fields commented out; "
        "commercial-use issue #15 unanswered). Evaluation only.",
        "",
        "## Per-image",
        "",
        "| id | model_s | peak_xy | center_mass | top_5pct_mass | density_sum |",
        "|----|---------|---------|-------------|---------------|-------------|",
    ]
    for c in checks:
        md_lines.append(
            f"| {c['id']} | {c['model_s']:.3f} | {c['peak_xy']} | "
            f"{c['center_mass']:.3f} | {c['top_5pct_mass']:.3f} | {c['sum']:.6f} |"
        )
    md_lines.extend(
        [
            "",
            f"Contact sheet: `{sheet_path.as_posix()}`",
            "",
            "## Visual sanity",
            "",
            "- GOV.UK: cookie heading / logo / hero / menu hotspots look plausible.",
            "- Stripe: headline, Sign in, CTA, chat faces.",
            "- Shopify: hero headline, person, Start for free CTAs.",
            "",
        ]
    )
    (run / "EVALUATION_REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print("wrote", report_path)
    print("wrote", sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
