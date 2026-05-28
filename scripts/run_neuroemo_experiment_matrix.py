#!/usr/bin/env python3
"""Build and optionally run NeuroEmo experiment matrices.

This runner covers phases 2-end of
`.cursor/plans/neuroemo_remaining_high_leverage_opportunities_2026-05-27.plan.md`.
It intentionally does not run or mutate the Modal vertex-equivalence proof
workflow from phase 1.

By default the script writes a manifest of reproducible commands. Pass
`--execute` to run them locally with the current Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_NPZ = Path("scout_data/neuroemo/tribev2_surface/neuroemo_tribev2_train.npz")
DEFAULT_OUTPUT_DIR = Path("scout_data/neuroemo/models/2026-05-27_surface_annot_matrix")
DEFAULT_TIMING_DIR = Path("scout_data/neuroemo/timing_grid")
DEFAULT_PREPROCESSED_DIR = Path("scout_data/neuroemo/tribev2_surface_preprocessed")
DEFAULT_PREPROC_CACHE_DIR = Path("scout_data/neuroemo/preprocessed")
DEFAULT_PROJECTED_BASELINE_DIR = Path("scout_data/neuroemo/models/2026-05-25_postfix_matrix")


@dataclass(frozen=True)
class RunSpec:
    name: str
    phase: str
    description: str
    commands: tuple[tuple[str, ...], ...]
    metrics_path: str
    model_path: str | None = None


def _rel(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _python(script: str) -> tuple[str, str]:
    return (sys.executable, script)


def _model_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"{name}.joblib"


def _metrics_path(output_dir: Path, name: str) -> Path:
    underscored = output_dir / f"{name}_metrics.json"
    dotted = output_dir / f"{name}.metrics.json"
    return underscored if underscored.exists() else dotted


def _base_train_args(
    *,
    train_npz: Path,
    model_path: Path,
    metrics_path: Path,
    temporal_stride_trs: int = 1,
    temporal_reducer: str = "mean",
    roi_reducers: str = "mean,std,mean_abs",
    exclude_labels: str = "neutral",
) -> list[str]:
    return [
        "--train-npz",
        _rel(train_npz),
        "--temporal-window-trs",
        "10",
        "--temporal-stride-trs",
        str(temporal_stride_trs),
        "--temporal-contiguity",
        "contiguous",
        "--temporal-reducer",
        temporal_reducer,
        "--roi-reducers",
        roi_reducers,
        "--exclude-labels",
        exclude_labels,
        "--metrics-json",
        _rel(metrics_path),
        "--output-model",
        _rel(model_path),
    ]


def surface_postfix_specs(output_dir: Path, train_npz: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for name, model_type in [
        ("logistic_saga_5class_10tr", "logistic_saga"),
        ("sgd_logistic_5class_10tr", "sgd_logistic"),
        ("linear_svc_5class_10tr", "linear_svc"),
    ]:
        model_path = _model_path(output_dir, name)
        metrics_path = _metrics_path(output_dir, name)
        cmd = (
            *_python("scripts/train_neuroemo_emotion_model.py"),
            "--model-type",
            model_type,
            *_base_train_args(train_npz=train_npz, model_path=model_path, metrics_path=metrics_path),
        )
        specs.append(
            RunSpec(
                name=name,
                phase="phase2_surface_postfix_matrix",
                description="Flat classical 5-class surface-native baseline.",
                commands=(cmd,),
                metrics_path=_rel(metrics_path),
                model_path=_rel(model_path),
            )
        )

    mlp_model = _model_path(output_dir, "mlp_5class_10tr")
    mlp_metrics = _metrics_path(output_dir, "mlp_5class_10tr")
    specs.append(
        RunSpec(
            name="mlp_5class_10tr",
            phase="phase2_surface_postfix_matrix",
            description="Small MLP 5-class surface-native baseline.",
            commands=(
                (
                    *_python("scripts/train_neuroemo_mlp_model.py"),
                    "--hidden-layers",
                    "64",
                    "--alpha",
                    "0.01",
                    "--no-epoch-metrics",
                    *_base_train_args(train_npz=train_npz, model_path=mlp_model, metrics_path=mlp_metrics),
                ),
            ),
            metrics_path=_rel(mlp_metrics),
            model_path=_rel(mlp_model),
        )
    )

    for name, calibration in [
        ("specialist_none_5class_10tr", "none"),
        ("specialist_sigmoid_5class_10tr", "sigmoid"),
    ]:
        model_path = _model_path(output_dir, name)
        metrics_path = _metrics_path(output_dir, name)
        cmd = (
            *_python("scripts/train_neuroemo_specialist_models.py"),
            "--calibration",
            calibration,
            *_base_train_args(train_npz=train_npz, model_path=model_path, metrics_path=metrics_path),
        )
        specs.append(
            RunSpec(
                name=name,
                phase="phase2_surface_postfix_matrix",
                description="One-vs-rest specialist 5-class surface-native baseline.",
                commands=(cmd,),
                metrics_path=_rel(metrics_path),
                model_path=_rel(model_path),
            )
        )

    valence_model = _model_path(output_dir, "mlp_valence_binary_10tr")
    valence_metrics = _metrics_path(output_dir, "mlp_valence_binary_10tr")
    specs.append(
        RunSpec(
            name="mlp_valence_binary_10tr",
            phase="phase2_surface_postfix_matrix",
            description="Binary valence MLP; calm and neutral excluded, remaining labels merged to positive/negative.",
            commands=(
                (
                    *_python("scripts/train_neuroemo_mlp_model.py"),
                    "--hidden-layers",
                    "64",
                    "--alpha",
                    "0.01",
                    "--no-epoch-metrics",
                    "--label-merge",
                    "negative=afraid,depressed;positive=delighted,excited",
                    *_base_train_args(
                        train_npz=train_npz,
                        model_path=valence_model,
                        metrics_path=valence_metrics,
                        exclude_labels="calm,neutral",
                    ),
                ),
            ),
            metrics_path=_rel(valence_metrics),
            model_path=_rel(valence_model),
        )
    )
    return specs


def timing_grid_specs(output_dir: Path, timing_dir: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for lag_s in (2.0, 4.0, 6.0, 8.0):
        for drop_trs in (0, 1, 2):
            prep_dir = timing_dir / f"lag{lag_s:g}s_drop{drop_trs}tr"
            train_npz = prep_dir / "neuroemo_tribev2_train.npz"
            prep_cmd = (
                *_python("scripts/prepare_neuroemo_tribev2.py"),
                "--skip-download",
                "--out-dir",
                _rel(prep_dir),
                "--bold-lag-s",
                str(lag_s),
                "--drop-transition-trs",
                str(drop_trs),
            )
            for stride in (1, 2, 5, 10):
                name = f"timing_lag{lag_s:g}s_drop{drop_trs}tr_stride{stride}"
                model_path = _model_path(output_dir, name)
                metrics_path = _metrics_path(output_dir, name)
                train_cmd = (
                    *_python("scripts/train_neuroemo_emotion_model.py"),
                    "--model-type",
                    "logistic_saga",
                    *_base_train_args(
                        train_npz=train_npz,
                        model_path=model_path,
                        metrics_path=metrics_path,
                        temporal_stride_trs=stride,
                    ),
                )
                specs.append(
                    RunSpec(
                        name=name,
                        phase="phase3_timing_grid",
                        description="Lag/drop/stride timing sweep with flat logistic_saga.",
                        commands=(prep_cmd, train_cmd),
                        metrics_path=_rel(metrics_path),
                        model_path=_rel(model_path),
                    )
                )
    return specs


def dynamic_feature_specs(output_dir: Path, train_npz: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for reducer in (
        "early_mean",
        "late_mean",
        "late_minus_early",
        "slope",
        "within_window_std",
        "last_minus_first",
        "dynamic_basic",
        "static_dynamic",
    ):
        name = f"logistic_saga_5class_10tr_{reducer}"
        model_path = _model_path(output_dir, name)
        metrics_path = _metrics_path(output_dir, name)
        cmd = (
            *_python("scripts/train_neuroemo_emotion_model.py"),
            "--model-type",
            "logistic_saga",
            *_base_train_args(
                train_npz=train_npz,
                model_path=model_path,
                metrics_path=metrics_path,
                temporal_reducer=reducer,
            ),
        )
        specs.append(
            RunSpec(
                name=name,
                phase="phase4_dynamic_window_features",
                description="Dynamic temporal reducer comparison under the frozen ROI contract.",
                commands=(cmd,),
                metrics_path=_rel(metrics_path),
                model_path=_rel(model_path),
            )
        )
    return specs


def preprocessing_roi_ablation_specs(
    output_dir: Path,
    train_npz: Path,
    preprocessed_dir: Path,
    preproc_cache_dir: Path,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for reducers in ("mean", "mean,std,mean_abs", "mean,std,mean_abs,max_abs"):
        suffix = reducers.replace(",", "_")
        name = f"raw_roi_reducers_{suffix}"
        model_path = _model_path(output_dir, name)
        metrics_path = _metrics_path(output_dir, name)
        cmd = (
            *_python("scripts/train_neuroemo_emotion_model.py"),
            "--model-type",
            "logistic_saga",
            *_base_train_args(
                train_npz=train_npz,
                model_path=model_path,
                metrics_path=metrics_path,
                roi_reducers=reducers,
            ),
        )
        specs.append(
            RunSpec(
                name=name,
                phase="phase5_preprocessing_roi_ablation",
                description="Raw surface ROI reducer ablation.",
                commands=(cmd,),
                metrics_path=_rel(metrics_path),
                model_path=_rel(model_path),
            )
        )

    prep_train_npz = preprocessed_dir / "neuroemo_tribev2_train.npz"
    prep_cmd = (
        *_python("scripts/prepare_neuroemo_tribev2.py"),
        "--skip-download",
        "--preprocess-bold",
        "--out-dir",
        _rel(preprocessed_dir),
        "--preprocessed-dir",
        _rel(preproc_cache_dir),
    )
    model_path = _model_path(output_dir, "preprocessed_logistic_saga_5class_10tr")
    metrics_path = _metrics_path(output_dir, "preprocessed_logistic_saga_5class_10tr")
    train_cmd = (
        *_python("scripts/train_neuroemo_emotion_model.py"),
        "--model-type",
        "logistic_saga",
        *_base_train_args(train_npz=prep_train_npz, model_path=model_path, metrics_path=metrics_path),
    )
    specs.append(
        RunSpec(
            name="preprocessed_logistic_saga_5class_10tr",
            phase="phase5_preprocessing_roi_ablation",
            description="Preprocessed-vs-raw surface A/B using the same logistic/ROI contract.",
            commands=(prep_cmd, train_cmd),
            metrics_path=_rel(metrics_path),
            model_path=_rel(model_path),
        )
    )
    return specs


def structured_decoding_specs(output_dir: Path, train_npz: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for model_type in ("logistic_saga", "linear_svc"):
        name = f"hierarchical_{model_type}_5class_10tr"
        model_path = _model_path(output_dir, name)
        metrics_path = _metrics_path(output_dir, name)
        cmd = (
            *_python("scripts/train_neuroemo_hierarchical_model.py"),
            "--model-type",
            model_type,
            *_base_train_args(train_npz=train_npz, model_path=model_path, metrics_path=metrics_path),
        )
        specs.append(
            RunSpec(
                name=name,
                phase="phase6_structured_decoding",
                description="Coarse-to-fine valence hierarchy with fine emotion refiners.",
                commands=(cmd,),
                metrics_path=_rel(metrics_path),
                model_path=_rel(model_path),
            )
        )
    return specs


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    output_dir = args.output_dir
    specs: list[RunSpec] = []
    phases = set(args.phase)
    if "all" in phases or "surface-postfix" in phases:
        specs.extend(surface_postfix_specs(output_dir, args.train_npz))
    if "all" in phases or "timing-grid" in phases:
        specs.extend(timing_grid_specs(output_dir / "timing_grid", args.timing_dir))
    if "all" in phases or "dynamic-features" in phases:
        specs.extend(dynamic_feature_specs(output_dir / "dynamic_features", args.train_npz))
    if "all" in phases or "preprocessing-roi" in phases:
        specs.extend(
            preprocessing_roi_ablation_specs(
                output_dir / "preprocessing_roi_ablation",
                args.train_npz,
                args.preprocessed_dir,
                args.preproc_cache_dir,
            )
        )
    if "all" in phases or "structured-decoding" in phases:
        specs.extend(structured_decoding_specs(output_dir / "structured_decoding", args.train_npz))
    return specs


def _command_text(command: tuple[str, ...]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _prepared_train_npz_from_command(command: tuple[str, ...]) -> Path | None:
    if "scripts/prepare_neuroemo_tribev2.py" not in command:
        return None
    try:
        out_dir = Path(command[command.index("--out-dir") + 1])
    except (ValueError, IndexError):
        return None
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    return out_dir / "neuroemo_tribev2_train.npz"


def _load_metric_summary(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cv = metrics.get("cross_validation") or {}
    return {
        "model_type": metrics.get("model_type"),
        "labels": metrics.get("labels"),
        "n_samples": metrics.get("n_samples"),
        "n_features": metrics.get("n_features"),
        "mean_accuracy": cv.get("mean_accuracy"),
        "mean_balanced_accuracy": cv.get("mean_balanced_accuracy"),
        "mean_macro_f1": cv.get("mean_macro_f1"),
        "pooled_log_loss": cv.get("pooled_log_loss"),
        "temporal_aggregation": metrics.get("temporal_aggregation"),
        "roi_reducers": metrics.get("roi_reducers"),
    }


def write_manifest(specs: list[RunSpec], path: Path, *, executed: bool) -> None:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        metrics_path = PROJECT_ROOT / spec.metrics_path
        rows.append(
            {
                **asdict(spec),
                "commands_text": [_command_text(command) for command in spec.commands],
                "metrics_summary": _load_metric_summary(metrics_path),
            }
        )
    payload = {
        "created_at_unix_ms": int(time.time() * 1000),
        "executed": executed,
        "phase1_vertex_equivalence": "skipped_by_design",
        "project_root": str(PROJECT_ROOT),
        "n_runs": len(rows),
        "runs": rows,
        "baseline_dirs": {
            "projected_postfix": _rel(DEFAULT_PROJECTED_BASELINE_DIR),
            "surface_native": _rel(DEFAULT_OUTPUT_DIR),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Manifest: {path}")


def execute_specs(specs: list[RunSpec], *, skip_existing: bool) -> None:
    for spec in specs:
        metrics_path = PROJECT_ROOT / spec.metrics_path
        if skip_existing and metrics_path.is_file():
            print(f"[skip] {spec.name}: metrics already exist at {metrics_path}")
            continue
        print(f"[run] {spec.name} ({spec.phase})")
        for command in spec.commands:
            prepared_npz = _prepared_train_npz_from_command(command)
            if prepared_npz is not None and prepared_npz.is_file():
                print(f"  [skip prep] {prepared_npz} already exists")
                continue
            print(f"  $ {_command_text(command)}")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase",
        action="append",
        choices=("all", "surface-postfix", "timing-grid", "dynamic-features", "preprocessing-roi", "structured-decoding"),
        default=[],
        help="Phase to include. May be repeated. Default: all.",
    )
    parser.add_argument("--train-npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timing-dir", type=Path, default=DEFAULT_TIMING_DIR)
    parser.add_argument("--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--preproc-cache-dir", type=Path, default=DEFAULT_PREPROC_CACHE_DIR)
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("scout_data/neuroemo/models/neuroemo_remaining_high_leverage_matrix_manifest.json"),
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of only writing the manifest.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip run specs whose metrics JSON already exists.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.phase:
        args.phase = ["all"]
    specs = build_specs(args)
    if args.execute:
        execute_specs(specs, skip_existing=args.skip_existing)
    write_manifest(specs, args.manifest_json, executed=args.execute)


if __name__ == "__main__":
    main()
