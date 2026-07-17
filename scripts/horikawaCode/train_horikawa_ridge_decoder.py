"""Train Horikawa ridge/MLP decoder and export to scout_models/."""



from __future__ import annotations



import argparse

import json

import sys

from pathlib import Path



import numpy as np

import yaml



PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))



from scout_core.horikawaCode.constants import DEFAULT_PRODUCT_CATEGORIES, HORIKAWA_14_DIMENSIONS

from scout_core.horikawaCode.cv import lovo_cv_r, make_ridge_pipeline

from scout_core.horikawaCode.labels import build_label_map

from scout_core.horikawaCode.mlp import group_kfold_mean_r, make_mlp_pipeline

from scout_core.mvpa_engine import export_decoder_bundle





def _load_train_npz(path: Path) -> dict[str, np.ndarray]:

    data = np.load(path, allow_pickle=True)

    return {k: data[k] for k in data.files}





def _resolve_status(corpus: str | None) -> str:

    if corpus in ("full_2181", "full"):

        return "full"

    if corpus in ("pilot_150", "pilot", "pilot_275", "ready_all"):

        return "pilot"

    return "bootstrap"





def _resolve_cv_scheme(cfg: dict) -> str:

    return str((cfg.get("cv") or {}).get("scheme", "leave_one_video_out"))





def _resolve_model_id(cfg: dict, model: str) -> str:

    base = str(cfg.get("model_id", "horikawa_ridge_v1"))

    if model == "mlp":

        return base.replace("ridge", "mlp")

    return base





def _mlp_kwargs(cfg: dict) -> dict:

    mlp_cfg = cfg.get("mlp") or {}

    hidden = mlp_cfg.get("hidden_layer_sizes", [64])

    return {

        "hidden_layer_sizes": tuple(int(x) for x in hidden),

        "alpha": float(mlp_cfg.get("alpha", 0.01)),

        "learning_rate_init": float(mlp_cfg.get("learning_rate_init", 0.001)),

        "max_iter": int(mlp_cfg.get("max_iter", 500)),

        "random_state": int(mlp_cfg.get("random_state", 42)),

    }





def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(

        "--config",

        type=Path,

        default=PROJECT_ROOT / "configs" / "horikawa_decoding.yaml",

    )

    parser.add_argument("--model", choices=("ridge", "mlp"), default="ridge")

    args = parser.parse_args()



    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    train_path = Path(cfg.get("data", {}).get(

        "train_npz", "scout_data/horikawaCode/tribev2_fused/train.npz"

    ))

    if not train_path.is_absolute():

        train_path = PROJECT_ROOT / train_path

    if not train_path.is_file():

        raise SystemExit(f"Missing training NPZ: {train_path}. Run prepare_horikawa_tribev2.py --demo")



    data = _load_train_npz(train_path)

    X = np.asarray(data["X"], dtype=np.float32)

    y = np.asarray(data["y"], dtype=np.float32)

    groups = np.asarray(data["groups"], dtype=np.int32)



    target_type = cfg.get("target_type", "product_8")

    if "label_names" in data:

        categories = [str(x) for x in np.asarray(data["label_names"]).tolist()]

    elif target_type in ("dimensions", "dimensions_14"):

        categories = list(HORIKAWA_14_DIMENSIONS)

    else:

        categories = list(cfg.get("categories") or DEFAULT_PRODUCT_CATEGORIES)



    label_map = build_label_map(categories)

    alphas = [float(a) for a in cfg.get("ridge", {}).get("alphas", [0.1, 1.0, 10.0])]

    cv_scheme = _resolve_cv_scheme(cfg)

    model_id = _resolve_model_id(cfg, args.model)



    if args.model == "mlp":

        mlp_kw = _mlp_kwargs(cfg)

        cv_splits = int((cfg.get("mlp") or {}).get("cv_splits", 5))

        cv_mean_r = group_kfold_mean_r(X, y, groups, n_splits=cv_splits, **mlp_kw)

        cv_metrics = {"mean_r": cv_mean_r}

        pipeline = make_mlp_pipeline(**mlp_kw)

    else:

        cv_metrics = lovo_cv_r(X, y, groups, alphas)

        cv_mean_r = float(np.mean(list(cv_metrics.values()))) if cv_metrics else 0.0

        pipeline = make_ridge_pipeline(alphas)



    pipeline.fit(X, y)



    corpus = str(data["corpus"]) if "corpus" in data else None

    data_hash = str(data["data_hash"]) if "data_hash" in data else None

    atlas_sha = str(data["atlas_sha256"]) if "atlas_sha256" in data else None



    meta = {

        "model_id": model_id,

        "model_type": args.model,

        "feature_spec": cfg.get("feature_spec", "fused_schaefer400_subcortical_v1"),

        "target_type": target_type,

        "n_classes": len(categories),

        "class_names": categories,

        "cv_scheme": cv_scheme if args.model == "ridge" else "group_kfold",

        "cv_metrics": cv_metrics,

        "cv_mean_r": cv_mean_r,

        "n_samples": int(X.shape[0]),

        "n_features": int(X.shape[1]),

        "status": _resolve_status(corpus),

        "corpus": corpus,

        "data_hash": data_hash,

        "atlas_sha256": atlas_sha,

    }



    bundle_dir = export_decoder_bundle(

        pipeline,

        model_id=model_id,

        label_map=label_map,

        meta=meta,

    )



    try:

        train_npz_rel = str(train_path.relative_to(PROJECT_ROOT))

    except ValueError:

        train_npz_rel = str(train_path)



    manifest = {

        "model_id": model_id,

        "model_type": args.model,

        "train_npz": train_npz_rel,

        "n_features": int(X.shape[1]),

        "n_targets": len(categories),

        "label_names": categories,

        "feature_spec": meta["feature_spec"],

        "atlas_sha256": atlas_sha,

        "data_hash": data_hash,

        "corpus": corpus,

        "cv_scheme": meta["cv_scheme"],

        "cv_mean_r": cv_mean_r,

        "training_argv": sys.argv,

    }

    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")



    reports_dir = PROJECT_ROOT / cfg.get("data", {}).get("reports_dir", "scout_data/horikawaCode/reports")

    reports_dir.mkdir(parents=True, exist_ok=True)

    report_name = cfg.get("data", {}).get("cv_report_name", "cv_report_v1.json")

    if args.model == "mlp":

        stem = Path(report_name).stem

        report_name = f"{stem.replace('_ridge', '').replace('ridge', 'mlp')}_mlp.json"

        if report_name == f"{stem}_mlp.json" and "mlp" not in stem:

            report_name = f"{stem}_mlp.json"

    report_path = reports_dir / report_name

    report_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")



    print(f"Exported decoder -> {bundle_dir}")

    print(f"cv_mean_r={cv_mean_r:.4f}  model={args.model}  status={meta['status']}")

    print(f"CV report -> {report_path}")





if __name__ == "__main__":

    main()

