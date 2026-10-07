#!/usr/bin/env python3
"""Static-contract runner for the matched YOLOv8n P2/P3 report backfill."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
CASES = ("V8_B1_P23_NOFTSC", "V8_H2_P23_FTSC")
DEFAULT_SEEDS = (42, 43)
CANONICAL_SEEDS = (42, 43, 44)
SEEDS = CANONICAL_SEEDS
CONFIGS = {
    "V8_B1_P23_NOFTSC": CONFIG_ROOT / "yolov8n_p2_levir_b1_p23_noftsc.yaml",
    "V8_H2_P23_FTSC": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml",
}
HISTORICAL_SUITE = "levir_yolov8n_p2_ftsc_head_pruning"
HISTORICAL_CASE = "H2_p2_p3_nomosaic"
HISTORICAL_ROOT = ROOT / "runs" / HISTORICAL_SUITE / HISTORICAL_CASE
RUNS_ROOT = ROOT / "runs/levir_v8_p23_portability_backfill_v1"
RESULTS_ROOT = ROOT / "results"
TRAINING = {
    "dataset": "LEVIR-Ship", "split_seed": 42, "epochs": 100, "imgsz": 512,
    "batch": 8, "workers": 4, "patience": 20, "deterministic": True,
    "amp": True, "plots": False, "optimizer": "AdamW", "lr0": .002,
    "momentum": .9, "beta2_effective": .999, "weight_decay": .0005,
    "cos_lr": False, "lrf": .01, "warmup_epochs": 3.,
    "warmup_momentum": .8, "warmup_bias_lr": 0., "nbs": 64,
    "mosaic": 0., "close_mosaic": 0, "hsv_h": .015, "hsv_s": .7,
    "hsv_v": .4, "degrees": 0., "translate": .1, "scale": .5,
    "shear": 0., "perspective": 0., "flipud": 0., "fliplr": .5,
    "mixup": 0., "cutmix": 0., "copy_paste": 0., "rect": False,
    "fitness_metric": "map50_95", "nms_iou": .5,
}
REQUIRED_METRICS = (
    "val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)",
    "test/metrics/precision(B)", "test/metrics/recall(B)",
    "test/metrics/mAP50(B)", "test/metrics/mAP75(B)",
    "test/metrics/mAP50-95(B)", "test_size/AP50-Small", "model/parameters",
    "model/GFLOPs", "test_speed/inference_ms_per_image",
)
REQUIRED_ARTIFACTS = ("weights/best.pt", "experiment_manifest.json", "evaluation_metrics_extended.json")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid model YAML: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir_for(case: str, seed: int) -> Path:
    require(case in CASES, f"Unsupported case: {case}")
    require(seed in CANONICAL_SEEDS, f"Seed outside canonical universe: {seed}")
    return RUNS_ROOT / case / f"seed_{seed}"


def historical_checkpoint(seed: int) -> Path:
    require(seed in CANONICAL_SEEDS, f"Seed outside canonical universe: {seed}")
    path = HISTORICAL_ROOT / f"seed_{seed}" / "weights/best.pt"
    require(path.is_file(), f"Historical H2 checkpoint missing: {path}")
    return path


def normalized_graph(payload: dict[str, Any]) -> dict[str, Any]:
    graph = copy.deepcopy(payload)
    graph.pop("ftsc", None)
    return graph


def static_graph_parity() -> dict[str, Any]:
    b1, h2 = load_yaml(CONFIGS[CASES[0]]), load_yaml(CONFIGS[CASES[1]])
    require(normalized_graph(b1) == normalized_graph(h2), "B1/H2 graph differs outside FTSC configuration")
    require(b1["head"][-1][0] == [19, 22], "B1 Detect inputs must be [19, 22]")
    require(h2["head"][-1][0] == [19, 22], "H2 Detect inputs must be [19, 22]")
    require(b1.get("ftsc", {}).get("enabled") is False, "B1 FTSC must be disabled")
    require(h2.get("ftsc", {}).get("enabled") is True, "H2 FTSC must be enabled")
    return {"detect_inputs": [19, 22], "b1_ftsc": False, "h2_ftsc": True}


def pretrained_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"requested": str(path), "status": "PENDING_PRETRAINED"}
    return {"requested": str(path), "canonical_path": str(path.resolve()), "sha256": digest(path), "status": "READY"}


def historical_provenance(seed: int) -> dict[str, Any]:
    source = historical_checkpoint(seed)
    source_yaml = CONFIGS["V8_H2_P23_FTSC"]
    return {
        "execution_mode": "evaluation_only_reuse",
        "source_suite": HISTORICAL_SUITE, "source_case": HISTORICAL_CASE,
        "source_seed": seed, "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": digest(source),
        "source_model_yaml": str(source_yaml.resolve()),
        "source_model_yaml_sha256": digest(source_yaml),
    }


def scientific_manifest(case: str, seed: int, data_yaml: Path, pretrained: dict[str, Any], requested_seeds: tuple[int, ...], *, require_historical_source: bool = True) -> dict[str, Any]:
    config = CONFIGS[case]
    manifest_path = data_yaml.parent / "manifest.json"
    contract = {
        "schema": "levir_v8_p23_portability_backfill_v1", "dataset": "LEVIR-Ship",
        "case": case, "seed": seed, "requested_seed_universe": list(requested_seeds),
        "canonical_seed_universe": list(CANONICAL_SEEDS), "split_seed": 42,
        "dataset_manifest_sha256": digest(manifest_path) if manifest_path.is_file() else None,
        "model_config_sha256": digest(config), "detect_inputs": [19, 22],
        "ftsc_enabled": case == "V8_H2_P23_FTSC", "training": TRAINING,
        "pretrained_sha256": pretrained.get("sha256"),
    }
    if case == "V8_H2_P23_FTSC" and require_historical_source:
        contract["historical_provenance"] = historical_provenance(seed)
    return {"scientific_contract": contract, "artifact_provenance": {"model_config": str(config), "model_config_sha256": digest(config), "pretrained": pretrained}, "execution_context": {"requested_workers": TRAINING["workers"]}}


def metrics_complete(run_dir: Path) -> bool:
    if any(not (run_dir / item).is_file() for item in REQUIRED_ARTIFACTS):
        return False
    try:
        from evaluate_test.standard_detection_metrics import load_merged_metrics
        metrics = load_merged_metrics(run_dir)
    except (ImportError, OSError, ValueError, json.JSONDecodeError):
        return False
    return all(metrics.get(key) is not None for key in REQUIRED_METRICS)


def sample_std(values: list[float]) -> float | None:
    """Return sample standard deviation; a singleton has no defined std."""
    return statistics.stdev(values) if len(values) >= 2 else None


def run_status(case: str, run_dir: Path, manifest: dict[str, Any], pretrained: dict[str, Any], data_yaml: Path) -> str:
    if pretrained.get("status") != "READY":
        return "PENDING_PRETRAINED"
    if not data_yaml.is_file() or not (data_yaml.parent / "manifest.json").is_file():
        return "PENDING_DATA_PREPARATION"
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return "NEW_RUN_REQUIRED"
    manifest_path = run_dir / "experiment_manifest.json"
    require(manifest_path.is_file(), f"Refusing non-empty run without manifest: {run_dir}")
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(existing.get("scientific_contract") == manifest["scientific_contract"], f"Scientific manifest mismatch: {run_dir}")
    if metrics_complete(run_dir):
        return "COMPLETE"
    if (run_dir / "weights/best.pt").is_file():
        return "EVAL_BACKFILL_REQUIRED"
    return "NEW_RUN_REQUIRED"


def validate_args(args: argparse.Namespace) -> None:
    require(args.seeds and tuple(args.seeds) == tuple(dict.fromkeys(args.seeds)), "seeds must be unique and non-empty")
    require(args.seeds and set(args.seeds).issubset(CANONICAL_SEEDS), "seeds must be a subset of 42,43,44")
    require(args.cases and tuple(args.cases) == tuple(dict.fromkeys(args.cases)) and set(args.cases).issubset(CASES), "cases must be the exact supported subset")


def build_preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    parity = static_graph_parity()
    data_yaml = args.dataset_root / "levir_ship_yolo_seed42/levir_ship.yaml"
    pretrained = pretrained_identity(Path(args.pretrained))
    rows = []
    for case in CASES:
        for seed in CANONICAL_SEEDS:
            requested = case in args.cases and seed in args.seeds
            manifest = scientific_manifest(case, seed, data_yaml, pretrained, tuple(args.seeds), require_historical_source=requested)
            status = run_status(case, run_dir_for(case, seed), manifest, pretrained, data_yaml) if requested else "NOT_REQUESTED"
            rows.append({"case": case, "seed": seed, "requested": requested, "status": status, "config": str(CONFIGS[case]), "config_sha256": digest(CONFIGS[case]), "parity": parity, "manifest": manifest, "data_yaml": str(data_yaml)})
    return rows


def train_b1(args: argparse.Namespace, row: dict[str, Any], data_yaml: Path, pretrained: dict[str, Any]) -> Path:
    from h2_generalization_completion_common import ensure_manifest
    from train_levir_scripts.train_all_levir_yolov8n_p2_routing import local_ultralytics, seed_everything
    from ultralytics import YOLO
    run_dir = run_dir_for(row["case"], row["seed"])
    ensure_manifest(run_dir, row["manifest"])
    local_ultralytics(); seed_everything(row["seed"])
    model = YOLO(str(CONFIGS["V8_B1_P23_NOFTSC"]), task="detect")
    last = run_dir / "weights/last.pt"
    if last.is_file():
        YOLO(str(last), task="detect").train(resume=str(last))
        require((run_dir / "weights/best.pt").is_file(), f"B1 resumed checkpoint missing: {run_dir}")
        return run_dir
    model.load(pretrained["canonical_path"], smart_transfer=True)
    kwargs = dict(TRAINING); kwargs.update(data=str(data_yaml), seed=row["seed"], workers=args.workers, device=args.device)
    kwargs.pop("dataset", None); kwargs.pop("split_seed", None); kwargs.pop("beta2_effective", None); kwargs.pop("nms_iou", None); kwargs.pop("fitness_metric", None)
    model.train(**kwargs, project=str(run_dir.parent), name=run_dir.name, exist_ok=True)
    require((run_dir / "weights/best.pt").is_file(), f"B1 training checkpoint missing: {run_dir}")
    return run_dir


def prepare_h2_reuse(row: dict[str, Any]) -> Path:
    source = historical_checkpoint(row["seed"])
    run_dir = run_dir_for(row["case"], row["seed"])
    from h2_generalization_completion_common import ensure_manifest
    ensure_manifest(run_dir, row["manifest"])
    destination = run_dir / "weights/best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require(digest(source) == digest(destination), "H2 source/copied checkpoint SHA mismatch")
    manifest_path = run_dir / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifact_provenance", {}).update({
        "original_source_checkpoint": str(source.resolve()),
        "original_source_sha256": digest(source),
        "copied_checkpoint_sha256": digest(destination),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def evaluate_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    from evaluate_test.standard_detection_metrics import evaluate_run, load_merged_metrics
    run_dir = run_dir_for(row["case"], row["seed"])
    evaluate_run(run_dir=run_dir, data_yaml=Path(row["data_yaml"]), dataset="levir", imgsz=512, batch=8, workers=args.workers, device=args.device, nms_iou=.5, include_size=True)
    row["metrics"] = load_merged_metrics(run_dir)
    row["status"] = "COMPLETE" if metrics_complete(run_dir) else "EVAL_BACKFILL_REQUIRED"
    return row


def dispatch(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pretrained = pretrained_identity(Path(args.pretrained))
    for row in rows:
        if not row["requested"]:
            continue
        if row["case"] == "V8_B1_P23_NOFTSC" and row["status"] == "NEW_RUN_REQUIRED":
            train_b1(args, row, Path(row["data_yaml"]), pretrained)
            row["status"] = "EVAL_BACKFILL_REQUIRED"
        elif row["case"] == "V8_H2_P23_FTSC" and row["status"] == "NEW_RUN_REQUIRED":
            prepare_h2_reuse(row)
            row["status"] = "EVAL_BACKFILL_REQUIRED"
        if row["status"] == "EVAL_BACKFILL_REQUIRED":
            evaluate_one(args, row)
        elif row["status"] == "COMPLETE":
            from evaluate_test.standard_detection_metrics import load_merged_metrics
            row["metrics"] = load_merged_metrics(run_dir_for(row["case"], row["seed"]))
            row["status"] = "REUSED"
    return rows


def write_summaries(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    from h2_generalization_completion_common import write_csv
    flat = [{"case": row["case"], "seed": row["seed"], "requested": row["requested"], "status": row["status"], **row.get("metrics", {})} for row in rows]
    write_csv(RESULTS_ROOT / "levir_v8_p23_portability_backfill_v1_summary_runs.csv", flat)
    aggregate = []
    for case in CASES:
        completed = [row for row in flat if row["case"] == case and row["status"] in {"COMPLETE", "REUSED"}]
        seeds = {int(row["seed"]) for row in completed}
        summary = {"case": case, "n_complete": len(completed), "requested_seed_count": len(args.seeds), "canonical_seed_count": len(CANONICAL_SEEDS), "requested_complete": set(args.seeds).issubset(seeds), "canonical_complete": set(CANONICAL_SEEDS).issubset(seeds)}
        for metric in REQUIRED_METRICS:
            values = [float(row[metric]) for row in completed if row.get(metric) is not None]
            summary[f"{metric}/mean"] = statistics.fmean(values) if values else None
            summary[f"{metric}/std"] = sample_std(values)
        aggregate.append(summary)
    write_csv(RESULTS_ROOT / "levir_v8_p23_portability_backfill_v1_summary_aggregate.csv", aggregate)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); validate_args(args); rows = build_preflight(args)
    print(json.dumps({"suite": "levir_v8_p23_portability_backfill_v1", "rows": rows}, indent=2, default=str))
    if args.confirm_run:
        write_summaries(args, dispatch(args, rows))


if __name__ == "__main__":
    main()
