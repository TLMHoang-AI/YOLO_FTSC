#!/usr/bin/env python3
"""Varroa B1/H2 cross-YOLO portability suite; read-only by default."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ULTRA = ROOT / "models_related/ultralytics"
for entry in (ROOT, ULTRA):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from utils import cross_yolo_portability as common

FAMILIES, CASES, SEEDS = common.FAMILIES, common.CASES, common.SEEDS
NAMESPACE = "varroa_ftsc_cross_yolo_portability_v1"
RUNS_ROOT = ROOT / "runs" / NAMESPACE
CANONICAL_H2 = ROOT / "models_related/models_config/yolov8/varroa/yolov8n_varroa_ftsc_v1_p2p3.yaml"
SPECS = {
    "yolov9t": {"p2": 18, "p3": 21, "detect": "Detect", "native": ("ELAN1", "RepNCSPELAN4", "AConv"), "pretrained": "yolov9t.pt"},
    "yolov10n": {"p2": 19, "p3": 22, "detect": "v10Detect", "native": ("C2f", "SCDown", "PSA"), "pretrained": "yolov10n.pt"},
    "yolo11n": {"p2": 19, "p3": 22, "detect": "Detect", "native": ("C3k2", "C2PSA"), "pretrained": "yolo11n.pt"},
}
TRAINING = {
    "epochs": 100, "imgsz": 640, "batch": 4, "patience": 20,
    "deterministic": True, "amp": True, "optimizer": "auto", "mosaic": 1.0,
    "close_mosaic": 10, "fitness_metric": "map50_95", "plots": False,
    "optimizer_resolution": {"expected": "AdamW", "iterations": 4400, "branch": "iterations<=10000", "lr0": 0.002, "momentum": 0.9, "weight_decay": 0.0005, "lrf": 0.01, "cos_lr": False, "warmup_epochs": 3.0},
}
EVALUATOR = {"standard_nms_iou": 0.7, "size_bucket_nms_iou": 0.5, "latency_unit": "ms/image"}
REQUIRED_METRICS = (
    "test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small",
    "model/parameters", "model/GFLOPs", "test_speed/inference_ms_per_image",
)


def model_config_for(family: str, case: str) -> Path:
    common.require(family in FAMILIES and case in CASES, f"Unsupported identity: {family}/{case}")
    stem = {"yolov9t": "yolov9t_p2_varroa", "yolov10n": "yolov10n_p2_varroa", "yolo11n": "yolo11n_p2_varroa"}[family]
    suffix = "b1_noftsc" if case == "B1_P23_NOFTSC" else "ftsc_h2"
    family_dir = {"yolov9t": "yolov9", "yolov10n": "yolov10", "yolo11n": "yolo11"}[family]
    return ROOT / f"models_related/models_config/{family_dir}/varroa/{stem}_{suffix}.yaml"


def resolved_payload(family: str, case: str) -> dict[str, Any]: return common.load_yaml(model_config_for(family, case))
def normalized_graph(payload: dict[str, Any]) -> dict[str, Any]: return common.normalized_graph(payload)


def static_preflight(family: str, case: str) -> dict[str, Any]:
    return common.static_model_preflight(family, case, model_config_for(family, case), SPECS[family], common.load_yaml(CANONICAL_H2)["ftsc"])


def optimizer_contract(batch_size: int = 4, epochs: int = 100) -> dict[str, Any]:
    iterations = math.ceil(2762 / max(batch_size, 64)) * epochs
    common.require(iterations == 4400, f"Varroa optimizer horizon drift: {iterations}")
    return common.canonical(TRAINING["optimizer_resolution"])


def run_dir_for(family: str, case: str, seed: int) -> Path:
    common.require(seed in SEEDS, f"Unsupported seed: {seed}")
    return RUNS_ROOT / family / case / f"seed_{seed}"


def data_yaml_for(dataset_root: Path, seed: int) -> Path: return dataset_root / f"varroa_yolo_seed{seed}/varroa.yaml"


def dataset_fingerprint(dataset_root: Path, seed: int) -> dict[str, Any] | None:
    data_yaml = data_yaml_for(dataset_root, seed)
    if not data_yaml.is_file(): return None
    import yaml
    value = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    required = {"gt_source": "gt_one", "only_positives": True, "class_policy": "map-3-to-1", "seed": seed}
    if any(value.get(key) != expected for key, expected in required.items()): return None
    root = Path(value.get("path", data_yaml.parent)); root = root if root.is_absolute() else (data_yaml.parent / root).resolve()
    fingerprints: dict[str, str] = {}
    for split, expected in {"train": 2762, "val": 592, "test": 592}.items():
        directory = root / str(value.get(split, ""))
        if not directory.is_dir(): return None
        names = sorted(path.name for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        if len(names) != expected: return None
        fingerprints[split] = __import__("hashlib").sha256("\n".join(names).encode()).hexdigest()
    return {**required, "counts": {"train": 2762, "val": 592, "test": 592}, "member_sha256": fingerprints, "data_yaml_sha256": common.sha256(data_yaml)}


def pretrained_for(args: argparse.Namespace, family: str) -> dict[str, Any]:
    return common.pretrained_identity(ROOT, family, SPECS[family]["pretrained"], getattr(args, f"pretrained_{family}"))


def source_identity() -> dict[str, str]:
    files = (Path(__file__), ROOT / "utils/cross_yolo_portability.py", ROOT / "train_levir_scripts/train_varroa_dbss_hit.py", ROOT / "evaluate_test/standard_detection_metrics.py", ROOT / "evaluate_test/size_bucket_evaluator.py", ROOT / "models_related/ultralytics/ultralytics/utils/loss.py")
    return {str(path.relative_to(ROOT)): common.sha256(path) for path in files}


def scientific_manifest(args: argparse.Namespace, family: str, case: str, seed: int, pretrained: dict[str, Any]) -> dict[str, Any]:
    config = model_config_for(family, case)
    return common.canonical({
        "scientific_contract": {
            "schema": NAMESPACE, "dataset": "Varroa", "family": family, "case": case, "seed": seed,
            "canonical_seeds": list(SEEDS), "model_yaml_sha256": common.sha256(config), "model_graph": static_preflight(family, case),
            "ftsc": resolved_payload(family, case).get("ftsc", {"enabled": False}),
            "pretrained": {"artifact": SPECS[family]["pretrained"], "sha256": pretrained.get("sha256"), "smart_transfer": True},
            "dataset_fingerprint": dataset_fingerprint(args.dataset_root, seed), "training": TRAINING,
            "evaluation": EVALUATOR, "checkpoint_selection": "weights/best.pt by map50_95", "source_sha256": source_identity(),
        }, "execution_context": {"device": args.device, "workers": args.workers},
    })


def validate_metrics(metrics: dict[str, Any]) -> None:
    common.require(float(metrics.get("evaluation/nms_iou", -1)) == 0.7, "Varroa standard NMS must be 0.7")
    protocol = str(metrics.get("test_size/protocol", ""))
    common.require("TinyBenchmark" in protocol, "Varroa native size-bucket evaluator missing")
    common.require(metrics.get("benchmark/latency_unit") == "ms/image", "Varroa latency must be ms/image")
    common.validate_metrics(metrics, REQUIRED_METRICS)


def status_for(args: argparse.Namespace, family: str, case: str, seed: int) -> tuple[str, str | None, dict[str, Any]]:
    pretrained = pretrained_for(args, family); manifest = scientific_manifest(args, family, case, seed, pretrained)
    ready = pretrained["status"] == "READY" and manifest["scientific_contract"]["dataset_fingerprint"] is not None
    status, reason = common.run_status(run_dir_for(family, case, seed), manifest, prerequisites_ready=ready, required_metrics=REQUIRED_METRICS, metric_validator=validate_metrics)
    if status == "BLOCKED" and not ready:
        missing = []
        if pretrained["status"] != "READY": missing.append(f"pretrained:{SPECS[family]['pretrained']}")
        if manifest["scientific_contract"]["dataset_fingerprint"] is None: missing.append(f"prepared_dataset:seed_{seed}")
        reason = "missing prerequisites: " + ", ".join(missing)
    return status, reason, manifest


def build_preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for family in args.families:
        common.require(normalized_graph(resolved_payload(family, CASES[0])) == normalized_graph(resolved_payload(family, CASES[1])), f"{family}: B1/H2 graph mismatch")
        for case in args.cases:
            static = static_preflight(family, case)
            for seed in args.seeds:
                status, reason, manifest = status_for(args, family, case, seed)
                rows.append({"dataset": "Varroa", "family": family, "case": case, "seed": seed, "status": status, "reason": reason, "static": static, "manifest": manifest})
    return rows


def prepare_dataset(args: argparse.Namespace, seed: int) -> Path:
    from train_levir_scripts import train_varroa_dbss_hit as native
    data_yaml = native.dataset_for_seed(args, seed)
    common.require(dataset_fingerprint(args.dataset_root, seed) is not None, "Prepared Varroa split violates gt_one/map-3-to-1/count contract")
    return data_yaml


def train_kwargs(args: argparse.Namespace, family: str, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    return {"data": str(data_yaml), "epochs": 100, "imgsz": 640, "batch": 4, "device": args.device, "workers": args.workers, "patience": 20, "amp": True, "plots": False, "seed": seed, "deterministic": True, "optimizer": "auto", "mosaic": 1.0, "close_mosaic": 10, "fitness_metric": "map50_95", "project": str(RUNS_ROOT / family / case), "name": f"seed_{seed}", "exist_ok": True}


def runtime_preflight(family: str, case: str) -> dict[str, Any]:
    return common.runtime_model_preflight(family, case, model_config_for(family, case), epochs=100, fitness_metric="map50_95")


def verify_initialization_parity(args: argparse.Namespace, family: str) -> None:
    from train_levir_scripts.train_varroa_dbss_hit import seed_everything
    common.verify_initialization_parity(family, model_config_for(family, CASES[0]), model_config_for(family, CASES[1]), pretrained_for(args, family), seed_everything)


def evaluate_one(args: argparse.Namespace, family: str, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    from train_levir_scripts import train_varroa_dbss_hit as native
    run = run_dir_for(family, case, seed); args.data_yaml = data_yaml
    common.quarantine_unverified_evaluation_cache(run)
    metrics = native.evaluate(run, args); metrics["benchmark/latency_unit"] = "ms/image"
    metrics["evaluation/size_bucket_nms_iou"] = 0.5
    metrics["evaluation/checkpoint_sha256"] = common.sha256(run / "weights/best.pt")
    validate_metrics(metrics)
    common.write_json(run / "evaluation_metrics_extended.json", metrics)
    common.mark_evaluation_complete(run, metrics)
    return metrics


def execute_one(args: argparse.Namespace) -> dict[str, Any]:
    family, case, seed = args.families[0], args.cases[0], args.seeds[0]
    data_yaml = prepare_dataset(args, seed); pretrained = pretrained_for(args, family)
    common.require(pretrained["status"] == "READY", pretrained.get("reason", "pretrained unavailable"))
    manifest = scientific_manifest(args, family, case, seed, pretrained); run = run_dir_for(family, case, seed)
    status, reason = common.run_status(run, manifest, prerequisites_ready=True, required_metrics=REQUIRED_METRICS, metric_validator=validate_metrics)
    common.require(status != "BLOCKED", reason or "run blocked")
    if args.action == "evaluate":
        if status == "COMPLETE":
            return {"status": "COMPLETE", **common.read_metrics(run)}
        common.require(status == "EVAL_BACKFILL_REQUIRED",
                       f"Evaluation requires finalized training, got {status}")
        return {"status": "COMPLETE", **evaluate_one(args, family, case, seed, data_yaml)}
    if status == "COMPLETE": return {"status": "COMPLETE", **common.read_metrics(run)}
    if status == "EVAL_BACKFILL_REQUIRED": return {"status": "COMPLETE", **evaluate_one(args, family, case, seed, data_yaml)}
    from train_levir_scripts.train_varroa_dbss_hit import local_ultralytics, seed_everything
    local_ultralytics(); from ultralytics import YOLO
    if status == "RESUME_REQUIRED": YOLO(run / "weights/last.pt", task="detect").train(resume=str(run / "weights/last.pt"))
    else:
        verify_initialization_parity(args, family); runtime_preflight(family, case)
        common.ensure_manifest(run, manifest); shutil.copy2(model_config_for(family, case), run / "config.yaml")
        seed_everything(seed); model = YOLO(model_config_for(family, case), task="detect"); model.load(pretrained["path"], smart_transfer=True); model.train(**train_kwargs(args, family, case, seed, data_yaml))
    common.require(all((run / item).is_file() for item in common.TRAIN_ARTIFACTS), f"Incomplete training: {run}")
    common.mark_training_complete(run, manifest)
    return {"status": "COMPLETE", **evaluate_one(args, family, case, seed, data_yaml)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "train", "evaluate", "aggregate", "archive"), default="preflight")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)); parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES)); parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "Varroa"); parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--imgsz", type=int, default=640); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--device", default="cuda"); parser.add_argument("--seed-scope", default="all", choices=("all",)); parser.add_argument("--live-model-smoke", action="store_true")
    parser.add_argument("--pretrained-yolov9t"); parser.add_argument("--pretrained-yolov10n"); parser.add_argument("--pretrained-yolo11n"); parser.add_argument("--hf-repo-id", default="TLMHoang/FTSC_results")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    common.require((args.epochs, args.imgsz, args.batch_size, args.patience) == (100, 640, 4, 20), "Varroa frozen protocol drift"); optimizer_contract(args.batch_size, args.epochs)
    for name, values, universe in (("families", args.families, FAMILIES), ("cases", args.cases, CASES), ("seeds", args.seeds, SEEDS)):
        common.require(values and len(values) == len(set(values)) and set(values).issubset(universe), f"Invalid {name}")
    common.require(args.workers >= 0 and bool(args.device.strip()), "Invalid execution context")
    if args.action in {"train", "evaluate", "archive"}: common.require(len(args.families) == len(args.cases) == len(args.seeds) == 1, f"{args.action} requires exactly one family, case, and seed")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); validate_args(args)
    if args.action == "preflight":
        rows = build_preflight(args)
        if args.live_model_smoke:
            smoke = {}
            for row in rows:
                key = (row["family"], row["case"])
                if key not in smoke: smoke[key] = runtime_preflight(*key)
                row["runtime"] = smoke[key]
        print(json.dumps({"suite": NAMESPACE, "optimizer": optimizer_contract(), "rows": rows}, indent=2, sort_keys=True, default=str)); return
    if args.action == "aggregate":
        rows = []
        for family in args.families:
            for case in args.cases:
                for seed in SEEDS:
                    status, reason, _ = status_for(args, family, case, seed); row = {"dataset": "Varroa", "family": family, "case": case, "seed": seed, "status": status, "reason": reason}
                    if status == "COMPLETE": row.update(common.read_metrics(run_dir_for(family, case, seed)))
                    rows.append(row)
        common.write_summaries(RUNS_ROOT, rows, REQUIRED_METRICS)
        common.write_json(RUNS_ROOT / "suite_manifest.json", {"schema": NAMESPACE, "dataset": "Varroa", "families": list(FAMILIES), "cases": list(CASES), "canonical_seeds": list(SEEDS), "archive_repository": args.hf_repo_id, "archive_namespace": NAMESPACE})
        return
    if args.action == "archive":
        family, case, seed = args.families[0], args.cases[0], args.seeds[0]; status, reason, _ = status_for(args, family, case, seed); common.require(status == "COMPLETE", reason or f"Archive requires COMPLETE, got {status}")
        print(json.dumps(common.archive_run(namespace=NAMESPACE, family=family, case=case, seed=seed, run_dir=run_dir_for(family, case, seed), repo_id=args.hf_repo_id), indent=2)); return
    print(json.dumps(execute_one(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__": main()
