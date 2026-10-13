#!/usr/bin/env python3
"""Varroa V1_FTSC H2 fixed-alpha sensitivity suite (static-safe by default)."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/varroa"
H2 = CONFIG_ROOT / "yolov8n_varroa_ftsc_v1_p2p3.yaml"
CASES = ("ALPHA4_rerun", "ALPHA8_rerun")
SEEDS = (42, 43, 44)
ALPHA = {"ALPHA4_rerun": 4.0, "ALPHA8_rerun": 8.0}
YAMLS = {
    "ALPHA4_rerun": CONFIG_ROOT / "yolov8n_varroa_ftsc_h2_alpha4.yaml",
    "ALPHA8_rerun": CONFIG_ROOT / "yolov8n_varroa_ftsc_h2_alpha8.yaml",
}
NAMESPACE = "varroa_ftsc_position_alpha_h2_v1"
HISTORICAL_ROOT = ROOT / "runs/varroa_ftsc_causal_suite/V1_FTSC"
TRAIN_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "config.yaml")
METRICS_ARTIFACT = "evaluation_metrics_alpha_contract_v1.json"
COMPLETE_ARTIFACT = "alpha_suite_complete.json"
CORE_METRICS = ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small", "model/parameters", "model/GFLOPs", "test_speed/inference_ms_per_image")


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, default=str))


def contracts_compatible(recorded: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Ignore Git location metadata only; source hashes remain mandatory."""
    left, right = copy.deepcopy(canonical(recorded)), copy.deepcopy(canonical(expected))
    for value in (left, right):
        source = value.get("source", {})
        source.pop("commit", None)
        source.pop("clean", None)
    return left == right


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")); require(isinstance(value, dict), f"Expected JSON object: {path}"); return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(canonical(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"); temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8")); require(isinstance(value, dict), f"Invalid YAML: {path}"); return value


def normalized_alpha(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload); value["ftsc"]["position_alpha"] = 6.0; value["ftsc"].pop("position_alpha_mode", None); return value


def source_identity() -> dict[str, Any]:
    files = ("train_levir_scripts/train_varroa_ftsc_alpha_sensitivity_v1.py", "train_levir_scripts/train_varroa_dbss_hit.py", "evaluate_test/standard_detection_metrics.py", "evaluate_test/size_bucket_evaluator.py", "models_related/ultralytics/ultralytics/nn/modules/ftsc.py", "models_related/ultralytics/ultralytics/utils/tal.py", "models_related/ultralytics/ultralytics/utils/loss.py")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(); clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit, clean = None, False
    return {"commit": commit, "clean": clean, "sha256": {name: sha(ROOT / name) for name in files}}


def source_preflight(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_yaml(H2); require(baseline["head"][-1][0] == [18, 21], "Varroa V1_FTSC Detect taps drifted")
    require(baseline["ftsc"].get("policy") == "f5", "Varroa FTSC policy drifted")
    cases = {}
    for case in args.cases:
        payload = load_yaml(YAMLS[case]); require(normalized_alpha(payload) == baseline, f"{case}: differs from V1_FTSC beyond position_alpha")
        require(payload["ftsc"].get("position_alpha") == ALPHA[case] and payload["ftsc"].get("position_alpha_mode", "fixed") == "fixed", f"{case}: wrong fixed-alpha policy")
        cases[case] = {"yaml": str(YAMLS[case]), "sha256": sha(YAMLS[case]), "alpha": ALPHA[case]}
    return {"protocol": NAMESPACE, "cases": cases, "source": source_identity(), "standard_nms_iou": 0.7, "size_bucket_nms_iou": 0.5}


def shared_module():
    from train_levir_scripts import train_varroa_dbss_hit as shared
    return shared


def optimizer_contract(args: argparse.Namespace) -> dict[str, Any]:
    iterations = math.ceil(2762 / max(args.batch_size, 64)) * args.epochs
    require(iterations == 4400, f"Varroa optimizer horizon drift: {iterations}")
    return {"requested": "auto", "resolved_expected": "AdamW", "iterations": iterations, "branch": "iterations<=10000", "lr0_resolved": 0.002, "momentum_resolved": 0.9, "weight_decay": 0.0005, "lrf": 0.01, "cos_lr": False, "warmup_epochs": 3.0}


def train_kwargs(args: argparse.Namespace, seed: int, data_yaml: Path) -> dict[str, Any]:
    values = shared_module().train_kwargs(args, data_yaml, seed, True)
    values.update(optimizer="auto", mosaic=1.0, close_mosaic=10, project="", name="", exist_ok=True)
    return values


def dataset_contract(data_yaml: Path, seed: int) -> dict[str, Any]:
    import yaml
    shared = shared_module(); shared.validate_dataset(data_yaml); payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    for key, expected in {"gt_source": "gt_one", "only_positives": True, "class_policy": "map-3-to-1", "seed": seed}.items(): require(payload.get(key) == expected, f"Varroa dataset drift: {key}")
    root = Path(payload["path"]); root = root if root.is_absolute() else (data_yaml.parent / root).resolve()
    fingerprints = {}
    for split, count in {"train": 2762, "val": 592, "test": 592}.items():
        names = sorted(path.name for path in (root / payload[split]).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}); require(len(names) == count, f"Varroa {split} count drift")
        fingerprints[split] = hashlib.sha256("\n".join(names).encode()).hexdigest()
    return {"seed": seed, "yaml": str(data_yaml.resolve()), "gt_source": "gt_one", "only_positives": True, "class_policy": "map-3-to-1", "counts": {"train": 2762, "val": 592, "test": 592}, "member_sha256": fingerprints}


def scientific_kwargs(args: argparse.Namespace, seed: int, data_yaml: Path) -> dict[str, Any]:
    values = train_kwargs(args, seed, data_yaml)
    for key in ("data", "device", "workers", "project", "name", "exist_ok"): values.pop(key, None)
    return canonical(values)


def contract(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    return canonical({"schema": "varroa_ftsc_alpha_contract_v1", "case": case, "seed": seed, "alpha": {"mode": "fixed", "value": ALPHA[case]}, "model_yaml_sha256": sha(YAMLS[case]), "baseline_yaml_sha256": sha(H2), "dataset": dataset_contract(data_yaml, seed), "pretrained_sha256": sha(Path(args.pretrained)), "train_args": scientific_kwargs(args, seed, data_yaml), "optimizer": optimizer_contract(args), "source": source_identity(), "evaluation": {"standard_nms_iou": 0.7, "size_bucket_nms_iou": 0.5, "imgsz": 640, "batch": 4}})


def run_dir(args: argparse.Namespace, case: str, seed: int) -> Path: return args.project / case / f"seed_{seed}"


def ensure_manifest(run: Path, expected: dict[str, Any]) -> None:
    path = run / "experiment_manifest.json"
    if path.is_file(): require(contracts_compatible(read_json(path).get("contract", {}), expected), f"CONTRACT_MISMATCH: {run}")
    elif run.exists() and any(run.iterdir()): raise RuntimeError(f"Untracked non-empty run: {run}")
    else: write_json(path, {"contract": expected})


def validate_metrics(metrics: dict[str, Any]) -> None:
    require(metrics.get("evaluation/nms_iou") == 0.7, "Varroa standard NMS must remain 0.7")
    require("test_size/AP50-Small" in metrics and metrics.get("test_size/protocol"), "Varroa native size-bucket metrics missing")
    for index, key in enumerate(CORE_METRICS):
        value = metrics.get(key); require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"METRIC_INCOMPLETE: {key}")
        require((0.0 <= float(value) <= 1.0) if index < 3 else float(value) > 0.0, f"Invalid metric: {key}")


def model_preflight(args: argparse.Namespace) -> dict[str, Any]:
    shared = shared_module(); shared.local_ultralytics(); import torch; from ultralytics import YOLO
    shared.seed_everything(0); baseline_model = YOLO(H2); baseline_model.load(args.pretrained, smart_transfer=True)
    reference = {name: value.detach().cpu() for name, value in baseline_model.model.state_dict().items()}; reports = {}
    for case in args.cases:
        shared.seed_everything(0); model = YOLO(YAMLS[case]); model.load(args.pretrained, smart_transfer=True); head = model.model.model[-1]; provider = head.ftsc_calibrator.providers["position_gaussian"]
        require(list(head.f) == [18, 21] and [int(v) for v in head.stride.tolist()] == [4, 8], f"{case}: runtime topology drift")
        require(provider.alpha_mode == "fixed" and provider.alpha == ALPHA[case], f"{case}: runtime alpha drift")
        state = {name: value.detach().cpu() for name, value in model.model.state_dict().items()}
        require(set(state) == set(reference) and all(torch.equal(state[k], reference[k]) for k in state), f"{case}: pretrained state differs from V1_FTSC H2")
        reports[case] = {"alpha": provider.alpha, "mode": provider.alpha_mode, "detect_from": list(head.f), "strides": [int(v) for v in head.stride.tolist()]}
    return reports


def train_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> Path:
    run = run_dir(args, case, seed); expected = contract(args, case, seed, data_yaml); ensure_manifest(run, expected)
    if (run / COMPLETE_ARTIFACT).is_file(): validate_completed(run, expected); return run
    shared = shared_module(); shared.local_ultralytics(); from ultralytics import YOLO
    if not (run / "config.yaml").is_file(): shutil.copy2(YAMLS[case], run / "config.yaml")
    last = run / "weights/last.pt"
    if last.is_file(): YOLO(last).train(resume=True)
    else:
        kwargs = train_kwargs(args, seed, data_yaml); kwargs.update(project=str(args.project / case), name=f"seed_{seed}")
        shared.seed_everything(seed); model = YOLO(YAMLS[case]); model.load(args.pretrained, smart_transfer=True); model.train(**kwargs)
    require(all((run / item).is_file() for item in TRAIN_ARTIFACTS), f"Incomplete training: {run}"); return run


def evaluate_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    run = run_dir(args, case, seed); expected = contract(args, case, seed, data_yaml); ensure_manifest(run, expected); require(all((run / item).is_file() for item in TRAIN_ARTIFACTS), f"Training artifacts missing: {run}")
    shared = shared_module(); args.data_yaml = data_yaml; metrics = shared.evaluate(run, args); validate_metrics(metrics); write_json(run / METRICS_ARTIFACT, metrics)
    artifacts = {relative: sha(run / relative) for relative in (*TRAIN_ARTIFACTS, "experiment_manifest.json", METRICS_ARTIFACT)}
    write_json(run / COMPLETE_ARTIFACT, {"status": "COMPLETE", "case": case, "seed": seed, "checkpoint_sha256": sha(run / "weights/best.pt"), "metrics_sha256": sha(run / METRICS_ARTIFACT), "artifact_sha256": artifacts}); return metrics


def validate_completed(run: Path, expected: dict[str, Any]) -> dict[str, Any]:
    require(contracts_compatible(read_json(run / "experiment_manifest.json").get("contract", {}), expected), f"CONTRACT_MISMATCH: {run}"); marker = read_json(run / COMPLETE_ARTIFACT); metrics = read_json(run / METRICS_ARTIFACT); validate_metrics(metrics)
    require(marker.get("checkpoint_sha256") == sha(run / "weights/best.pt") and marker.get("metrics_sha256") == sha(run / METRICS_ARTIFACT), f"Artifact hash mismatch: {run}")
    for relative in (*TRAIN_ARTIFACTS, "experiment_manifest.json", METRICS_ARTIFACT): require(marker.get("artifact_sha256", {}).get(relative) == sha(run / relative), f"Artifact hash mismatch: {run / relative}")
    return metrics


def historical_status(seed: int) -> dict[str, Any]:
    source = HISTORICAL_ROOT / f"seed_{seed}"; checkpoint = source / "weights/best.pt"
    config, train_args = source / "config.yaml", source / "args.yaml"
    verified = checkpoint.is_file() and config.is_file() and train_args.is_file() and load_yaml(config) == load_yaml(H2)
    if verified:
        saved = load_yaml(train_args)
        verified = saved.get("seed") == seed and saved.get("imgsz") == 640 and saved.get("batch") == 4 and saved.get("amp") is True and saved.get("mosaic") == 1.0 and saved.get("close_mosaic") == 10
    return {"case": "HISTORICAL_H2_ALPHA6", "seed": seed, "status": "EVAL_BACKFILL_REQUIRED" if verified else "BLOCKED", "source_run": str(source), "source_checkpoint": str(checkpoint), "provenance": "varroa_ftsc_causal_suite/V1_FTSC/read_only"}


def backfill_historical(args: argparse.Namespace, seed: int, data_yaml: Path) -> dict[str, Any]:
    """Evaluate canonical V1_FTSC best.pt without touching its source run."""
    source = HISTORICAL_ROOT / f"seed_{seed}"; checkpoint = source / "weights/best.pt"; config = source / "config.yaml"; args_path = source / "args.yaml"
    require(checkpoint.is_file(), f"BLOCKED: missing historical Varroa checkpoint {checkpoint}")
    require(config.is_file() and load_yaml(config) == load_yaml(H2), f"BLOCKED: unverified V1_FTSC config lineage {source}")
    require(args_path.is_file(), f"BLOCKED: missing historical Varroa args {source}")
    saved = load_yaml(args_path)
    require(saved.get("seed") == seed and saved.get("imgsz") == 640 and saved.get("batch") == 4 and saved.get("amp") is True and saved.get("mosaic") == 1.0 and saved.get("close_mosaic") == 10, f"BLOCKED: historical Varroa recipe mismatch {source}")
    require(Path(saved.get("data", "")).resolve() == data_yaml.resolve(), f"BLOCKED: historical Varroa split mismatch {source}")
    target = args.project / "HISTORICAL_H2_ALPHA6" / f"seed_{seed}"; target_checkpoint = target / "weights/best.pt"; target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if target_checkpoint.is_file(): require(sha(target_checkpoint) == sha(checkpoint), "Historical backfill checkpoint changed")
    else: shutil.copy2(checkpoint, target_checkpoint)
    shared = shared_module(); args.data_yaml = data_yaml; metrics = shared.evaluate(target, args); validate_metrics(metrics); write_json(target / METRICS_ARTIFACT, metrics)
    write_json(target / "historical_reference.json", {"status": "COMPLETE", "case": "HISTORICAL_H2_ALPHA6", "seed": seed, "source_run": str(source), "source_checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "training": "read_only_no_retraining", "protocol": "varroa_ftsc_causal_suite/V1_FTSC"})
    return metrics


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for case in args.cases:
        for seed in args.seeds:
            data = args.dataset_root / f"varroa_yolo_seed{seed}/varroa.yaml"; metrics = validate_completed(run_dir(args, case, seed), contract(args, case, seed, data)); rows.append({"case": case, "seed": seed, "status": "COMPLETE", **metrics})
    for seed in args.seeds:
        path = args.project / "HISTORICAL_H2_ALPHA6" / f"seed_{seed}" / METRICS_ARTIFACT; status = historical_status(seed)
        if path.is_file(): status["status"] = "COMPLETE"
        rows.append({**status, **(read_json(path) if path.is_file() else {})})
    args.project.mkdir(parents=True, exist_ok=True); fields = sorted(set().union(*(row.keys() for row in rows)))
    with (args.project / f"{NAMESPACE}_summary_runs.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups = []
    for case in args.cases:
        group = [row for row in rows if row["case"] == case]; require(len(group) == len(args.seeds), f"Partial aggregate: {case}"); record: dict[str, Any] = {"case": case, "runs": len(group)}
        comparable = len({json.dumps(canonical({k: row.get(k) for k in ("benchmark/gpu_name", "benchmark/device", "benchmark/imgsz", "benchmark/batch", "benchmark/precision")}), sort_keys=True) for row in group}) == 1
        record["benchmark/speed_comparable"] = comparable
        for key in CORE_METRICS:
            if key.startswith("test_speed/") and not comparable: continue
            values = [float(row[key]) for row in group]; record[f"{key}/mean"] = statistics.fmean(values); record[f"{key}/std"] = statistics.stdev(values)
        groups.append(record)
    fields = sorted(set().union(*(row.keys() for row in groups)))
    with (args.project / f"{NAMESPACE}_summary_aggregate.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(groups)
    write_json(args.project / f"{NAMESPACE}_manifest.json", {"protocol": NAMESPACE, "cases": list(args.cases), "seeds": list(args.seeds), "historical": [historical_status(seed) for seed in args.seeds]})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--phase", choices=("preflight", "train", "evaluate", "validate", "aggregate", "archive", "backfill-historical"), default="preflight")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES)); parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS)); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets"); parser.add_argument("--project", type=Path, default=ROOT / f"runs/{NAMESPACE}"); parser.add_argument("--pretrained", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--imgsz", type=int, default=640); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--patience", type=int, default=20); parser.add_argument("--workers", type=int, default=8); parser.add_argument("--device", default="cuda"); parser.add_argument("--seed-scope", default="all", choices=("all",)); parser.add_argument("--live-model-preflight", action="store_true"); return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    require((args.epochs, args.imgsz, args.batch_size, args.patience) == (100, 640, 4, 20), "Varroa frozen protocol drift"); require(set(args.cases).issubset(CASES) and len(args.cases) == len(set(args.cases)), "Invalid cases"); require(set(args.seeds).issubset(SEEDS) and len(args.seeds) == len(set(args.seeds)), "Invalid seeds")


def main(argv=None) -> None:
    args = parse_args(argv); validate_args(args); report = {"source": source_preflight(args), "optimizer": optimizer_contract(args), "historical": [historical_status(seed) for seed in args.seeds]}
    if args.live_model_preflight: report["model"] = model_preflight(args)
    if args.phase == "preflight": print(json.dumps(report, indent=2, sort_keys=True)); return
    if args.phase == "aggregate": aggregate(args); return
    shared = shared_module()
    for seed in args.seeds:
        data = shared.dataset_for_seed(args, seed)
        if args.phase == "backfill-historical": backfill_historical(args, seed, data); continue
        for case in args.cases:
            if args.phase == "train": train_one(args, case, seed, data)
            elif args.phase == "evaluate": evaluate_one(args, case, seed, data)
            elif args.phase in {"validate", "archive"}: validate_completed(run_dir(args, case, seed), contract(args, case, seed, data))


if __name__ == "__main__": main()
