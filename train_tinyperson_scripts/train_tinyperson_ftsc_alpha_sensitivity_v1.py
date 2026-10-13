#!/usr/bin/env python3
"""TinyPerson H2_MOSAIC fixed-alpha sensitivity suite.

The trainable matrix is ALPHA4_rerun/ALPHA8_rerun x seeds 42/43/44. Historical
H2_MOSAIC (alpha=6) is read-only and may only be evaluated from its real
best.pt.  The default phase is source-only preflight and never trains.
"""
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
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8"
H2 = CONFIG_ROOT / "levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
CASES = ("ALPHA4_rerun", "ALPHA8_rerun")
SEEDS = (42, 43, 44)
ALPHA = {"ALPHA4_rerun": 4.0, "ALPHA8_rerun": 8.0}
YAMLS = {
    "ALPHA4_rerun": CONFIG_ROOT / "tinyperson/yolov8n_p2_tinyperson_ftsc_h2_alpha4.yaml",
    "ALPHA8_rerun": CONFIG_ROOT / "tinyperson/yolov8n_p2_tinyperson_ftsc_h2_alpha8.yaml",
}
NAMESPACE = "tinyperson_ftsc_position_alpha_h2_v1"
PROTOCOL = "tinyperson_ftsc_position_alpha_h2_v1"
HISTORICAL_VARIANT = "H2_MOSAIC"
HISTORICAL_ROOT = ROOT / "runs/tinyperson_ftsc_causal_suite/H2_MOSAIC"
TRAIN_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "config.yaml")
METRICS_ARTIFACT = "evaluation_metrics_alpha_contract_v1.json"
COMPLETE_ARTIFACT = "alpha_suite_complete.json"
CORE_METRICS = (
    "test_merged/AP50", "test_merged/mAP50-95", "test_merged/AP50-Small",
    "model/parameters", "model/GFLOPs", "test_speed/inference_ms_per_image",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(canonical(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid YAML: {path}")
    return value


def normalized_alpha(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    value["ftsc"]["position_alpha"] = 6.0
    value["ftsc"].pop("position_alpha_mode", None)
    return value


def source_identity() -> dict[str, Any]:
    files = (
        "train_tinyperson_scripts/train_tinyperson_ftsc_alpha_sensitivity_v1.py",
        "train_tinyperson_scripts/train_tinyperson_ftsc_causal_suite.py",
        "train_tinyperson_scripts/train_all_tinyperson.py",
        "vendor/tinyperson_cocoeval.py",
        "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
        "models_related/ultralytics/ultralytics/utils/tal.py",
        "models_related/ultralytics/ultralytics/utils/loss.py",
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, "unavailable"
    return {"commit": commit, "clean": dirty == "", "sha256": {name: sha(ROOT / name) for name in files}}


def source_preflight(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_yaml(H2)
    require(baseline["head"][-1][0] == [19, 22], "TinyPerson H2 Detect taps drifted")
    require(baseline["ftsc"].get("policy") == "f5", "TinyPerson H2 FTSC policy drifted")
    reports = {}
    for case in args.cases:
        payload = load_yaml(YAMLS[case])
        require(normalized_alpha(payload) == baseline, f"{case}: differs from H2 beyond position_alpha")
        require(payload["ftsc"].get("position_alpha") == ALPHA[case], f"{case}: wrong position_alpha")
        require(payload["ftsc"].get("position_alpha_mode", "fixed") == "fixed", f"{case}: adaptive alpha forbidden")
        reports[case] = {"yaml": str(YAMLS[case]), "sha256": sha(YAMLS[case]), "alpha": ALPHA[case]}
    tiny_source = (ROOT / "train_tinyperson_scripts/train_all_tinyperson.py").read_text(encoding="utf-8")
    for fragment in ("sw640_sh512", "conf=0.001", "iou=0.5", "max_det=300", "nms_detections(items, iou_threshold=0.5)", "test_merged/mAP50-95"):
        require(fragment in tiny_source, f"TinyPerson native protocol missing {fragment}")
    return {"protocol": PROTOCOL, "cases": reports, "source": source_identity(), "corner_geometry": "sw640_sh512", "model_imgsz": 640}


def native_modules():
    from train_tinyperson_scripts import train_all_tinyperson as workflow
    from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as causal
    return workflow, causal


def train_kwargs(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    _, causal = native_modules()
    values = causal.train_kwargs(args, HISTORICAL_VARIANT, seed, data_yaml)
    values.update(project=str(args.project / case), name=f"seed_{seed}_corner_sw640_sh512", exist_ok=True)
    return values


def dataset_contract(data_yaml: Path, seed: int) -> dict[str, Any]:
    manifest = data_yaml.parent / "corner_manifest.json"
    require(manifest.is_file(), f"Missing TinyPerson corner manifest: {manifest}")
    payload = read_json(manifest)
    train_sources = sorted({str(item["file_name"]) for item in payload.get("train", [])})
    val_sources = sorted({str(item["file_name"]) for item in payload.get("val", [])})
    require(set(train_sources).isdisjoint(val_sources), "TinyPerson original-image train/val leakage")
    return {"seed": seed, "yaml": str(data_yaml.resolve()), "manifest_sha256": sha(manifest), "train_source_sha256": hashlib.sha256("\n".join(train_sources).encode()).hexdigest(), "val_source_sha256": hashlib.sha256("\n".join(val_sources).encode()).hexdigest(), "corner_protocol": "sw640_sh512"}


def scientific_kwargs(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    values = train_kwargs(args, case, seed, data_yaml)
    for key in ("data", "device", "workers", "project", "name", "exist_ok"):
        values.pop(key, None)
    return canonical(values)


def contract(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    return canonical({
        "schema": "tinyperson_ftsc_alpha_contract_v1", "case": case, "seed": seed,
        "alpha": {"mode": "fixed", "value": ALPHA[case]},
        "model_yaml_sha256": sha(YAMLS[case]), "baseline_yaml_sha256": sha(H2),
        "dataset": dataset_contract(data_yaml, seed), "pretrained_sha256": sha(Path(args.pretrained)),
        "train_args": scientific_kwargs(args, case, seed, data_yaml), "source": source_identity(),
        "evaluation": {"coordinate_space": "merged_original", "window_conf": 0.001, "window_nms_iou": 0.5, "cross_window_nms_iou": 0.5, "max_det": 300, "iou_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)]},
    })


def run_dir(args: argparse.Namespace, case: str, seed: int) -> Path:
    return args.project / case / f"seed_{seed}_corner_sw640_sh512"


def validate_metrics(metrics: dict[str, Any]) -> None:
    require(metrics.get("test_merged/evaluator_contract") == "tinybenchmark_merged_original_iou50_95_v2", "Wrong TinyPerson evaluator contract")
    require(float(metrics.get("test_merged/available", 0.0)) == 1.0, "Merged TinyPerson evaluation unavailable")
    for key in CORE_METRICS:
        value = metrics.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"METRIC_INCOMPLETE: {key}")
    for key in CORE_METRICS[:3]:
        require(0.0 <= float(metrics[key]) <= 1.0, f"Out-of-range AP: {key}")
    for key in CORE_METRICS[3:]:
        require(float(metrics[key]) > 0.0, f"Non-positive complexity/latency: {key}")


def model_preflight(args: argparse.Namespace) -> dict[str, Any]:
    _, causal = native_modules()
    causal.local_ultralytics()
    import torch
    from ultralytics import YOLO
    causal._seed_everything(0); baseline_model = YOLO(H2); baseline_model.load(args.pretrained, smart_transfer=True)
    reference = {name: value.detach().cpu() for name, value in baseline_model.model.state_dict().items()}
    reports = {}
    for case in args.cases:
        causal._seed_everything(0)
        model = YOLO(YAMLS[case]); model.load(args.pretrained, smart_transfer=True)
        head = model.model.model[-1]
        provider = head.ftsc_calibrator.providers["position_gaussian"]
        require(list(head.f) == [19, 22] and [int(v) for v in head.stride.tolist()] == [4, 8], f"{case}: runtime topology drift")
        require(provider.alpha_mode == "fixed" and provider.alpha == ALPHA[case], f"{case}: runtime alpha drift")
        state = {name: value.detach().cpu() for name, value in model.model.state_dict().items()}
        require(set(state) == set(reference) and all(torch.equal(state[k], reference[k]) for k in state), f"{case}: pretrained state differs from H2_MOSAIC")
        reports[case] = {"alpha": provider.alpha, "mode": provider.alpha_mode, "detect_from": list(head.f), "strides": [int(v) for v in head.stride.tolist()]}
    return reports


def evaluate_checkpoint(run: Path, data_yaml: Path, test_out: Path, args: argparse.Namespace) -> dict[str, Any]:
    workflow, causal = native_modules(); metrics = workflow.evaluate(run, data_yaml, test_out, args.data_root, args)
    causal.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    model = YOLO(run / "weights/best.pt")
    metrics.update({"model/parameters": int(sum(p.numel() for p in model.model.parameters())), "model/GFLOPs": float(get_flops(model.model, imgsz=args.imgsz)), "model/GFLOPs_imgsz": args.imgsz, "model/GFLOPs_method": "ultralytics.get_flops", "benchmark/latency_unit": "ms/window"})
    validate_metrics(metrics); return metrics


def ensure_manifest(run: Path, expected: dict[str, Any]) -> None:
    path = run / "experiment_manifest.json"
    if path.is_file(): require(contracts_compatible(read_json(path).get("contract", {}), expected), f"CONTRACT_MISMATCH: {run}")
    elif run.exists() and any(run.iterdir()): raise RuntimeError(f"Untracked non-empty run: {run}")
    else: write_json(path, {"contract": expected})


def train_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> Path:
    require(case in CASES, "Historical alpha6 is read-only")
    run = run_dir(args, case, seed); expected = contract(args, case, seed, data_yaml); ensure_manifest(run, expected)
    if (run / COMPLETE_ARTIFACT).is_file(): validate_completed(run, expected); return run
    _, causal = native_modules(); causal.local_ultralytics(); from ultralytics import YOLO
    if not (run / "config.yaml").is_file(): shutil.copy2(YAMLS[case], run / "config.yaml")
    last = run / "weights/last.pt"
    if last.is_file(): YOLO(last).train(resume=True)
    else:
        causal._seed_everything(seed); model = YOLO(YAMLS[case]); model.load(args.pretrained, smart_transfer=True); model.train(**train_kwargs(args, case, seed, data_yaml))
    require(all((run / item).is_file() for item in TRAIN_ARTIFACTS), f"Incomplete training: {run}")
    return run


def validate_completed(run: Path, expected: dict[str, Any]) -> dict[str, Any]:
    require(contracts_compatible(read_json(run / "experiment_manifest.json").get("contract", {}), expected), f"CONTRACT_MISMATCH: {run}")
    marker = read_json(run / COMPLETE_ARTIFACT); metrics = read_json(run / METRICS_ARTIFACT); validate_metrics(metrics)
    require(marker.get("checkpoint_sha256") == sha(run / "weights/best.pt") and marker.get("metrics_sha256") == sha(run / METRICS_ARTIFACT), f"Artifact hash mismatch: {run}")
    for relative in (*TRAIN_ARTIFACTS, "experiment_manifest.json", METRICS_ARTIFACT):
        require(marker.get("artifact_sha256", {}).get(relative) == sha(run / relative), f"Artifact hash mismatch: {run / relative}")
    return metrics


def evaluate_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path, test_out: Path) -> dict[str, Any]:
    run = run_dir(args, case, seed); expected = contract(args, case, seed, data_yaml); ensure_manifest(run, expected)
    require(all((run / item).is_file() for item in TRAIN_ARTIFACTS), f"Training artifacts missing: {run}")
    metrics = evaluate_checkpoint(run, data_yaml, test_out, args); write_json(run / METRICS_ARTIFACT, metrics)
    artifacts = {relative: sha(run / relative) for relative in (*TRAIN_ARTIFACTS, "experiment_manifest.json", METRICS_ARTIFACT)}
    write_json(run / COMPLETE_ARTIFACT, {"status": "COMPLETE", "case": case, "seed": seed, "checkpoint_sha256": sha(run / "weights/best.pt"), "metrics_sha256": sha(run / METRICS_ARTIFACT), "artifact_sha256": artifacts})
    return metrics


def historical_source(seed: int) -> Path:
    return HISTORICAL_ROOT / f"seed_{seed}_corner_sw640_sh512"


def historical_status(seed: int) -> dict[str, Any]:
    source = historical_source(seed); checkpoint = source / "weights/best.pt"
    manifest = source / "experiment_manifest.json"
    verified = checkpoint.is_file() and manifest.is_file()
    if verified:
        contract_value = read_json(manifest).get("contract", {})
        verified = contract_value.get("variant") == "H2_MOSAIC" and contract_value.get("config_sha256") == sha(H2)
    return {"case": "HISTORICAL_H2_ALPHA6", "seed": seed, "status": "EVAL_BACKFILL_REQUIRED" if verified else "BLOCKED", "source_run": str(source), "source_checkpoint": str(checkpoint), "provenance": "tinyperson_causal_suite/H2_MOSAIC/read_only"}


def backfill_historical(args: argparse.Namespace, seed: int, data_yaml: Path, test_out: Path) -> dict[str, Any]:
    """Evaluate the real H2_MOSAIC checkpoint in an additive shadow directory."""
    source = historical_source(seed); checkpoint = source / "weights/best.pt"; manifest = source / "experiment_manifest.json"
    require(checkpoint.is_file(), f"BLOCKED: missing historical TinyPerson checkpoint {checkpoint}")
    require(manifest.is_file(), f"BLOCKED: missing H2_MOSAIC manifest {source}")
    source_contract = read_json(manifest).get("contract", {})
    require(source_contract.get("variant") == "H2_MOSAIC" and source_contract.get("config_sha256") == sha(H2), f"BLOCKED: unverified H2_MOSAIC config lineage {source}")
    require(source_contract.get("data_yaml_sha256") == sha(data_yaml), f"BLOCKED: historical TinyPerson split mismatch {source}")
    training = source_contract.get("training", {})
    require(training.get("mosaic") == 1.0 and training.get("close_mosaic") == 10 and training.get("seed") == seed, f"BLOCKED: historical TinyPerson recipe mismatch {source}")
    target = args.project / "HISTORICAL_H2_ALPHA6" / f"seed_{seed}_corner_sw640_sh512"
    target_checkpoint = target / "weights/best.pt"; target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if target_checkpoint.is_file(): require(sha(target_checkpoint) == sha(checkpoint), "Historical backfill checkpoint changed")
    else: shutil.copy2(checkpoint, target_checkpoint)
    metrics = evaluate_checkpoint(target, data_yaml, test_out, args); write_json(target / METRICS_ARTIFACT, metrics)
    write_json(target / "historical_reference.json", {"status": "COMPLETE", "case": "HISTORICAL_H2_ALPHA6", "seed": seed, "source_run": str(source), "source_checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "training": "read_only_no_retraining", "protocol": "tinyperson_causal_suite/H2_MOSAIC"})
    return metrics


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for case in args.cases:
        for seed in args.seeds:
            run = run_dir(args, case, seed); metrics = validate_completed(run, contract(args, case, seed, args.dataset_root / f"tinyperson_seed_{seed}_corner_sw640_sh512/tinyperson.yaml"))
            rows.append({"case": case, "seed": seed, "status": "COMPLETE", **metrics})
    for seed in args.seeds:
        path = args.project / "HISTORICAL_H2_ALPHA6" / f"seed_{seed}_corner_sw640_sh512" / METRICS_ARTIFACT
        status = historical_status(seed)
        if path.is_file():
            status["status"] = "COMPLETE"
        rows.append({**status, **(read_json(path) if path.is_file() else {})})
    args.project.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (args.project / f"{NAMESPACE}_summary_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    groups = []
    for case in args.cases:
        group = [row for row in rows if row["case"] == case]; require(len(group) == len(args.seeds), f"Partial aggregate: {case}")
        record: dict[str, Any] = {"case": case, "runs": len(group)}
        comparable = len({json.dumps(canonical({k: row.get(k) for k in ("benchmark/gpu_name", "benchmark/device", "benchmark/imgsz", "benchmark/batch", "benchmark/precision", "benchmark/latency_unit")}), sort_keys=True) for row in group}) == 1
        record["benchmark/speed_comparable"] = comparable
        for key in CORE_METRICS:
            if key.startswith("test_speed/") and not comparable: continue
            values = [float(row[key]) for row in group]; record[f"{key}/mean"] = statistics.fmean(values); record[f"{key}/std"] = statistics.stdev(values)
        groups.append(record)
    fields = sorted(set().union(*(row.keys() for row in groups)))
    with (args.project / f"{NAMESPACE}_summary_aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(groups)
    write_json(args.project / f"{NAMESPACE}_manifest.json", {"protocol": PROTOCOL, "cases": list(args.cases), "seeds": list(args.seeds), "historical": [historical_status(seed) for seed in args.seeds]})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "train", "evaluate", "validate", "aggregate", "archive", "backfill-historical"), default="preflight")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES)); parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / f"runs/{NAMESPACE}"); parser.add_argument("--pretrained", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--imgsz", type=int, default=640); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--device", default="cuda"); parser.add_argument("--live-model-preflight", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    require((args.epochs, args.imgsz, args.batch_size, args.patience) == (100, 640, 8, 20), "TinyPerson frozen protocol drift")
    require(set(args.cases).issubset(CASES) and len(args.cases) == len(set(args.cases)), "Invalid cases")
    require(set(args.seeds).issubset(SEEDS) and len(args.seeds) == len(set(args.seeds)), "Invalid seeds")


def main(argv=None) -> None:
    args = parse_args(argv); validate_args(args); report = {"source": source_preflight(args), "historical": [historical_status(seed) for seed in args.seeds]}
    if args.live_model_preflight: report["model"] = model_preflight(args)
    if args.phase == "preflight": print(json.dumps(report, indent=2, sort_keys=True)); return
    if args.phase == "aggregate": aggregate(args); return
    workflow, _ = native_modules(); test_out = workflow.prepare_test_set(args.data_root, args.dataset_root)
    for seed in args.seeds:
        data_yaml = workflow.prepare_seed_dataset(args.data_root, args.dataset_root, test_out, seed) / "tinyperson.yaml"
        if args.phase == "backfill-historical": backfill_historical(args, seed, data_yaml, test_out); continue
        for case in args.cases:
            if args.phase == "train": train_one(args, case, seed, data_yaml)
            elif args.phase == "evaluate": evaluate_one(args, case, seed, data_yaml, test_out)
            elif args.phase in {"validate", "archive"}: validate_completed(run_dir(args, case, seed), contract(args, case, seed, data_yaml))


if __name__ == "__main__":
    main()
