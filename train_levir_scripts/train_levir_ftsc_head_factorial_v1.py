#!/usr/bin/env python3
"""LEVIR-Ship: controlled 2x2 factorial (P345/P23) x (FTSC absent/present).

All four arms are NEW training runs, from the same pretrained checkpoint, with
the historical LEVIR F5 no-Mosaic recipe. Historical results are references
ONLY and are never folded into this cohort. Default is read-only preflight.

Examples (from yolo_code/):
  python -m train_levir_scripts.train_levir_ftsc_head_factorial_v1
  python -m train_levir_scripts.train_levir_ftsc_head_factorial_v1 --model-preflight
  python -m train_levir_scripts.train_levir_ftsc_head_factorial_v1 --confirm-run
  python -m train_levir_scripts.train_levir_ftsc_head_factorial_v1 --aggregate
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as historical
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "models_related/models_config/yolov8/levir"
STOCK_YAML = ROOT / "models_related/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml"
RUNS = ROOT / "runs/levir_ftsc_head_factorial_v1"
CASES = {
    "P345_NOFTSC": CONFIG / "yolov8n_p345_levir_factorial_noftsc.yaml",
    "P345_FTSC": CONFIG / "yolov8n_p345_levir_factorial_ftsc.yaml",
    "P23_NOFTSC": CONFIG / "yolov8n_p2_levir_b1_p23_noftsc.yaml",
    "P23_FTSC": CONFIG / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml",
}
FACTORS = {
    "P345_NOFTSC": ("P345", False),
    "P345_FTSC": ("P345", True),
    "P23_NOFTSC": ("P23", False),
    "P23_FTSC": ("P23", True),
}
DETECT = {"P345": [15, 18, 21], "P23": [19, 22]}
STRIDES = {"P345": [8.0, 16.0, 32.0], "P23": [4.0, 8.0]}
SEEDS = (42, 43, 44)
METRICS = {
    "AP50": "test/metrics/mAP50(B)",
    "mAP50_95": "test/metrics/mAP50-95(B)",
    "AP50_small": "test_size/AP50-Small",
}
# Relevant changes to code, preprocessing or measurement invalidate comparability.
SOURCE_FILES = (
    "train_levir_scripts/train_levir_ftsc_head_factorial_v1.py",
    "train_levir_scripts/train_all_levir_yolov8n_p2_ftsc_head_pruning.py",
    "train_levir_scripts/train_all_levir_yolov8n_p2_routing.py",
    "models_related/ultralytics/ultralytics/nn/tasks.py",
    "models_related/ultralytics/ultralytics/nn/modules/head.py",
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
    "models_related/ultralytics/ultralytics/engine/trainer.py",
    "models_related/ultralytics/ultralytics/cfg/default.yaml",
    "models_related/ultralytics/ultralytics/utils/loss.py",
    "models_related/ultralytics/ultralytics/utils/tal.py",
    "evaluate_test/standard_detection_metrics.py",
    "evaluate_test/size_bucket_evaluator.py",
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Invalid YAML: {path}")
    return payload


def graph(payload: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy({key: payload[key] for key in ("nc", "scale", "scales", "backbone", "head")})
    # parse_model evaluates the stock YAML string "None" to Python None.
    # Normalize only Upsample's size argument; preserve all architecture checks.
    for section in ("backbone", "head"):
        for layer in result[section]:
            module, args = layer[2], layer[3]
            if module == "nn.Upsample" and args and args[0] == "None":
                args[0] = None
    return result


def source_preflight() -> dict[str, Any]:
    configs = {case: load(path) for case, path in CASES.items()}
    reference = load(STOCK_YAML)
    reference["nc"] = 1
    reference["scale"] = "n"
    reference["scales"] = {"n": reference["scales"]["n"]}
    require(graph(configs["P345_NOFTSC"]) == graph(reference),
            "P345 control is not the exact stock YOLOv8n model graph")
    for scale in ("P345", "P23"):
        off, on = configs[scale + "_NOFTSC"], configs[scale + "_FTSC"]
        require(graph(off) == graph(on), f"{scale}: changing FTSC changes the network graph")
        require(off["head"][-1][0] == DETECT[scale], f"{scale}: wrong Detect inputs")
        require(off.get("ftsc", {}).get("enabled") is False, f"{scale}: control accidentally enables FTSC")
        require(on.get("ftsc", {}).get("enabled") is True, f"{scale}: treatment disables FTSC")
        require(off.get("p2_offset_regression") is False and on.get("p2_offset_regression") is False,
                f"{scale}: P2 offset is a confound")
    require(configs["P345_FTSC"]["ftsc"] == configs["P23_FTSC"]["ftsc"],
            "FTSC F5 settings differ across detection heads")
    require(configs["P345_NOFTSC"]["backbone"] == configs["P23_NOFTSC"]["backbone"],
            "Backbone differs between head configurations")
    ftsc = configs["P23_FTSC"]["ftsc"]
    require(ftsc["policy"] == "f5" and ftsc["position_alpha"] == 6.0
            and ftsc["evidence"] == ["position_gaussian", "dfl_distribution"]
            and ftsc["fixed_strengths"] == {"dfl_distribution": 1.0},
            "Historical FTSC treatment contract changed")
    require(historical.FITNESS_METRIC == "map50_95", "Selection metric drift")
    return {
        "status": "STATIC_CONFIG_VERIFIED",
        "configs_sha256": {case: sha(path) for case, path in CASES.items()},
        "source_sha256": {name: sha(ROOT / name) for name in SOURCE_FILES},
        "stock_yaml_sha256": sha(STOCK_YAML),
        "factorial": {case: {"head": FACTORS[case][0], "ftsc": FACTORS[case][1]}
                      for case in CASES},
    }


def training(args: argparse.Namespace, data_yaml: Path, seed: int) -> dict[str, Any]:
    values = historical.train_kwargs(args, data_yaml, seed, True)
    require(values["optimizer"] == "auto" and values["mosaic"] == 0.0
            and values["close_mosaic"] == 0 and values["fitness_metric"] == "map50_95"
            and values["deterministic"] is True and values["amp"] is True,
            "No-Mosaic historical recipe drift")
    require((values["epochs"], values["imgsz"], values["batch"], values["patience"])
            == (100, 512, 8, 20), "Frozen LEVIR training recipe drift")
    return values


def dataset(args: argparse.Namespace) -> Path:
    data_yaml = args.dataset_root / "levir_ship_yolo_seed42/levir_ship.yaml"
    require(data_yaml.is_file(), f"Frozen split missing: {data_yaml}. Prepare it separately; do not auto-generate it")
    workflow.validate_split(data_yaml, args.data_root)
    return data_yaml


def dataset_fingerprint(data_yaml: Path) -> str:
    """Stable identity includes split membership and label bytes, not just counts."""
    payload = load(data_yaml)
    base = Path(payload["path"]).resolve()
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        images = sorted((base / payload[split]).glob("*.png"))
        require(images, f"Empty {split} split")
        for image in images:
            label = base / "labels" / split / (image.stem + ".txt")
            require(image.is_file() and label.is_file(), f"Missing {split} pair {image.stem}")
            digest.update(f"{split}/{image.name}\0".encode())
            digest.update(str(image.resolve()).encode())
            digest.update(b"\0")
            digest.update(sha(label).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def contract(args: argparse.Namespace, case: str, seed: int, data_yaml: Path,
             source: dict[str, Any]) -> dict[str, Any]:
    train = training(args, data_yaml, seed)
    scientific_train = {key: value for key, value in train.items()
                        if key not in {"device", "workers", "data"}}
    return {
        "schema": "levir_ftsc_head_factorial_v1",
        "case": case, "seed": seed, "seed_universe": list(SEEDS), "split_seed": 42,
        "head_configuration": FACTORS[case][0], "ftsc_enabled": FACTORS[case][1],
        "model_config_sha256": source["configs_sha256"][case],
        "source_sha256": source["source_sha256"],
        "stock_yaml_sha256": source["stock_yaml_sha256"],
        "pretrained_sha256": sha(args.pretrained),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha(data_yaml),
        "dataset_fingerprint_sha256": dataset_fingerprint(data_yaml),
        "scientific_train_args": scientific_train,
        "evaluation": {"imgsz": 512, "batch": 8, "nms_iou": 0.5,
                       "dataset": "levir", "evaluator": "standard_detection_metrics"},
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def current_state(run: Path, expected: dict[str, Any]) -> str:
    if not run.exists():
        return "NEW"
    require((run / "factorial_contract.json").is_file(),
            f"Non-empty or orphan run has no contract: {run}")
    saved = json.loads((run / "factorial_contract.json").read_text(encoding="utf-8"))
    require(saved == expected, f"Scientific contract mismatch; refusing overwrite: {run}")
    manifest = run / "experiment_manifest.json"
    require(manifest.is_file() and json.loads(manifest.read_text(encoding="utf-8")) == expected,
            f"Canonical HF experiment manifest missing or mismatched: {run}")
    completion = run / "factorial_completion.json"
    if completion.is_file():
        result = json.loads(completion.read_text(encoding="utf-8"))
        require(result["contract_sha256"] == sha(run / "factorial_contract.json")
                and result["best_sha256"] == sha(run / "weights/best.pt")
                and result["metrics_sha256"] == sha(run / "evaluation_metrics_extended.json"),
                f"Completed artifacts changed: {run}")
        return "COMPLETE"
    if (run / "weights/best.pt").is_file() and (run / "weights/last.pt").is_file() and (run / "results.csv").is_file():
        return "EVALUATE"
    if (run / "weights/last.pt").is_file():
        return "PARTIAL_TRAINING"
    return "ORPHAN_OR_PREPARING"


def model_preflight(pretrained: Path | None = None) -> dict[str, Any]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import Detect
    import torch
    reports = {}
    backbone_checksums = {}
    for case, path in CASES.items():
        torch.manual_seed(0)
        model = DetectionModel(str(path), verbose=False)
        head = model.model[-1]
        require(isinstance(head, Detect), f"{case}: not an Ultralytics Detect module")
        actual = [float(x) for x in head.stride.detach().cpu().tolist()]
        require(actual == STRIDES[FACTORS[case][0]], f"{case}: expected {STRIDES[FACTORS[case][0]]}, got {actual}")
        calib = getattr(head, "ftsc_calibrator", None)
        require((calib is not None) == FACTORS[case][1], f"{case}: FTSC runtime mismatch")
        if calib is not None:
            require(calib.policy == "f5"
                    and calib.evidence_names == ("position_gaussian", "dfl_distribution"),
                    f"{case}: FTSC calibration mechanism drift")
        reports[case] = {"stride": actual, "ftsc": calib is not None,
                         "model_params": sum(p.numel() for p in model.parameters())}
        if pretrained is not None:
            require(pretrained.is_file(), f"Missing pretrained weights: {pretrained}")
            from ultralytics import YOLO
            torch.manual_seed(0)
            loaded = YOLO(str(path), task="detect")
            loaded.load(str(pretrained), smart_transfer=True)
            digest = hashlib.sha256()
            found = 0
            for name, tensor in sorted(loaded.model.state_dict().items()):
                parts = name.split(".")
                if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit():
                    continue
                if int(parts[1]) > 9:
                    continue
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                found += 1
            require(found > 50, f"{case}: pretrained backbone state incomplete")
            backbone_checksums[case] = digest.hexdigest()
            reports[case]["pretrained_backbone_sha256"] = digest.hexdigest()
    if backbone_checksums:
        require(len(set(backbone_checksums.values())) == 1,
                f"Pretrained backbone differs across factorial cells: {backbone_checksums}")
    return reports


def train_and_evaluate(args: argparse.Namespace, case: str, seed: int, data_yaml: Path,
                       source: dict[str, Any]) -> None:
    run = args.project / case / f"seed_{seed}"
    expected = contract(args, case, seed, data_yaml, source)
    state = current_state(run, expected)
    if state == "COMPLETE":
        print(f"[REUSE MATCHED] {case}/seed_{seed}", flush=True)
        return
    if state == "ORPHAN_OR_PREPARING":
        require(False, f"Orphan/partially initialized run; inspect manually: {run}")
    if state == "NEW":
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "factorial_contract.json", expected)
        # The canonical HF archive restore contract expects this standard filename.
        write_json(run / "experiment_manifest.json", expected)
        workflow.local_ultralytics()
        from ultralytics import YOLO
        workflow.seed_everything(seed)
        model = YOLO(str(CASES[case]), task="detect")
        model.load(str(args.pretrained), smart_transfer=True)
        options = training(args, data_yaml, seed)
        model.train(**options, project=str(run.parent), name=run.name, exist_ok=True)
        require((run / "weights/best.pt").is_file() and (run / "weights/last.pt").is_file()
                and (run / "results.csv").is_file(), f"Missing training artifacts: {run}")
    elif state == "PARTIAL_TRAINING":
        require(args.resume, f"Partial training: {run}; explicit --resume required")
        workflow.local_ultralytics()
        from ultralytics import YOLO
        # Never resume a completed run or silently restart without optimizer state.
        import torch
        checkpoint = torch.load(run / "weights/last.pt", map_location="cpu", weights_only=False)
        require(isinstance(checkpoint, dict) and checkpoint.get("optimizer") is not None,
                f"Checkpoint not resumable (optimizer missing): {run}")
        prior = checkpoint.get("train_args", {})
        require(isinstance(prior, dict), f"Missing checkpoint train_args: {run}")
        options = training(args, data_yaml, seed)
        for key in ("seed", "epochs", "imgsz", "batch", "optimizer", "mosaic", "close_mosaic"):
            require(str(prior.get(key)) == str(options[key]),
                    f"Resume train_args mismatch for {key}: {run}")
        YOLO(str(run / "weights/last.pt")).train(resume=True, device=args.device, workers=args.workers)
        require((run / "weights/best.pt").is_file(), f"Resume produced no best checkpoint: {run}")
    if state in ("NEW", "PARTIAL_TRAINING", "EVALUATE"):
        from evaluate_test.standard_detection_metrics import evaluate_run, load_merged_metrics
        evaluate_run(run_dir=run, data_yaml=data_yaml, dataset="levir",
                     imgsz=512, batch=8, workers=args.workers, device=args.device,
                     nms_iou=0.5, include_size=True)
        scores = load_merged_metrics(run)
        for name, key in METRICS.items():
            require(key in scores and math.isfinite(float(scores[key])),
                    f"Evaluation missing/invalid {name}: {run}")
        # Copy immutable architecture YAML for replayability and HF archival.
        import shutil
        config_copy = run / "config.yaml"
        if config_copy.is_file():
            require(sha(config_copy) == sha(CASES[case]), f"Model YAML copy drift: {run}")
        else:
            shutil.copy2(CASES[case], config_copy)
        write_json(run / "factorial_completion.json", {
            "state": "COMPLETE",
            "contract_sha256": sha(run / "factorial_contract.json"),
            "best_sha256": sha(run / "weights/best.pt"),
            "metrics_sha256": sha(run / "evaluation_metrics_extended.json"),
        })
        require(current_state(run, expected) == "COMPLETE", f"Completion integrity check failed: {run}")
        print(f"[COMPLETE] {case}/seed_{seed}", flush=True)


def mean_std(values: list[float]) -> dict[str, float]:
    require(len(values) == len(SEEDS), "Cannot report 3-seed statistics from incomplete values")
    return {"mean": statistics.fmean(values), "sample_std": statistics.stdev(values)}


def factorial_effects(scores: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, Any]]:
    """Five paired seed-level effects per metric; all 12 cells required."""
    require(set(scores) == set(CASES), "Missing factorial arm")
    require(all(set(scores[case]) == set(SEEDS) for case in CASES), "Incomplete paired seeds")
    result = []
    factors = {
        "ftsc_effect_p345": ("P345_FTSC", "P345_NOFTSC"),
        "ftsc_effect_p23": ("P23_FTSC", "P23_NOFTSC"),
        "head_effect_noftsc": ("P23_NOFTSC", "P345_NOFTSC"),
        "head_effect_ftsc": ("P23_FTSC", "P345_FTSC"),
    }
    for metric in METRICS:
        deltas = {name: [scores[a][s][metric] - scores[b][s][metric]
                         for s in SEEDS] for name, (a, b) in factors.items()}
        deltas["interaction_p23_vs_p345"] = [
            p23 - p345 for p23, p345 in zip(
                deltas["ftsc_effect_p23"], deltas["ftsc_effect_p345"], strict=True
            )
        ]
        for name, values in deltas.items():
            result.append({"metric": metric, "effect": name, "n": len(SEEDS),
                           **mean_std(values),
                           **{f"seed_{seed}": values[i] for i, seed in enumerate(SEEDS)}})
    return result


def aggregate(args: argparse.Namespace, data_yaml: Path, source: dict[str, Any]) -> dict[str, Any]:
    from evaluate_test.standard_detection_metrics import load_merged_metrics
    scores: dict[str, dict[int, dict[str, float]]] = {}
    rows = []
    for case in CASES:
        scores[case] = {}
        for seed in SEEDS:
            run = args.project / case / f"seed_{seed}"
            require(current_state(run, contract(args, case, seed, data_yaml, source)) == "COMPLETE",
                    f"Cannot aggregate incomplete factorial: {case}/seed_{seed}")
            data = load_merged_metrics(run)
            value = {name: float(data[key]) for name, key in METRICS.items()}
            require(all(math.isfinite(v) for v in value.values()), f"Non-finite metric {run}")
            scores[case][seed] = value
            rows.append({"case": case, "seed": seed, **value})
    summaries = []
    for case in CASES:
        item = {"case": case, "n": len(SEEDS)}
        for metric in METRICS:
            for k, v in mean_std([scores[case][s][metric] for s in SEEDS]).items():
                item[f"{metric}_{k}"] = v
        summaries.append(item)
    effects = factorial_effects(scores)
    def write_csv(name: str, data: list[dict[str, Any]]) -> None:
        file = args.project / name
        with file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    args.project.mkdir(parents=True, exist_ok=True)
    write_csv("factorial_runs.csv", rows)
    write_csv("factorial_aggregate.csv", summaries)
    write_csv("factorial_effects.csv", effects)
    return {"status": "FULL_12_RUNS_AGGREGATED", "runs": len(rows),
            "effects": len(effects), "output_dir": str(args.project)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true", help="Train/evaluate selected NEW-cohort arms")
    parser.add_argument("--model-preflight", action="store_true", help="Instantiate and inspect four models; no training")
    parser.add_argument("--aggregate", action="store_true", help="Require 12 fully completed runs and compute paired effects")
    parser.add_argument("--resume", action="store_true", help="Explicitly resume a contract-verified partial checkpoint")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--project", type=Path, default=RUNS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    require(args.split_seed == 42, "LEVIR split seed must stay at 42")
    require(args.workers >= 0, "Invalid workers")
    require(len(args.cases) == len(set(args.cases)), "Duplicate cases")
    require(len(args.seeds) == len(set(args.seeds)) and set(args.seeds).issubset(SEEDS),
            "Seeds must be unique members of (42,43,44)")
    require(sum(bool(x) for x in (args.confirm_run, args.model_preflight, args.aggregate)) <= 1,
            "Select at most one action")
    require(not args.resume or args.confirm_run, "--resume requires --confirm-run")
    source = source_preflight()
    result: dict[str, Any] = {"status": "STATIC_PREFLIGHT_ONLY", "source": source}
    if args.model_preflight:
        result = {"status": "MODEL_PREFLIGHT_PASSED", "models": model_preflight()}
    elif args.confirm_run or args.aggregate:
        args.pretrained = args.pretrained.expanduser().resolve()
        args.data_root = args.data_root.expanduser().resolve()
        args.dataset_root = args.dataset_root.expanduser().resolve()
        args.project = args.project.expanduser().resolve()
        require(args.pretrained.is_file(), f"Missing local pretrained checkpoint: {args.pretrained}")
        data_yaml = dataset(args)
        if args.aggregate:
            require(set(args.cases) == set(CASES) and set(args.seeds) == set(SEEDS),
                    "Aggregation must include the full 2x2 x 3 seed design")
            result = aggregate(args, data_yaml, source)
        else:
            model_preflight(args.pretrained)  # Verify heads and identical pretrained backbone.
            for seed in args.seeds:
                for case in args.cases:
                    train_and_evaluate(args, case, seed, data_yaml, source)
            result = {"status": "REQUESTED_RUNS_PROCESSED",
                      "cases": args.cases, "seeds": args.seeds,
                      "note": "Run --aggregate only after all four cases x 3 seeds complete."}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
