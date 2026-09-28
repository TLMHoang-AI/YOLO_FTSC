#!/usr/bin/env python3
"""Preflight/run the isolated LEVIR FTSC H2 MuSGD Copy-Paste V1 ablation.

Default execution is read-only preflight.  Training requires --confirm-run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

EXPERIMENT = "levir_ftsc_musgd_cp_ablation_v1"
H2_MODEL_CONFIG = REPO / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
CASES = (
    "MG0_H2_MUSGD_BASE", "MG1_H2_MUSGD_CP2_RANDOM", "MG2_H2_MUSGD_CP2_COLLISION",
    "MG3_H2_MUSGD_CP2_COLLISION_P030", "MG4_H2_MUSGD_CP2_COLLISION_P070",
    "MG5_H2_MUSGD_CP1_COLLISION", "MG6_H2_MUSGD_CP2_POSITIVE_ONLY",
)
SEEDS = (42, 43, 44)
COMMON_AUGMENTATION = {
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0,
    "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5, "mosaic": 0.0, "close_mosaic": 0,
    "adaptive_zoom": False, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
    "rect": False, "mosaic_policy": "standard", "hard_negative_tile": False,
    "hardneg_mosaic_prob": 0.30, "hard_negative_bank": "",
}
CP2 = {
    "copy_paste_enabled": True, "copy_paste_mode": "single", "copy_paste_unit": "single",
    "copy_paste_copies": 2, "copy_paste_p": 0.5, "copy_paste_scale": 1.0,
    "copy_paste_padding": 0.0, "copy_paste_blend": "hard", "copy_paste_placement": "random",
    "copy_paste_max_overlap": 0.0, "copy_paste_max_trials": 30,
    "copy_paste_allow_empty_target": True, "copy_paste_allow_same_source": True,
}


def augmentation_for(case: str) -> dict[str, object]:
    """Return a complete, explicit augmentation configuration for ``case``."""
    if case not in CASES:
        raise ValueError(case)
    config: dict[str, object] = {**COMMON_AUGMENTATION, **CP2}
    if case == "MG0_H2_MUSGD_BASE":
        config["copy_paste_enabled"] = False
    elif case != "MG1_H2_MUSGD_CP2_RANDOM":
        config["copy_paste_placement"] = "collision_aware"
        if case == "MG3_H2_MUSGD_CP2_COLLISION_P030":
            config["copy_paste_p"] = 0.30
        elif case == "MG4_H2_MUSGD_CP2_COLLISION_P070":
            config["copy_paste_p"] = 0.70
        elif case == "MG5_H2_MUSGD_CP1_COLLISION":
            config["copy_paste_copies"] = 1
        elif case == "MG6_H2_MUSGD_CP2_POSITIVE_ONLY":
            config["copy_paste_allow_empty_target"] = False
    return config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepared_data_yaml(args: argparse.Namespace) -> Path:
    return args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}" / "levir_ship.yaml"


def train_kwargs(args: argparse.Namespace, case: str, data_yaml: Path, seed: int, amp: bool = True) -> dict[str, object]:
    return {
        "data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch_size,
        "device": args.device, "workers": args.workers, "patience": args.patience, "seed": seed,
        "deterministic": True, "amp": amp, "plots": False, "optimizer": "MuSGD", "lr0": 0.01,
        "momentum": 0.9, "cos_lr": False, "lrf": 0.01, "fitness_metric": "map50_95",
        **augmentation_for(case),
    }


def source_preflight(args: argparse.Namespace) -> dict[str, object]:
    import yaml

    payload = yaml.safe_load(H2_MODEL_CONFIG.read_text(encoding="utf-8"))
    if payload["head"][-1][0] != [19, 22]:
        raise ValueError("FTSC H2 P2/P3 Detect inputs changed")
    ftsc = payload["ftsc"]
    if ftsc["evidence"] != ["position_gaussian", "dfl_distribution"] or ftsc["fixed_strengths"] != {"dfl_distribution": 1.0}:
        raise ValueError("canonical H2 FTSC configuration changed")
    configs = {case: augmentation_for(case) for case in CASES}
    for case, config in configs.items():
        if config["mosaic"] != 0.0 or config["close_mosaic"] != 0 or config["copy_paste"] != 0.0:
            raise ValueError(f"{case} violates the no-Mosaic/stock-CopyPaste contract")
    data_yaml = prepared_data_yaml(args)
    dataset_status = "PENDING_DATA_PREPARATION"
    if data_yaml.is_file():
        workflow.validate_split(data_yaml)
        dataset_status = "VALIDATED_FIXED_SPLIT"
    return {
        "status": "PREPARED — NOT YET RUN", "experiment": EXPERIMENT, "cases": list(CASES),
        "seeds": list(args.seeds), "dataset": "LEVIR-Ship", "split_seed": args.split_seed,
        "data_yaml": str(data_yaml.resolve()), "dataset_preflight": dataset_status, "model_config": str(H2_MODEL_CONFIG),
        "model_config_sha256": sha256(H2_MODEL_CONFIG), "head_inputs": payload["head"][-1][0],
        "ftsc": ftsc, "augmentation": configs,
        "invariants": {key: getattr(args, key) for key in ("epochs", "imgsz", "batch_size", "workers", "patience", "pretrained")},
    }


def musgd_routing_preflight() -> dict[str, object]:
    """Mirror Trainer.build_optimizer's MuSGD grouping without constructing an optimizer."""
    workflow.local_ultralytics()
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn.tasks import DetectionModel

    # Fail closed if the local trainer no longer contains the routing behavior
    # this report mirrors.  The special routing is deliberately not changed.
    trainer_source = Path(sys.modules[BaseTrainer.__module__].__file__).read_text(encoding="utf-8")
    routing_rule = r"(?=.*23)(?=.*cv3)|proto\.semseg|SemanticSegment"
    if routing_rule not in trainer_source or "lr * 3" not in trainer_source:
        raise RuntimeError("local MuSGD lr*3 routing implementation changed; refusing preflight")

    model = DetectionModel(H2_MODEL_CONFIG, verbose=False)
    pattern = re.compile(routing_rule)
    muon_names, sgd_names, boosted_names = [], [], []
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            (muon_names if parameter.ndim >= 2 else sgd_names).append(name)
            if pattern.search(name):
                boosted_names.append(name)
    report = {
        "optimizer_requested": "MuSGD", "optimizer_resolved": "MuSGD", "base_lr": 0.01,
        "momentum": 0.9, "muon_parameter_count": len(muon_names), "sgd_parameter_count": len(sgd_names),
        "lr3_parameter_names": boosted_names, "lr3_parameter_count": len(boosted_names),
        "lr3_multiplier": 3, "lr3": 0.03,
        "routing_rule": routing_rule,
    }
    # All cases deliberately reference this same YAML.  Retain explicit per-case evidence in the preflight report.
    report["by_case"] = {case: {**report, "by_case": None} for case in CASES}
    signatures = {json.dumps(value, sort_keys=True) for value in report["by_case"].values()}
    if len(signatures) != 1:
        raise RuntimeError("MuSGD parameter routing differs across MG0–MG6")
    return report


def model_preflight() -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(H2_MODEL_CONFIG, verbose=False)
    head = model.model[-1]
    strides = [float(x) for x in head.stride.detach().cpu().tolist()]
    if strides != [4.0, 8.0] or list(head.f) != [19, 22]:
        raise ValueError("FTSC H2 P2/P3 graph changed")
    calibrator = head.ftsc_calibrator
    if calibrator is None or calibrator.policy != "f5":
        raise ValueError("FTSC f5 calibrator is inactive")
    return {"class": type(model).__name__, "head": type(head).__name__, "strides": strides,
            "detect_inputs": list(head.f), "ftsc_policy": calibrator.policy,
            "ftsc_evidence": list(calibrator.evidence_names)}


def compact_matrix() -> list[dict[str, object]]:
    columns = {
        "mosaic": "mosaic", "close_mosaic": "close_mosaic", "copy_paste_enabled": "copy_paste_enabled",
        "copies": "copy_paste_copies", "p": "copy_paste_p", "placement": "copy_paste_placement",
        "max_overlap": "copy_paste_max_overlap", "max_trials": "copy_paste_max_trials",
        "allow_empty_target": "copy_paste_allow_empty_target", "allow_same_source": "copy_paste_allow_same_source",
    }
    return [{"case": case, "model": H2_MODEL_CONFIG.name, "optimizer": "MuSGD", "lr0": 0.01, "momentum": 0.9,
             **{column: augmentation_for(case)[key] for column, key in columns.items()}}
            for case in CASES]


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def write_manifest(run_dir: Path, args: argparse.Namespace, case: str, seed: int, data_yaml: Path, routing: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "RUN AUTHORIZED — METRICS PENDING", "experiment": EXPERIMENT, "case": case, "seed": seed,
                "dataset": "LEVIR-Ship", "split_seed": args.split_seed, "data_yaml": str(data_yaml),
                "model_config": str(H2_MODEL_CONFIG), "model_config_sha256": sha256(H2_MODEL_CONFIG),
                "pretrained": args.pretrained, "optimizer_requested": "MuSGD", "optimizer_resolved": "MuSGD",
                "lr0": 0.01, "momentum": 0.9, "cos_lr": False, "lrf": 0.01,
                "augmentation": augmentation_for(case), "musgd_lr_routing": routing, "commit_sha": git_sha()}
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path, routing: dict[str, object]) -> Path:
    run_dir = args.project / case / f"seed_{seed}"
    required = (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv")
    if all(path.is_file() for path in required):
        return run_dir
    write_manifest(run_dir, args, case, seed, data_yaml, routing)
    workflow.local_ultralytics()
    from ultralytics import YOLO
    workflow.seed_everything(seed)
    model = YOLO(H2_MODEL_CONFIG, task="detect")
    model.load(args.pretrained, smart_transfer=True)
    model.train(**train_kwargs(args, case, data_yaml, seed), project=str(args.project / case), name=f"seed_{seed}", exist_ok=True)
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete training artifacts: {run_dir}")
    return run_dir


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    for case in args.cases:
        for seed in args.seeds:
            metric_path = args.project / case / f"seed_{seed}" / "evaluation_metrics.json"
            if metric_path.is_file():
                rows.append({"case": case, "seed": seed, **json.loads(metric_path.read_text(encoding="utf-8"))})
    if not rows:
        return
    args.project.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"case", "seed"}, key))
    with (args.project / "summary_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = []
    for case in args.cases:
        group = [row for row in rows if row["case"] == case]
        if not group: continue
        record = {"case": case, "runs": len(group)}
        for key in sorted(set.intersection(*(set(row) for row in group)) - {"case", "seed"}):
            try: values = [float(row[key]) for row in group]
            except (TypeError, ValueError): continue
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    fields = sorted({key for row in summary for key in row}, key=lambda key: (key not in {"case", "runs"}, key))
    with (args.project / "summary_aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES)); parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--split-seed", type=int, default=42); parser.add_argument("--data-root", type=Path, default=REPO / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=REPO / "datasets"); parser.add_argument("--project", type=Path, default=REPO / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt"); parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--confirm-run", action="store_true"); parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); args.project = args.project.resolve()
    report = source_preflight(args); report["matrix"] = compact_matrix(); report["model"] = model_preflight(); report["musgd_routing"] = musgd_routing_preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.aggregate_only: aggregate(args); return
    if not args.confirm_run:
        print("PREPARED — NOT YET RUN (pass --confirm-run to authorize local training)"); return
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"suite seeds are fixed to {SEEDS}")
    pretrained = Path(args.pretrained).expanduser()
    if not pretrained.is_file(): raise FileNotFoundError("--pretrained must name an existing local checkpoint; downloads are disabled")
    args.pretrained = str(pretrained.resolve()); data_yaml = workflow.prepare_fixed_split(args); routing = report["musgd_routing"]
    for seed in args.seeds:
        for case in args.cases:
            run_dir = train_one(args, case, seed, data_yaml, routing); workflow.evaluate(run_dir, data_yaml, args); aggregate(args)


if __name__ == "__main__":
    main()
