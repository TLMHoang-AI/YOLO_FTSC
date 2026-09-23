#!/usr/bin/env python3
"""Prepare or run the FTSC H2 R1/R4 negative-canvas screen.

The default action is source/dataset preflight only. Training requires the
explicit ``--confirm-run`` flag and a local checkpoint. No upload path exists.
Only LEVIR-Ship and TinyPerson are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ULTRALYTICS_ROOT = ROOT / "models_related/ultralytics"
if str(ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))

from project_ultralytics.copy_paste import copy_paste_config
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as levir_workflow
from train_tinyperson_scripts import train_all_tinyperson as tiny_workflow

MODEL_CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
DATASETS = ("levir", "tinyperson")
CASES = ("r1", "r4")
SEEDS = (42, 43, 44)
RUN_NAMES = {"r1": "FTSC_H2_R1_NEGCANVAS", "r4": "FTSC_H2_R4_NEGCANVAS"}
CONTROL_REFERENCES = {
    "levir": "existing matched FTSC-H2 no-Mosaic reference",
    "tinyperson": "existing H2_NM reference",
}

COMMON_NEGATIVE_CANVAS = {
    "mosaic": 0.0,
    "close_mosaic": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_enabled": True,
    "copy_paste_mode": "negative_canvas",
    "copy_paste_unit": "single",
    "copy_paste_copies": 1,
    "copy_paste_p": 0.5,
    "copy_paste_scale": 1.0,
    "copy_paste_padding": 0.0,
    "copy_paste_blend": "hard",
    "copy_paste_placement": "collision_aware",
    "copy_paste_max_overlap": 0.0,
    "copy_paste_max_trials": 30,
    "copy_paste_allow_empty_target": True,
    "copy_paste_allow_same_source": True,
    "negative_cp_p": 0.30,
    "negative_cp_target_max_size": 20.0,
    "negative_cp_deficit_gamma": 0.5,
    "negative_cp_max_weight_ratio": 3.0,
    "negative_cp_matched_ratio_max": 1.5,
    "negative_cp_large_ratio_min": 1.5,
    "negative_cp_large_ratio_max": 2.5,
    "negative_cp_blur_sigma": 0.5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def augmentation_for(case: str) -> dict[str, object]:
    if case not in CASES:
        raise ValueError(f"Unsupported case {case!r}; expected one of {CASES}")
    if case == "r1":
        specific = {
            "negative_cp_target_policy": "empirical",
            "negative_cp_donor_policy": "matched",
            "negative_cp_degradation": "none",
        }
    else:
        specific = {
            "negative_cp_target_policy": "deficit",
            "negative_cp_donor_policy": "larger",
            "negative_cp_degradation": "weak_blur",
        }
    return {**COMMON_NEGATIVE_CANVAS, **specific}


def _resolve_split(data_yaml: Path, split: str) -> Path:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    path = Path(payload[split])
    return path if path.is_absolute() else root / path


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Cannot map image path to labels path: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def dataset_eligibility(data_yaml: Path) -> dict[str, int]:
    """Count original train positives/negatives, donors, and <=20 px targets."""
    train = _resolve_split(data_yaml, "train")
    if not train.is_dir():
        raise FileNotFoundError(f"Prepared training image directory is missing: {train}")
    images = sorted(
        path for path in train.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    positive_images = negative_images = donor_objects = eligible_small = 0
    for image_path in images:
        label_path = _label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(f"Prepared image has no label file: {image_path}")
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            positive_images += 1
        else:
            negative_images += 1
        with Image.open(image_path) as image:
            width, height = image.size
        for line_number, values in enumerate(rows, 1):
            if len(values) != 5:
                raise ValueError(f"Malformed YOLO label {label_path}:{line_number}")
            _, _, _, normalized_width, normalized_height = map(float, values)
            size = ((normalized_width * width) * (normalized_height * height)) ** 0.5
            donor_objects += 1
            eligible_small += int(size <= 20.0)
    if negative_images == 0:
        raise RuntimeError(f"Negative-canvas preflight failed: {data_yaml} has zero original-negative train images")
    if eligible_small == 0:
        raise RuntimeError(f"Negative-canvas preflight failed: {data_yaml} has zero eligible <=20 px targets")
    return {
        "training_images": len(images),
        "original_positive_images": positive_images,
        "original_negative_images": negative_images,
        "donor_objects": donor_objects,
        "eligible_small_targets_le_20px": eligible_small,
    }


def source_preflight() -> dict[str, Any]:
    payload = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    if payload["head"][-1][0] != [19, 22]:
        raise RuntimeError("Canonical H2 P2/P3 Detect taps changed")
    ftsc = payload["ftsc"]
    if ftsc.get("policy") != "f5" or ftsc.get("evidence") != ["position_gaussian", "dfl_distribution"]:
        raise RuntimeError("Canonical FTSC F5 evidence contract changed")
    configs = {case: copy_paste_config(SimpleNamespace(**augmentation_for(case))) for case in CASES}
    if any(config["negative_bank_required"] for config in configs.values()):
        raise RuntimeError("R1/R4 unexpectedly require a negative bank")
    return {
        "model_config": str(MODEL_CONFIG),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "head_inputs": [19, 22],
        "ftsc": ftsc,
        "augmentation": {case: augmentation_for(case) for case in CASES},
        "serialized_copy_paste": configs,
        "negative_bank_required": False,
        "enabled_datasets": list(DATASETS),
        "disabled_datasets": ["varroa", "visdrone"],
        "control_references": {
            dataset: {"description": reference, "training_enabled": False}
            for dataset, reference in CONTROL_REFERENCES.items()
        },
    }


def prepared_yaml(args: argparse.Namespace, dataset: str, seed: int) -> Path:
    if dataset == "levir":
        return args.dataset_root / "levir_ship_yolo_seed42/levir_ship.yaml"
    return args.dataset_root / f"tinyperson_seed_{seed}_corner_sw640_sh512/tinyperson.yaml"


def prepare_datasets(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Reuse the existing LEVIR and TinyPerson preparation functions exactly."""
    reports: dict[str, dict[str, Any]] = {}
    if "levir" in args.datasets:
        namespace = SimpleNamespace(data_root=args.levir_data_root, dataset_root=args.dataset_root, split_seed=42)
        yaml_path = levir_workflow.prepare_fixed_split(namespace)
        reports["levir"] = {"seed_42": {"data_yaml": str(yaml_path), **dataset_eligibility(yaml_path)}}
    if "tinyperson" in args.datasets:
        test_dir = tiny_workflow.prepare_test_set(args.tinyperson_data_root, args.dataset_root)
        seed_reports = {}
        for seed in args.seeds:
            seed_dir = tiny_workflow.prepare_seed_dataset(
                args.tinyperson_data_root, args.dataset_root, test_dir, seed
            )
            yaml_path = seed_dir / "tinyperson.yaml"
            seed_reports[f"seed_{seed}"] = {"data_yaml": str(yaml_path), **dataset_eligibility(yaml_path)}
        reports["tinyperson"] = seed_reports
    return reports


def inspect_prepared_datasets(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for dataset in args.datasets:
        seed_reports = {}
        seeds = (42,) if dataset == "levir" else args.seeds
        for seed in seeds:
            yaml_path = prepared_yaml(args, dataset, seed)
            key = f"seed_{seed}"
            if yaml_path.is_file():
                seed_reports[key] = {"data_yaml": str(yaml_path), **dataset_eligibility(yaml_path)}
            else:
                seed_reports[key] = {"data_yaml": str(yaml_path), "status": "PENDING_DATA_PREPARATION"}
        reports[dataset] = seed_reports
    return reports


def _base_train_kwargs(args: argparse.Namespace, dataset: str, data_yaml: Path, seed: int) -> dict[str, Any]:
    common = {
        "data": str(data_yaml), "epochs": 100, "batch": 8, "device": args.device,
        "workers": 4, "patience": 20, "seed": seed, "deterministic": True,
        "amp": True, "plots": False, "fitness_metric": "map50_95",
    }
    if dataset == "levir":
        return {**common, "imgsz": 512, "optimizer": "auto", "cos_lr": False, "lrf": 0.01}
    return {
        **common, "imgsz": 640, "optimizer": "AdamW", "lr0": 0.002,
        "momentum": 0.9, "weight_decay": 0.0005, "cos_lr": False, "lrf": 0.01,
        "warmup_epochs": 3.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.0, "nbs": 64,
    }


def train_one(args: argparse.Namespace, dataset: str, case: str, seed: int, data_yaml: Path) -> Path:
    """Future local execution path; unreachable without explicit confirmation."""
    levir_workflow.local_ultralytics()
    from ultralytics import YOLO

    run_dir = args.project / dataset / RUN_NAMES[case] / f"seed_{seed}"
    if (run_dir / "weights/best.pt").is_file() and (run_dir / "results.csv").is_file():
        return run_dir
    model = YOLO(MODEL_CONFIG, task="detect")
    model.load(str(args.pretrained), smart_transfer=True)
    kwargs = {**_base_train_kwargs(args, dataset, data_yaml, seed), **augmentation_for(case)}
    kwargs.update(project=str(run_dir.parent), name=run_dir.name, exist_ok=True)
    model.train(**kwargs)
    if not (run_dir / "weights/best.pt").is_file():
        raise RuntimeError(f"Training did not produce best.pt: {run_dir}")
    return run_dir


def evaluate_one(
    args: argparse.Namespace, dataset: str, run_dir: Path, data_yaml: Path,
    test_dir: Path | None,
) -> dict[str, Any]:
    if dataset == "levir":
        eval_args = SimpleNamespace(imgsz=512, batch_size=8, device=args.device, workers=4)
        return levir_workflow.evaluate(run_dir, data_yaml, eval_args)
    if test_dir is None:
        raise RuntimeError("TinyPerson merged evaluation requires its prepared test windows")
    eval_args = SimpleNamespace(imgsz=640, batch_size=8, device=args.device, workers=4)
    return tiny_workflow.evaluate(
        run_dir, data_yaml, test_dir, args.tinyperson_data_root, eval_args
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--levir-data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--tinyperson-data-root", type=Path, default=ROOT.parent / "TinyPerson/tiny_set")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/ftsc_negative_canvas_suite")
    parser.add_argument("--pretrained", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if set(args.seeds) - set(SEEDS):
        raise ValueError(f"Seeds must be selected from {SEEDS}")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Duplicate seeds are not allowed")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    for field in ("dataset_root", "levir_data_root", "tinyperson_data_root", "project"):
        setattr(args, field, getattr(args, field).resolve())
    report = source_preflight()
    report["status"] = "PREPARED_NOT_RUN"
    report["dataset_preflight"] = prepare_datasets(args) if args.prepare_data else inspect_prepared_datasets(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.confirm_run:
        return
    if not args.pretrained.is_file():
        raise FileNotFoundError("--pretrained must be an existing local checkpoint; downloads are disabled")
    args.pretrained = args.pretrained.resolve()
    tiny_test_dir = None
    if "tinyperson" in args.datasets:
        tiny_test_dir = tiny_workflow.prepare_test_set(args.tinyperson_data_root, args.dataset_root)
    for dataset in args.datasets:
        for seed in args.seeds:
            data_yaml = prepared_yaml(args, dataset, seed)
            dataset_eligibility(data_yaml)
            for case in args.cases:
                run_dir = train_one(args, dataset, case, seed, data_yaml)
                evaluate_one(args, dataset, run_dir, data_yaml, tiny_test_dir)


if __name__ == "__main__":
    main()
