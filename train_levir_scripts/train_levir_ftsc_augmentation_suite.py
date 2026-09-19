#!/usr/bin/env python3
"""Prepare/run the five-case FTSC H2 augmentation suite.

The default action is a read-only preflight. Training is unreachable unless
``--confirm-run`` is supplied. No upload or remote-service path exists here.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

MODEL_CONFIG = REPO / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
EXPERIMENT = "levir_ftsc_h2_augmentation_suite"
CASES = (
    "A0_FTSC",
    "A1_OACP_R2",
    "A2_M5",
    "A3_CP2",
    "A4_OACP_R2_M5",
)
SEEDS = (42, 43, 44)
OACP_ENV_KEYS = (
    "YOLO_CONTEXT_AUG",
    "OACP_PROFILE",
    "OACP_VARIANT",
    "OACP_PLACEMENT",
    "OACP_P",
    "OACP_STRENGTH_MIN",
    "OACP_STRENGTH_MAX",
    "OACP_SCALE_MIN",
    "OACP_SCALE_MAX",
    "OACP_PROTECTED_EXPAND",
    "YOLO_LEGACY_DOUBLE_OACP",
)

COMMON_AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,  # stock segmentation transform is always disabled
    "rect": False,
    "adaptive_zoom": False,
}


def augmentation_for(case: str, hard_negative_bank: Path | None = None) -> dict[str, object]:
    """Return the complete explicit augmentation configuration for one case."""
    if case not in CASES:
        raise ValueError(case)
    settings: dict[str, object] = {
        **COMMON_AUGMENTATION,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mosaic_policy": "standard",
        "hard_negative_tile": False,
        "hardneg_mosaic_prob": 0.30,
        "hard_negative_bank": "",
        "copy_paste_enabled": False,
        "copy_paste_mode": "flip",
        "copy_paste_unit": "single",
        "copy_paste_copies": 2,
        "copy_paste_p": 0.5,
        "copy_paste_scale": 1.0,
        "copy_paste_padding": 0.0,
        "copy_paste_blend": "hard",
        "copy_paste_placement": "random",
        "copy_paste_max_overlap": 0.0,
        "copy_paste_max_trials": 30,
        "copy_paste_allow_empty_target": True,
        "copy_paste_allow_same_source": True,
    }
    if case in {"A2_M5", "A4_OACP_R2_M5"}:
        settings.update(
            mosaic=1.0,
            close_mosaic=10,
            mosaic_policy="hard_negative",
            hard_negative_tile=True,
            hard_negative_bank=str(hard_negative_bank or ""),
        )
    if case == "A3_CP2":
        settings.update(copy_paste_enabled=True, copy_paste_mode="single")
    return settings


def environment_for(case: str) -> dict[str, str]:
    """Return source-compatible OACP environment settings.

    A4 deliberately omits OACP_PLACEMENT. Duy's ``levir_m5_oacp_r2`` runner
    did the same, and its pipeline default therefore applies OACP after
    M5/RandomPerspective. A1 explicitly uses pre_transform.
    """
    if case not in CASES:
        raise ValueError(case)
    if case not in {"A1_OACP_R2", "A4_OACP_R2_M5"}:
        return {}
    env = {
        "YOLO_CONTEXT_AUG": "oacp",
        "OACP_PROFILE": "r2",
        "OACP_VARIANT": "current",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
    }
    if case == "A1_OACP_R2":
        env["OACP_PLACEMENT"] = "pre_transform"
    return env


def transform_order_for(case: str) -> tuple[str, ...]:
    if case == "A1_OACP_R2":
        return ("OACP_R2", "Mosaic_disabled", "RandomPerspective")
    if case == "A4_OACP_R2_M5":
        return ("M5_HardNegativeMosaic", "RandomPerspective", "OACP_R2")
    if case == "A2_M5":
        return ("M5_HardNegativeMosaic", "RandomPerspective")
    if case == "A3_CP2":
        return ("Mosaic_disabled", "RandomPerspective", "DetectionCP2")
    return ("Mosaic_disabled", "RandomPerspective")


@contextlib.contextmanager
def configured_environment(case: str) -> Iterator[dict[str, str]]:
    previous = {key: os.environ.get(key) for key in OACP_ENV_KEYS}
    try:
        for key in OACP_ENV_KEYS:
            os.environ.pop(key, None)
        selected = environment_for(case)
        os.environ.update(selected)
        yield selected
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_kwargs(args: argparse.Namespace, case: str, data_yaml: Path, seed: int, amp: bool = True) -> dict[str, object]:
    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": seed,
        "deterministic": True,
        "amp": amp,
        "plots": False,
        "optimizer": "auto",
        "cos_lr": False,
        "lrf": 0.01,
        "fitness_metric": "map50_95",  # H2 checkpoint-selection protocol; AP50 is the suite's primary report metric.
        **augmentation_for(case, args.hard_negative_bank),
    }


def source_preflight(args: argparse.Namespace) -> dict[str, object]:
    import yaml

    payload = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    if payload["head"][-1][0] != [19, 22]:
        raise ValueError("FTSC H2 P2/P3 Detect inputs changed")
    ftsc = payload["ftsc"]
    if ftsc["evidence"] != ["position_gaussian", "dfl_distribution"]:
        raise ValueError("FTSC evidence changed")
    if ftsc["fixed_strengths"] != {"dfl_distribution": 1.0}:
        raise ValueError("FTSC DFL strength changed")
    configs = {case: augmentation_for(case, args.hard_negative_bank) for case in CASES}
    invariant_keys = (
        "epochs", "imgsz", "batch_size", "workers", "patience", "split_seed", "pretrained"
    )
    return {
        "status": "PREPARED — NOT YET RUN",
        "cases": list(CASES),
        "seeds": list(args.seeds),
        "model_config": str(MODEL_CONFIG),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "head_inputs": payload["head"][-1][0],
        "ftsc": ftsc,
        "invariants": {key: getattr(args, key) for key in invariant_keys},
        "augmentation": configs,
        "oacp_environment": {case: environment_for(case) for case in CASES},
        "transform_order": {case: transform_order_for(case) for case in CASES},
        "primary_metric": "AP50",
        "secondary_metrics": ["mAP50-95", "AP75", "precision", "recall"],
    }


def model_preflight() -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(MODEL_CONFIG, verbose=False)
    head = model.model[-1]
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    if strides != [4.0, 8.0]:
        raise ValueError(f"expected P2/P3 strides [4, 8], got {strides}")
    calibrator = head.ftsc_calibrator
    if calibrator is None or calibrator.policy != "f5":
        raise ValueError("FTSC H2 calibrator is inactive")
    return {
        "class": type(model).__name__,
        "head": type(head).__name__,
        "strides": strides,
        "ftsc_policy": calibrator.policy,
        "ftsc_evidence": list(calibrator.evidence_names),
    }


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def write_manifest(run_dir: Path, args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUN AUTHORIZED — METRICS PENDING",
        "experiment": EXPERIMENT,
        "case": case,
        "seed": seed,
        "split_seed": args.split_seed,
        "model_config": str(MODEL_CONFIG),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "data_yaml": str(data_yaml),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pretrained": args.pretrained,
        "augmentation": augmentation_for(case, args.hard_negative_bank),
        "oacp_environment": environment_for(case),
        "transform_order": transform_order_for(case),
        "primary_metric": "val/test AP50",
        "secondary_metrics": ["val/test mAP50-95", "val/test AP75", "val/test precision", "val/test recall"],
        "commit_sha": git_sha(),
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def train_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> Path:
    run_dir = args.project / case / f"seed_{seed}"
    required = (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv")
    if all(path.is_file() for path in required):
        return run_dir
    write_manifest(run_dir, args, case, seed, data_yaml)
    workflow.local_ultralytics()
    from ultralytics import YOLO

    # Reset Python/NumPy/Torch before every case so matched seeds start from
    # the same transferred model state; only the augmentation path may differ.
    workflow.seed_everything(seed)
    with configured_environment(case):
        model = YOLO(MODEL_CONFIG, task="detect")
        model.load(args.pretrained, smart_transfer=True)
        model.train(
            **train_kwargs(args, case, data_yaml, seed),
            project=str(args.project / case),
            name=f"seed_{seed}",
            exist_ok=True,
        )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete training artifacts: {run_dir}")
    return run_dir


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    for case in args.cases:
        for seed in args.seeds:
            path = args.project / case / f"seed_{seed}" / "evaluation_metrics.json"
            if path.is_file():
                rows.append({"case": case, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})
    if not rows:
        return
    args.project.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"case", "seed"}, key))
    with (args.project / "summary_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = []
    for case in args.cases:
        group = [row for row in rows if row["case"] == case]
        if not group:
            continue
        record: dict[str, object] = {"case": case, "runs": len(group)}
        for key in sorted(set.intersection(*(set(row) for row in group)) - {"case", "seed"}):
            try:
                values = [float(row[key]) for row in group]
            except (TypeError, ValueError):
                continue
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    summary_fields = sorted({key for row in summary for key in row}, key=lambda key: (key not in {"case", "runs"}, key))
    with (args.project / "summary_aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields); writer.writeheader(); writer.writerows(summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=REPO / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=REPO / "datasets")
    parser.add_argument("--project", type=Path, default=REPO / f"runs/{EXPERIMENT}")
    parser.add_argument("--hard-negative-bank", type=Path)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.project = args.project.resolve()
    report = source_preflight(args)
    report["model"] = model_preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.aggregate_only:
        aggregate(args)
        return
    if not args.confirm_run:
        print("PREPARED — NOT YET RUN (pass --confirm-run to authorize local training)")
        return
    if any(seed not in SEEDS for seed in args.seeds):
        raise ValueError(f"suite seeds are fixed to {SEEDS}")
    pretrained = Path(args.pretrained).expanduser()
    if not pretrained.is_file():
        raise FileNotFoundError("--pretrained must name an existing local checkpoint; downloads are disabled")
    args.pretrained = str(pretrained.resolve())
    needs_bank = any(case in {"A2_M5", "A4_OACP_R2_M5"} for case in args.cases)
    if needs_bank and (args.hard_negative_bank is None or not args.hard_negative_bank.is_file()):
        raise FileNotFoundError("A2/A4 require an existing --hard-negative-bank JSON")
    data_yaml = workflow.prepare_fixed_split(args)
    for seed in args.seeds:
        for case in args.cases:
            run_dir = train_one(args, case, seed, data_yaml)
            workflow.evaluate(run_dir, data_yaml, args)
            aggregate(args)


if __name__ == "__main__":
    main()
