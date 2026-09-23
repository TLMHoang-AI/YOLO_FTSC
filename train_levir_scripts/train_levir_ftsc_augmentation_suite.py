#!/usr/bin/env python3
"""Prepare/run the nine-case FTSC H2/ES1 augmentation suite.

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

H2_MODEL_CONFIG = REPO / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
ES1_MODEL_CONFIG = REPO / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml"
# Backward-compatible alias used by the existing A0--A6 integrations.
MODEL_CONFIG = H2_MODEL_CONFIG
ES1_MODEL_CONFIG_SHA256 = "29d7bc1c09d970426e61bf7f15d5d8d687aad441319374ed95963f9892730e79"
EXPERIMENT = "levir_ftsc_h2_augmentation_suite"
CASES = (
    "A0_FTSC",
    "A1_OACP_R2",
    "A2_M5",
    "A3_CP2",
    "A4_OACP_R2_M5",
    "A5_NEGCANVAS_R1",
    "A6_NEGCANVAS_R4",
    "A7_ES1_NEGCANVAS_R1",
    "A8_ES1_NEGCANVAS_R4",
)
HARD_NEGATIVE_BANK_CASES = frozenset({"A2_M5", "A4_OACP_R2_M5"})
NEGATIVE_CANVAS_CASES = {
    "A5_NEGCANVAS_R1": "r1",
    "A6_NEGCANVAS_R4": "r4",
    "A7_ES1_NEGCANVAS_R1": "r1",
    "A8_ES1_NEGCANVAS_R4": "r4",
}
ES1_CASES = frozenset({"A7_ES1_NEGCANVAS_R1", "A8_ES1_NEGCANVAS_R4"})
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
    if case in NEGATIVE_CANVAS_CASES:
        # Keep R1/R4 in lockstep with the already-ported canonical recipe.
        from train_levir_scripts.train_ftsc_negative_canvas_suite import (
            augmentation_for as negative_canvas_augmentation_for,
        )

        settings.update(negative_canvas_augmentation_for(NEGATIVE_CANVAS_CASES[case]))
    return settings


def model_config_for(case: str) -> Path:
    """Select the frozen model graph for a suite case."""
    if case not in CASES:
        raise ValueError(case)
    return ES1_MODEL_CONFIG if case in ES1_CASES else H2_MODEL_CONFIG


def hard_negative_bank_required(cases: list[str] | tuple[str, ...]) -> bool:
    """Return whether any selected case consumes the M5 hard-negative bank."""
    return bool(HARD_NEGATIVE_BANK_CASES.intersection(cases))


def prepared_data_yaml(args: argparse.Namespace) -> Path:
    """Return the fixed-split YAML path without preparing or downloading data."""
    return args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}" / "levir_ship.yaml"


def dataset_preflight(args: argparse.Namespace) -> dict[str, object]:
    """Report negative-canvas eligibility while keeping default preflight read-only."""
    selected = [case for case in args.cases if case in NEGATIVE_CANVAS_CASES]
    data_yaml = prepared_data_yaml(args)
    report: dict[str, object] = {
        "required": bool(selected),
        "selected_cases": selected,
        "data_yaml": str(data_yaml.resolve()),
    }
    if not selected:
        report["status"] = "NOT_REQUIRED"
        return report
    if not data_yaml.is_file():
        report["status"] = "PENDING_DATA_PREPARATION"
        return report

    from train_levir_scripts.train_ftsc_negative_canvas_suite import dataset_eligibility

    report.update(dataset_eligibility(data_yaml))
    report["status"] = "READY"
    return report


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
    if case in NEGATIVE_CANVAS_CASES:
        return (
            "Mosaic_disabled",
            "RandomPerspective",
            "NegativeCanvasCopyPaste",
            "photometric_transforms/flips",
        )
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

    payload = yaml.safe_load(H2_MODEL_CONFIG.read_text(encoding="utf-8"))
    if payload["head"][-1][0] != [19, 22]:
        raise ValueError("FTSC H2 P2/P3 Detect inputs changed")
    ftsc = payload["ftsc"]
    if ftsc["evidence"] != ["position_gaussian", "dfl_distribution"]:
        raise ValueError("FTSC evidence changed")
    if ftsc["fixed_strengths"] != {"dfl_distribution": 1.0}:
        raise ValueError("FTSC DFL strength changed")

    es1_sha256 = sha256(ES1_MODEL_CONFIG)
    if es1_sha256 != ES1_MODEL_CONFIG_SHA256:
        raise ValueError(
            "historical ES1 model YAML changed: "
            f"expected {ES1_MODEL_CONFIG_SHA256}, got {es1_sha256}"
        )
    es1_payload = yaml.safe_load(ES1_MODEL_CONFIG.read_text(encoding="utf-8"))
    if es1_payload["head"][9] != [-1, 1, "nn.Identity", []]:
        raise ValueError("historical ES1 layer 19 Identity changed")
    if es1_payload["head"][10] != [
        -1,
        1,
        "P2EdgeCueFusion",
        [32, 0.25, "constant", 5, 15, "learned"],
    ]:
        raise ValueError("historical ES1 layer 20 P2EdgeCueFusion changed")
    if es1_payload["head"][-1][0] != [19, 22]:
        raise ValueError("historical ES1 Detect inputs must remain [19, 22]")
    if es1_payload["ftsc"] != ftsc:
        raise ValueError("ES1 must retain the canonical H2 FTSC configuration")

    configs = {case: augmentation_for(case, args.hard_negative_bank) for case in CASES}
    for case in NEGATIVE_CANVAS_CASES:
        config = configs[case]
        if config["hard_negative_tile"] or config["hard_negative_bank"]:
            raise RuntimeError(f"{case} must not depend on a hard-negative bank")
    model_configs = {case: model_config_for(case) for case in CASES}
    if any(model_configs[case] != H2_MODEL_CONFIG for case in CASES[:7]):
        raise RuntimeError("A0--A6 must retain the canonical H2 model YAML")
    if any(model_configs[case] != ES1_MODEL_CONFIG for case in ES1_CASES):
        raise RuntimeError("A7/A8 must use the historical ES1 model YAML")
    invariant_keys = (
        "epochs", "imgsz", "batch_size", "workers", "patience", "split_seed", "pretrained"
    )
    return {
        "status": "PREPARED — NOT YET RUN",
        "cases": list(CASES),
        "seeds": list(args.seeds),
        "model_config": str(H2_MODEL_CONFIG),
        "model_config_sha256": sha256(H2_MODEL_CONFIG),
        "model_config_by_case": {case: str(path) for case, path in model_configs.items()},
        "model_config_sha256_by_case": {
            case: sha256(path) for case, path in model_configs.items()
        },
        "head_inputs": payload["head"][-1][0],
        "ftsc": ftsc,
        "es1_graph": {
            "yaml_sha256": es1_sha256,
            "identity_layer": 19,
            "edge_fusion_layer": 20,
            "edge_fusion_args": es1_payload["head"][10][3],
            "detect_inputs": es1_payload["head"][-1][0],
            "edge_stability": es1_payload["edge_stability"],
        },
        "invariants": {key: getattr(args, key) for key in invariant_keys},
        "augmentation": configs,
        "oacp_environment": {case: environment_for(case) for case in CASES},
        "transform_order": {case: transform_order_for(case) for case in CASES},
        "hard_negative_bank": {
            "required": hard_negative_bank_required(args.cases),
            "selected_cases": [case for case in args.cases if case in HARD_NEGATIVE_BANK_CASES],
            "path": str(args.hard_negative_bank.resolve()) if args.hard_negative_bank else None,
        },
        "dataset_eligibility": dataset_preflight(args),
        "primary_metric": "AP50",
        "secondary_metrics": ["mAP50-95", "AP75", "precision", "recall"],
    }


def model_preflight(model_config: Path = MODEL_CONFIG) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    model_config = model_config.resolve()
    if model_config not in {H2_MODEL_CONFIG.resolve(), ES1_MODEL_CONFIG.resolve()}:
        raise ValueError(f"unsupported suite model config: {model_config}")
    model = DetectionModel(model_config, verbose=False)
    head = model.model[-1]
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    if strides != [4.0, 8.0]:
        raise ValueError(f"expected P2/P3 strides [4, 8], got {strides}")
    calibrator = head.ftsc_calibrator
    if calibrator is None or calibrator.policy != "f5":
        raise ValueError("FTSC f5 calibrator is inactive")
    if list(calibrator.evidence_names) != ["position_gaussian", "dfl_distribution"]:
        raise ValueError("FTSC evidence changed")
    if list(head.f) != [19, 22]:
        raise ValueError(f"Detect topology changed for {model_config.name}: {head.f}")

    report: dict[str, object] = {
        "class": type(model).__name__,
        "head": type(head).__name__,
        "strides": strides,
        "ftsc_policy": calibrator.policy,
        "ftsc_evidence": list(calibrator.evidence_names),
    }
    if model_config == ES1_MODEL_CONFIG.resolve():
        edge = model.model[20]
        runtime_signature = {
            "identity_layer_19": type(model.model[19]).__name__,
            "edge_layer_20": type(edge).__name__,
            "edge_from": edge.f,
            "residual_scale": float(edge.residual_scale),
            "residual_schedule": edge.residual_schedule,
            "ramp_start_epoch": int(edge.ramp_start_epoch),
            "ramp_end_epoch": int(edge.ramp_end_epoch),
            "orientation_gate_mode": edge.orientation_gate_mode,
            "hidden": int(edge.hidden),
            "detect_inputs": list(head.f),
        }
        expected_signature = {
            "identity_layer_19": "Identity",
            "edge_layer_20": "P2EdgeCueFusion",
            "edge_from": -1,
            "residual_scale": 0.25,
            "residual_schedule": "constant",
            "ramp_start_epoch": 5,
            "ramp_end_epoch": 15,
            "orientation_gate_mode": "learned",
            "hidden": 32,
            "detect_inputs": [19, 22],
        }
        if runtime_signature != expected_signature:
            raise ValueError(
                "historical ES1 runtime graph changed: "
                f"expected {expected_signature}, got {runtime_signature}"
            )
        report["es1_runtime_graph"] = runtime_signature
    return report


def model_preflights(cases: list[str] | tuple[str, ...]) -> dict[str, dict[str, object]]:
    """Build-check every model family needed by the selected cases."""
    reports = {"H2": model_preflight(H2_MODEL_CONFIG)}
    if ES1_CASES.intersection(cases):
        reports["ES1"] = model_preflight(ES1_MODEL_CONFIG)
    return reports


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def write_manifest(run_dir: Path, args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    model_config = model_config_for(case)
    manifest = {
        "status": "RUN AUTHORIZED — METRICS PENDING",
        "experiment": EXPERIMENT,
        "case": case,
        "seed": seed,
        "split_seed": args.split_seed,
        "model_config": str(model_config),
        "model_config_sha256": sha256(model_config),
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
    # the same transferred checkpoint before the explicit case model/augmentation.
    workflow.seed_everything(seed)
    with configured_environment(case):
        model = YOLO(model_config_for(case), task="detect")
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
    model_reports = model_preflights(args.cases)
    report["model"] = model_reports["H2"]
    report["models"] = model_reports
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
    if hard_negative_bank_required(args.cases) and (
        args.hard_negative_bank is None or not args.hard_negative_bank.is_file()
    ):
        raise FileNotFoundError("A2/A4 require an existing --hard-negative-bank JSON")
    data_yaml = workflow.prepare_fixed_split(args)
    if any(case in NEGATIVE_CANVAS_CASES for case in args.cases):
        from train_levir_scripts.train_ftsc_negative_canvas_suite import dataset_eligibility

        dataset_eligibility(data_yaml)
    for seed in args.seeds:
        for case in args.cases:
            run_dir = train_one(args, case, seed, data_yaml)
            workflow.evaluate(run_dir, data_yaml, args)
            aggregate(args)


if __name__ == "__main__":
    main()
