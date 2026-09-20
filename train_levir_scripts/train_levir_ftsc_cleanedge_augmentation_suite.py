#!/usr/bin/env python3
"""Prepare the three future LEVIR FTSC Clean Edge augmentation experiments.

This runner is deliberately fail-closed: its default action only validates the
local code contract.  Training requires an explicit ``--confirm-run`` and is
not performed by this preparation task.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow
from train_levir_scripts import train_levir_ftsc_augmentation_suite as augmentation_suite

H2_CONFIG = REPO / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
ES1_CONFIG = REPO / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml"
MODEL_CONFIG = REPO / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_cleanedge025_nomosaic.yaml"
EXPERIMENT = "levir_ftsc_h2_cleanedge_augmentation_suite"
CASES = ("CE_A1_OACP_R2", "CE_A2_M5", "CE_A3_CP2")
SEEDS = (42, 43, 44)
EXPECTED_DETECT_FROM = (20, 23)
EXPECTED_STRIDES = (4.0, 8.0)
EDGE_PARAMETERS = {
    "residual_scale": 0.25,
    "residual_schedule": "constant",
    "orientation_gate_mode": "learned",
    "hidden": 32,
}
_AUGMENTATION_CASE = {
    "CE_A1_OACP_R2": "A1_OACP_R2",
    "CE_A2_M5": "A2_M5",
    "CE_A3_CP2": "A3_CP2",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    _require(path.is_file(), f"Missing model config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Invalid model YAML: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def augmentation_for(case: str, hard_negative_bank: Path | None = None) -> dict[str, object]:
    """Reuse the completed augmentation-suite contract for one CE case."""
    if case not in CASES:
        raise ValueError(case)
    return augmentation_suite.augmentation_for(
        _AUGMENTATION_CASE[case], hard_negative_bank
    )


def environment_for(case: str) -> dict[str, str]:
    if case not in CASES:
        raise ValueError(case)
    return augmentation_suite.environment_for(_AUGMENTATION_CASE[case])


def transform_order_for(case: str) -> tuple[str, ...]:
    if case not in CASES:
        raise ValueError(case)
    return (
        "CleanEdge_image_aware_fusion",
        *augmentation_suite.transform_order_for(_AUGMENTATION_CASE[case]),
        "Detect_post_edge_P2_post_C2f_P3",
    )


def _layers(payload: dict[str, Any]) -> list[list[Any]]:
    return [*payload["backbone"], *payload["head"]]


def _edge_indices(payload: dict[str, Any]) -> list[int]:
    backbone_length = len(payload["backbone"])
    return [
        backbone_length + index
        for index, layer in enumerate(payload["head"])
        if layer[2] == "P2EdgeCueFusion"
    ]


def _validate_augmentation(case: str, settings: dict[str, object], env: dict[str, str]) -> None:
    oacp = case == "CE_A1_OACP_R2"
    m5 = case == "CE_A2_M5"
    cp2 = case == "CE_A3_CP2"
    _require(bool(env) is oacp, f"{case}: OACP environment leakage")
    _require(settings["mosaic"] == (1.0 if m5 else 0.0), f"{case}: wrong mosaic")
    _require(settings["close_mosaic"] == (10 if m5 else 0), f"{case}: wrong close_mosaic")
    _require(settings["mosaic_policy"] == ("hard_negative" if m5 else "standard"), f"{case}: wrong mosaic policy")
    _require(settings["hard_negative_tile"] is m5, f"{case}: hard-negative leakage")
    _require(settings["hardneg_mosaic_prob"] == 0.30, f"{case}: hard-negative probability changed")
    _require(settings["copy_paste_enabled"] is cp2, f"{case}: Copy-Paste leakage")
    _require(settings["copy_paste"] == 0.0, f"{case}: stock segmentation CopyPaste must stay disabled")
    _require(settings["mixup"] == 0.0 and settings["cutmix"] == 0.0, f"{case}: mix augmentation leakage")
    if oacp:
        expected = {
            "YOLO_CONTEXT_AUG": "oacp",
            "OACP_PROFILE": "r2",
            "OACP_VARIANT": "current",
            "OACP_PLACEMENT": "pre_transform",
            "YOLO_LEGACY_DOUBLE_OACP": "0",
        }
        _require(env == expected, f"{case}: not corrected single-pass OACP R2")
        # Keep source preflight dependency-free. Runtime parity is covered by
        # the existing augmentation tests when cv2/numpy are installed.
        source = (REPO / "project_ultralytics/context_augment.py").read_text(encoding="utf-8")
        for fragment in (
            '"p": 0.40 if profile == "r2"',
            '"strength": (0.10, 0.25) if profile == "r2"',
            '"scale": (0.80, 0.95) if profile == "r2"',
            '"placement": "pre_transform" if profile == "r2"',
            '"protected_expand": float(os.environ.get("OACP_PROTECTED_EXPAND", 3.0))',
            'def build_context_augment(dataset) -> list[OACP]:',
        ):
            _require(fragment in source, f"{case}: corrected OACP R2 source contract changed: {fragment}")
    if cp2:
        expected = {
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
        _require(
            {key: settings[key] for key in expected} == expected,
            f"{case}: CP2 contract changed",
        )


def source_preflight(args: argparse.Namespace) -> dict[str, Any]:
    h2 = _load_yaml(H2_CONFIG)
    es1 = _load_yaml(ES1_CONFIG)
    model = _load_yaml(MODEL_CONFIG)
    _require(h2["head"][-1][0] == [19, 22], "H2 P2/P3 detector contract changed")
    _require(model["ftsc"] == h2["ftsc"], "Clean Edge FTSC differs from current H2")
    _require(model["head"][-1][0] == list(EXPECTED_DETECT_FROM), "Clean Edge Detect taps changed")
    edge_indices = _edge_indices(model)
    _require(edge_indices == [20], f"Expected one Clean Edge module at layer 20, got {edge_indices}")
    layers = _layers(model)
    _require(layers[20][2] == "P2EdgeCueFusion", "Detect P2 is not post-Edge")
    _require(
        layers[20][3] == [32, 0.25, "constant", 5, 15, "learned"],
        "Clean Edge P2EdgeCueFusion parameters changed",
    )
    _require(layers[23][2] == "C2f", "Detect P3 is not downstream post-C2f")
    _require(model["backbone"] == es1["backbone"], "Clean Edge backbone drifted from ES1")
    _require(model["head"][:-1] == es1["head"][:-1], "Clean Edge path drifted from ES1")
    for key, value in EDGE_PARAMETERS.items():
        _require(model["edge_stability"].get(key) == value, f"Clean Edge {key} changed")

    variants: dict[str, Any] = {}
    shared_settings: dict[str, object] | None = None
    for case in args.cases:
        settings = augmentation_for(case, args.hard_negative_bank)
        env = environment_for(case)
        _validate_augmentation(case, settings, env)
        unchanged = {
            key: value
            for key, value in settings.items()
            if key not in {
                "mosaic", "close_mosaic", "mosaic_policy", "hard_negative_tile",
                "hard_negative_bank", "copy_paste_enabled", "copy_paste_mode",
                "copy_paste_unit", "copy_paste_copies", "copy_paste_p",
                "copy_paste_scale", "copy_paste_padding", "copy_paste_blend",
                "copy_paste_placement", "copy_paste_max_overlap", "copy_paste_max_trials",
                "copy_paste_allow_empty_target", "copy_paste_allow_same_source",
            }
        }
        if shared_settings is None:
            shared_settings = unchanged
        else:
            _require(unchanged == shared_settings, f"{case}: common augmentation protocol drifted")
        variants[case] = {
            "model_config": str(MODEL_CONFIG),
            "comparison_reference": "existing_clean_edge_es1",
            "detect_from": list(EXPECTED_DETECT_FROM),
            "detect_source_layers": {"P2": 20, "P3": 23},
            "detect_source_types": {"P2": "P2EdgeCueFusion", "P3": "C2f"},
            "ftsc": copy.deepcopy(model["ftsc"]),
            "edge": {"layer": 20, **EDGE_PARAMETERS},
            "augmentation": settings,
            "oacp_environment": env,
            "transform_order": list(transform_order_for(case)),
            "primary_metric": "AP50",
            "secondary_metrics": ["mAP50-95", "AP75", "precision", "recall"],
        }
    return {
        "status": "PREPARED — NOT YET RUN",
        "experiment": EXPERIMENT,
        "git_commit": git_sha(),
        "cases": variants,
        "seeds": list(args.seeds),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "h2_config_sha256": sha256(H2_CONFIG),
    }


def model_preflight(args: argparse.Namespace) -> dict[str, Any]:
    workflow.local_ultralytics()
    import torch
    from ultralytics.nn.modules import Detect, P2EdgeCueFusion
    from ultralytics.nn.tasks import DetectionModel

    reports: dict[str, Any] = {}
    for case in args.cases:
        torch.manual_seed(0)
        model = DetectionModel(MODEL_CONFIG, verbose=False)
        head = model.model[-1]
        _require(isinstance(head, Detect), f"{case}: final module is not Detect")
        detect_from = [int(index) for index in head.f]
        strides = [float(value) for value in head.stride.detach().cpu().tolist()]
        _require(detect_from == list(EXPECTED_DETECT_FROM), f"{case}: runtime Detect taps changed")
        _require(strides == list(EXPECTED_STRIDES), f"{case}: runtime strides changed: {strides}")
        _require(head.ftsc_calibrator is not None and head.ftsc_calibrator.policy == "f5", f"{case}: FTSC H2 inactive")
        edges = [module for module in model.modules() if isinstance(module, P2EdgeCueFusion)]
        _require(len(edges) == 1, f"{case}: expected exactly one P2EdgeCueFusion")
        edge = edges[0]
        for key, value in EDGE_PARAMETERS.items():
            _require(getattr(edge, key) == value, f"{case}: runtime Edge {key} changed")
        source_types = [type(model.model[index]).__name__ for index in detect_from]
        _require(source_types == ["P2EdgeCueFusion", "C2f"], f"{case}: wrong Detect sources {source_types}")
        reports[case] = {
            "model_config": str(MODEL_CONFIG),
            "detect_from": detect_from,
            "detect_source_types": source_types,
            "strides": strides,
            "ftsc_policy": head.ftsc_calibrator.policy,
            "ftsc_evidence": list(head.ftsc_calibrator.evidence_names),
            "edge": EDGE_PARAMETERS,
        }
    return reports


def train_kwargs(args: argparse.Namespace, case: str, data_yaml: Path, seed: int) -> dict[str, object]:
    """Keep the established H2 recipe, varying only the selected augmentation."""
    return augmentation_suite.train_kwargs(
        args, _AUGMENTATION_CASE[case], data_yaml, seed
    )


def validate_execution_args(args: argparse.Namespace) -> None:
    _require(len(args.cases) == len(set(args.cases)), "Duplicate CE cases are not allowed")
    _require(len(args.seeds) == len(set(args.seeds)), "Duplicate seeds are not allowed")
    _require(set(args.seeds).issubset(SEEDS), f"Seeds must be selected from {list(SEEDS)}")
    if "CE_A2_M5" in args.cases:
        _require(args.hard_negative_bank is not None, "CE_A2_M5 requires --hard-negative-bank")
        _require(args.hard_negative_bank.is_file(), "CE_A2_M5 hard-negative bank does not exist")


def write_manifest(run_dir: Path, args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUN AUTHORIZED — METRICS PENDING",
        "experiment": EXPERIMENT,
        "case": case,
        "seed": seed,
        "split_seed": args.split_seed,
        "comparison_reference": "existing_clean_edge_es1",
        "model_config": str(MODEL_CONFIG),
        "model_config_sha256": sha256(MODEL_CONFIG),
        "data_yaml": str(data_yaml),
        "training": train_kwargs(args, case, data_yaml, seed),
        "oacp_environment": environment_for(case),
        "transform_order": transform_order_for(case),
        "primary_metric": "AP50",
        "secondary_metrics": ["mAP50-95", "AP75", "precision", "recall"],
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

    workflow.seed_everything(seed)
    with augmentation_suite.configured_environment(_AUGMENTATION_CASE[case]):
        model = YOLO(MODEL_CONFIG, task="detect")
        model.load(args.pretrained, smart_transfer=True)
        model.train(
            **train_kwargs(args, case, data_yaml, seed),
            project=str(args.project / case),
            name=f"seed_{seed}",
            exist_ok=True,
        )
    _require(all(path.is_file() for path in required), f"Incomplete artifacts: {run_dir}")
    return run_dir


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.project = args.project.resolve()
    report = source_preflight(args)
    try:
        report["model"] = model_preflight(args)
    except ModuleNotFoundError as error:
        report["model"] = {
            "status": "SKIPPED — optional local runtime dependency unavailable",
            "missing_dependency": error.name,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.confirm_run:
        print("PREPARED — NOT YET RUN (pass --confirm-run to authorize local training)")
        return
    validate_execution_args(args)
    pretrained = Path(args.pretrained).expanduser()
    _require(pretrained.is_file(), "--pretrained must name an existing local checkpoint; downloads are disabled")
    args.pretrained = str(pretrained.resolve())
    data_yaml = workflow.prepare_fixed_split(args)
    for seed in args.seeds:
        for case in args.cases:
            run_dir = train_one(args, case, seed, data_yaml)
            workflow.evaluate(run_dir, data_yaml, args)
            augmentation_suite.aggregate(args)


if __name__ == "__main__":
    main()
