#!/usr/bin/env python3
"""Preflight or run the seven-case VisDrone2019-DET FTSC augmentation suite.

The default action is local, read-only preflight. Training is unreachable
without ``--confirm-run`` and a local ``yolov8n.pt`` checkpoint. This runner
contains no downloader, uploader, or other remote-service integration.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "models_related/ultralytics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_visdrone_scripts import visdrone_protocol

H2_SOURCE_CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
ES1_SOURCE_CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml"
H2_MODEL_CONFIG = ROOT / "models_related/models_config/yolov8/visdrone/yolov8n_p2_visdrone_ftsc_h2_p2_p3.yaml"
ES1_MODEL_CONFIG = ROOT / "models_related/models_config/yolov8/visdrone/yolov8n_p2_visdrone_ftsc_es1_p2_p3.yaml"

EXPERIMENT = "visdrone2019_det_ftsc_ablation_suite"
CASES = (
    "V0_H2_MOSAIC",
    "V1_H2_NOMOSAIC",
    "V2_ES1_MOSAIC",
    "V3_H2_CP2_MOSAIC",
    "V4_H2_R4_MOSAIC",
    "V5_ES1_R4_MOSAIC",
    "V6_H2_OACP_R2_MOSAIC",
)
SEEDS = (42, 43, 44)
ES1_CASES = frozenset({"V2_ES1_MOSAIC", "V5_ES1_R4_MOSAIC"})
R4_CASES = frozenset({"V4_H2_R4_MOSAIC", "V5_ES1_R4_MOSAIC"})
OACP_CASES = frozenset({"V6_H2_OACP_R2_MOSAIC"})
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

COMMON_TRAINING = {
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "workers": 8,
    "patience": 0,
    "optimizer": "auto",
    "deterministic": True,
    "amp": True,
    "cos_lr": False,
    "lrf": 0.01,
    "fitness_metric": "map50_95",
    "iou": 0.5,
}

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
    "copy_paste": 0.0,
    "rect": False,
    "adaptive_zoom": False,
    "mosaic": 1.0,
    "close_mosaic": 10,
    "mosaic_policy": "standard",
    "hard_negative_tile": False,
    "hard_negative_bank": "",
    "copy_paste_enabled": False,
    "copy_paste_mode": "single",
    "copy_paste_unit": "single",
    "copy_paste_copies": 2,
    "copy_paste_p": 0.5,
    "copy_paste_scale": 1.0,
    "copy_paste_padding": 0.0,
    "copy_paste_max_overlap": 0.0,
    "copy_paste_max_trials": 30,
    "copy_paste_blend": "hard",
    "copy_paste_placement": "random",
    "copy_paste_allow_empty_target": True,
    "copy_paste_allow_same_source": True,
}

EVALUATION_PROTOCOL = {
    "standard": "native ten-class VisDrone val and test-dev evaluation",
    "standard_metrics": ["val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95"],
    "size_buckets": "evaluate_test.size_bucket_evaluator.evaluate_native_test_size_buckets",
    "nms_iou": 0.5,
    "class_agnostic_standard_eval": False,
}

EXPECTED_ARTIFACTS = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def local_ultralytics() -> None:
    if str(ULTRALYTICS_ROOT) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def model_config_for(case: str) -> Path:
    if case not in CASES:
        raise ValueError(case)
    return ES1_MODEL_CONFIG if case in ES1_CASES else H2_MODEL_CONFIG


def augmentation_for(case: str) -> dict[str, object]:
    """Return the complete, explicit augmentation configuration for one case."""
    if case not in CASES:
        raise ValueError(case)
    settings = deepcopy(COMMON_AUGMENTATION)
    if case == "V1_H2_NOMOSAIC":
        settings.update(mosaic=0.0, close_mosaic=0)
    elif case == "V3_H2_CP2_MOSAIC":
        settings["copy_paste_enabled"] = True
    elif case in R4_CASES:
        # Reuse the existing port rather than duplicating NegativeCanvasCopyPaste.
        from train_levir_scripts.train_ftsc_negative_canvas_suite import augmentation_for as negative_canvas_for

        settings.update(negative_canvas_for("r4"))
        settings.update(mosaic=1.0, close_mosaic=10, mosaic_policy="standard")
    return settings


def environment_for(case: str) -> dict[str, str]:
    if case not in CASES:
        raise ValueError(case)
    if case not in OACP_CASES:
        return {}
    return {
        "YOLO_CONTEXT_AUG": "oacp",
        "OACP_PROFILE": "r2",
        "OACP_VARIANT": "current",
        "OACP_PLACEMENT": "pre_transform",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
    }


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


def transform_order_for(case: str) -> tuple[str, ...]:
    if case not in CASES:
        raise ValueError(case)
    mosaic = "Mosaic_disabled" if case == "V1_H2_NOMOSAIC" else "StandardMosaic"
    if case == "V3_H2_CP2_MOSAIC":
        return (mosaic, "RandomPerspective", "DetectionCP2", "photometric_transforms/flips")
    if case in R4_CASES:
        return (mosaic, "RandomPerspective", "NegativeCanvasCopyPaste", "photometric_transforms/flips")
    if case in OACP_CASES:
        return ("OACP_R2", mosaic, "RandomPerspective", "photometric_transforms/flips")
    return (mosaic, "RandomPerspective", "photometric_transforms/flips")


def _dict_differences(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _assert_nc_only_parity(source_path: Path, target_path: Path) -> dict[str, object]:
    source, target = _load_yaml(source_path), _load_yaml(target_path)
    _require(source.get("nc") == 1, f"Unexpected source nc in {source_path}")
    _require(target.get("nc") == 10, f"VisDrone model must use nc=10: {target_path}")
    normalized = deepcopy(source)
    normalized["nc"] = 10
    _require(normalized == target, f"{target_path.name} differs from its LEVIR source by more than nc")
    return {
        "source": str(source_path),
        "target": str(target_path),
        "source_sha256": sha256(source_path),
        "target_sha256": sha256(target_path),
        "allowed_difference": {"nc": [1, 10]},
    }


def source_preflight() -> dict[str, Any]:
    parity = {
        "H2": _assert_nc_only_parity(H2_SOURCE_CONFIG, H2_MODEL_CONFIG),
        "ES1": _assert_nc_only_parity(ES1_SOURCE_CONFIG, ES1_MODEL_CONFIG),
    }
    h2, es1 = _load_yaml(H2_MODEL_CONFIG), _load_yaml(ES1_MODEL_CONFIG)
    expected_ftsc = {
        "policy": "f5",
        "evidence": ["position_gaussian", "dfl_distribution"],
        "warmup_epochs": 5,
        "ramp_epochs": 10,
        "fixed_strengths": {"dfl_distribution": 1.0},
    }
    for name, payload in (("H2", h2), ("ES1", es1)):
        _require(payload["nc"] == 10, f"{name}: nc changed")
        _require(payload["head"][-1][0] == [19, 22], f"{name}: Detect inputs changed")
        for key, expected in expected_ftsc.items():
            _require(payload["ftsc"].get(key) == expected, f"{name}: FTSC {key} changed")
    _require(es1["head"][9] == [-1, 1, "nn.Identity", []], "ES1 layer 19 changed")
    _require(
        es1["head"][10] == [-1, 1, "P2EdgeCueFusion", [32, 0.25, "constant", 5, 15, "learned"]],
        "Historical ES1 P2EdgeCueFusion graph changed",
    )
    expected_edge = {
        "residual_scale": 0.25,
        "residual_schedule": "constant",
        "ramp_start_epoch": 5,
        "ramp_end_epoch": 15,
        "orientation_gate_mode": "learned",
        "hidden": 32,
    }
    for key, expected in expected_edge.items():
        _require(es1["edge_stability"].get(key) == expected, f"ES1 edge_stability.{key} changed")

    configs = {case: augmentation_for(case) for case in CASES}
    _require(
        _dict_differences(configs["V0_H2_MOSAIC"], configs["V1_H2_NOMOSAIC"])
        == {"mosaic", "close_mosaic"},
        "V0/V1 are not isolated to Mosaic",
    )
    _require(
        _dict_differences(configs["V0_H2_MOSAIC"], configs["V3_H2_CP2_MOSAIC"])
        == {"copy_paste_enabled"},
        "V0/V3 are not isolated to CP2",
    )
    r4_differences = _dict_differences(configs["V0_H2_MOSAIC"], configs["V4_H2_R4_MOSAIC"])
    _require(r4_differences == _dict_differences(configs["V2_ES1_MOSAIC"], configs["V5_ES1_R4_MOSAIC"]),
             "H2 and ES1 do not receive the same R4 factor")
    _require("copy_paste_enabled" in r4_differences and "negative_cp_target_policy" in r4_differences,
             "R4 factor is incomplete")
    serialized_r4: dict[str, dict[str, Any]] = {}
    for case in R4_CASES:
        settings = configs[case]
        _require(not settings["hard_negative_tile"] and not settings["hard_negative_bank"],
                 f"{case}: R4 must not use a negative bank")
        _require(settings["negative_cp_p"] == 0.30, f"{case}: R4 probability changed")
        _require(settings["negative_cp_target_policy"] == "deficit", f"{case}: target policy changed")
        _require(settings["negative_cp_deficit_gamma"] == 0.5, f"{case}: deficit gamma changed")
        _require(settings["negative_cp_max_weight_ratio"] == 3.0, f"{case}: max weight ratio changed")
        _require(settings["negative_cp_donor_policy"] == "larger", f"{case}: donor policy changed")
        _require(settings["negative_cp_large_ratio_min"] == 1.5, f"{case}: donor minimum changed")
        _require(settings["negative_cp_large_ratio_max"] == 2.5, f"{case}: donor maximum changed")
        _require(settings["negative_cp_target_max_size"] == 20.0, f"{case}: target size changed")
        _require(settings["negative_cp_degradation"] == "weak_blur", f"{case}: degradation changed")
        _require(settings["negative_cp_blur_sigma"] == 0.5, f"{case}: blur sigma changed")
        _require(settings["copy_paste_copies"] == 1, f"{case}: copy count changed")
        _require(settings["copy_paste_placement"] == "collision_aware", f"{case}: placement changed")
        _require(settings["copy_paste_max_overlap"] == 0.0, f"{case}: overlap changed")
        from project_ultralytics.copy_paste import copy_paste_config

        serialized_r4[case] = copy_paste_config(SimpleNamespace(**settings))
        _require(not serialized_r4[case]["negative_bank_required"], f"{case}: negative-bank dependency detected")
    r4_source = (ROOT / "project_ultralytics/negative_canvas_copy_paste.py").read_text(encoding="utf-8")
    for fragment in (
        "cv2.GaussianBlur(crop, (3, 3), sigmaX=self.blur_sigma, sigmaY=self.blur_sigma)",
        "interpolation=cv2.INTER_AREA",
        "copies=1",
    ):
        _require(fragment in r4_source, f"NegativeCanvasCopyPaste implementation contract changed: {fragment}")
    oacp_env = environment_for("V6_H2_OACP_R2_MOSAIC")
    _require(oacp_env["YOLO_LEGACY_DOUBLE_OACP"] == "0", "OACP legacy double path enabled")
    _require(oacp_env["OACP_PLACEMENT"] == "pre_transform", "OACP R2 placement changed")
    with configured_environment("V6_H2_OACP_R2_MOSAIC"):
        from project_ultralytics.context_augment import augmentation_config as oacp_config

        resolved_oacp = oacp_config()
    expected_oacp = {
        "oacp_probability": 0.40,
        "oacp_strength": [0.10, 0.25],
        "oacp_resolution_scale": [0.80, 0.95],
        "protected_expand": 3.0,
        "oacp_variant": "current",
        "oacp_placement": "pre_transform",
        "legacy_double_oacp": False,
    }
    for key, expected in expected_oacp.items():
        _require(resolved_oacp[key] == expected, f"OACP R2 {key} changed")

    return {
        "status": "PASS",
        "yaml_parity": parity,
        "ftsc": expected_ftsc,
        "detect_inputs": [19, 22],
        "es1": expected_edge,
        "case_augmentation": configs,
        "case_environment": {case: environment_for(case) for case in CASES},
        "r4_negative_bank_required": False,
        "r4_serialized_copy_paste": serialized_r4,
        "oacp_resolved": resolved_oacp,
    }


class _PipelineProbeDataset:
    data = {"flip_idx": []}
    use_keypoints = False
    cache = False
    labels: list[dict[str, Any]] = []
    im_files: list[str] = []
    buffer: list[int] = []


def transform_pipeline_preflight(case: str) -> dict[str, Any]:
    """Construct the real augmentation graph and verify its effective order."""
    local_ultralytics()
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import Compose, Mosaic, RandomPerspective, v8_transforms
    from project_ultralytics.context_augment import OACP
    from project_ultralytics.copy_paste import SmallObjectCopyPaste
    from project_ultralytics.negative_canvas_copy_paste import NegativeCanvasCopyPaste

    settings = augmentation_for(case)
    with configured_environment(case):
        transforms = v8_transforms(_PipelineProbeDataset(), 640, get_cfg(overrides=settings))
    top = transforms.transforms
    oacp_count = sum(isinstance(item, OACP) for item in top)
    if case in OACP_CASES:
        _require(oacp_count == 1 and isinstance(top[0], OACP) and isinstance(top[1], Compose),
                 f"{case}: OACP is not a single pre_transform pass")
        spatial, after_spatial = top[1], 2
    else:
        _require(oacp_count == 0 and isinstance(top[0], Compose), f"{case}: unexpected OACP/Compose layout")
        spatial, after_spatial = top[0], 1
    _require(isinstance(spatial.transforms[0], Mosaic), f"{case}: standard Mosaic implementation changed")
    _require(isinstance(spatial.transforms[1], RandomPerspective), f"{case}: RandomPerspective order changed")
    _require(float(spatial.transforms[0].p) == float(settings["mosaic"]), f"{case}: Mosaic probability drifted")
    _require(type(spatial.transforms[0]).__name__ == "Mosaic", f"{case}: nonstandard Mosaic selected")
    _require(all(type(item).__name__ != "CopyPaste" for item in spatial.transforms),
             f"{case}: stock segmentation CopyPaste leaked into spatial transforms")

    if case == "V3_H2_CP2_MOSAIC":
        _require(type(top[after_spatial]) is SmallObjectCopyPaste, "V3: CP2 insertion point changed")
        after_spatial += 1
    elif case in R4_CASES:
        _require(type(top[after_spatial]) is NegativeCanvasCopyPaste, f"{case}: R4 insertion point changed")
        after_spatial += 1
    else:
        _require(not any(isinstance(item, (SmallObjectCopyPaste, NegativeCanvasCopyPaste)) for item in top),
                 f"{case}: Copy-Paste leakage")
    tail = [type(item).__name__ for item in top[after_spatial : after_spatial + 6]]
    _require(tail == ["MixUp", "CutMix", "Albumentations", "RandomHSV", "RandomFlip", "RandomFlip"],
             f"{case}: photometric/flip tail changed: {tail}")
    return {
        "reported_order": list(transform_order_for(case)),
        "top_level_types": [type(item).__name__ for item in top],
        "spatial_types": [type(item).__name__ for item in spatial.transforms],
        "oacp_instances": oacp_count,
    }


def model_preflight(model_config: Path) -> dict[str, Any]:
    local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(model_config, verbose=False)
    head = model.model[-1]
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    _require(strides == [4.0, 8.0], f"{model_config.name}: expected strides [4, 8], got {strides}")
    _require(list(head.f) == [19, 22], f"{model_config.name}: runtime Detect inputs changed")
    calibrator = head.ftsc_calibrator
    _require(calibrator is not None and calibrator.policy == "f5", f"{model_config.name}: FTSC f5 inactive")
    _require(list(calibrator.evidence_names) == ["position_gaussian", "dfl_distribution"],
             f"{model_config.name}: FTSC evidence changed")
    report: dict[str, Any] = {
        "model_config": str(model_config),
        "model_config_sha256": sha256(model_config),
        "nc": int(head.nc),
        "strides": strides,
        "detect_inputs": list(head.f),
        "ftsc_policy": calibrator.policy,
        "ftsc_evidence": list(calibrator.evidence_names),
    }
    if model_config == ES1_MODEL_CONFIG:
        edge = model.model[20]
        signature = {
            "layer_19": type(model.model[19]).__name__,
            "layer_20": type(edge).__name__,
            "hidden": int(edge.hidden),
            "residual_scale": float(edge.residual_scale),
            "residual_schedule": edge.residual_schedule,
            "orientation_gate_mode": edge.orientation_gate_mode,
            "ramp_start_epoch": int(edge.ramp_start_epoch),
            "ramp_end_epoch": int(edge.ramp_end_epoch),
            "detect_inputs": list(head.f),
        }
        expected = {
            "layer_19": "Identity",
            "layer_20": "P2EdgeCueFusion",
            "hidden": 32,
            "residual_scale": 0.25,
            "residual_schedule": "constant",
            "orientation_gate_mode": "learned",
            "ramp_start_epoch": 5,
            "ramp_end_epoch": 15,
            "detect_inputs": [19, 22],
        }
        _require(signature == expected, f"Historical ES1 runtime graph changed: {signature}")
        report["es1_runtime_graph"] = signature
    return report


def dataset_preflight(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.is_file():
        return {"status": "PENDING_DATA_PREPARATION", "data_yaml": str(data_yaml.resolve())}
    validation = visdrone_protocol.validate_converted_dataset(data_yaml.parent)
    return {
        "status": "READY",
        "data_yaml": str(data_yaml.resolve()),
        "official_splits": dict(visdrone_protocol.OFFICIAL_SPLITS),
        "expected_images": dict(visdrone_protocol.EXPECTED_SPLIT_IMAGES),
        "validation": validation,
    }


def train_kwargs(case: str, data_yaml: Path, seed: int, device: str) -> dict[str, object]:
    if seed not in SEEDS:
        raise ValueError(f"Unsupported seed {seed}; expected one of {SEEDS}")
    return {
        "data": str(data_yaml),
        **COMMON_TRAINING,
        "device": device,
        "seed": seed,
        "plots": False,
        "val": True,
        **augmentation_for(case),
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    source = source_preflight()
    selected_families = {"ES1" if case in ES1_CASES else "H2" for case in args.cases}
    models = {}
    if "H2" in selected_families:
        models["H2"] = model_preflight(H2_MODEL_CONFIG)
    if "ES1" in selected_families:
        models["ES1"] = model_preflight(ES1_MODEL_CONFIG)
    transforms = {case: transform_pipeline_preflight(case) for case in args.cases}
    return {
        "status": "PREFLIGHT_PASS — NO TRAINING RUN",
        "experiment": EXPERIMENT,
        "cases": list(args.cases),
        "seeds": list(args.seeds),
        "git_commit": git_sha(),
        "model_config_by_case": {case: str(model_config_for(case)) for case in args.cases},
        "common_training": COMMON_TRAINING,
        "pretrained": str(args.pretrained),
        "source": source,
        "runtime_models": models,
        "runtime_transforms": transforms,
        "dataset": dataset_preflight(args.data_yaml),
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "expected_artifacts": list(EXPECTED_ARTIFACTS),
        "remote_services": False,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def patch_musgd_noncontiguous() -> None:
    """Preserve the partner VisDrone baseline's MuSGD compatibility patch."""
    local_ultralytics()
    from ultralytics.optim import muon

    if getattr(muon, "_project_noncontiguous_patch", False):
        return
    original = muon.muon_update

    def compatible(grad, momentum, beta=0.95, nesterov=True):
        contiguous_grad = grad.contiguous()
        contiguous_momentum = momentum.contiguous()
        update = original(contiguous_grad, contiguous_momentum, beta=beta, nesterov=nesterov)
        momentum.copy_(contiguous_momentum)
        return update

    muon.muon_update = compatible
    muon._project_noncontiguous_patch = True


def _manifest(
    args: argparse.Namespace,
    case: str,
    seed: int,
    validation: dict[str, Any],
    status: str,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = model_config_for(case)
    payload = _load_yaml(config)
    return {
        "status": status,
        "experiment": EXPERIMENT,
        "dataset": "VisDrone2019-DET",
        "case": case,
        "seed": seed,
        "git_commit": git_sha(),
        "model_config": str(config),
        "model_config_sha256": sha256(config),
        "pretrained": str(args.pretrained),
        "data_yaml": str(args.data_yaml),
        "visdrone_conversion_validation": validation,
        "training_kwargs": train_kwargs(case, args.data_yaml, seed, args.device),
        "augmentation_config": augmentation_for(case),
        "oacp_environment": environment_for(case),
        "transform_order": list(transform_order_for(case)),
        "ftsc_config": payload["ftsc"],
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "expected_artifacts": list(EXPECTED_ARTIFACTS),
        "metrics": metrics or {},
    }


def _write_manifest(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _training_complete(run_dir: Path) -> bool:
    required = (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv", run_dir / "args.yaml")
    return all(path.is_file() for path in required)


def train_one(args: argparse.Namespace, case: str, seed: int, validation: dict[str, Any]) -> Path:
    run_dir = args.project / case / f"seed_{seed}"
    if _training_complete(run_dir):
        return run_dir
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite incomplete run directory: {run_dir}")
    _write_manifest(run_dir, _manifest(args, case, seed, validation, "RUN_AUTHORIZED — METRICS_PENDING"))
    local_ultralytics()
    patch_musgd_noncontiguous()
    from ultralytics import YOLO

    seed_everything(seed)
    with configured_environment(case):
        model = YOLO(model_config_for(case), task="detect")
        model.load(str(args.pretrained), smart_transfer=True)
        model.train(
            **train_kwargs(case, args.data_yaml, seed, args.device),
            project=str(args.project / case),
            name=f"seed_{seed}",
            exist_ok=True,
        )
    if not _training_complete(run_dir):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    return run_dir


def evaluate_one(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    local_ultralytics()
    from train_visdrone_scripts.evaluate_visdrone import evaluate_checkpoint

    return evaluate_checkpoint(
        run_dir,
        args.data_yaml,
        imgsz=640,
        batch=8,
        device=args.device,
        workers=8,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "datasets/visdrone2019_det/visdrone.yaml")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/visdrone_ftsc_ablation_suite")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.data_yaml = args.data_yaml.resolve()
    args.project = args.project.resolve()
    args.pretrained = args.pretrained.resolve()
    report = preflight(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.confirm_run:
        return
    _require(tuple(args.seeds) == SEEDS, f"Confirmed suite runs require seeds {SEEDS}")
    _require(report["dataset"]["status"] == "READY", "Canonical prepared VisDrone dataset is required")
    _require(args.pretrained.is_file(), f"Local pretrained checkpoint required: {args.pretrained}")
    validation = report["dataset"]["validation"]
    for case in args.cases:
        for seed in args.seeds:
            run_dir = train_one(args, case, seed, validation)
            metrics_path = run_dir / "evaluation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else evaluate_one(args, run_dir)
            _write_manifest(run_dir, _manifest(args, case, seed, validation, "COMPLETE", metrics))
            missing = [path for path in EXPECTED_ARTIFACTS if not (run_dir / path).is_file()]
            _require(not missing, f"Incomplete result provenance for {run_dir}: {missing}")


if __name__ == "__main__":
    main()
