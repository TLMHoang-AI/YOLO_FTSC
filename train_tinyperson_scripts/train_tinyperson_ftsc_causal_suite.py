#!/usr/bin/env python3
"""Run the frozen TinyPerson FTSC causal-ablation suite.

The four default variants are new experiments. Completed H2/ES1 no-Mosaic
results are read-only historical controls used by aggregation; this runner
never schedules them for training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import random
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

CONFIG_ROOT = ROOT / "models_related/models_config/yolov8"
ULTRALYTICS_ROOT = ROOT / "models_related/ultralytics"
LEViR_CONFIG_ROOT = CONFIG_ROOT / "levir"
TINYPERSON_CONFIG_ROOT = CONFIG_ROOT / "tinyperson"
H2_CONFIG = LEViR_CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
ES1_CONFIG = LEViR_CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml"
H2_NOFTSC_CONFIG = TINYPERSON_CONFIG_ROOT / "yolov8n_p2_tinyperson_h2_noftsc_nomosaic.yaml"
CLEAN_EDGE_CONFIG = (
    TINYPERSON_CONFIG_ROOT / "yolov8n_p2_tinyperson_ftsc_h2_clean_edge025_nomosaic.yaml"
)

PROTOCOL_VERSION = "tinyperson_ftsc_causal_suite_v1"
FITNESS_METRIC = "map50_95"
EXPECTED_STRIDES = (4.0, 8.0)
DEFAULT_SEEDS = (42, 43, 44)
HISTORICAL_CONFIG_SHA256 = {
    "H2_NM": "93c90262592a15962553a7f4ffe7f9cf9545679d8dbc702c614fae9d97842676",
    "ES1_NM": "29d7bc1c09d970426e61bf7f15d5d8d687aad441319374ed95963f9892730e79",
}
HISTORICAL_RESULTS_DEFAULT = ROOT.parent / "results/tinyperson_h2_es1_results.json"
TINY_WORKFLOW_MODULE = "train_tinyperson_scripts.train_all_tinyperson"
TINY_WORKFLOW_PATH = SCRIPT_DIR / "train_all_tinyperson.py"
REQUIRED_TINY_METRICS = (
    "test_merged/AP50",
    "test_merged/AP75",
    "test_merged/mAP50-75",
    "test_merged/AP-Tiny1",
    "test_merged/AP-Tiny2",
    "test_merged/AP-Tiny3",
    "test_merged/AP-Small",
    "test_merged/AP-Medium",
)


@dataclass(frozen=True)
class VariantSpec:
    config: Path
    ftsc_enabled: bool
    edge_enabled: bool
    detect_from: tuple[int, int]
    mosaic: float
    close_mosaic: int
    graph_contract: str


VARIANTS: dict[str, VariantSpec] = {
    "H2_NOFTSC_NM": VariantSpec(
        config=H2_NOFTSC_CONFIG,
        ftsc_enabled=False,
        edge_enabled=False,
        detect_from=(19, 22),
        mosaic=0.0,
        close_mosaic=0,
        graph_contract="h2_p2_p3_ftsc_disabled_v1",
    ),
    "CLEAN_EDGE_NM": VariantSpec(
        config=CLEAN_EDGE_CONFIG,
        ftsc_enabled=True,
        edge_enabled=True,
        detect_from=(20, 23),
        mosaic=0.0,
        close_mosaic=0,
        graph_contract="clean_edge_postfusion_p2_postc2f_p3_v1",
    ),
    "H2_MOSAIC": VariantSpec(
        config=H2_CONFIG,
        ftsc_enabled=True,
        edge_enabled=False,
        detect_from=(19, 22),
        mosaic=1.0,
        close_mosaic=10,
        graph_contract="h2_p2_p3_ftsc_v1",
    ),
    "ES1_MOSAIC": VariantSpec(
        config=ES1_CONFIG,
        ftsc_enabled=True,
        edge_enabled=True,
        detect_from=(19, 22),
        mosaic=1.0,
        close_mosaic=10,
        graph_contract="exact_current_levir",
    ),
}
DEFAULT_VARIANTS = tuple(VARIANTS)

HISTORICAL_CONTROLS = {
    "H2_NM": {
        "result_variant": "H2",
        "config": H2_CONFIG,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "graph_contract": "h2_p2_p3_ftsc_v1",
    },
    "ES1_NM": {
        "result_variant": "ES1",
        "config": ES1_CONFIG,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "graph_contract": "exact_current_levir",
    },
}

COMPARISONS = (
    ("H2_NM-H2_NOFTSC_NM", "H2_NM", "H2_NOFTSC_NM"),
    ("H2_MOSAIC-H2_NM", "H2_MOSAIC", "H2_NM"),
    ("ES1_MOSAIC-ES1_NM", "ES1_MOSAIC", "ES1_NM"),
    ("ES1_NM-H2_NM", "ES1_NM", "H2_NM"),
    ("ES1_MOSAIC-H2_MOSAIC", "ES1_MOSAIC", "H2_MOSAIC"),
    ("CLEAN_EDGE_NM-H2_NM", "CLEAN_EDGE_NM", "H2_NM"),
    ("CLEAN_EDGE_NM-ES1_NM", "CLEAN_EDGE_NM", "ES1_NM"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Invalid model YAML: {path}")
    return payload


def _tiny_workflow():
    """Load the existing TinyPerson workflow only when runtime work needs it."""
    return importlib.import_module(TINY_WORKFLOW_MODULE)


def local_ultralytics() -> Path:
    """Hard-lock imports to the vendored, patched Ultralytics source tree."""
    package = ULTRALYTICS_ROOT / "ultralytics/__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"Missing vendored Ultralytics fork: {package}")
    root = str(ULTRALYTICS_ROOT.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != root]
    sys.path.insert(0, root)
    loaded = sys.modules.get("ultralytics")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if not loaded_file.is_relative_to(ULTRALYTICS_ROOT.resolve()):
            for name in list(sys.modules):
                if name == "ultralytics" or name.startswith("ultralytics."):
                    sys.modules.pop(name, None)
    return ULTRALYTICS_ROOT.resolve()


def _edge_layer_indices(payload: dict[str, Any]) -> list[int]:
    offset = len(payload["backbone"])
    return [
        offset + index
        for index, layer in enumerate(payload["head"])
        if layer[2] == "P2EdgeCueFusion"
    ]


def _source_layer(payload: dict[str, Any], index: int) -> list[Any]:
    layers = [*payload["backbone"], *payload["head"]]
    _require(0 <= index < len(layers), f"Layer index {index} is out of range")
    return layers[index]


def _frozen_tinyperson_protocol() -> dict[str, Any]:
    source = TINY_WORKFLOW_PATH.read_text(encoding="utf-8")
    prediction_start = source.index("def predict_merged_test(")
    prediction_end = source.index("\ndef mean_ap50_75", prediction_start)
    evaluator_start = source.index("def evaluate_tiny_benchmark(")
    evaluator_end = source.index("\ndef evaluate_merged_test", evaluator_start)
    prediction_source = source[prediction_start:prediction_end]
    evaluator_source = source[evaluator_start:evaluator_end]
    _require("conf=0.001" in prediction_source, "TinyPerson prediction conf changed")
    _require("iou=0.5" in prediction_source, "TinyPerson prediction NMS changed")
    _require("max_det=300" in prediction_source, "TinyPerson max_det changed")
    _require(
        "nms_detections(items, iou_threshold=0.5)" in prediction_source,
        "TinyPerson cross-window NMS changed",
    )
    _require('Params.EVAL_STRANDARD = "tiny"' in evaluator_source, "Tiny evaluation standard changed")
    _require(
        "np.linspace(0.50, 0.75, 6)" in evaluator_source,
        "TinyBenchmark IoU sweep changed",
    )
    expected_paths = {
        "train_corner_json": "erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json",
        "test_corner_json": "annotations/corner/task/tiny_set_test_sw640_sh512_all.json",
        "test_merged_json": "annotations/task/tiny_set_test_all.json",
    }
    for name, value in expected_paths.items():
        _require(f'Path("{value}")' in source, f"TinyPerson {name} changed")
    return {
        **expected_paths,
        "corner_window": "sw640_sh512",
        "prediction_conf": 0.001,
        "prediction_nms_iou": 0.5,
        "max_det": 300,
        "cross_window_nms_iou": 0.5,
        "tiny_evaluation_standard": "tiny",
        "tiny_iou_thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    }


def source_preflight(args: argparse.Namespace) -> dict[str, Any]:
    historical = {"H2_NM": _load_yaml(H2_CONFIG), "ES1_NM": _load_yaml(ES1_CONFIG)}
    for name, path in (("H2_NM", H2_CONFIG), ("ES1_NM", ES1_CONFIG)):
        actual = _sha256(path)
        _require(
            actual == HISTORICAL_CONFIG_SHA256[name],
            f"Historical {name} YAML changed: {actual}",
        )

    h2 = historical["H2_NM"]
    es1 = historical["ES1_NM"]
    _require(h2["head"][-1][0] == [19, 22], "Historical H2 Detect taps changed")
    _require(es1["head"][-1][0] == [19, 22], "Historical ES1 Detect taps changed")
    _require(_edge_layer_indices(h2) == [], "Historical H2 unexpectedly contains Edge")
    _require(_edge_layer_indices(es1) == [20], "Historical ES1 Edge index changed")

    expected_ftsc = copy.deepcopy(h2["ftsc"])
    _require(expected_ftsc.get("enabled") is True, "Historical H2 FTSC is not enabled")
    _require(es1["ftsc"] == expected_ftsc, "Historical H2/ES1 FTSC contracts differ")

    reports: dict[str, dict[str, Any]] = {}
    for variant in args.variants:
        spec = VARIANTS[variant]
        payload = _load_yaml(spec.config)
        detect_from = payload["head"][-1][0]
        edge_indices = _edge_layer_indices(payload)
        _require(detect_from == list(spec.detect_from), f"{variant}: wrong Detect taps {detect_from}")
        _require(
            bool(payload["ftsc"].get("enabled", True)) is spec.ftsc_enabled,
            f"{variant}: wrong FTSC enabled state",
        )
        expected_variant_ftsc = copy.deepcopy(expected_ftsc)
        expected_variant_ftsc["enabled"] = spec.ftsc_enabled
        _require(payload["ftsc"] == expected_variant_ftsc, f"{variant}: FTSC settings drifted")
        _require(bool(edge_indices) is spec.edge_enabled, f"{variant}: wrong Edge presence")

        if variant == "H2_NOFTSC_NM":
            normalized = copy.deepcopy(payload)
            normalized["ftsc"] = copy.deepcopy(h2["ftsc"])
            _require(normalized == h2, "H2_NOFTSC_NM differs from H2 beyond FTSC enabled=false")
        elif variant == "CLEAN_EDGE_NM":
            normalized = copy.deepcopy(payload)
            normalized["head"][-1][0] = [19, 22]
            _require(normalized == es1, "CLEAN_EDGE_NM differs from ES1 beyond Detect taps")
            _require(edge_indices == [20], "CLEAN_EDGE_NM Edge must be layer 20")
            _require(_source_layer(payload, 20)[2] == "P2EdgeCueFusion", "Clean P2 tap is not Edge")
            _require(_source_layer(payload, 23)[2] == "C2f", "Clean P3 tap is not post-C2f")
        elif variant == "H2_MOSAIC":
            _require(payload == h2, "H2_MOSAIC must reuse exact historical H2 graph")
        elif variant == "ES1_MOSAIC":
            _require(payload == es1, "ES1_MOSAIC must reuse exact historical ES1 graph")
            _require(detect_from != [20, 23], "ES1_MOSAIC accidentally uses clean Edge graph")

        edge_report: dict[str, Any] | None = None
        if spec.edge_enabled:
            edge = payload["edge_stability"]
            _require(edge["residual_scale"] == 0.25, f"{variant}: Edge alpha changed")
            _require(edge["residual_schedule"] == "constant", f"{variant}: Edge schedule changed")
            _require(edge["orientation_gate_mode"] == "learned", f"{variant}: Edge gate changed")
            _require(edge["hidden"] == 32, f"{variant}: Edge hidden width changed")
            edge_report = {
                "layer": edge_indices[0],
                "alpha": 0.25,
                "schedule": "constant",
                "orientation_gate": "learned",
                "hidden": 32,
            }

        reports[variant] = {
            "config": str(spec.config),
            "config_sha256": _sha256(spec.config),
            "topology_sha256": _json_hash({"backbone": payload["backbone"], "head": payload["head"]}),
            "detect_from": detect_from,
            "expected_strides": list(EXPECTED_STRIDES),
            "ftsc": copy.deepcopy(payload["ftsc"]),
            "edge": edge_report,
            "mosaic": spec.mosaic,
            "close_mosaic": spec.close_mosaic,
            "graph_contract": spec.graph_contract,
        }

    return {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "variants": reports,
        "historical_controls": {
            name: {
                "config": str(control["config"]),
                "config_sha256": HISTORICAL_CONFIG_SHA256[name],
                "training_default": False,
                "graph_contract": control["graph_contract"],
            }
            for name, control in HISTORICAL_CONTROLS.items()
        },
        "tinyperson": _frozen_tinyperson_protocol(),
    }


def _state_hash(model, *, max_layer: int | None = None) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if max_layer is not None:
            pieces = name.split(".")
            if len(pieces) < 2 or pieces[0] != "model" or not pieces[1].isdigit():
                continue
            if int(pieces[1]) > max_layer:
                continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_preflight(args: argparse.Namespace) -> dict[str, Any]:
    expected_ultralytics = local_ultralytics()
    import torch
    import ultralytics
    from ultralytics.nn.modules import Detect, P2EdgeCueFusion
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops

    loaded_ultralytics = Path(ultralytics.__file__).resolve()
    _require(
        loaded_ultralytics.is_relative_to(expected_ultralytics),
        f"Refusing non-vendored Ultralytics: {loaded_ultralytics}",
    )

    reports: dict[str, dict[str, Any]] = {}
    backbone_hashes: dict[str, str] = {}
    for variant in args.variants:
        spec = VARIANTS[variant]
        torch.manual_seed(0)
        model = DetectionModel(spec.config, verbose=False)
        head = model.model[-1]
        _require(isinstance(head, Detect), f"{variant}: final runtime module is not Detect")
        detect_from = [int(index) for index in head.f]
        strides = [float(value) for value in head.stride.detach().cpu().tolist()]
        _require(detect_from == list(spec.detect_from), f"{variant}: runtime Detect taps changed")
        _require(strides == list(EXPECTED_STRIDES), f"{variant}: runtime strides changed to {strides}")
        _require(
            (head.ftsc_calibrator is not None) is spec.ftsc_enabled,
            f"{variant}: runtime FTSC state is wrong",
        )

        edge_modules = [module for module in model.modules() if isinstance(module, P2EdgeCueFusion)]
        _require(len(edge_modules) == int(spec.edge_enabled), f"{variant}: runtime Edge count is wrong")
        edge_report: dict[str, Any] | None = None
        if edge_modules:
            edge = edge_modules[0]
            _require(edge.residual_scale == 0.25, f"{variant}: runtime Edge alpha changed")
            _require(edge.residual_schedule == "constant", f"{variant}: runtime Edge schedule changed")
            _require(edge.orientation_gate_mode == "learned", f"{variant}: runtime Edge gate changed")
            _require(edge.hidden == 32, f"{variant}: runtime Edge hidden changed")
            zero_init = bool(
                torch.count_nonzero(edge.projection.weight).item() == 0
                and torch.count_nonzero(edge.projection.bias).item() == 0
            )
            _require(zero_init, f"{variant}: Edge residual projection is not zero-initialized")
            edge_report = {
                "alpha": edge.residual_scale,
                "schedule": edge.residual_schedule,
                "orientation_gate": edge.orientation_gate_mode,
                "hidden": edge.hidden,
                "gate_trainable": edge.gate is not None
                and any(parameter.requires_grad for parameter in edge.gate.parameters()),
                "zero_initialized_residual": zero_init,
                "effective_alpha_epoch0": edge.effective_residual_scale(0),
            }

        source_types = [type(model.model[index]).__name__ for index in detect_from]
        if variant == "CLEAN_EDGE_NM":
            _require(source_types == ["P2EdgeCueFusion", "C2f"], f"Clean Edge sources wrong: {source_types}")
        if variant == "ES1_MOSAIC":
            _require(source_types == ["Identity", "Concat"], f"Historical ES1 sources wrong: {source_types}")

        reports[variant] = {
            "config": str(spec.config),
            "detect_from": detect_from,
            "detect_source_types": source_types,
            "detect_strides": strides,
            "ftsc_enabled": head.ftsc_calibrator is not None,
            "ftsc_policy": getattr(head.ftsc_calibrator, "policy", None),
            "ftsc_evidence": list(getattr(head.ftsc_calibrator, "evidence_names", ())),
            "edge_enabled": bool(edge_modules),
            "edge": edge_report,
            "mosaic": spec.mosaic,
            "close_mosaic": spec.close_mosaic,
            "graph_contract": spec.graph_contract,
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_params_before_trainer_freeze": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "gflops_at_imgsz": float(get_flops(model, imgsz=args.imgsz)),
            "imgsz": args.imgsz,
            "ultralytics": str(loaded_ultralytics),
        }
        backbone_hashes[variant] = _state_hash(model, max_layer=18)

    _require(
        len(set(backbone_hashes.values())) == 1,
        f"Backbone initialization differs across causal variants: {backbone_hashes}",
    )
    return {
        "variants": reports,
        "initial_backbone_hashes": backbone_hashes,
        "same_initial_backbone": True,
    }


def train_kwargs(
    args: argparse.Namespace, variant: str, seed: int, data_yaml: Path
) -> dict[str, Any]:
    spec = VARIANTS[variant]
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
        "amp": True,
        "plots": False,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "cos_lr": False,
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "nbs": 64,
        "fitness_metric": FITNESS_METRIC,
        "mosaic": spec.mosaic,
        "close_mosaic": spec.close_mosaic,
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
        "project": str(args.project / variant),
        "name": f"seed_{seed}_corner_sw640_sh512",
        "exist_ok": True,
    }


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "pretrained": str(args.pretrained),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "optimizer": "AdamW",
        "lr0": 0.002,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "scheduler": "linear",
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.0,
        "fitness_metric": FITNESS_METRIC,
        "deterministic": True,
        "amp": True,
        **_frozen_tinyperson_protocol(),
    }


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pretrained_model(variant: str, pretrained: str, seed: int):
    """Build one variant and verify the exact smart-transfer result."""
    local_ultralytics()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import intersect_dicts

    _seed_everything(seed)
    model = YOLO(VARIANTS[variant].config)
    initial_backbone_hash = _state_hash(model.model, max_layer=18)
    model.load(pretrained, smart_transfer=True)
    checkpoint_model = model.ckpt["model"] if isinstance(model.ckpt, dict) else model.ckpt
    source_state = checkpoint_model.float().state_dict()
    target_state = model.model.state_dict()
    transferred = intersect_dicts(source_state, target_state, smart_transfer=True)
    verified = sum(torch.equal(target_state[name].detach().cpu(), value.detach().cpu()) for name, value in transferred.items())
    _require(verified == len(transferred), f"{variant}: pretrained transfer verification failed")
    report = {
        "pretrained": str(pretrained),
        "smart_transfer": True,
        "transferred_items": len(transferred),
        "target_state_items": len(target_state),
        "verified_items": verified,
        "initial_backbone_hash": initial_backbone_hash,
        "transferred_backbone_hash": _state_hash(model.model, max_layer=18),
    }
    return model, report


def _run_dir(args: argparse.Namespace, variant: str, seed: int) -> Path:
    return args.project / variant / f"seed_{seed}_corner_sw640_sh512"


def lock_seed_dataset(args: argparse.Namespace, seed: int, data_yaml: Path) -> dict[str, Any]:
    """Lock every variant of a seed to one prepared TinyPerson dataset YAML."""
    contract = {
        "seed": seed,
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": _sha256(data_yaml),
    }
    path = args.project / "dataset_contracts" / f"seed_{seed}.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _require(existing == contract, f"Seed {seed} dataset contract changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _run_contract(
    args: argparse.Namespace,
    variant: str,
    seed: int,
    data_yaml: Path,
    source_report: dict[str, Any],
    model_report: dict[str, Any],
) -> dict[str, Any]:
    spec = VARIANTS[variant]
    return {
        "variant": variant,
        "seed": seed,
        "config": str(spec.config.resolve()),
        "config_sha256": _sha256(spec.config),
        "git_commit": _git_commit(),
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": _sha256(data_yaml),
        "data_root": str(args.data_root),
        "dataset_root": str(args.dataset_root),
        "protocol": _protocol(args),
        "training": {
            key: value
            for key, value in train_kwargs(args, variant, seed, data_yaml).items()
            if key not in {"data", "project", "name", "exist_ok"}
        },
        # Contracts are persisted as JSON. Keep tuple-valued dataclass fields
        # JSON-native here so a resumed manifest compares equal after a
        # write/read round trip.
        "variant_contract": asdict(spec)
        | {
            "config": str(spec.config),
            "detect_from": list(spec.detect_from),
        },
        "source_preflight": source_report,
        "model_preflight": model_report,
    }


def _require_tinybenchmark(metrics: dict[str, Any], variant: str, seed: int) -> None:
    missing = [name for name in REQUIRED_TINY_METRICS if name not in metrics]
    _require(
        float(metrics.get("test_merged/available", 0.0)) == 1.0 and not missing,
        f"{variant}/seed{seed}: TinyBenchmark incomplete; missing={missing}",
    )


def run_one(
    args: argparse.Namespace,
    variant: str,
    seed: int,
    data_yaml: Path,
    test_out_dir: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    tiny = _tiny_workflow()
    local_ultralytics()
    from ultralytics import YOLO

    run_dir = _run_dir(args, variant, seed)
    manifest_path = run_dir / "experiment_manifest.json"
    metrics_path = run_dir / "evaluation_metrics.json"
    complete_path = run_dir / "causal_suite_complete.json"
    contract = _run_contract(
        args,
        variant,
        seed,
        data_yaml,
        preflight["source"]["variants"][variant],
        preflight["model"]["variants"][variant],
    )

    existing_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(existing_manifest.get("contract") == contract, f"Incompatible existing run: {run_dir}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Refusing untracked non-empty run directory: {run_dir}")

    if complete_path.is_file() and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        _require_tinybenchmark(metrics, variant, seed)
        return {"variant": variant, "seed": seed, "run_dir": str(run_dir), **metrics}

    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "weights/last.pt"
    trained_artifacts = all(
        (run_dir / relative).is_file()
        for relative in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "config.yaml")
    )
    if trained_artifacts:
        _require(existing_manifest is not None, f"Missing manifest for trained run: {run_dir}")
        print(f"Reusing trained artifacts before evaluation: {run_dir}", flush=True)
    elif last.is_file():
        _require(existing_manifest is not None, f"Missing manifest for resumable run: {run_dir}")
        YOLO(last).train(resume=True)
    else:
        model, transfer = build_pretrained_model(variant, args.pretrained, seed)
        manifest_path.write_text(
            json.dumps({"contract": contract, "pretrained_transfer": transfer}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(VARIANTS[variant].config, run_dir / "config.yaml")
        model.train(**train_kwargs(args, variant, seed, data_yaml))

    for relative in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "config.yaml"):
        _require((run_dir / relative).is_file(), f"{variant}/seed{seed}: missing {relative}")

    metrics = tiny.evaluate(run_dir, data_yaml, test_out_dir, args.data_root, args)
    _require_tinybenchmark(metrics, variant, seed)
    complete_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "variant": variant,
                "seed": seed,
                "protocol_version": PROTOCOL_VERSION,
                "checkpoint": "weights/best.pt",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"variant": variant, "seed": seed, "run_dir": str(run_dir), **metrics}


def _historical_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports = payload.get("preflight", {})
    output = []
    for row in payload.get("runs", []):
        mapped = "H2_NM" if row.get("variant") == "H2" else "ES1_NM" if row.get("variant") == "ES1" else None
        if mapped is None:
            continue
        report = reports.get(row["variant"], {})
        output.append(
            {
                **row,
                "variant": mapped,
                "origin": "historical_read_only",
                "graph_contract": HISTORICAL_CONTROLS[mapped]["graph_contract"],
                "mosaic": 0.0,
                "close_mosaic": 0,
                "params": report.get("params"),
                "gflops_at_imgsz": report.get("gflops_at_imgsz"),
            }
        )
    return output


def _new_rows(args: argparse.Namespace, preflight: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    model_reports = (preflight or {}).get("model", {}).get("variants", {})
    for variant, spec in VARIANTS.items():
        for seed in args.seeds:
            run_dir = _run_dir(args, variant, seed)
            metrics_path = run_dir / "evaluation_metrics.json"
            if not metrics_path.is_file():
                continue
            report = model_reports.get(variant, {})
            manifest_path = run_dir / "experiment_manifest.json"
            if not report and manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                report = manifest.get("contract", {}).get("model_preflight", {})
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "origin": "causal_suite",
                    "graph_contract": spec.graph_contract,
                    "mosaic": spec.mosaic,
                    "close_mosaic": spec.close_mosaic,
                    "params": report.get("params"),
                    "gflops_at_imgsz": report.get("gflops_at_imgsz"),
                    **json.loads(metrics_path.read_text(encoding="utf-8")),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted(
        {key for row in rows for key in row},
        key=lambda key: (key not in {"variant", "seed", "origin", "run_dir"}, key),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["variant", "seed"])
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in (*HISTORICAL_CONTROLS, *VARIANTS):
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        record: dict[str, Any] = {"variant": variant, "runs": len(group)}
        common = set.intersection(*(set(row) for row in group))
        for key in sorted(common - {"variant", "seed", "origin", "run_dir", "checkpoint", "graph_contract"}):
            values = [row[key] for row in group]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                continue
            numeric = [float(value) for value in values]
            record[f"{key}/mean"] = statistics.fmean(numeric)
            record[f"{key}/std"] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        output.append(record)
    return output


def _comparison_summary(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {row["variant"]: row for row in aggregate}
    comparisons, missing = [], []
    for name, left, right in COMPARISONS:
        if left not in by_variant or right not in by_variant:
            missing.append({"comparison": name, "missing": [v for v in (left, right) if v not in by_variant]})
            continue
        left_row, right_row = by_variant[left], by_variant[right]
        common = {
            key
            for key in set(left_row) & set(right_row)
            if key.endswith("/mean")
            and key.startswith(("val/", "test/", "test_merged/"))
            and isinstance(left_row[key], (int, float))
            and isinstance(right_row[key], (int, float))
        }
        comparisons.append(
            {
                "comparison": name,
                "left": left,
                "right": right,
                "delta_left_minus_right": {
                    key.removesuffix("/mean"): float(left_row[key]) - float(right_row[key])
                    for key in sorted(common)
                },
            }
        )

    interaction_variants = ("ES1_MOSAIC", "H2_MOSAIC", "ES1_NM", "H2_NM")
    interactions = []
    if all(name in by_variant for name in interaction_variants):
        es1_m, h2_m, es1_nm, h2_nm = (by_variant[name] for name in interaction_variants)
        common = {
            key
            for key in set(es1_m) & set(h2_m) & set(es1_nm) & set(h2_nm)
            if key.endswith("/mean")
            and key.startswith(("val/", "test/", "test_merged/"))
            and all(isinstance(row[key], (int, float)) for row in (es1_m, h2_m, es1_nm, h2_nm))
        }
        interactions.append(
            {
                "interaction": "(ES1_MOSAIC-H2_MOSAIC)-(ES1_NM-H2_NM)",
                "delta": {
                    key.removesuffix("/mean"): (
                        float(es1_m[key])
                        - float(h2_m[key])
                        - float(es1_nm[key])
                        + float(h2_nm[key])
                    )
                    for key in sorted(common)
                },
            }
        )
    else:
        missing.append(
            {
                "interaction": "(ES1_MOSAIC-H2_MOSAIC)-(ES1_NM-H2_NM)",
                "missing": [name for name in interaction_variants if name not in by_variant],
            }
        )
    return {"comparisons": comparisons, "interactions": interactions, "missing": missing}


def write_summaries(args: argparse.Namespace, preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    args.project.mkdir(parents=True, exist_ok=True)
    rows = [*_historical_rows(args.historical_results), *_new_rows(args, preflight)]
    aggregate = _aggregate(rows)
    comparisons = _comparison_summary(aggregate)
    _write_csv(args.project / "summary_runs.csv", rows)
    (args.project / "summary_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.project / "summary_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"runs": rows, "aggregate": aggregate, **comparisons}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "TinyPerson/tiny_set")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/tinyperson_ftsc_causal_suite")
    parser.add_argument("--historical-results", type=Path, default=HISTORICAL_RESULTS_DEFAULT)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "epochs": (args.epochs, 100),
        "patience": (args.patience, 20),
        "imgsz": (args.imgsz, 640),
        "batch_size": (args.batch_size, 8),
    }
    changed = {name: actual for name, (actual, expected) in frozen.items() if actual != expected}
    _require(not changed, f"Frozen TinyPerson protocol changed: {changed}")
    _require(Path(args.pretrained).name == "yolov8n.pt", "Pretrained must be yolov8n.pt")
    _require(len(args.variants) == len(set(args.variants)), "Duplicate variants are not allowed")
    _require(len(args.seeds) == len(set(args.seeds)), "Duplicate seeds are not allowed")
    _require(set(args.seeds).issubset(DEFAULT_SEEDS), f"Seeds must be selected from {list(DEFAULT_SEEDS)}")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.historical_results = args.historical_results.resolve()
    validate_args(args)

    preflight: dict[str, Any] = {"source": source_preflight(args)}
    if args.aggregate_only:
        summary = write_summaries(args, preflight)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    preflight["model"] = model_preflight(args)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return

    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tiny = _tiny_workflow()
    test_out_dir = tiny.prepare_test_set(args.data_root, args.dataset_root)
    for seed in args.seeds:
        seed_dir = tiny.prepare_seed_dataset(args.data_root, args.dataset_root, test_out_dir, seed)
        data_yaml = seed_dir / "tinyperson.yaml"
        lock_seed_dataset(args, seed, data_yaml)
        for variant in args.variants:
            run_one(args, variant, seed, data_yaml, test_out_dir, preflight)
            write_summaries(args, preflight)
    write_summaries(args, preflight)
    print("TinyPerson FTSC causal suite complete.")


if __name__ == "__main__":
    main()
