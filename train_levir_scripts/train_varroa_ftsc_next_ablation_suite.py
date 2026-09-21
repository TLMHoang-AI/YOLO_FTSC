#!/usr/bin/env python3
"""Prepare/run the nine-case Varroa FTSC next-ablation suite.

FTSC, TAL, loss, split, optimizer, scheduler, and evaluation behavior are
frozen. Cases change only detector topology or the registered Mosaic/CP2
training factors. The default action is preparation-only; training requires
the explicit ``--confirm-run`` gate.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_levir_scripts import train_levir_ftsc_augmentation_suite as levir_augmentation
from train_levir_scripts import train_varroa_dbss_hit as shared
from train_levir_scripts import train_varroa_ftsc_edge_placement_headscale as placement


CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/varroa"
RESULTS_ROOT = ROOT / "results"
PREFLIGHT_PATH = RESULTS_ROOT / "varroa_ftsc_next_ablation_suite_preflight.json"
SUMMARY_RUNS_PATH = RESULTS_ROOT / "varroa_ftsc_next_ablation_suite_summary_runs.csv"
SUMMARY_AGGREGATE_PATH = RESULTS_ROOT / "varroa_ftsc_next_ablation_suite_summary_aggregate.csv"
CANONICAL_FTSC_CONFIG = CONFIG_ROOT / "yolov8n_varroa_ftsc_v1_p2p3.yaml"
P2_EDGE_REFERENCE_CONFIG = CONFIG_ROOT / "yolov8n_varroa_ftsc_edge_p2_detect_only.yaml"
P234_REFERENCE_CONFIG = CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p234.yaml"
DEFAULT_CFG = ROOT / "models_related/ultralytics/ultralytics/cfg/default.yaml"
DEFAULT_SEEDS = (42, 43, 44)
EDGE_ARGS = [32, 0.25, "constant", 5, 15, "learned"]
CURRENT_VARROA_MOSAIC = 1.0
CURRENT_VARROA_CLOSE_MOSAIC = 10
CP2_KEYS = (
    "copy_paste",
    "copy_paste_enabled",
    "copy_paste_mode",
    "copy_paste_unit",
    "copy_paste_copies",
    "copy_paste_p",
    "copy_paste_scale",
    "copy_paste_padding",
    "copy_paste_blend",
    "copy_paste_placement",
    "copy_paste_max_overlap",
    "copy_paste_max_trials",
    "copy_paste_allow_empty_target",
    "copy_paste_allow_same_source",
)
UNWANTED_TOP_LEVEL = {
    "loc_assign",
    "dfl_residual",
    "api",
    "dgfe",
    "tal_topk",
    "box",
    "cls",
    "dfl",
}
UNWANTED_MODULES = {
    "AdversarialPerturbationInjection",
    "BoundaryFeatureBlock",
    "FeatureDGFE",
    "LocalDetailRepC2f",
    "PoolingEdgeRepC2f",
    "KVCompressedAttention",
    "EnSimAM",
    "AdaptiveZoom",
    "DBSS",
    "DualIrreducibilityHIT",
    "MaskedP2DetailReconstruction",
    "P2ColorCueFusion",
}


@dataclass(frozen=True)
class CaseSpec:
    config: Path
    stage: str
    architecture_base: str
    detect_from: tuple[int, ...]
    pyramid_levels: tuple[str, ...]
    detect_source_types: tuple[str, ...]
    edge_index: int | None = None
    clean_downstream_start: tuple[int, int] | None = None
    no_mosaic: bool = False
    cp2_enabled: bool = False


ARCHITECTURE_CASES = (
    "FTSC_P234_EDGE_P2_DETECT_ONLY",
    "FTSC_P34",
    "FTSC_P34_EDGE_P3_DETECT_ONLY",
)
AUGMENTATION_CORE_CASES = (
    "FTSC_P23_NOMOSAIC",
    "FTSC_P23_CP2",
    "FTSC_P23_NOMOSAIC_CP2",
)
AUGMENTATION_EDGE_CASES = (
    "EDGE_P2_DETECT_ONLY_NOMOSAIC",
    "EDGE_P2_DETECT_ONLY_CP2",
    "EDGE_P2_DETECT_ONLY_NOMOSAIC_CP2",
)
ALL_CASES = ARCHITECTURE_CASES + AUGMENTATION_CORE_CASES + AUGMENTATION_EDGE_CASES
STAGES = {
    "architecture": ARCHITECTURE_CASES,
    "augmentation-core": AUGMENTATION_CORE_CASES,
    "augmentation-edge": AUGMENTATION_EDGE_CASES,
    "all": ALL_CASES,
}

CASES: dict[str, CaseSpec] = {
    "FTSC_P234_EDGE_P2_DETECT_ONLY": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_p234_edge_p2_detect_only.yaml",
        "architecture", "REF_HEAD_P234", (19, 22, 25), ("P2", "P3", "P4"),
        ("P2EdgeCueFusion", "RepC2f", "RepC2f"), 19, (20, 18),
    ),
    "FTSC_P34": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p34.yaml",
        "architecture", "REF_FTSC_P23", (21, 24), ("P3", "P4"),
        ("RepC2f", "RepC2f"),
    ),
    "FTSC_P34_EDGE_P3_DETECT_ONLY": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p34_edge_p3.yaml",
        "architecture", "REF_EDGE_P3_DETECT_ONLY", (22, 25), ("P3", "P4"),
        ("P3EdgeCueFusion", "RepC2f"), 22, (23, 21),
    ),
    "FTSC_P23_NOMOSAIC": CaseSpec(
        CANONICAL_FTSC_CONFIG, "augmentation-core", "REF_FTSC_P23",
        (18, 21), ("P2", "P3"), ("RepC2f", "RepC2f"), no_mosaic=True,
    ),
    "FTSC_P23_CP2": CaseSpec(
        CANONICAL_FTSC_CONFIG, "augmentation-core", "REF_FTSC_P23",
        (18, 21), ("P2", "P3"), ("RepC2f", "RepC2f"), cp2_enabled=True,
    ),
    "FTSC_P23_NOMOSAIC_CP2": CaseSpec(
        CANONICAL_FTSC_CONFIG, "augmentation-core", "REF_FTSC_P23",
        (18, 21), ("P2", "P3"), ("RepC2f", "RepC2f"), no_mosaic=True, cp2_enabled=True,
    ),
    "EDGE_P2_DETECT_ONLY_NOMOSAIC": CaseSpec(
        P2_EDGE_REFERENCE_CONFIG, "augmentation-edge", "REF_EDGE_P2_DETECT_ONLY",
        (19, 22), ("P2", "P3"), ("P2EdgeCueFusion", "RepC2f"),
        19, (20, 18), no_mosaic=True,
    ),
    "EDGE_P2_DETECT_ONLY_CP2": CaseSpec(
        P2_EDGE_REFERENCE_CONFIG, "augmentation-edge", "REF_EDGE_P2_DETECT_ONLY",
        (19, 22), ("P2", "P3"), ("P2EdgeCueFusion", "RepC2f"),
        19, (20, 18), cp2_enabled=True,
    ),
    "EDGE_P2_DETECT_ONLY_NOMOSAIC_CP2": CaseSpec(
        P2_EDGE_REFERENCE_CONFIG, "augmentation-edge", "REF_EDGE_P2_DETECT_ONLY",
        (19, 22), ("P2", "P3"), ("P2EdgeCueFusion", "RepC2f"),
        19, (20, 18), no_mosaic=True, cp2_enabled=True,
    ),
}

HISTORICAL_REFERENCES = {
    "REF_FTSC_P23": {
        "config": str(CANONICAL_FTSC_CONFIG),
        "meaning": "completed canonical FTSC Detect P2/P3",
        "training_enabled": False,
    },
    "REF_EDGE_P2_DETECT_ONLY": {
        "config": str(P2_EDGE_REFERENCE_CONFIG),
        "meaning": "completed P2 prediction-local Edge with clean downstream P3",
        "training_enabled": False,
    },
    "REF_HEAD_P234": {
        "config": str(P234_REFERENCE_CONFIG),
        "meaning": "completed clean FTSC Detect P2/P3/P4",
        "training_enabled": False,
    },
    "REF_EDGE_P3_DETECT_ONLY": {
        "config": str(CONFIG_ROOT / "yolov8n_varroa_ftsc_edge_p3_detect_only.yaml"),
        "meaning": "completed P3 prediction-local Edge with Detect P2/P3",
        "training_enabled": False,
    },
    "REF_HEAD_P234_EDGE_P3": {
        "config": str(CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p234_edge_p3.yaml"),
        "meaning": "completed P3 prediction-local Edge with Detect P2/P3/P4",
        "training_enabled": False,
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_strides(spec: CaseSpec) -> tuple[float, ...]:
    return tuple(float(2 ** int(level[1:])) for level in spec.pyramid_levels)


def validated_cp2_settings() -> dict[str, object]:
    """Return the exact detection CP2 settings from validated LEVIR A3_CP2."""
    source = levir_augmentation.augmentation_for("A3_CP2")
    settings = {key: source[key] for key in CP2_KEYS}
    _require(settings["copy_paste"] == 0.0, "stock segmentation Copy-Paste must stay disabled")
    _require(settings["copy_paste_enabled"] is True, "LEVIR A3_CP2 detection transform is inactive")
    _require(
        (settings["copy_paste_unit"], settings["copy_paste_copies"])
        == ("single", 2),
        "LEVIR A3_CP2 is not two copies of one raw detection object",
    )
    return settings


def augmentation_overrides(case: str) -> dict[str, object]:
    """Return only explicit augmentation controls for one Varroa case."""
    spec = CASES[case]
    settings: dict[str, object] = {
        "mosaic_policy": "standard",
        "hard_negative_tile": False,
        "hard_negative_bank": "",
        "copy_paste": 0.0,
        "copy_paste_enabled": False,
    }
    if spec.no_mosaic:
        settings.update(mosaic=0.0, close_mosaic=0)
    if spec.cp2_enabled:
        settings.update(validated_cp2_settings())
    return settings


def augmentation_contract(case: str) -> dict[str, object]:
    spec = CASES[case]
    return {
        "mosaic": 0.0 if spec.no_mosaic else CURRENT_VARROA_MOSAIC,
        "close_mosaic": 0 if spec.no_mosaic else CURRENT_VARROA_CLOSE_MOSAIC,
        "mosaic_source": "explicit_off" if spec.no_mosaic else "inherited_current_varroa_default",
        "cp2_enabled": spec.cp2_enabled,
        "cp2_parameters": validated_cp2_settings() if spec.cp2_enabled else None,
        "cp2_parameter_fingerprint": _json_fingerprint(validated_cp2_settings()),
        "stock_segmentation_copy_paste": 0.0,
        "oacp_enabled": False,
        "m5_enabled": False,
        "transform_order": [
            "Mosaic_disabled" if spec.no_mosaic else "StandardMosaic",
            "RandomPerspective",
            *(("DetectionCP2",) if spec.cp2_enabled else ()),
        ],
        "train_overrides": augmentation_overrides(case),
    }


def _remove_edge_for_comparison(payload: dict[str, Any], edge_index: int, detect_from: list[int]) -> dict[str, Any]:
    normalized = copy.deepcopy(payload)
    head_index = edge_index - len(normalized["backbone"])
    normalized["head"].pop(head_index)
    normalized["head"][head_index][0] = -1
    normalized["head"][-1][0] = detect_from
    return normalized


def source_preflight() -> dict[str, Any]:
    """Validate all static graph, FTSC, augmentation, and registry contracts."""
    import yaml

    _require(tuple(CASES) == ALL_CASES, "case registry order/content differs from the exact nine-case plan")
    _require(set(CASES).isdisjoint(HISTORICAL_REFERENCES), "historical references entered trainable registry")
    canonical = placement._load_yaml(CANONICAL_FTSC_CONFIG)
    canonical_ftsc = canonical["ftsc"]
    ftsc_fingerprint = _json_fingerprint(canonical_ftsc)
    defaults = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    _require(defaults["mosaic"] == CURRENT_VARROA_MOSAIC, "current Mosaic default changed")
    _require(defaults["close_mosaic"] == CURRENT_VARROA_CLOSE_MOSAIC, "current close_mosaic default changed")
    _require(defaults["copy_paste"] == 0.0, "stock segmentation Copy-Paste default is not off")
    _require(defaults["copy_paste_enabled"] is False, "detection CP2 default is not off")

    p234 = placement._load_yaml(P234_REFERENCE_CONFIG)
    cases: dict[str, Any] = {}
    for name, spec in CASES.items():
        payload = placement._load_yaml(spec.config)
        layers = placement._layers(payload)
        modules = [layer[2] for layer in layers]
        detect = layers[-1]
        edge_indices = [i for i, module in enumerate(modules) if module in placement.EDGE_MODULES]
        source_types = tuple(layers[index][2] for index in spec.detect_from)

        _require(payload["ftsc"] == canonical_ftsc, f"{name}: FTSC config changed")
        _require(_json_fingerprint(payload["ftsc"]) == ftsc_fingerprint, f"{name}: FTSC fingerprint changed")
        _require(not (UNWANTED_TOP_LEVEL & set(payload)), f"{name}: TAL/loss/mechanism override present")
        _require(not (UNWANTED_MODULES & set(modules)), f"{name}: forbidden representation module present")
        _require(detect[2] == "Detect" and tuple(detect[0]) == spec.detect_from, f"{name}: Detect taps changed")
        _require(source_types == spec.detect_source_types, f"{name}: Detect source types changed")
        _require(edge_indices == ([] if spec.edge_index is None else [spec.edge_index]), f"{name}: Edge count/index changed")

        clean_downstream_verified = spec.edge_index is None
        if spec.edge_index is not None:
            edge_layer = layers[spec.edge_index]
            _require(edge_layer[3] == EDGE_ARGS, f"{name}: Edge constructor parameters changed")
            _require(spec.clean_downstream_start is not None, f"{name}: clean downstream contract missing")
            downstream, clean_source = spec.clean_downstream_start
            _require(placement._resolved_inputs(layers, downstream) == (clean_source,),
                     f"{name}: downstream does not start from clean source {clean_source}")
            _require(spec.edge_index not in placement._ancestors(layers, downstream),
                     f"{name}: Edge leaks into downstream at layer {downstream}")
            for level, source in zip(spec.pyramid_levels, spec.detect_from):
                if source != spec.edge_index:
                    _require(spec.edge_index not in placement._ancestors(layers, source),
                             f"{name}: Edge leaks into clean {level} Detect source")
            clean_downstream_verified = True

        if name == "FTSC_P234_EDGE_P2_DETECT_ONLY":
            _require(placement._resolved_inputs(layers, 23) == (22,), "A1: P4 does not start from clean P3")
            normalized = _remove_edge_for_comparison(payload, 19, [18, 21, 24])
            _require(normalized == p234, "A1 differs from clean P234 beyond prediction-local Edge")
        if name == "FTSC_P34_EDGE_P3_DETECT_ONLY":
            _require(placement._resolved_inputs(layers, 23) == (21,), "A3: P4 does not start from clean P3")

        augmentation = augmentation_contract(name)
        _require(augmentation["oacp_enabled"] is False and augmentation["m5_enabled"] is False,
                 f"{name}: OACP/M5 must be off")
        cases[name] = {
            "status": "PREPARED_NOT_RUN",
            "stage": spec.stage,
            "architecture_base": spec.architecture_base,
            "config": str(spec.config),
            "config_sha256": _sha256(spec.config),
            "ftsc_fingerprint": ftsc_fingerprint,
            "detect_from": list(spec.detect_from),
            "detect_source_types": list(source_types),
            "detect_strides": list(_expected_strides(spec)),
            "edge_enabled": spec.edge_index is not None,
            "edge_index": spec.edge_index,
            "edge_detect_only": spec.edge_index is not None,
            "clean_downstream_verified": clean_downstream_verified,
            "augmentation": augmentation,
        }

    return {
        "status": "PREPARED_NOT_RUN",
        "suite": "varroa_ftsc_next_ablation_suite",
        "cases": cases,
        "case_names": list(ALL_CASES),
        "stages": {name: list(values) for name, values in STAGES.items()},
        "seeds": list(DEFAULT_SEEDS),
        "canonical_ftsc_config": str(CANONICAL_FTSC_CONFIG),
        "ftsc_config": canonical_ftsc,
        "ftsc_fingerprint": ftsc_fingerprint,
        "cp2_source": "train_levir_ftsc_augmentation_suite.A3_CP2",
        "cp2_parameters": validated_cp2_settings(),
        "cp2_parameter_fingerprint": _json_fingerprint(validated_cp2_settings()),
        "historical_references": copy.deepcopy(HISTORICAL_REFERENCES),
        "references_scheduled": False,
        "configs_reused": sorted({str(spec.config) for spec in CASES.values() if spec.config.name != "yolov8n_varroa_ftsc_p234_edge_p2_detect_only.yaml"}),
        "configs_created": [str(CASES["FTSC_P234_EDGE_P2_DETECT_ONLY"].config)],
        "training_protocol": {
            "epochs": 100,
            "patience": 20,
            "imgsz": 640,
            "batch": 4,
            "seeds": list(DEFAULT_SEEDS),
            "deterministic": True,
            "pretrained": "yolov8n.pt with smart_transfer=True",
            "dataset": {
                "gt_source": "gt_one",
                "only_positives": True,
                "class_policy": "map-3-to-1",
                "expected_counts": dict(shared.EXPECTED_SPLITS),
            },
            "optimizer_scheduler": "unchanged current Varroa/Ultralytics defaults",
            "tal_loss_ftsc": "frozen",
        },
        "summary_paths": {
            "runs": str(SUMMARY_RUNS_PATH),
            "aggregate": str(SUMMARY_AGGREGATE_PATH),
        },
    }


def _shape_summary(value: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        return type(value).__name__
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {str(key): _shape_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape_summary(item) for item in value]
    return type(value).__name__


def model_preflight() -> dict[str, Any]:
    """Instantiate and forward every unique graph on CPU."""
    shared.local_ultralytics()
    import torch
    from ultralytics.nn.modules import Detect, EdgeCueFusion
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

    by_config: dict[Path, dict[str, Any]] = {}
    reference_model = DetectionModel(P2_EDGE_REFERENCE_CONFIG, verbose=False).cpu()
    reference_edge = next(module for module in reference_model.modules() if isinstance(module, EdgeCueFusion))
    reference_kernel = reference_edge.orientation_kernels.detach().cpu()
    reference_kernel_sha = hashlib.sha256(reference_kernel.numpy().tobytes()).hexdigest()

    for config in dict.fromkeys(spec.config for spec in CASES.values()):
        model = DetectionModel(config, verbose=False).cpu().eval()
        head = model.model[-1]
        _require(isinstance(head, Detect), f"{config.name}: runtime head is not Detect")
        edges = [module for module in model.modules() if isinstance(module, EdgeCueFusion)]
        for edge in edges:
            _require(torch.equal(edge.orientation_kernels.detach().cpu(), reference_kernel),
                     f"{config.name}: directional kernels differ from completed P2 Edge reference")
            _require(edge.orientation_count == 4, f"{config.name}: orientation count changed")
            _require(edge.projection.weight.detach().count_nonzero().item() == 0,
                     f"{config.name}: projection is not zero initialized")
            _require(
                (edge.hidden, edge.residual_scale, edge.residual_schedule,
                 edge.ramp_start_epoch, edge.ramp_end_epoch, edge.orientation_gate_mode)
                == (32, 0.25, "constant", 5, 15, "learned"),
                f"{config.name}: Edge runtime parameters changed",
            )
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64))
        by_config[config] = {
            "detect_from": list(head.f),
            "detect_source_types": [type(model.model[index]).__name__ for index in head.f],
            "detect_strides": [float(value) for value in head.stride.tolist()],
            "edge_count": len(edges),
            "edge_kernel_sha256": reference_kernel_sha if edges else None,
            "parameters": int(get_num_params(model)),
            "gradients": int(get_num_gradients(model)),
            "gflops_640": float(value) if (value := get_flops(model, imgsz=640)) else None,
            "forward_input_shape": [1, 3, 64, 64],
            "forward_output_shapes": _shape_summary(output),
        }

    report: dict[str, Any] = {}
    for name, spec in CASES.items():
        item = copy.deepcopy(by_config[spec.config])
        _require(tuple(item["detect_from"]) == spec.detect_from, f"{name}: runtime Detect taps changed")
        _require(tuple(item["detect_source_types"]) == spec.detect_source_types,
                 f"{name}: runtime Detect source types changed")
        _require(tuple(item["detect_strides"]) == _expected_strides(spec), f"{name}: runtime strides changed")
        _require(item["edge_count"] == int(spec.edge_index is not None), f"{name}: runtime Edge count changed")
        report[name] = item
    return report


def model_for(case: str):
    shared.local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(CASES[case].config)
    model.load("yolov8n.pt", smart_transfer=True)
    return model


def train_kwargs(args: argparse.Namespace, case: str, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = shared.train_kwargs(args, data_yaml, seed, amp)
    kwargs.update(augmentation_overrides(case))
    return kwargs


def _run_dir(args: argparse.Namespace, case: str, seed: int) -> Path:
    return args.project / case / f"seed_{seed}"


def train_one(case: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    """Use established Varroa resume/AMP behavior with only case augmentation overrides."""
    run_dir = _run_dir(args, case, seed)
    if shared.complete(run_dir):
        print(f"Reusing completed run: {case}/seed_{seed}", flush=True)
        return run_dir
    last = run_dir / "weights/last.pt"
    with levir_augmentation.configured_environment("A0_FTSC"):
        if last.is_file():
            shared.local_ultralytics()
            from ultralytics import YOLO

            shared.seed_everything(seed)
            YOLO(last).train(resume=True)
        else:
            kwargs = train_kwargs(args, case, data_yaml, seed, amp)
            kwargs.update(project=str(args.project / case), name=f"seed_{seed}", exist_ok=True)
            shared.seed_everything(seed)
            try:
                model_for(case).train(**kwargs)
            except Exception as error:
                if not amp:
                    raise
                archived = shared.archive_failed_amp(run_dir)
                print(
                    f"AMP failed for {case}/seed_{seed}: {error!r}; "
                    f"archived={archived}; retrying amp=False",
                    flush=True,
                )
                kwargs["amp"] = False
                shared.seed_everything(seed)
                model_for(case).train(**kwargs)
    if not shared.complete(run_dir):
        raise FileNotFoundError(f"Training ended without required artifacts: {run_dir}")
    return run_dir


def _metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    names = {
        "AP50": "metrics/mAP50(B)",
        "mAP50_95": "metrics/mAP50-95(B)",
        "AP75": "metrics/mAP75(B)",
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
    }
    for split in ("val", "test"):
        for alias, source in names.items():
            key = f"{split}/{source}"
            if key in metrics:
                aliases[f"{split}_{alias}"] = metrics[key]
    return aliases


def write_summaries(args: argparse.Namespace) -> bool:
    """Write future metric summaries only for a complete selected matrix."""
    source = source_preflight()
    models = model_preflight()
    rows: list[dict[str, Any]] = []
    for case in args.cases:
        spec = CASES[case]
        for seed in args.seeds:
            run_dir = _run_dir(args, case, seed)
            path = run_dir / "evaluation_metrics.json"
            if not path.is_file():
                return False
            metrics = json.loads(path.read_text(encoding="utf-8"))
            augmentation = source["cases"][case]["augmentation"]
            model = models[case]
            rows.append({
                "seed": seed,
                "variant": case,
                "architecture_base": spec.architecture_base,
                "detect_strides": json.dumps(model["detect_strides"]),
                "edge_enabled": spec.edge_index is not None,
                "edge_detect_only": spec.edge_index is not None,
                "mosaic": augmentation["mosaic"],
                "close_mosaic": augmentation["close_mosaic"],
                "cp2_enabled": spec.cp2_enabled,
                "params": model["parameters"],
                "GFLOPs": model["gflops_640"],
                "checkpoint_used": str(run_dir / "weights/best.pt"),
                "run_dir": str(run_dir),
                **_metric_aliases(metrics),
                **metrics,
            })

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
    with SUMMARY_RUNS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregates: list[dict[str, Any]] = []
    metadata = {
        "architecture_base", "detect_strides", "edge_enabled", "edge_detect_only",
        "mosaic", "close_mosaic", "cp2_enabled", "params", "GFLOPs",
    }
    for case in args.cases:
        group = [row for row in rows if row["variant"] == case]
        _require(len(group) == len(args.seeds), f"partial aggregate for {case}")
        record: dict[str, Any] = {"variant": case, "runs": len(group)}
        for key in metadata:
            record[key] = group[0][key]
        common = set.intersection(*(set(row) for row in group)) - {"variant", "seed", "run_dir", "checkpoint_used"} - metadata
        for key in sorted(common):
            values = [row[key] for row in group]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                numeric = [float(value) for value in values]
                record[f"{key}/mean"] = statistics.fmean(numeric)
                record[f"{key}/std"] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        aggregates.append(record)
    aggregate_fields = sorted({key for row in aggregates for key in row}, key=lambda key: (key not in {"variant", "runs"}, key))
    with SUMMARY_AGGREGATE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=ALL_CASES)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/varroa_ftsc_next_ablation_suite")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--preflight-json", type=Path, default=PREFLIGHT_PATH)
    args = parser.parse_args(argv)
    if args.cases is None:
        args.cases = list(STAGES[args.stage])
    return args


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "epochs": (args.epochs, 100),
        "patience": (args.patience, 20),
        "imgsz": (args.imgsz, 640),
        "batch": (args.batch_size, 4),
    }
    changed = {name: actual for name, (actual, expected) in frozen.items() if actual != expected}
    _require(not changed, f"frozen Varroa protocol changed: {changed}")
    _require(len(args.cases) == len(set(args.cases)), "duplicate cases are not allowed")
    _require(set(args.cases).issubset(CASES), "unknown case selected")
    _require(set(args.cases).isdisjoint(HISTORICAL_REFERENCES), "historical references cannot be trained")
    _require(len(args.seeds) == len(set(args.seeds)), "duplicate seeds are not allowed")
    _require(set(args.seeds).issubset(DEFAULT_SEEDS), f"seeds must come from {list(DEFAULT_SEEDS)}")
    if args.cases != list(STAGES[args.stage]) and args.stage != "all":
        raise RuntimeError("use either --stage or --cases, not both")


def build_preflight() -> dict[str, Any]:
    source = source_preflight()
    return {
        "status": source["status"],
        "suite": source["suite"],
        "case_names": source["case_names"],
        "source": source,
        "model": model_preflight(),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    preflight = build_preflight()
    output = json.dumps(preflight, indent=2, sort_keys=True) + "\n"
    args.preflight_json.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_json.write_text(output, encoding="utf-8")
    print(output, end="")
    if args.aggregate_only:
        _require(write_summaries(args), "refusing to aggregate an incomplete selected matrix")
        return
    if args.preflight_only or not args.confirm_run:
        print("PREPARED_NOT_RUN — pass --confirm-run to authorize future training")
        return

    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.seed_scope = "all"
    args.amp_overrides = {}
    for seed in args.seeds:
        data_yaml = shared.dataset_for_seed(args, seed)
        args.data_yaml = data_yaml
        for case in args.cases:
            run_dir = train_one(case, seed, data_yaml, args.amp, args)
            shared.evaluate(run_dir, args)
    _require(write_summaries(args), "refusing to aggregate an incomplete selected matrix")


if __name__ == "__main__":
    main()
