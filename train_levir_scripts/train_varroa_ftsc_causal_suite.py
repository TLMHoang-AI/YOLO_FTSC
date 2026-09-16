#!/usr/bin/env python3
"""Run the four-case Varroa FTSC causal suite.

This is a thin specialization of ``train_varroa_dbss_hit.py``. Dataset
preparation, validation, deterministic seeding, completion checks, AMP retry,
and val/test evaluation remain owned by that established Varroa runner.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_varroa_dbss_hit as shared


CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/varroa"
BASE_REFERENCE = ROOT / "models_related/models_config/yolov8/tried/yolov8n_varroa_p2p3_only_detect_dfl25.yaml"
H2_REFERENCE = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
EXPECTED_STRIDES = (4.0, 8.0)
DEFAULT_SEEDS = (42, 43, 44)
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
}
UNWANTED_TOP_LEVEL = {
    "loc_assign",
    "dfl_residual",
    "api",
    "dgfe",
    "tal_topk",
}


@dataclass(frozen=True)
class VariantSpec:
    config: Path
    ftsc_enabled: bool
    cue_module: str | None
    detect_from: tuple[int, int]


VARIANTS: dict[str, VariantSpec] = {
    "V0_NOFTSC": VariantSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_v0_p2p3_noftsc.yaml", False, None, (18, 21)
    ),
    "V1_FTSC": VariantSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_v1_p2p3.yaml", True, None, (18, 21)
    ),
    "V2_EDGE025": VariantSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_v2_p2p3_edge025.yaml",
        True,
        "P2EdgeCueFusion",
        (19, 22),
    ),
    "V3_COLOR": VariantSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_v3_p2p3_color.yaml",
        True,
        "P2ColorCueFusion",
        (19, 22),
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"Invalid YAML mapping: {path}")
    return payload


def _layers(payload: dict[str, Any]) -> list[list[Any]]:
    return [*payload["backbone"], *payload["head"]]


def _module_indices(payload: dict[str, Any], module: str) -> list[int]:
    return [index for index, layer in enumerate(_layers(payload)) if layer[2] == module]


def _remove_cue(payload: dict[str, Any], cue_module: str) -> dict[str, Any]:
    """Normalize one cue insertion back to the V1 graph."""
    normalized = copy.deepcopy(payload)
    cue_positions = [i for i, layer in enumerate(normalized["head"]) if layer[2] == cue_module]
    _require(cue_positions == [9], f"{cue_module}: expected one cue at head position 9")
    normalized["head"].pop(9)
    normalized["head"][-1][0] = [18, 21]
    return normalized


def source_preflight() -> dict[str, Any]:
    """Validate the YAML-level scientific and graph contracts."""
    _require(BASE_REFERENCE.is_file(), f"Missing Varroa base reference: {BASE_REFERENCE}")
    _require(H2_REFERENCE.is_file(), f"Missing canonical FTSC reference: {H2_REFERENCE}")
    payloads = {name: _load_yaml(spec.config) for name, spec in VARIANTS.items()}
    v0, v1 = payloads["V0_NOFTSC"], payloads["V1_FTSC"]

    normalized_v0 = copy.deepcopy(v0)
    normalized_v0["ftsc"]["enabled"] = True
    _require(normalized_v0 == v1, "V0 and V1 differ beyond ftsc.enabled")

    expected_ftsc = copy.deepcopy(v1["ftsc"])
    _require(expected_ftsc == _load_yaml(H2_REFERENCE)["ftsc"], "V1 FTSC differs from canonical H2")
    for name, spec in VARIANTS.items():
        payload = payloads[name]
        layers = _layers(payload)
        modules = [layer[2] for layer in layers]
        _require(payload.get("scale") == "n", f"{name}: scale must be YOLOv8n")
        _require(payload["head"][-1][2] == "Detect", f"{name}: final layer must be Detect")
        _require(payload["head"][-1][0] == list(spec.detect_from), f"{name}: wrong Detect taps")
        _require(payload["ftsc"].get("enabled") is spec.ftsc_enabled, f"{name}: wrong FTSC state")
        expected = copy.deepcopy(expected_ftsc)
        expected["enabled"] = spec.ftsc_enabled
        _require(payload["ftsc"] == expected, f"{name}: FTSC settings drifted from V1")
        _require(not (UNWANTED_MODULES & set(modules)), f"{name}: forbidden module present")
        _require(not (UNWANTED_TOP_LEVEL & set(payload)), f"{name}: forbidden mechanism flag present")
        _require(len(_module_indices(payload, "P2EdgeCueFusion")) == int(spec.cue_module == "P2EdgeCueFusion"),
                 f"{name}: wrong Edge cue count")
        _require(len(_module_indices(payload, "P2ColorCueFusion")) == int(spec.cue_module == "P2ColorCueFusion"),
                 f"{name}: wrong Color cue count")

    for name, cue_module in (("V2_EDGE025", "P2EdgeCueFusion"), ("V3_COLOR", "P2ColorCueFusion")):
        payload = payloads[name]
        _require(_remove_cue(payload, cue_module) == v1, f"{name}: differs from V1 beyond cue insertion")
        layers = _layers(payload)
        _require(layers[19][2] == cue_module, f"{name}: Detect P2 source is not post-cue")
        _require(layers[20][2] == "Conv" and layers[20][0] == -1,
                 f"{name}: P3 path does not start from fused P2")
        _require(layers[22][2] == "RepC2f", f"{name}: Detect P3 source is not post-RepC2f")

    edge_layer = _layers(payloads["V2_EDGE025"])[19]
    _require(edge_layer[3] == [32, 0.25, "constant", 5, 15, "learned"], "V2: Edge parameters drifted")
    color_layer = _layers(payloads["V3_COLOR"])[19]
    _require(color_layer[3] == [32], "V3: Color hidden width drifted")

    variants = {
        name: {
            "config": str(spec.config),
            "ftsc_enabled": spec.ftsc_enabled,
            "cue_module": spec.cue_module,
            "detect_from": list(spec.detect_from),
            "p2_source_type": _layers(payloads[name])[spec.detect_from[0]][2],
            "p3_source_type": _layers(payloads[name])[spec.detect_from[1]][2],
        }
        for name, spec in VARIANTS.items()
    }
    return {
        "base_reference": str(BASE_REFERENCE),
        "ftsc_reference": str(H2_REFERENCE),
        "base_derivation": (
            "Retained the full Varroa RepC2f backbone/PAN and P2/P3-only Detect intent; "
            "removed PoolingEdgeRepC2f, API, and EnSimAM."
        ),
        "variants": variants,
    }


def model_preflight() -> dict[str, Any]:
    """Construct all models on CPU and verify resolved Detect semantics."""
    shared.local_ultralytics()
    from ultralytics.nn.modules import Detect, P2ColorCueFusion, P2EdgeCueFusion
    from ultralytics.nn.tasks import DetectionModel

    report: dict[str, Any] = {}
    for name, spec in VARIANTS.items():
        model = DetectionModel(spec.config, verbose=False)
        head = model.model[-1]
        _require(isinstance(head, Detect), f"{name}: runtime head is not Detect")
        strides = tuple(float(value) for value in head.stride.tolist())
        _require(strides == EXPECTED_STRIDES, f"{name}: strides resolved to {strides}")
        _require(list(head.f) == list(spec.detect_from), f"{name}: runtime Detect taps drifted")
        _require((head.ftsc_calibrator is not None) is spec.ftsc_enabled, f"{name}: runtime FTSC state wrong")
        edge = [module for module in model.modules() if isinstance(module, P2EdgeCueFusion)]
        color = [module for module in model.modules() if isinstance(module, P2ColorCueFusion)]
        _require(len(edge) == int(spec.cue_module == "P2EdgeCueFusion"), f"{name}: runtime Edge count wrong")
        _require(len(color) == int(spec.cue_module == "P2ColorCueFusion"), f"{name}: runtime Color count wrong")
        if edge:
            _require(edge[0].hidden == 32 and edge[0].residual_scale == 0.25, f"{name}: Edge width/alpha wrong")
            _require(edge[0].residual_schedule == "constant", f"{name}: Edge schedule is not constant")
            _require(edge[0].orientation_gate_mode == "learned", f"{name}: Edge gate is not learned")
        if color:
            _require(color[0].encoder[0].conv.out_channels == 32, f"{name}: Color hidden width wrong")
        report[name] = {
            "detect_from": list(head.f),
            "detect_source_types": [type(model.model[index]).__name__ for index in head.f],
            "detect_strides": list(strides),
            "ftsc_enabled": head.ftsc_calibrator is not None,
            "edge_count": len(edge),
            "color_count": len(color),
        }
    return report


def model_for(variant: str):
    shared.local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(VARIANTS[variant].config)
    model.load("yolov8n.pt", smart_transfer=True)
    return model


def _run_dir(args: argparse.Namespace, variant: str, seed: int) -> Path:
    return args.project / variant / f"seed_{seed}"


def train_one(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    """Reuse the established Varroa completion/resume and AMP fallback policy."""
    run_dir = _run_dir(args, variant, seed)
    if shared.complete(run_dir):
        print(f"Reusing completed run: {variant}/seed_{seed}", flush=True)
        return run_dir

    last = run_dir / "weights/last.pt"
    if last.is_file():
        shared.local_ultralytics()
        from ultralytics import YOLO

        shared.seed_everything(seed)
        YOLO(last).train(resume=True)
    else:
        kwargs = shared.train_kwargs(args, data_yaml, seed, amp)
        kwargs.update(project=str(args.project / variant), name=f"seed_{seed}", exist_ok=True)
        shared.seed_everything(seed)
        try:
            model_for(variant).train(**kwargs)
        except Exception as error:
            if not amp:
                raise
            archived = shared.archive_failed_amp(run_dir)
            print(f"AMP failed for {variant}/seed_{seed}: {error!r}; archived={archived}; retrying amp=False", flush=True)
            kwargs["amp"] = False
            shared.seed_everything(seed)
            model_for(variant).train(**kwargs)
    if not shared.complete(run_dir):
        raise FileNotFoundError(f"Training ended without required artifacts: {run_dir}")
    return run_dir


def write_summaries(args: argparse.Namespace) -> bool:
    """Write summaries only for a complete selected matrix; never label partial groups final."""
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        for seed in args.seeds:
            path = _run_dir(args, variant, seed) / "evaluation_metrics.json"
            if not path.is_file():
                return False
            rows.append({"variant": variant, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})

    args.project.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
    with (args.project / "summary_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for variant in args.variants:
        group = [row for row in rows if row["variant"] == variant]
        _require(len(group) == len(args.seeds), f"Partial aggregate for {variant}")
        record: dict[str, Any] = {"variant": variant, "runs": len(group)}
        common = set.intersection(*(set(row) for row in group)) - {"variant", "seed"}
        for key in sorted(common):
            values = [row[key] for row in group]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                numeric = [float(value) for value in values]
                record[f"{key}/mean"] = statistics.fmean(numeric)
                record[f"{key}/std"] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        aggregates.append(record)
    aggregate_fields = sorted({key for row in aggregates for key in row}, key=lambda key: (key not in {"variant", "runs"}, key))
    with (args.project / "summary_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/varroa_ftsc_causal_suite")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    frozen = {"epochs": (args.epochs, 100), "patience": (args.patience, 20), "imgsz": (args.imgsz, 640), "batch": (args.batch_size, 4)}
    changed = {name: actual for name, (actual, expected) in frozen.items() if actual != expected}
    _require(not changed, f"Frozen Varroa protocol changed: {changed}")
    _require(len(args.variants) == len(set(args.variants)), "Duplicate variants are not allowed")
    _require(len(args.seeds) == len(set(args.seeds)), "Duplicate seeds are not allowed")
    _require(set(args.seeds).issubset(DEFAULT_SEEDS), f"Seeds must come from {list(DEFAULT_SEEDS)}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    preflight: dict[str, Any] = {"source": source_preflight()}
    try:
        preflight["model"] = model_preflight()
    except ModuleNotFoundError as error:
        if not args.preflight_only:
            raise
        preflight["model"] = {
            "status": "unavailable",
            "reason": f"missing local Python dependency: {error.name}",
        }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return

    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    args.seed_scope = "all"
    args.amp_overrides = {}
    for seed in args.seeds:
        data_yaml = shared.dataset_for_seed(args, seed)
        args.data_yaml = data_yaml
        for variant in args.variants:
            run_dir = train_one(variant, seed, data_yaml, args.amp, args)
            shared.evaluate(run_dir, args)
    _require(write_summaries(args), "Refusing to aggregate an incomplete selected matrix")


if __name__ == "__main__":
    main()
