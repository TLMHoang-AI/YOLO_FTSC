#!/usr/bin/env python3
"""Prepare/run the Varroa FTSC Edge-placement and head-scale causal suite.

The default matrix is exactly the four new stage-1 cases. Historical references
are comparison metadata only, and optional stage-2 cases require an explicit
``--include-stage2`` opt-in. Use ``--preflight-only`` for local CPU checks.
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_varroa_dbss_hit as shared


CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/varroa"
REFERENCE_CONFIG = CONFIG_ROOT / "yolov8n_varroa_ftsc_v1_p2p3.yaml"
HISTORICAL_REFERENCES = {
    "REF_FTSC_P23": {
        "config": str(REFERENCE_CONFIG),
        "meaning": "completed FTSC P2/P3 without Edge",
        "training_enabled": False,
    },
    "REF_FTSC_P23_EDGE_EARLYP2": {
        "config": str(CONFIG_ROOT / "yolov8n_varroa_ftsc_v2_p2p3_edge025.yaml"),
        "meaning": "completed early-P2 Edge whose fused P2 propagates into P3",
        "training_enabled": False,
    },
}
DEFAULT_SEEDS = (42, 43, 44)
EDGE_ARGS = [32, 0.25, "constant", 5, 15, "learned"]
EDGE_MODULES = frozenset({"EdgeCueFusion", "P2EdgeCueFusion", "P3EdgeCueFusion"})
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
UNWANTED_TOP_LEVEL = {"loc_assign", "dfl_residual", "api", "dgfe", "tal_topk"}


@dataclass(frozen=True)
class CaseSpec:
    config: Path
    stage: int
    detect_from: tuple[int, ...]
    pyramid_levels: tuple[str, ...]
    expected_source_types: tuple[str, ...]
    edge_index: int | None = None
    clean_downstream_start: tuple[int, int] | None = None


CASES: dict[str, CaseSpec] = {
    "EDGE_P2_DETECT_ONLY": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_edge_p2_detect_only.yaml",
        1, (19, 22), ("P2", "P3"), ("P2EdgeCueFusion", "RepC2f"), 19, (20, 18),
    ),
    "EDGE_P3_DETECT_ONLY": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_edge_p3_detect_only.yaml",
        1, (18, 22), ("P2", "P3"), ("RepC2f", "P3EdgeCueFusion"), 22, (23, 21),
    ),
    "HEAD_P234": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p234.yaml",
        1, (18, 21, 24), ("P2", "P3", "P4"), ("RepC2f", "RepC2f", "RepC2f"),
    ),
    "HEAD_P234_EDGE_P3": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p234_edge_p3.yaml",
        1, (18, 22, 25), ("P2", "P3", "P4"), ("RepC2f", "P3EdgeCueFusion", "RepC2f"),
        22, (23, 21),
    ),
    "HEAD_P34": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p34.yaml",
        2, (21, 24), ("P3", "P4"), ("RepC2f", "RepC2f"),
    ),
    "HEAD_P34_EDGE_P3": CaseSpec(
        CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p34_edge_p3.yaml",
        2, (22, 25), ("P3", "P4"), ("P3EdgeCueFusion", "RepC2f"), 22, (23, 21),
    ),
}
STAGE1_CASES = tuple(name for name, spec in CASES.items() if spec.stage == 1)
STAGE2_CASES = tuple(name for name, spec in CASES.items() if spec.stage == 2)


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


def _resolved_inputs(layers: list[list[Any]], index: int) -> tuple[int, ...]:
    source = layers[index][0]
    values = source if isinstance(source, list) else [source]
    return tuple(index - 1 if value == -1 else int(value) for value in values)


def _ancestors(layers: list[list[Any]], index: int) -> set[int]:
    found: set[int] = set()
    pending = [index]
    while pending:
        current = pending.pop()
        for source in _resolved_inputs(layers, current):
            if source >= 0 and source not in found:
                found.add(source)
                pending.append(source)
    return found


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _architecture_without_detect(payload: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    return payload["backbone"], payload["head"][:-1]


def _remove_edge_for_comparison(payload: dict[str, Any], edge_index: int) -> dict[str, Any]:
    """Remove a detect-only Edge node and restore the canonical clean sequential edge."""
    normalized = copy.deepcopy(payload)
    head_index = edge_index - len(normalized["backbone"])
    normalized["head"].pop(head_index)
    downstream = normalized["head"][head_index]
    # The explicit clean absolute source becomes the previous layer after Edge removal.
    downstream[0] = -1
    return normalized


def source_preflight() -> dict[str, Any]:
    """Validate YAML, FTSC parity, exact taps, and clean-branch ancestry."""
    reference = _load_yaml(REFERENCE_CONFIG)
    expected_ftsc = reference["ftsc"]
    reference_architecture = _architecture_without_detect(reference)
    report: dict[str, Any] = {}

    for name, spec in CASES.items():
        payload = _load_yaml(spec.config)
        layers = _layers(payload)
        modules = [layer[2] for layer in layers]
        detect = layers[-1]
        edge_indices = [index for index, module in enumerate(modules) if module in EDGE_MODULES]

        _require(payload["ftsc"] == expected_ftsc, f"{name}: FTSC differs from REF_FTSC_P23")
        _require(payload.get("scale") == "n", f"{name}: scale must remain YOLOv8n")
        _require(detect[2] == "Detect", f"{name}: final layer is not Detect")
        _require(tuple(detect[0]) == spec.detect_from, f"{name}: Detect taps are {detect[0]}")
        _require(not (UNWANTED_MODULES & set(modules)), f"{name}: forbidden module present")
        _require(not (UNWANTED_TOP_LEVEL & set(payload)), f"{name}: forbidden mechanism flag present")
        _require(edge_indices == ([] if spec.edge_index is None else [spec.edge_index]),
                 f"{name}: expected Edge index {spec.edge_index}, got {edge_indices}")

        source_types = tuple(layers[index][2] for index in spec.detect_from)
        _require(source_types == spec.expected_source_types,
                 f"{name}: Detect source types {source_types} != {spec.expected_source_types}")

        if spec.edge_index is not None:
            edge_layer = layers[spec.edge_index]
            _require(edge_layer[3] == EDGE_ARGS, f"{name}: Edge parameters changed")
            _require(spec.clean_downstream_start is not None, f"{name}: clean downstream contract missing")
            downstream_index, clean_source = spec.clean_downstream_start
            _require(_resolved_inputs(layers, downstream_index) == (clean_source,),
                     f"{name}: downstream layer {downstream_index} does not start from clean {clean_source}")
            _require(spec.edge_index not in _ancestors(layers, downstream_index),
                     f"{name}: Edge contaminates downstream PAN at layer {downstream_index}")
            for level, source in zip(spec.pyramid_levels, spec.detect_from):
                if source != spec.edge_index:
                    _require(spec.edge_index not in _ancestors(layers, source),
                             f"{name}: Edge contaminates clean {level} Detect source {source}")
            normalized = _remove_edge_for_comparison(payload, spec.edge_index)
            _require(_architecture_without_detect(normalized) == reference_architecture,
                     f"{name}: architecture differs from reference beyond Edge insertion/Detect taps")
        else:
            _require(_architecture_without_detect(payload) == reference_architecture,
                     f"{name}: architecture differs from reference beyond Detect taps")

        report[name] = {
            "status": "PREPARED — NOT YET RUN",
            "stage": spec.stage,
            "config": str(spec.config),
            "model_yaml_sha256": _sha256(spec.config),
            "ftsc": copy.deepcopy(payload["ftsc"]),
            "detect_from": list(spec.detect_from),
            "detect_source_types": list(source_types),
            "expected_detect_strides": [float(2 ** int(level[1:])) for level in spec.pyramid_levels],
            "edge_location": None if spec.edge_index is None else spec.pyramid_levels[list(spec.detect_from).index(spec.edge_index)],
            "edge_propagates_downstream": False,
        }
    return {
        "suite": "varroa_ftsc_edge_placement_headscale",
        "status": "PREPARED — NOT YET RUN",
        "default_cases": list(STAGE1_CASES),
        "optional_stage2_cases": list(STAGE2_CASES),
        "historical_references": copy.deepcopy(HISTORICAL_REFERENCES),
        "cases": report,
    }


def model_preflight() -> dict[str, Any]:
    """Instantiate every new model on CPU and inspect resolved heads/modules."""
    shared.local_ultralytics()
    from ultralytics.nn.modules import Detect, EdgeCueFusion
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

    report: dict[str, Any] = {}
    for name, spec in CASES.items():
        model = DetectionModel(spec.config, verbose=False).cpu()
        head = model.model[-1]
        _require(isinstance(head, Detect), f"{name}: runtime head is not Detect")
        strides = tuple(float(value) for value in head.stride.tolist())
        expected_strides = tuple(float(2 ** int(level[1:])) for level in spec.pyramid_levels)
        _require(strides == expected_strides, f"{name}: strides {strides} != {expected_strides}")
        _require(tuple(head.f) == spec.detect_from, f"{name}: runtime Detect taps changed")
        _require(head.ftsc_calibrator is not None, f"{name}: FTSC is not active")
        source_types = tuple(type(model.model[index]).__name__ for index in head.f)
        _require(source_types == spec.expected_source_types,
                 f"{name}: runtime source types {source_types} != {spec.expected_source_types}")
        edges = [module for module in model.modules() if isinstance(module, EdgeCueFusion)]
        _require(len(edges) == int(spec.edge_index is not None), f"{name}: runtime Edge count wrong")
        if edges:
            edge = edges[0]
            _require(
                (edge.hidden, edge.residual_scale, edge.residual_schedule,
                 edge.ramp_start_epoch, edge.ramp_end_epoch, edge.orientation_gate_mode)
                == (32, 0.25, "constant", 5, 15, "learned"),
                f"{name}: runtime Edge hyperparameters changed",
            )
            _require(edge.orientation_count == 4, f"{name}: directional filter count changed")
            _require(edge.projection.weight.detach().count_nonzero().item() == 0,
                     f"{name}: Edge projection is not zero initialized")
        params = get_num_params(model)
        gradients = get_num_gradients(model)
        gflops = get_flops(model, imgsz=640)
        report[name] = {
            "detect_from": list(head.f),
            "detect_source_types": list(source_types),
            "detect_strides": list(strides),
            "edge_count": len(edges),
            "parameters": int(params),
            "gradients": int(gradients),
            "gflops_640": float(gflops) if gflops else None,
        }
    return report


def model_for(case: str):
    shared.local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(CASES[case].config)
    model.load("yolov8n.pt", smart_transfer=True)
    return model


def _run_dir(args: argparse.Namespace, case: str, seed: int) -> Path:
    return args.project / case / f"seed_{seed}"


def train_one(case: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    """Use the established Varroa completion/resume and AMP fallback policy."""
    run_dir = _run_dir(args, case, seed)
    if shared.complete(run_dir):
        print(f"Reusing completed run: {case}/seed_{seed}", flush=True)
        return run_dir
    last = run_dir / "weights/last.pt"
    if last.is_file():
        shared.local_ultralytics()
        from ultralytics import YOLO

        shared.seed_everything(seed)
        YOLO(last).train(resume=True)
    else:
        kwargs = shared.train_kwargs(args, data_yaml, seed, amp)
        kwargs.update(project=str(args.project / case), name=f"seed_{seed}", exist_ok=True)
        shared.seed_everything(seed)
        try:
            model_for(case).train(**kwargs)
        except Exception as error:
            if not amp:
                raise
            archived = shared.archive_failed_amp(run_dir)
            print(f"AMP failed for {case}/seed_{seed}: {error!r}; archived={archived}; retrying amp=False", flush=True)
            kwargs["amp"] = False
            shared.seed_everything(seed)
            model_for(case).train(**kwargs)
    if not shared.complete(run_dir):
        raise FileNotFoundError(f"Training ended without required artifacts: {run_dir}")
    return run_dir


def write_summaries(args: argparse.Namespace) -> bool:
    """Write summaries only when the entire selected matrix is complete."""
    rows: list[dict[str, Any]] = []
    for case in args.cases:
        for seed in args.seeds:
            path = _run_dir(args, case, seed) / "evaluation_metrics.json"
            if not path.is_file():
                return False
            rows.append({"case": case, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})
    args.project.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"case", "seed"}, key))
    with (args.project / "summary_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for case in args.cases:
        group = [row for row in rows if row["case"] == case]
        _require(len(group) == len(args.seeds), f"Partial aggregate for {case}")
        record: dict[str, Any] = {"case": case, "runs": len(group)}
        common = set.intersection(*(set(row) for row in group)) - {"case", "seed"}
        for key in sorted(common):
            values = [row[key] for row in group]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                numeric = [float(value) for value in values]
                record[f"{key}/mean"] = statistics.fmean(numeric)
                record[f"{key}/std"] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        aggregates.append(record)
    aggregate_fields = sorted({key for row in aggregates for key in row}, key=lambda key: (key not in {"case", "runs"}, key))
    with (args.project / "summary_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--include-stage2", action="store_true",
                        help="Permit optional stage-2 cases; without --cases, append them to stage 1.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/varroa_ftsc_edge_placement_headscale")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-json", type=Path,
                        help="Optionally write preparation metadata; never writes result rows.")
    args = parser.parse_args(argv)
    if args.cases is None:
        args.cases = list(STAGE1_CASES + (STAGE2_CASES if args.include_stage2 else ()))
    return args


def validate_args(args: argparse.Namespace) -> None:
    frozen = {
        "epochs": (args.epochs, 100), "patience": (args.patience, 20),
        "imgsz": (args.imgsz, 640), "batch": (args.batch_size, 4),
    }
    changed = {name: actual for name, (actual, expected) in frozen.items() if actual != expected}
    _require(not changed, f"Frozen Varroa protocol changed: {changed}")
    _require(len(args.cases) == len(set(args.cases)), "Duplicate cases are not allowed")
    _require(len(args.seeds) == len(set(args.seeds)), "Duplicate seeds are not allowed")
    _require(set(args.seeds).issubset(DEFAULT_SEEDS), f"Seeds must come from {list(DEFAULT_SEEDS)}")
    selected_stage2 = set(args.cases) & set(STAGE2_CASES)
    _require(not selected_stage2 or args.include_stage2,
             f"Stage-2 cases require --include-stage2: {sorted(selected_stage2)}")
    _require(not (set(args.cases) & set(HISTORICAL_REFERENCES)), "Historical references cannot be trained")


def main() -> None:
    args = parse_args()
    validate_args(args)
    preflight: dict[str, Any] = {"source": source_preflight()}
    try:
        preflight["model"] = model_preflight()
    except ModuleNotFoundError as error:
        if not args.preflight_only:
            raise
        preflight["model"] = {"status": "unavailable", "reason": f"missing local dependency: {error.name}"}
    output = json.dumps(preflight, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    print(output, end="")
    if args.preflight_json:
        args.preflight_json.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_json.write_text(output, encoding="utf-8")
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
        for case in args.cases:
            run_dir = train_one(case, seed, data_yaml, args.amp, args)
            shared.evaluate(run_dir, args)
    _require(write_summaries(args), "Refusing to aggregate an incomplete selected matrix")


if __name__ == "__main__":
    main()
