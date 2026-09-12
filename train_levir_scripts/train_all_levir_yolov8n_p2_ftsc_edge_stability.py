#!/usr/bin/env python3
"""Preflight and later run the H2+Edge residual/gate stability screen.

The default variants are ES1/ES2/ES3. ES0 reuses the existing E_H2 config and
is available only when explicitly selected; it is never retrained by default.
This file only prepares provenance and aggregation when used with
``--preflight-only`` or ``--aggregate-only`` and does not train during tests.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

_WORKFLOW_EVALUATE = workflow.evaluate
CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT = "ftsc_edge_stability"
FITNESS_METRIC = "map50_95"
DEFAULT_VARIANTS = ["ES1", "ES2", "ES3"]
VARIANTS = {
    "ES0": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edgecue.yaml",
    "ES1": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml",
    "ES2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es2_ramp025.yaml",
    "ES3": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es3_uniform025.yaml",
}
EXPECTED_STRIDES = [4.0, 8.0]
EDGE_EXPECTED = {
    "ES0": {"residual_scale": 1.0, "residual_schedule": "constant", "orientation_gate_mode": "learned"},
    "ES1": {"residual_scale": 0.25, "residual_schedule": "constant", "orientation_gate_mode": "learned"},
    "ES2": {"residual_scale": 0.25, "residual_schedule": "ramp", "orientation_gate_mode": "learned"},
    "ES3": {"residual_scale": 0.25, "residual_schedule": "constant", "orientation_gate_mode": "uniform"},
}
EDGE_DIAGNOSTIC_METRICS = (
    "edge_effective_residual_alpha",
    "edge_orientation_gate_weight_0",
    "edge_orientation_gate_weight_1",
    "edge_orientation_gate_weight_2",
    "edge_orientation_gate_weight_3",
    "edge_orientation_entropy",
    "edge_residual_norm",
    "edge_p2_norm",
    "edge_residual_p2_norm_ratio",
    "edge_projection_weight_norm",
    "edge_feature_activation_mean",
    "edge_feature_activation_std",
)
DETECTION_METRICS = (
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
    "metrics/mAP50-95(B)", "metrics/mAP75(B)",
)
REQUIRED_PROTOCOL = {
    "epochs": 100, "patience": 20, "imgsz": 512, "batch": 8,
    "optimizer": "auto", "cos_lr": False, "lrf": 0.01,
    "mosaic": 0.0, "close_mosaic": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    return {
        "data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch_size, "device": args.device, "workers": args.workers,
        "patience": args.patience, "seed": seed, "deterministic": True, "amp": amp,
        "plots": False, "optimizer": "auto", "cos_lr": False, "lrf": 0.01,
        "fitness_metric": FITNESS_METRIC, "mosaic": 0.0, "close_mosaic": 0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "rect": False,
    }


def _config_payloads() -> dict[str, dict]:
    import yaml

    return {name: yaml.safe_load(path.read_text(encoding="utf-8")) for name, path in VARIANTS.items()}


def source_preflight(args: argparse.Namespace) -> dict[str, object]:
    payloads = _config_payloads()
    configs = {}
    for variant in args.variants:
        payload = payloads[variant]
        _require(payload["head"][-1][2] == "Detect", f"{variant}: final layer is not Detect")
        _require(payload["head"][-1][0] == [19, 22], f"{variant}: Detect inputs changed")
        _require(payload["ftsc"]["policy"] == "f5", f"{variant}: FTSC policy changed")
        _require(payload["ftsc"]["evidence"] == ["position_gaussian", "dfl_distribution"], f"{variant}: FTSC evidence changed")
        _require(payload["ftsc"]["fixed_strengths"] == {"dfl_distribution": 1.0}, f"{variant}: DFL fixed strength changed")
        edge_layers = [layer for layer in payload["head"] if layer[2] == "P2EdgeCueFusion"]
        _require(len(edge_layers) == 1, f"{variant}: expected one P2EdgeCueFusion layer")
        edge = payload.get("edge_stability", {"residual_scale": 1.0, "residual_schedule": "constant", "orientation_gate_mode": "learned"})
        expected = EDGE_EXPECTED[variant]
        for key, value in expected.items():
            _require(edge.get(key) == value, f"{variant}: {key} expected {value!r}, got {edge.get(key)!r}")
        _require(edge.get("hidden", 32) == 32, f"{variant}: Edge hidden width changed")
        configs[variant] = {
            "path": str(VARIANTS[variant]), "sha256": _sha256(VARIANTS[variant]),
            "detect_from": payload["head"][-1][0], "active_strides": EXPECTED_STRIDES,
            "edge": {key: edge.get(key) for key in ("residual_scale", "residual_schedule", "ramp_start_epoch", "ramp_end_epoch", "orientation_gate_mode", "hidden")},
            "base_reference": "E_H2",
        }
    return {
        "suite": EXPERIMENT,
        "variants": list(args.variants),
        "configs": configs,
        "protocol": {**REQUIRED_PROTOCOL, "seed_values": list(args.seeds), "split_seed": args.split_seed, "fitness_metric": FITNESS_METRIC, "nms_iou": 0.5, "pretrained": args.pretrained},
        "ftsc": {"policy": "f5", "evidence": ["position_gaussian", "dfl_distribution"], "dfl_fixed_effective": 1.0, "position_strength_trainable": True},
        "teacher_student": False,
        "diagnostics": {"extra_forward_pass": False, "csv": "ftsc_edge_stability_diagnostics.csv"},
    }


def _state_hash(model, shared_prefix_limit: int | None = None) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if shared_prefix_limit is not None:
            parts = name.split(".")
            if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit() or int(parts[1]) > shared_prefix_limit:
                continue
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _static_flops(model, imgsz: int) -> float:
    """Estimate model multiply-adds without a training or validation pass."""
    import torch
    import torch.nn as nn

    total, hooks = 0, []

    def hook(module, _inputs, output):
        nonlocal total
        outputs = output if isinstance(output, (list, tuple)) else [output]
        for value in outputs:
            if not isinstance(value, torch.Tensor):
                continue
            locations = value.numel() // max(value.shape[1], 1) if value.ndim >= 2 else value.numel()
            if isinstance(module, nn.Conv2d):
                total += int(2 * module.in_channels * module.out_channels * math.prod(module.kernel_size) * locations / module.groups)
            elif isinstance(module, nn.Linear):
                total += int(2 * module.in_features * module.out_features * locations)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device))
    for handle in hooks:
        handle.remove()
    model.train(was_training)
    return total / 1e9


def _head_report(model, variant: str) -> dict[str, object]:
    from ultralytics.nn.modules import Detect, P2EdgeCueFusion

    head = model.model[-1]
    _require(isinstance(head, Detect), f"{variant}: expected Detect head")
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    _require(strides == EXPECTED_STRIDES, f"{variant}: expected {EXPECTED_STRIDES}, got {strides}")
    branch = [module for module in model.modules() if isinstance(module, P2EdgeCueFusion)]
    _require(len(branch) == 1, f"{variant}: expected one Edge branch")
    edge = branch[0]
    edge_hidden = getattr(edge, "hidden", edge.encoder[0].conv.out_channels)
    _require(edge_hidden == 32, f"{variant}: Edge hidden width changed")
    return {
        "active_strides": strides,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "edge_hidden": edge.encoder[0].conv.out_channels,
        "edge_residual_scale": edge.residual_scale,
        "edge_residual_schedule": edge.residual_schedule,
        "edge_orientation_gate_mode": edge.orientation_gate_mode,
        "edge_gate_trainable": edge.gate is not None and any(parameter.requires_grad for parameter in edge.gate.parameters()),
        "edge_gflops": round(edge.estimated_gflops(512), 6),
    }


def model_preflight(args: argparse.Namespace) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    import torch

    reports, hashes = {}, {}
    for variant in args.variants:
        torch.manual_seed(0)
        model = DetectionModel(VARIANTS[variant], verbose=False)
        reports[variant] = _head_report(model, variant)
        reports[variant]["GFLOPs"] = round(_static_flops(model, args.imgsz), 6)
        # ES3 omits the learned gate, so parameters after the Edge layer have
        # different RNG draw order. Compare only the shared H2 graph before
        # Edge (model.0..model.19) to verify matched initialization.
        hashes[variant] = _state_hash(model, shared_prefix_limit=19)
    _require(len(set(hashes.values())) == 1, f"matched H2 initialization differs: {hashes}")
    return {"variant_reports": reports, "matched_backbone_hashes": hashes, "same_non_edge_initial_state": True}


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    return model


def _read_results(path: Path, variant: str, seed: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", ""))) - 1
        except (TypeError, ValueError):
            continue
        record: dict[str, object] = {"variant": variant, "seed": seed, "epoch": epoch}
        for key in EDGE_DIAGNOSTIC_METRICS + DETECTION_METRICS:
            if row.get(key) not in (None, ""):
                try:
                    record[key] = float(row[key])
                except (TypeError, ValueError):
                    pass
        output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed", "epoch"}, key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["variant", "seed"])
        writer.writeheader()
        writer.writerows(rows)


def aggregate_outputs(args: argparse.Namespace, preflight: dict[str, object] | None = None) -> dict[str, object]:
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            rows.extend(_read_results(args.project / variant / f"seed_{seed}" / "results.csv", variant, seed))
    _write_csv(args.results_dir / "ftsc_edge_stability_diagnostics.csv", [row for row in rows if any(key in row for key in EDGE_DIAGNOSTIC_METRICS)])

    summary_runs = []
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = args.project / variant / f"seed_{seed}"
            record: dict[str, object] = {"variant": variant, "seed": seed}
            if isinstance(preflight, dict):
                model_report = preflight.get("model", {}).get("variant_reports", {}).get(variant, {})
                if isinstance(model_report, dict):
                    record.update(model_report)
            metrics_path = run_dir / "evaluation_metrics.json"
            if metrics_path.is_file():
                record.update(json.loads(metrics_path.read_text(encoding="utf-8")))
            summary_runs.append(record)
    _write_csv(args.results_dir / "ftsc_edge_stability_summary_runs.csv", summary_runs)

    aggregate = []
    for variant in args.variants:
        group = [row for row in summary_runs if row["variant"] == variant]
        record: dict[str, object] = {"variant": variant, "runs": len(group)}
        for key in sorted(set.intersection(*(set(row) for row in group)) - {"variant", "seed"}) if group else ():
            try:
                values = [float(row[key]) for row in group]
            except (TypeError, ValueError):
                continue
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(record)
    _write_csv(args.results_dir / "ftsc_edge_stability_summary_aggregate.csv", aggregate)
    payload = {"preflight": preflight or {}, "summary_runs": summary_runs, "diagnostics": rows}
    (args.results_dir / "ftsc_edge_stability_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
    parser.add_argument("--results-dir", type=Path, default=ROOT.parent.parent / "results")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.project = args.project.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.split_seed != 42:
        raise ValueError("split_seed must remain 42")
    preflight = {"source": source_preflight(args)}
    if args.aggregate_only:
        aggregate_outputs(args, preflight)
        return
    try:
        preflight["model"] = model_preflight(args)
    except (ImportError, ModuleNotFoundError) as error:
        preflight["model"] = {"status": "blocked_missing_runtime_dependency", "error": str(error)}
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return
    workflow.EXPERIMENT = EXPERIMENT
    workflow.VARIANTS = VARIANTS
    workflow.model_for = model_for
    workflow.train_kwargs = train_kwargs
    data_yaml = workflow.prepare_fixed_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, True, args)
            metrics = _WORKFLOW_EVALUATE(run_dir, data_yaml, args)
            (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate_outputs(args, preflight)


if __name__ == "__main__":
    main()
