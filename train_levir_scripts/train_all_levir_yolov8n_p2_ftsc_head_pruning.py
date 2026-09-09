#!/usr/bin/env python3
"""Prepare/run the structurally pruned FTSC legacy-compatible head ablation.

The default command is a preflight only.  A future run without
``--preflight-only`` uses the matched no-Mosaic protocol and writes results under
``runs/levir_yolov8n_p2_ftsc_head_pruning``.  This module never changes the
legacy reference tree.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT = "levir_yolov8n_p2_ftsc_head_pruning"
FITNESS_METRIC = "map50_95"
PROTOCOL_VERSION = "ftsc_head_pruning_nomosaic_v1"
VARIANTS = {
    "H0_p2_p3_p4_p5_nomosaic": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_n0.yaml",
    "H1_p2_p3_p4_nomosaic": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h1_p2_p3_p4.yaml",
    "H2_p2_p3_nomosaic": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml",
    "H3_p2_only_nomosaic": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h3_p2_only.yaml",
}
EXPECTED_STRIDES = {
    "H0_p2_p3_p4_p5_nomosaic": [4.0, 8.0, 16.0, 32.0],
    "H1_p2_p3_p4_nomosaic": [4.0, 8.0, 16.0],
    "H2_p2_p3_nomosaic": [4.0, 8.0],
    "H3_p2_only_nomosaic": [4.0],
}
# H0 is the historical control.  New pruning cases default to one seed so the
# user can expand the comparison explicitly after inspecting this preflight.
DEFAULT_VARIANTS = list(VARIANTS)[1:]
DEFAULT_SEEDS = [42]


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(model, *, common_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if common_only:
            parts = name.split(".")
            # DetectionModel state keys are model.<index>.*.  Modules through
            # P2 (index 19) are shared by every H case.
            if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit() or int(parts[1]) > 19:
                continue
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _level_label(stride: float) -> str:
    stride = float(stride)
    exponent = round(math.log2(stride)) if stride > 0 else -1
    return f"p{exponent}" if 2**exponent == stride else f"stride{int(stride)}"


def _head_preflight(model, expected_strides: list[float]) -> dict[str, object]:
    from ultralytics.nn.modules import DFL, Detect
    from ultralytics.nn.modules.head import P2OffsetRegression
    from ultralytics.engine.trainer import BaseTrainer

    del P2OffsetRegression  # import check: this class must remain available, even when disabled.
    head = model.model[-1]
    _require(isinstance(head, Detect), f"expected Detect head, got {type(head).__name__}")
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    _require(strides == expected_strides, f"expected strides {expected_strides}, got {strides}")
    calibrator = head.ftsc_calibrator
    _require(calibrator is not None and calibrator.policy == "f5", "FTSC F5 is not active")
    _require(calibrator.fixed_strengths == {"dfl_distribution": 1.0}, "DFL FTSC strength is not fixed at 1")
    _require("dfl_distribution" not in calibrator.strength_logits, "fixed DFL strength became trainable")
    _require(calibrator.strength_logits["position_gaussian"].requires_grad, "Position strength is frozen")
    _require(calibrator.evidence_names == ("position_gaussian", "dfl_distribution"), "Y4 evidence changed")
    _require(calibrator.providers["dfl_distribution"].detach, "DFL evidence must be detached")
    _require(calibrator.dfl_apply_cls and not calibrator.dfl_apply_box and not calibrator.dfl_apply_dfl, "DFL route changed")
    _require(calibrator.apply_cls and calibrator.apply_box and calibrator.apply_dfl, "Position route changed")
    projection_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, DFL)
        for parameter in module.parameters(recurse=True)
    }
    projection = {name: parameter for name, parameter in model.named_parameters() if id(parameter) in projection_ids}
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(freeze=None)
    trainer._freeze_layers([])
    _require(projection and all(not parameter.requires_grad for parameter in projection.values()), "actual DFL projection is not frozen")
    return {
        "head_strides": strides,
        "active_level_labels": [_level_label(stride) for stride in strides],
        "detection_level_count": len(strides),
        "ftsc_policy": calibrator.policy,
        "ftsc_evidence": list(calibrator.evidence_names),
        "dfl_strength_fixed": True,
        "dfl_strength_effective": float(calibrator.strength("dfl_distribution", calibrator.strength_logits["position_gaussian"]).detach().cpu()),
        "position_strength_trainable": True,
        "dfl_projection_parameters": sorted(projection),
    }


def _static_flops(model, imgsz: int) -> tuple[float, bool]:
    """Estimate Conv/Linear FLOPs from one static dummy forward.

    This fallback keeps the preflight useful when the optional ``thop`` package
    is absent.  It is observational and does not alter model weights.
    """
    import torch
    import torch.nn as nn

    total = 0
    hooks = []

    def hook(module, _inputs, output):
        nonlocal total
        if isinstance(output, (tuple, list)):
            outputs = [item for item in output if isinstance(item, torch.Tensor)]
        elif isinstance(output, torch.Tensor):
            outputs = [output]
        else:
            outputs = []
        for value in outputs:
            locations = value.numel() // max(value.shape[1], 1) if value.ndim >= 2 else value.numel()
            if isinstance(module, nn.Conv2d):
                kernel = math.prod(module.kernel_size)
                total += int(2 * module.in_channels * module.out_channels * kernel * locations / module.groups)
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
    return total / 1e9, True


def _transfer_report(model, pretrained: str) -> dict[str, object]:
    """Count matching tensors after smart transfer without forcing a download."""
    from pathlib import Path as _Path
    source_path = _Path(pretrained)
    if not source_path.is_file():
        return {"status": "checkpoint_not_local", "checkpoint": pretrained}
    from ultralytics import YOLO
    source = YOLO(source_path).model.state_dict()
    target = model.model.state_dict()
    matching = [key for key, value in source.items() if key in target and target[key].shape == value.shape]
    transferred = sum(torch_equal(target[key], source[key]) for key in matching)
    total = len(target)
    return {
        "status": "ok",
        "checkpoint": str(source_path),
        "transferred_items": int(transferred),
        "compatible_items": int(len(matching)),
        "total_model_items": int(total),
        "transfer_ratio": float(transferred / max(total, 1)),
    }


def torch_equal(left, right) -> bool:
    import torch
    return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    return {
        "data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch_size, "device": args.device, "workers": args.workers,
        "patience": args.patience, "seed": seed, "deterministic": True,
        "amp": amp, "plots": False, "optimizer": "auto", "cos_lr": False,
        "lrf": 0.01, "fitness_metric": FITNESS_METRIC,
        "mosaic": 0.0, "close_mosaic": 0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "rect": False,
    }


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    return model


def source_preflight(args: argparse.Namespace) -> dict[str, object]:
    import yaml
    configs = {}
    for variant, path in VARIANTS.items():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        _require(payload["ftsc"] == yaml.safe_load(VARIANTS["H0_p2_p3_p4_p5_nomosaic"].read_text())["ftsc"], f"{variant}: FTSC contract differs from H0")
        configs[variant] = {
            "path": str(path),
            "sha256": _sha256_path(path),
            "detect_from": payload["head"][-1][0],
            "head_layers": len(payload["head"]),
        }
    loss_source = ROOT.parent / "models_related/ultralytics/ultralytics/utils/loss.py"
    tal_source = ROOT.parent / "models_related/ultralytics/ultralytics/utils/tal.py"
    return {
        "variants": list(VARIANTS), "configs": configs,
        "protocol": {
            "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz,
            "batch": args.batch_size, "optimizer_requested": "auto", "optimizer_expected": "AdamW",
            "scheduler": "linear", "lrf": 0.01, "fitness_metric": FITNESS_METRIC,
            "mosaic": 0.0, "close_mosaic": 0, "nms_iou": 0.5,
        },
        "ftsc_contract": {
            "policy": "f5", "evidence": ["position_gaussian", "dfl_distribution"],
            "position_alpha": 6.0, "dfl_detach": True, "dfl_apply": {"cls": True, "box": False, "dfl": False},
            "apply": {"cls": True, "box": True, "dfl": True}, "fixed_strengths": {"dfl_distribution": 1.0},
        },
        "sources": {"loss_sha256": _sha256_path(loss_source), "tal_sha256": _sha256_path(tal_source)},
        "diagnostics": {"dfl_near_regmax_threshold": "reg_max - 1.5", "extra_forward_pass": False},
    }


def model_preflight(args: argparse.Namespace) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    import torch

    reports, init_hashes, shared_hashes, complexity = {}, {}, {}, {}
    transfer, shared_pretrained_hashes = {}, {}
    for variant, config in VARIANTS.items():
        torch.manual_seed(0)
        model = DetectionModel(config, verbose=False)
        reports[variant] = _head_preflight(model, EXPECTED_STRIDES[variant])
        init_hashes[variant] = _state_hash(model)
        shared_hashes[variant] = _state_hash(model, common_only=True)
        params = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        gflops, estimated = _static_flops(model, args.imgsz)
        complexity[variant] = {
            "params": int(params), "trainable_params": int(trainable), "GFLOPs": round(gflops, 6),
            "GFLOPs_method": "static Conv/Linear hook" if estimated else "unavailable",
            "strides": EXPECTED_STRIDES[variant], "detection_level_count": len(EXPECTED_STRIDES[variant]),
        }
        if args.pretrained:
            # Transfer counts are only computed when the checkpoint already
            # exists locally; preflight never downloads weights.
            try:
                from ultralytics import YOLO
                if Path(args.pretrained).is_file():
                    torch.manual_seed(0)
                    loaded = YOLO(config)
                    loaded.load(args.pretrained, smart_transfer=True)
                    transfer[variant] = _transfer_report(loaded, args.pretrained)
                    shared_pretrained_hashes[variant] = _state_hash(loaded.model, common_only=True)
                else:
                    transfer[variant] = {"status": "checkpoint_not_local", "checkpoint": args.pretrained}
            except (ImportError, ModuleNotFoundError, RuntimeError) as error:
                transfer[variant] = {"status": "unavailable", "error": str(error)}
    expected_shared = shared_hashes["H0_p2_p3_p4_p5_nomosaic"]
    _require(all(value == expected_shared for value in shared_hashes.values()), f"shared initialization hashes diverged: {shared_hashes}")
    ordered = [complexity[name]["params"] for name in VARIANTS]
    _require(ordered == sorted(ordered, reverse=True), f"parameter complexity is not monotonic: {complexity}")
    gflops = [complexity[name]["GFLOPs"] for name in VARIANTS]
    if all(gflops):
        _require(gflops == sorted(gflops, reverse=True), f"GFLOPs complexity is not monotonic: {complexity}")
    if shared_pretrained_hashes:
        expected_pretrained = shared_pretrained_hashes["H0_p2_p3_p4_p5_nomosaic"]
        _require(
            all(value == expected_pretrained for value in shared_pretrained_hashes.values()),
            f"shared pretrained initialization hashes diverged: {shared_pretrained_hashes}",
        )
    return {
        "variant_reports": reports, "student_init_hashes": init_hashes,
        "shared_initialization_hashes": shared_hashes, "shared_initialization_equal": True,
        "complexity": complexity, "pretrained_transfer": transfer,
        "shared_pretrained_initialization_hashes": shared_pretrained_hashes,
        "shared_pretrained_initialization_equal": bool(shared_pretrained_hashes),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _collect_outputs(args: argparse.Namespace, preflight: dict[str, object]) -> None:
    rows, levels = [], []
    complexity = preflight.get("model", {}).get("complexity", {}) if isinstance(preflight.get("model"), dict) else {}
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = args.project / variant / f"seed_{seed}"
            metrics = run_dir / "evaluation_metrics.json"
            if metrics.is_file():
                record = {"variant": variant, "seed": seed, **json.loads(metrics.read_text(encoding="utf-8"))}
                record.update({
                    "active_levels": len(EXPECTED_STRIDES[variant]),
                    "active_strides": ",".join(str(int(value)) for value in EXPECTED_STRIDES[variant]),
                    **{key: value for key, value in complexity.get(variant, {}).items() if key in {"params", "trainable_params", "GFLOPs"}},
                })
                rows.append(record)
            results_csv = run_dir / "results.csv"
            if results_csv.is_file():
                with results_csv.open(newline="", encoding="utf-8") as handle:
                    for result_row in csv.DictReader(handle):
                        epoch = result_row.get("epoch", "")
                        for metric, value in result_row.items():
                            if not metric.startswith("ftsc_level_") or value in (None, ""):
                                continue
                            levels.append({"variant": variant, "seed": seed, "epoch": epoch, "metric": metric, "value": float(value)})
            diagnostic = run_dir / "level_diagnostics.json"
            if diagnostic.is_file():
                payload = json.loads(diagnostic.read_text(encoding="utf-8"))
                levels.extend({"variant": variant, "seed": seed, **item} for item in payload)
    _write_csv(args.project / "ftsc_head_pruning_summary_runs.csv", rows)
    _write_csv(args.project / "ftsc_head_pruning_level_diagnostics.csv", levels)
    aggregate = []
    for variant in args.variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group: continue
        record = {"variant": variant, "runs": len(group)}
        for key in sorted(set.intersection(*(set(row) for row in group)) - {"variant", "seed"}):
            try:
                values = [float(row[key]) for row in group]
            except (TypeError, ValueError):
                continue
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(record)
    _write_csv(args.project / "ftsc_head_pruning_summary_aggregate.csv", aggregate)
    (args.project / "ftsc_head_pruning_results.json").write_text(
        json.dumps({"preflight": preflight, "runs": rows, "level_diagnostics": levels}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    """Evaluate through the routing runner without replacing its evaluator."""
    metrics = workflow.evaluate(run_dir, data_yaml, args)
    metrics.update({"protocol/version": PROTOCOL_VERSION, "protocol/fitness_metric": FITNESS_METRIC,
                    "protocol/nms_iou": 0.5, "protocol/mosaic": 0.0, "protocol/close_mosaic": 0})
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.project = args.project.resolve()
    preflight: dict[str, object] = {"source": source_preflight(args)}
    if args.preflight_only:
        try:
            preflight["model"] = model_preflight(args)
        except (ImportError, ModuleNotFoundError) as error:
            preflight["model"] = {"status": "blocked_missing_runtime_dependency", "error": str(error)}
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return
    workflow.EXPERIMENT = EXPERIMENT
    workflow.VARIANTS = VARIANTS
    workflow.model_for = model_for
    workflow.train_kwargs = train_kwargs
    args.fitness_metric = FITNESS_METRIC
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    preflight["model"] = model_preflight(args)
    data_yaml = workflow.prepare_fixed_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, True, args)
            evaluate(run_dir, data_yaml, args)
    _collect_outputs(args, preflight)


if __name__ == "__main__":
    main()
