#!/usr/bin/env python3
"""Preflight and (when explicitly requested) run the preregistered H1/H2 cases.

This suite is deliberately separate from the historical head-pruning runner. Its
first invocation should be ``--source-preflight-only`` or ``--preflight-only``;
it never uploads artifacts and has no teacher/student or distillation path.
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
EXPERIMENT = "ftsc_h1_h2_nextgen_v1"
FITNESS_METRIC = "map50_95"
PROTOCOL_VERSION = "y4_legacy_nomosaic_nextgen_v1"
VARIANTS = {
    "AZ_H1": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h1_zoomdet.yaml",
    "AZ_H2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_zoomdet.yaml",
    "C_H1": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h1_colorcue.yaml",
    "C_H2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_colorcue.yaml",
    "E_H1": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h1_edgecue.yaml",
    "E_H2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edgecue.yaml",
    "H4_P2_P3_P5": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h4_p2_p3_p5.yaml",
}
BASE_REFERENCE = {"AZ_H1": "H1", "C_H1": "H1", "E_H1": "H1", "AZ_H2": "H2", "C_H2": "H2", "E_H2": "H2", "H4_P2_P3_P5": "Y4"}
EXPECTED_DETECT_INPUTS = {
    "AZ_H1": [19, 22, 25], "C_H1": [19, 22, 25], "E_H1": [19, 22, 25],
    "AZ_H2": [19, 22], "C_H2": [19, 22], "E_H2": [19, 22],
    "H4_P2_P3_P5": [19, 22, 28],
}
EXPECTED_STRIDES = {
    "AZ_H1": [4.0, 8.0, 16.0], "C_H1": [4.0, 8.0, 16.0], "E_H1": [4.0, 8.0, 16.0],
    "AZ_H2": [4.0, 8.0], "C_H2": [4.0, 8.0], "E_H2": [4.0, 8.0],
    "H4_P2_P3_P5": [4.0, 8.0, 32.0],
}
DEFAULT_VARIANTS = list(VARIANTS)
DEFAULT_SEEDS = [42]
RESULTS_ROOT = ROOT.parent.parent / "results"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _level_label(stride: float) -> str:
    exponent = round(math.log2(float(stride)))
    return f"p{exponent}" if 2**exponent == float(stride) else f"stride{int(stride)}"


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    """Return the fixed Y4 no-Mosaic protocol; no teacher/student arguments exist."""
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


def source_preflight(args: argparse.Namespace) -> dict[str, object]:
    import yaml
    n0 = yaml.safe_load((CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_n0.yaml").read_text(encoding="utf-8"))
    configs = {}
    for variant, path in VARIANTS.items():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        _require(payload.get("ftsc") == n0.get("ftsc"), f"{variant}: FTSC contract differs from legacy Y4")
        _require(payload["head"][-1][2] == "Detect", f"{variant}: final layer is not Detect")
        _require(payload["head"][-1][0] == EXPECTED_DETECT_INPUTS[variant], f"{variant}: Detect inputs changed")
        if variant.startswith("AZ_"):
            z = payload.get("zoomdet", {})
            _require(z.get("status") == "BLOCKED_EXACT_PORT", f"{variant}: AZ must remain explicitly blocked")
            _require(z.get("legacy_adaptive_zoom_forbidden") is True, f"{variant}: legacy crop path not forbidden")
        configs[variant] = {
            "path": str(path), "sha256": _sha256_path(path), "detect_from": payload["head"][-1][0],
            "head_layers": len(payload["head"]), "base_reference": BASE_REFERENCE[variant],
            "zoomdet_status": payload.get("zoomdet", {}).get("status", "disabled"),
            "color_cue_enabled": variant.startswith("C_"), "edge_cue_enabled": variant.startswith("E_"),
            "h4_p4_intermediate": variant == "H4_P2_P3_P5",
        }
    return {
        "suite": EXPERIMENT, "variants": list(VARIANTS), "configs": configs,
        "protocol": {
            "epochs": args.epochs, "patience": args.patience, "imgsz": args.imgsz,
            "batch": args.batch_size, "workers": args.workers, "optimizer_requested": "auto",
            "optimizer_expected": "AdamW", "scheduler": "linear", "lrf": 0.01,
            "fitness_metric": FITNESS_METRIC, "mosaic": 0.0, "close_mosaic": 0,
            "split_seed": args.split_seed, "nms_iou": 0.5, "deterministic": True,
        },
        "ftsc_contract": {
            "policy": "f5", "evidence": ["position_gaussian", "dfl_distribution"],
            "position_alpha": 6.0, "dfl_detach": True,
            "dfl_apply": {"cls": True, "box": False, "dfl": False},
            "apply": {"cls": True, "box": True, "dfl": True},
            "fixed_strengths": {"dfl_distribution": 1.0}, "position_strength_trainable": True,
        },
        "sources": {
            "loss_sha256": _sha256_path(ROOT.parent / "models_related/ultralytics/ultralytics/utils/loss.py"),
            "tal_sha256": _sha256_path(ROOT.parent / "models_related/ultralytics/ultralytics/utils/tal.py"),
        },
        "teacher_student": False,
        "diagnostics": {"extra_forward_pass": False, "level_labels_from_stride": True},
    }


def _static_flops(model, imgsz: int) -> float:
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
    was_training = model.training; model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz, device=next(model.parameters()).device))
    for handle in hooks: handle.remove()
    model.train(was_training)
    return total / 1e9


def _state_hash(model, prefix_limit: int | None = None) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if prefix_limit is not None and (not name.startswith("model.") or not name.split(".")[1].isdigit() or int(name.split(".")[1]) > prefix_limit):
            continue
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _head_report(model, variant: str) -> dict[str, object]:
    from ultralytics.nn.modules import Detect, DFL
    head = model.model[-1]
    _require(isinstance(head, Detect), f"{variant}: expected Detect, got {type(head).__name__}")
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    _require(strides == EXPECTED_STRIDES[variant], f"{variant}: expected strides {EXPECTED_STRIDES[variant]}, got {strides}")
    calibrator = head.ftsc_calibrator
    _require(calibrator is not None and calibrator.policy == "f5", f"{variant}: FTSC F5 inactive")
    _require(calibrator.evidence_names == ("position_gaussian", "dfl_distribution"), f"{variant}: evidence changed")
    _require(calibrator.fixed_strengths == {"dfl_distribution": 1.0}, f"{variant}: DFL strength changed")
    _require("dfl_distribution" not in calibrator.strength_logits, f"{variant}: DFL became trainable")
    _require(calibrator.strength_logits["position_gaussian"].requires_grad, f"{variant}: Position is frozen")
    dfl_params = [p for m in model.modules() if isinstance(m, DFL) for p in m.parameters()]
    _require(dfl_params and all(not p.requires_grad for p in dfl_params), f"{variant}: DFL projection is not frozen")
    branch = [m for m in model.modules() if m.__class__.__name__ in {"P2ColorCueFusion", "P2EdgeCueFusion"}]
    return {
        "head_strides": strides, "active_level_labels": [_level_label(v) for v in strides],
        "detection_level_count": len(strides), "ftsc_policy": calibrator.policy,
        "ftsc_evidence": list(calibrator.evidence_names), "dfl_strength_fixed": True,
        "position_strength_trainable": True, "cue_branch": branch[0].__class__.__name__ if branch else None,
        "cue_branch_parameters": sum(p.numel() for p in branch[0].parameters()) if branch else 0,
        "cue_branch_gflops": round(branch[0].estimated_gflops(512), 6) if branch and hasattr(branch[0], "estimated_gflops") else 0.0,
        "h4_p4_intermediate": variant == "H4_P2_P3_P5",
        "h4_p4_detect": False if variant == "H4_P2_P3_P5" else None,
    }


def model_preflight(args: argparse.Namespace) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    import torch
    reports, complexity, hashes = {}, {}, {}
    for variant, config in VARIANTS.items():
        if variant.startswith("AZ_"):
            reports[variant] = {"status": "BLOCKED_EXACT_PORT", "reason": "See ultralytics/nn/modules/zoomdet.py"}
            continue
        torch.manual_seed(0)
        model = DetectionModel(config, verbose=False)
        reports[variant] = _head_report(model, variant)
        complexity[variant] = {
            "params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "GFLOPs": round(_static_flops(model, args.imgsz), 6), "strides": EXPECTED_STRIDES[variant],
        }
        limit = 18 if variant.startswith(("C_", "E_")) else 19
        hashes[variant] = _state_hash(model, limit)
    return {"variant_reports": reports, "complexity": complexity, "shared_initialization_hashes": hashes,
            "shared_initialization_note": "compare within matched H1/H2 family; cue branch excluded by prefix"}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row}, key=lambda k: (k not in {"variant", "seed"}, k))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def collect_outputs(args: argparse.Namespace, preflight: dict[str, object]) -> None:
    rows, diagnostics = [], []
    complexity = preflight.get("model", {}).get("complexity", {}) if isinstance(preflight.get("model"), dict) else {}
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = args.project / variant / f"seed_{seed}"
            metrics = run_dir / "evaluation_metrics.json"
            if metrics.is_file():
                record = {"variant": variant, "seed": seed, **json.loads(metrics.read_text(encoding="utf-8"))}
                record.update({"base_reference": BASE_REFERENCE[variant], "active_strides": ",".join(str(int(x)) for x in EXPECTED_STRIDES[variant]), **complexity.get(variant, {})})
                rows.append(record)
            diag = run_dir / "level_diagnostics.json"
            if diag.is_file():
                payload = json.loads(diag.read_text(encoding="utf-8"))
                diagnostics.extend({"variant": variant, "seed": seed, **item} for item in payload)
    out = args.results_dir
    _write_csv(out / "ftsc_h1_h2_nextgen_summary_runs.csv", rows)
    _write_csv(out / "ftsc_h1_h2_nextgen_diagnostics.csv", diagnostics)
    aggregate = []
    for variant in args.variants:
        group = [r for r in rows if r["variant"] == variant]
        if not group: continue
        record = {"variant": variant, "runs": len(group)}
        for key in sorted(set.intersection(*(set(r) for r in group)) - {"variant", "seed"}):
            try: values = [float(r[key]) for r in group]
            except (TypeError, ValueError): continue
            record[f"{key}/mean"] = statistics.fmean(values); record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(record)
    _write_csv(out / "ftsc_h1_h2_nextgen_summary_aggregate.csv", aggregate)
    (out / "ftsc_h1_h2_nextgen_results.json").write_text(json.dumps({"preflight": preflight, "runs": rows, "diagnostics": diagnostics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_for(variant: str, pretrained: str):
    if variant.startswith("AZ_"):
        from ultralytics.nn.modules.zoomdet import assert_exact_port_available
        assert_exact_port_available()
    workflow.local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(VARIANTS[variant]); model.load(pretrained, smart_transfer=True)
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20); parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--source-preflight-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true", default=True, help="retained for explicit no-upload protocol")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(); args.project = args.project.resolve(); args.results_dir = args.results_dir.resolve()
    preflight: dict[str, object] = {"source": source_preflight(args)}
    if args.source_preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True)); return
    try:
        preflight["model"] = model_preflight(args)
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        preflight["model"] = {"status": "blocked_missing_runtime_dependency", "error": str(error)}
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True)); return
    # Training is opt-in and is never invoked by this task. The caller must
    # explicitly omit --preflight-only after reviewing the generated table.
    workflow.EXPERIMENT = EXPERIMENT; workflow.VARIANTS = VARIANTS
    workflow.model_for = model_for; workflow.train_kwargs = train_kwargs
    args.fitness_metric = FITNESS_METRIC
    data_yaml = workflow.prepare_fixed_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, True, args)
            workflow.evaluate(run_dir, data_yaml, args)
    collect_outputs(args, preflight)


if __name__ == "__main__":
    main()
