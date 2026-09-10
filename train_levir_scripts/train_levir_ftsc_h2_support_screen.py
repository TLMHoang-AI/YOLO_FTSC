#!/usr/bin/env python3
"""Run the seed-42 H2 post-TAL support mechanism screen.

Use ``--preflight-only`` to inspect contracts without training. This runner
never changes TAL or FTSC; S1 only enables the existing training-time
AdaptiveZoom pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

_WORKFLOW_EVALUATE = workflow.evaluate
CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT = "levir_ftsc_h2_support_screen"
FITNESS_METRIC = "map50_95"
VARIANTS = {
    "S0_H2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_support_control.yaml",
    "S1_AZ_H2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_support_zoom.yaml",
}
ADAPTIVE_ZOOM = {"S0_H2": False, "S1_AZ_H2": True}
EXPECTED_STRIDES = {"S0_H2": [4.0, 8.0], "S1_AZ_H2": [4.0, 8.0]}
SUPPORT_METRICS = (
    "tal_gt_count", "tal_positive_count", "tal_mean_positives_per_gt", "tal_median_positives_per_gt",
    "tal_zero_positive_gt_fraction", "tal_single_positive_gt_fraction", "tal_two_positive_gt_fraction",
    "tal_three_positive_gt_fraction", "tal_four_positive_gt_fraction", "tal_fiveplus_positive_gt_fraction",
    "ftsc_activation_rate", "tal_positive_count_p2", "tal_positive_count_p3",
    "tal_mean_p2_positives_per_gt", "tal_mean_p3_positives_per_gt",
    "tal_gt_support_p2_only_fraction", "tal_gt_support_p3_only_fraction", "tal_gt_support_p2_p3_fraction",
    "tal_positive_share_p2", "tal_positive_share_p3",
)
DETECTION_METRICS = (
    "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
    "metrics/mAP50-95(B)", "metrics/mAP75(B)",
)
ZOOM_METRICS = (
    "zoom_applied_fraction", "zoom_factor_mean", "zoom_factor_std", "zoom_factor_max",
    "tiny_gt_count", "zoomed_gt_count", "boxes_dropped_by_zoom", "empty_after_zoom_count",
    "bbox_min_side_before_mean", "bbox_min_side_after_mean",
)
WINDOWS = {"early": (0, 9), "mid": (40, 59), "late": (90, 99)}


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
    variant = getattr(args, "_active_variant", "S0_H2")
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
        "adaptive_zoom": ADAPTIVE_ZOOM[variant],
        "adaptive_zoom_probability": 0.5, "adaptive_zoom_target_min_side": 16.0,
        "adaptive_zoom_max_factor": 2.0, "adaptive_zoom_visible_ratio": 0.5,
        "adaptive_zoom_center_jitter": 0.05,
    }


def _head_preflight(model, variant: str) -> dict[str, object]:
    from ultralytics.nn.modules import Detect
    head = model.model[-1]
    _require(isinstance(head, Detect), f"{variant}: expected Detect head")
    strides = [float(v) for v in head.stride.detach().cpu().tolist()]
    _require(strides == EXPECTED_STRIDES[variant], f"{variant}: expected {EXPECTED_STRIDES[variant]}, got {strides}")
    calibrator = head.ftsc_calibrator
    _require(calibrator is not None and calibrator.policy == "f5", f"{variant}: FTSC F5 inactive")
    _require(calibrator.evidence_names == ("position_gaussian", "dfl_distribution"), f"{variant}: evidence changed")
    _require(calibrator.fixed_strengths == {"dfl_distribution": 1.0}, f"{variant}: DFL strength changed")
    _require("dfl_distribution" not in calibrator.strength_logits, f"{variant}: DFL became trainable")
    _require(calibrator.strength_logits["position_gaussian"].requires_grad, f"{variant}: Position is frozen")
    _require(calibrator.providers["dfl_distribution"].detach, f"{variant}: DFL is not detached")
    _require(calibrator.dfl_apply_cls and not calibrator.dfl_apply_box and not calibrator.dfl_apply_dfl, f"{variant}: DFL routing changed")
    _require(calibrator.apply_cls and calibrator.apply_box and calibrator.apply_dfl, f"{variant}: Position routing changed")
    return {"head_strides": strides, "ftsc_policy": calibrator.policy, "ftsc_evidence": list(calibrator.evidence_names), "dfl_strength_effective": 1.0, "position_strength_trainable": True, "adaptive_zoom": ADAPTIVE_ZOOM[variant]}


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_preflight(args: argparse.Namespace) -> dict[str, object]:
    import yaml
    payloads = {name: yaml.safe_load(path.read_text(encoding="utf-8")) for name, path in VARIANTS.items()}
    for name, payload in payloads.items():
        _require(payload["ftsc"]["fixed_strengths"] == {"dfl_distribution": 1.0}, f"{name}: fixed DFL contract missing")
        _require(payload["head"][-1][0] == [19, 22], f"{name}: H2 Detect inputs changed")
        _require(payload["support_screen"]["case"] == name, f"{name}: support-screen metadata case mismatch")
    _require(payloads["S0_H2"]["backbone"] == payloads["S1_AZ_H2"]["backbone"], "backbone differs")
    _require(payloads["S0_H2"]["head"] == payloads["S1_AZ_H2"]["head"], "H2 topology differs")
    invariant_zero = {name: {key: value for key, value in payload.items() if key != "support_screen"} for name, payload in payloads.items()}
    _require(invariant_zero["S0_H2"] == invariant_zero["S1_AZ_H2"], "configs differ outside support_screen metadata")
    return {"variants": list(VARIANTS), "config_sha256": {n: _sha256(p) for n, p in VARIANTS.items()}, "configs": {n: str(p) for n, p in VARIANTS.items()}, "seed": args.seeds[0], "split_seed": args.split_seed, "epochs": args.epochs, "patience": args.patience, "optimizer_requested": "auto", "optimizer_expected": "AdamW", "lr0": 0.002, "lrf": 0.01, "scheduler": "linear", "fitness_metric": FITNESS_METRIC, "mosaic": 0.0, "close_mosaic": 0, "nms_iou": 0.5, "pretrained": args.pretrained, "project": str(args.project), "ftsc": {"policy": "f5", "evidence": ["position_gaussian", "dfl_distribution"], "dfl_fixed": 1.0, "position_trainable": True}, "adaptive_zoom": {name: ADAPTIVE_ZOOM[name] for name in VARIANTS}, "same_h2_topology": True, "tal_modified": False}


def model_preflight(args: argparse.Namespace) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    import torch
    reports, hashes = {}, {}
    for name, config in VARIANTS.items():
        torch.manual_seed(0); model = DetectionModel(config, verbose=False)
        reports[name] = _head_preflight(model, name); hashes[name] = _state_hash(model)
    _require(hashes["S0_H2"] == hashes["S1_AZ_H2"], f"initial H2 states differ: {hashes}")
    return {"variant_reports": reports, "student_init_hashes": hashes, "same_initial_state": True}


def print_preflight(report: dict[str, object], args: argparse.Namespace) -> None:
    print("variant\tseed\tyaml\tstrides\tFTSC\tDFL_fixed\tPosition_trainable\tmosaic\tadaptive_zoom\tpretrained\tproject")
    for name, path in VARIANTS.items(): print("\t".join(map(str, (name, args.seeds[0], path.name, EXPECTED_STRIDES[name], "f5", 1.0, True, 0.0, ADAPTIVE_ZOOM[name], args.pretrained, args.project))))


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(VARIANTS[variant]); model.load(pretrained, smart_transfer=True); return model


def _read_rows(run_dir: Path, variant: str, seed: int) -> list[dict[str, object]]:
    path = run_dir / "results.csv"
    if not path.is_file(): return []
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        try:
            # Ultralytics writes epoch numbers as 1..N; the preregistered
            # windows use zero-based epoch indices (0..N-1).
            epoch = int(float(row.get("epoch", ""))) - 1
        except (TypeError, ValueError): continue
        record = {"variant": variant, "seed": seed, "epoch": epoch}
        for key, value in row.items():
            if key in SUPPORT_METRICS or key in DETECTION_METRICS or key in ZOOM_METRICS:
                if value not in (None, ""):
                    try: record[key] = float(value)
                    except ValueError: pass
        output.append(record)
    return output


def aggregate_outputs(args: argparse.Namespace) -> dict[str, object]:
    all_rows, summary = [], []
    for variant in args.variants:
        rows = _read_rows(args.project / variant / f"seed_{args.seeds[0]}", variant, args.seeds[0]); all_rows.extend(rows)
        record = {"variant": variant, "seed": args.seeds[0]}
        metrics_path = args.project / variant / f"seed_{args.seeds[0]}" / "evaluation_metrics.json"
        if metrics_path.is_file(): record.update(json.loads(metrics_path.read_text(encoding="utf-8")))
        for window, (low, high) in WINDOWS.items():
            chosen = [r for r in rows if low <= int(r["epoch"]) <= high]
            if not chosen and rows: chosen = rows[-min(10, len(rows)):]
            record[f"{window}_epoch_start"] = min((r["epoch"] for r in chosen), default=None); record[f"{window}_epoch_end"] = max((r["epoch"] for r in chosen), default=None)
            for key in SUPPORT_METRICS:
                values = [float(r[key]) for r in chosen if key in r]
                if values: record[f"{window}_{key}"] = statistics.fmean(values)
        summary.append(record)
    args.project.mkdir(parents=True, exist_ok=True)
    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        fields = sorted({k for row in rows for k in row}, key=lambda k: (k not in {"variant", "seed", "epoch"}, k))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or ["variant", "seed"]); writer.writeheader(); writer.writerows(rows)
    write_csv(args.project / "support_screen_epoch_metrics.csv", all_rows); write_csv(args.project / "support_screen_comparison.csv", summary)
    (args.project / "support_screen_results.json").write_text(json.dumps({"summary": summary, "epoch_metrics": all_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "epoch_metrics": all_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)); parser.add_argument("--seeds", nargs="+", type=int, default=[42]); parser.add_argument("--split-seed", type=int, default=42); parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData"); parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets"); parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}"); parser.add_argument("--pretrained", default="yolov8n.pt"); parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--imgsz", type=int, default=512); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--patience", type=int, default=20); parser.add_argument("--device", default="cuda"); parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--aggregate-only", action="store_true"); return parser.parse_args(argv)


def main() -> None:
    args = parse_args(); args.project = args.project.resolve()
    if len(args.seeds) != 1 or args.seeds[0] != 42: raise ValueError("support screen requires exactly --seeds 42")
    if args.aggregate_only: aggregate_outputs(args); return
    preflight = {"source": source_preflight(args)}
    try: preflight["model"] = model_preflight(args)
    except (ImportError, ModuleNotFoundError) as error: preflight["model"] = {"status": "blocked_missing_runtime_dependency", "error": str(error)}
    print_preflight(preflight, args)
    if args.preflight_only: print(json.dumps(preflight, indent=2, sort_keys=True)); return
    workflow.EXPERIMENT = EXPERIMENT; workflow.VARIANTS = VARIANTS; workflow.model_for = model_for; workflow.train_kwargs = train_kwargs
    data_yaml = workflow.prepare_fixed_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            args._active_variant = variant; run_dir = workflow.train(variant, seed, data_yaml, True, args); metrics = _WORKFLOW_EVALUATE(run_dir, data_yaml, args)
            (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate_outputs(args)


if __name__ == "__main__": main()
