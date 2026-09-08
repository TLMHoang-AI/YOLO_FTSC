#!/usr/bin/env python3
"""Prepare the matched no-Mosaic FTSC N0--N3 first-screen suite.

No training is performed by this task. The runner is deliberately explicit so a
future launch cannot accidentally compare a branch against historical Y4+Mosaic.
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

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_nomosaic_nextgen"
FITNESS_METRIC = "map50_95"
PROTOCOL_VERSION = "ftsc_nomosaic_nextgen_v1"
VARIANTS = {
    "N0_y4_nomosaic": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_n0.yaml",
    "N1_y4_nomosaic_adaptive_zoom": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_n1_adaptive_zoom.yaml",
    "N2_y4_nomosaic_support_assignment": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_n2_support_assignment.yaml",
    "N3_y4_nomosaic_localization_distill": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_n3_localization_distill.yaml",
}
VARIANT_FLAGS = {
    "N0_y4_nomosaic": {"adaptive_zoom": False, "support_assignment": False, "localization_distill": False},
    "N1_y4_nomosaic_adaptive_zoom": {"adaptive_zoom": True, "support_assignment": False, "localization_distill": False},
    "N2_y4_nomosaic_support_assignment": {"adaptive_zoom": False, "support_assignment": True, "localization_distill": False},
    "N3_y4_nomosaic_localization_distill": {"adaptive_zoom": False, "support_assignment": False, "localization_distill": True},
}
ACTIVE_TEACHER: Path | None = None

workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.VARIANTS = VARIANTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model) -> str:
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    variant = getattr(args, "_active_variant", "N0_y4_nomosaic")
    flags = VARIANT_FLAGS[variant]
    kwargs: dict[str, object] = {
        "data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch_size, "device": args.device, "workers": args.workers,
        "patience": args.patience, "seed": seed, "deterministic": True,
        "amp": amp, "plots": False, "optimizer": "auto", "cos_lr": False,
        "lrf": 0.01, "fitness_metric": FITNESS_METRIC,
        # Critical generation boundary: Mosaic is disabled from epoch zero.
        "mosaic": 0.0, "close_mosaic": 0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "rect": False,
        "adaptive_zoom": flags["adaptive_zoom"],
        "adaptive_zoom_probability": 0.5,
        "adaptive_zoom_target_min_side": 16.0,
        "adaptive_zoom_max_factor": 2.0,
        "adaptive_zoom_visible_ratio": 0.5,
        "adaptive_zoom_center_jitter": 0.05,
        "support_assignment": flags["support_assignment"],
        "support_assignment_topk": 5,
        "localization_distill": flags["localization_distill"],
        "localization_distill_lambda": 0.1,
        "localization_distill_teacher": str(args.teacher) if args.teacher else "",
    }
    return kwargs


def _head_preflight(model) -> dict[str, object]:
    from ultralytics.nn.modules import DFL, Detect, P2OffsetRegression
    from types import SimpleNamespace
    from ultralytics.engine.trainer import BaseTrainer

    head = model.model[-1]
    _require(isinstance(head, Detect), f"expected Detect head, got {type(head).__name__}")
    _require(head.stride.tolist() == [4.0, 8.0, 16.0, 32.0], f"expected P2-P5, got {head.stride.tolist()}")
    _require(head.ftsc_calibrator is not None and head.ftsc_calibrator.policy == "f5", "FTSC F5 is not active")
    calibrator = head.ftsc_calibrator
    _require(calibrator.fixed_strengths == {"dfl_distribution": 1.0}, "DFL FTSC strength is not fixed at 1")
    _require("dfl_distribution" not in calibrator.strength_logits, "fixed DFL strength became trainable")
    _require(calibrator.strength_logits["position_gaussian"].requires_grad, "Position strength is frozen")
    _require(calibrator.evidence_names == ("position_gaussian", "dfl_distribution"), "Y4 evidence changed")
    _require(calibrator.providers["dfl_distribution"].detach, "DFL evidence must be detached")
    _require(calibrator.dfl_apply_cls and not calibrator.dfl_apply_box and not calibrator.dfl_apply_dfl, "DFL route changed")
    _require(calibrator.apply_cls and calibrator.apply_box and calibrator.apply_dfl, "Position route changed")
    projection_ids = {id(p) for module in model.modules() if isinstance(module, DFL) for p in module.parameters(recurse=True)}
    projection = {name: p for name, p in model.named_parameters() if id(p) in projection_ids}
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(freeze=None)
    trainer._freeze_layers([])
    _require(projection and all(not p.requires_grad for p in projection.values()), "actual DFL projection is not frozen")
    return {
        "head_strides": list(head.stride.tolist()),
        "ftsc_policy": calibrator.policy,
        "ftsc_evidence": list(calibrator.evidence_names),
        "dfl_strength_fixed": True,
        "dfl_strength_effective": float(calibrator.strength("dfl_distribution", calibrator.strength_logits["position_gaussian"]).detach().cpu()),
        "position_strength_trainable": True,
        "dfl_projection_parameters": sorted(projection),
    }


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    if variant == "N3_y4_nomosaic_localization_distill":
        if ACTIVE_TEACHER is None or not ACTIVE_TEACHER.is_file():
            raise FileNotFoundError("N3 requires an existing --teacher checkpoint")
        teacher = YOLO(ACTIVE_TEACHER).model
        teacher_head = teacher.model[-1]
        student_head = model.model.model[-1]
        _require(teacher_head.stride.tolist() == student_head.stride.tolist(), "teacher/student strides are incompatible")
        _require(teacher_head.reg_max == student_head.reg_max, "teacher/student DFL bins are incompatible")
        teacher.to(next(model.model.parameters()).device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        # Bypass nn.Module registration: teacher must never enter the student
        # optimizer or checkpoint/state hash.
        object.__setattr__(model.model, "_ftsc_localization_teacher", teacher)
    return model


def source_preflight(args: argparse.Namespace) -> dict[str, object]:
    import yaml
    payloads = {}
    for variant, path in VARIANTS.items():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payloads[variant] = payload
        _require(payload["ftsc"]["fixed_strengths"] == {"dfl_distribution": 1.0}, f"{variant}: fixed DFL contract missing")
    trainer_source = ROOT.parent / "models_related/ultralytics/ultralytics/engine/trainer.py"
    augment_source = ROOT.parent / "models_related/ultralytics/ultralytics/data/augment.py"
    loss_source = ROOT.parent / "models_related/ultralytics/ultralytics/utils/loss.py"
    trainer_text, augment_text, loss_text = (p.read_text(encoding="utf-8") for p in (trainer_source, augment_source, loss_source))
    _require('always_freeze_names = [".dfl"]' not in trainer_text, "broad .dfl freeze was restored")
    _require("class AdaptiveZoom" in augment_text and "p=hyp.mosaic" in augment_text, "zoom/mosaic hooks missing")
    _require("SupportAwareTaskAlignedAssigner" in loss_text, "support assignment hook missing")
    _require("localization_distill_enabled" in loss_text, "localization distillation hook missing")
    return {
        "variants": list(VARIANTS),
        "configs": {name: str(path) for name, path in VARIANTS.items()},
        "config_sha256": {name: sha256(path) for name, path in VARIANTS.items()},
        "epochs": args.epochs, "patience": args.patience,
        "optimizer_requested": "auto", "optimizer_expected": "AdamW",
        "lr0": 0.002, "lrf": 0.01, "scheduler": "linear",
        "fitness_metric": FITNESS_METRIC, "mosaic": 0.0, "close_mosaic": 0,
        "adaptive_zoom": {"target_min_side": 16.0, "p": 0.5, "z_max": 2.0, "visible_ratio": 0.5},
        "assignment": {"variant": "standard TAL posterior support filter", "support_topk": 5},
        "localization_distillation": {"enabled_only_in": "N3", "lambda": 0.1,
                                       "teacher": str(args.teacher) if args.teacher else None,
                                       "teacher_sha256": sha256(args.teacher) if args.teacher and args.teacher.is_file() else None},
        "freeze": {"actual_dfl_projection_only": True, "broad_name_freeze": False},
        "sources": {"trainer_sha256": sha256(trainer_source), "augment_sha256": sha256(augment_source), "loss_sha256": sha256(loss_source)},
    }


def model_preflight(args: argparse.Namespace) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    import torch
    reports, hashes = {}, {}
    for variant, config in VARIANTS.items():
        torch.manual_seed(0)
        model = DetectionModel(config, verbose=False)
        reports[variant] = _head_preflight(model)
        hashes[variant] = state_hash(model)
    expected = hashes["N0_y4_nomosaic"]
    _require(all(value == expected for value in hashes.values()), f"student initialization hashes diverged: {hashes}")
    return {"variant_reports": reports, "student_init_hashes": hashes, "init_hash_equal": True}


def print_preflight_table(report: dict[str, object], args: argparse.Namespace) -> None:
    """Print the compact human-readable contract before any future training."""
    source = report["source"]
    model = report.get("model", {})
    model_reports = model.get("variant_reports", {}) if isinstance(model, dict) else {}
    hashes = model.get("student_init_hashes", {}) if isinstance(model, dict) else {}
    columns = ("variant", "yaml", "seed", "split_seed", "epochs", "patience", "optimizer", "lr0/lrf",
               "fitness", "mosaic", "close_mosaic", "Position", "DFL fixed", "zoom", "assignment", "LD", "teacher", "student_init_hash")
    print("\t".join(columns))
    for variant in args.variants:
        flags = VARIANT_FLAGS[variant]
        print("\t".join(map(str, (
            variant, Path(source["configs"][variant]).name, args.seeds, args.split_seed,
            args.epochs, args.patience, "auto->AdamW", "0.002/0.01", FITNESS_METRIC,
            source["mosaic"], source["close_mosaic"],
            model_reports.get(variant, {}).get("position_strength_trainable", "pending"),
            model_reports.get(variant, {}).get("dfl_strength_effective", 1.0),
            flags["adaptive_zoom"], flags["support_assignment"], flags["localization_distill"],
            str(args.teacher) if variant.startswith("N3") and args.teacher else ("required" if variant.startswith("N3") else "-"),
            hashes.get(variant, "pending"),
        ))))


def _read_mechanism(run_dir: Path, variant: str, seed: int) -> list[dict[str, object]]:
    path = run_dir / "results.csv"
    if not path.is_file():
        return [{"variant": variant, "seed": seed}]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [{"variant": variant, "seed": seed}]
    names = ("zoom_", "support_", "ld_", "ftsc_positive", "ftsc_mean_positives", "ftsc_single_positive")
    output = []
    for row in rows:
        record = {"variant": variant, "seed": seed}
        if "epoch" in row and row["epoch"] not in ("", None):
            record["epoch"] = int(float(row["epoch"]))
        record.update({key: float(value) for key, value in row.items() if any(key.startswith(prefix) for prefix in names) and value not in ("", None)})
        output.append(record)
    return output


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    metrics = workflow.evaluate(run_dir, data_yaml, args)
    metrics.update({"protocol/version": PROTOCOL_VERSION, "protocol/fitness_metric": FITNESS_METRIC,
                    "protocol/best_checkpoint_metric": FITNESS_METRIC, "protocol/nms_iou": 0.5,
                    "protocol/mosaic": 0.0, "protocol/close_mosaic": 0})
    path = run_dir / "evaluation_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_outputs(args: argparse.Namespace, preflight: dict[str, object]) -> None:
    rows, mechanism = [], []
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = args.project / variant / f"seed_{seed}"
            metrics_path = run_dir / "evaluation_metrics.json"
            if metrics_path.is_file():
                rows.append({"variant": variant, "seed": seed, **json.loads(metrics_path.read_text())})
            mechanism.extend(_read_mechanism(run_dir, variant, seed))
    if rows:
        fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
        with (args.project / "ftsc_nomosaic_nextgen_summary_runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
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
        afields = sorted({key for row in aggregate for key in row}, key=lambda key: (key not in {"variant", "runs"}, key))
        with (args.project / "ftsc_nomosaic_nextgen_summary_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=afields); writer.writeheader(); writer.writerows(aggregate)
    args.project.mkdir(parents=True, exist_ok=True)
    mfields = sorted({key for row in mechanism for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
    with (args.project / "ftsc_nomosaic_nextgen_mechanism_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mfields); writer.writeheader(); writer.writerows(mechanism)
    (args.project / "ftsc_nomosaic_nextgen_results.json").write_text(json.dumps({"preflight": preflight, "runs": rows, "mechanism": mechanism}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT_SLUG}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--teacher", type=Path, default=None, help="explicit N3 teacher checkpoint; never auto-selected")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    args.fitness_metric = FITNESS_METRIC
    return args


def main() -> None:
    global ACTIVE_TEACHER
    args = parse_args()
    args.project = args.project.resolve()
    ACTIVE_TEACHER = args.teacher.resolve() if args.teacher else None
    preflight = {"source": source_preflight(args)}
    if args.preflight_only:
        try:
            preflight["model"] = model_preflight(args)
        except (ImportError, ModuleNotFoundError) as error:
            preflight["model"] = {"status": "blocked_missing_runtime_dependency", "error": str(error)}
        print_preflight_table(preflight, args)
        print(json.dumps(preflight, indent=2, sort_keys=True)); return
    if "N3_y4_nomosaic_localization_distill" in args.variants and not ACTIVE_TEACHER:
        raise ValueError("N3 was requested without --teacher; pass an explicit compatible checkpoint")
    workflow.model_for = model_for
    workflow.evaluate = evaluate
    workflow.train_kwargs = train_kwargs
    args.data_root, args.dataset_root = args.data_root.resolve(), args.dataset_root.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    preflight["model"] = model_preflight(args)
    print_preflight_table(preflight, args)
    for seed in args.seeds:
        for variant in args.variants:
            args._active_variant = variant
            run_dir = workflow.train(variant, seed, data_yaml, True, args)
            evaluate(run_dir, data_yaml, args)
            write_outputs(args, preflight)


if __name__ == "__main__":
    main()
