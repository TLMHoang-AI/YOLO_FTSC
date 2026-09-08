#!/usr/bin/env python3
"""Historical FTSC Y4 compatibility runner (training-time only).

This runner reproduces the historical Y4 semantics in the main codebase while
keeping the corrected FTSC runner and its narrow DFL freeze unchanged.  It is
safe to use for preflight/audit without launching training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
CONFIG = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y4_legacy_compat.yaml"
VARIANT = "Y4_legacy_compat"
VARIANTS = {VARIANT: CONFIG}
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_y4_legacy_compat"
FITNESS_METRIC = "map50_95"
PROTOCOL_VERSION = "ftsc_historical_y4_legacy_compat_v1"
HISTORICAL_COMMIT = "ca48644cd1abe08315de08008d138f84565bf6f7"
HISTORICAL_TEST_AP50 = {42: 0.7878924760784733, 43: 0.7784960372913239, 44: 0.8037524648720230}

workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.VARIANTS = VARIANTS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"{VARIANT}: {message}")


def _yaml_payload() -> dict:
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def preflight_model(model) -> dict[str, object]:
    """Check the semantic closure needed by historical Y4, without training."""
    workflow.local_ultralytics()
    from ultralytics.nn.modules import DFL, Detect, P2OffsetRegression
    head = model.model[-1] if hasattr(model, "model") else model[-1]
    _require(isinstance(head, Detect), f"expected Detect head, got {type(head).__name__}")
    _require(head.stride.tolist() == [4.0, 8.0, 16.0, 32.0], f"expected P2-P5 strides, got {head.stride.tolist()}")
    _require(not head.end2end, "end2end must be disabled")
    _require(not bool(getattr(head, "ghm_enabled", False)), "GHM must be disabled")
    _require(not bool(getattr(head, "quality_head", False)), "quality head must be disabled")
    _require(not bool(getattr(head, "hbs_enabled", False)), "HBS must be disabled")
    _require(not bool(getattr(head, "dfl_residual", False)), "DFL residual must be disabled")
    _require(not isinstance(head.cv2[0], P2OffsetRegression), "P2 offset regression must be disabled")
    calibrator = head.ftsc_calibrator
    _require(calibrator is not None and calibrator.policy == "f5", "expected F5 FTSC")
    _require(calibrator.evidence_names == ("position_gaussian", "dfl_distribution"), f"unexpected evidence {calibrator.evidence_names}")
    _require(calibrator.fixed_strengths == {"dfl_distribution": 1.0}, f"unexpected fixed strengths {calibrator.fixed_strengths}")
    _require("dfl_distribution" not in calibrator.strength_logits, "fixed DFL scalar must not be a Parameter")
    _require("position_gaussian" in calibrator.strength_logits, "Position scalar is missing")
    _require(calibrator.strength_logits["position_gaussian"].requires_grad, "Position scalar must remain trainable")
    _require(calibrator.providers["dfl_distribution"].detach, "DFL evidence must be detached")
    _require(calibrator.dfl_apply_cls and not calibrator.dfl_apply_box and not calibrator.dfl_apply_dfl, "DFL routing drifted")
    _require(calibrator.apply_cls and calibrator.apply_box and calibrator.apply_dfl, "Position routing drifted")
    _require(calibrator.classification_replacement == "bce" and calibrator.mal_loss is None, "MAL must be disabled")
    _require(not calibrator.um_enabled and not calibrator.gt_mass_rebalance_cls and not calibrator.gt_mass_shuffle, "extra corrected branches enabled")
    fixed_dfl_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, DFL)
        for parameter in module.parameters(recurse=True)
    }
    projection = {name: parameter for name, parameter in model.named_parameters() if id(parameter) in fixed_dfl_ids}
    # The current main freeze implementation is intentionally narrow: only the
    # actual DFL integral projection is frozen, while FTSC Position remains live.
    from ultralytics.engine.trainer import BaseTrainer
    from types import SimpleNamespace
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(freeze=None)
    trainer._freeze_layers([])
    _require(projection and all(not p.requires_grad for p in projection.values()), "DFL projection is not frozen narrowly")
    reference = calibrator.strength_logits["position_gaussian"]
    return {
        "head_strides": list(head.stride.tolist()),
        "ftsc_policy": calibrator.policy,
        "evidence": list(calibrator.evidence_names),
        "fixed_strengths": dict(calibrator.fixed_strengths),
        "position_strength_trainable": bool(reference.requires_grad),
        "dfl_strength_trainable": False,
        "dfl_strength_effective": float(calibrator.strength("dfl_distribution", reference).detach().cpu()),
        "fixed_dfl_parameters": sorted(projection),
    }


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    preflight_model(model.model)
    return model


def legacy_scheduler_factor(epoch: float, epochs: int = 100, lrf: float = 0.01) -> float:
    """Resolved factor used by the historical linear (non-cosine) schedule."""
    return max(1.0 - float(epoch) / float(epochs), 0.0) * (1.0 - float(lrf)) + float(lrf)


def legacy_auto_optimizer(iterations: int, nc: int = 1) -> dict[str, object]:
    """Resolve Ultralytics optimizer=auto for the historical short run."""
    lr = round(0.002 * 5 / (4 + int(nc)), 6)
    if int(iterations) > 10000:
        return {"name": "MuSGD", "lr": 0.01, "beta1_or_momentum": 0.9}
    return {"name": "AdamW", "lr": lr, "beta1_or_momentum": 0.9, "beta2": 0.999}


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    # Explicit values make the historical/default augmentation contract visible
    # in run metadata instead of relying on a mutable global default.yaml.
    return {
        "data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch_size, "device": args.device, "workers": args.workers,
        "patience": args.patience, "seed": seed, "deterministic": True,
        "amp": amp, "plots": False, "optimizer": "auto", "cos_lr": False,
        "lrf": 0.01, "fitness_metric": FITNESS_METRIC,
        "mosaic": 1.0, "close_mosaic": 10, "hsv_h": 0.015, "hsv_s": 0.7,
        "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "multi_scale": 0.0,
        "rect": False,
    }


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    metrics = workflow.evaluate(run_dir, data_yaml, args)
    metrics.update({"protocol/fitness_metric": FITNESS_METRIC, "protocol/best_checkpoint_metric": FITNESS_METRIC,
                    "protocol/version": PROTOCOL_VERSION, "protocol/legacy_y4_compat": 1.0,
                    "protocol/nms_iou": 0.5})
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def source_preflight() -> dict[str, object]:
    payload = _yaml_payload()
    _require(payload["ftsc"]["fixed_strengths"] == {"dfl_distribution": 1.0}, "compat YAML must fix DFL strength at one")
    _require(payload["ftsc"]["policy"] == "f5", "compat YAML must use F5")
    _require(payload["ftsc"]["evidence"] == ["position_gaussian", "dfl_distribution"], "compat evidence drifted")
    _require(payload["ftsc"]["dfl_detach"] is True and payload["ftsc"]["dfl_apply_cls"] is True, "DFL evidence routing drifted")
    _require(payload["ftsc"]["dfl_apply_box"] is False and payload["ftsc"]["dfl_apply_dfl"] is False, "DFL must remain cls-only")
    _require(payload["ftsc"]["warmup_epochs"] == 5 and payload["ftsc"]["ramp_epochs"] == 10, "warmup/ramp drifted")
    source = ROOT.parent / "models_related/ultralytics/ultralytics/engine/trainer.py"
    text = source.read_text(encoding="utf-8")
    _require("isinstance(module, DFL)" in text, "narrow DFL freeze marker missing")
    _require("always_freeze_names = [\".dfl\"]" not in text, "broad historical .dfl freeze was restored")
    return {"config_sha256": sha256(CONFIG), "trainer_sha256": sha256(source), "historical_commit": HISTORICAL_COMMIT,
            "fitness_metric": FITNESS_METRIC, "optimizer": "auto->AdamW(lr=0.002,beta1=0.9) for nc=1 and <=10000 iterations",
            "scheduler": "linear: max(1-epoch/epochs,0)*(1-lrf)+lrf, lrf=0.01", "augmentation": "YOLO default: mosaic=1.0, close_mosaic=10"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=[VARIANT])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT_SLUG}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--source-preflight-only", action="store_true")
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    args.fitness_metric = FITNESS_METRIC
    return args


def main() -> None:
    args = parse_args()
    report = {"source": source_preflight(), "train_kwargs": train_kwargs(args, Path("<fixed_split>"), args.seeds[0], True)}
    if args.source_preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True)); return
    workflow.model_for = model_for
    workflow.evaluate = evaluate
    workflow.train_kwargs = train_kwargs
    workflow.local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    report["model"] = preflight_model(DetectionModel(CONFIG, verbose=False))
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True)); return
    args.data_root, args.dataset_root, args.project = args.data_root.resolve(), args.dataset_root.resolve(), args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, True, args)
            evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)


if __name__ == "__main__":
    main()
