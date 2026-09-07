#!/usr/bin/env python3
"""Run the corrected F-learned versus intentional DFL-strength-fixed-at-one audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_corrected_controls as corrected
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_dfl_strength_audit"
HF_REPO = "duyle2408/levir-yolov8n-p2-ftsc-dfl-strength-audit"
FITNESS_METRIC = "map50"
PROTOCOL_VERSION = "ftsc_corrected_ap50_v1"
SUITE_NAME = "dfl_strength_fixed1_audit"

VARIANTS = {
    "F_learned": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml",
    "F_fixed1": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls_fixed1.yaml",
}


def _require(condition: bool, variant: str, message: str) -> None:
    if not condition:
        raise ValueError(f"{variant}: {message}")


def preflight_model(model, variant: str) -> dict[str, object]:
    """Verify the audit differs only in DFL-strength learning status."""
    if variant not in VARIANTS:
        raise KeyError(f"Unknown DFL-strength audit variant: {variant}")

    workflow.local_ultralytics()
    from ultralytics.cfg import get_cfg
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.nn.modules import DFL
    from ultralytics.nn.modules.head import P2OffsetRegression
    from ultralytics.utils.tal import TaskAlignedAssigner

    network, head = corrected._network_and_head(model)
    _require(
        head.stride.tolist() == [4.0, 8.0, 16.0, 32.0],
        variant,
        f"expected P2-P5 strides, got {head.stride.tolist()}",
    )
    _require(not head.end2end, variant, "end-to-end head must be disabled")
    _require(not bool(getattr(head, "ghm_enabled", False)), variant, "GHM must be disabled")
    _require(not bool(getattr(head, "quality_head", False)), variant, "quality head must be disabled")
    _require(not bool(getattr(head, "hbs_enabled", False)), variant, "HBS must be disabled")
    _require(not bool(getattr(head, "dfl_residual", False)), variant, "DFL residual must be disabled")
    _require(not bool(getattr(head, "p1_reg_injection", False)), variant, "P1 injection must be disabled")
    _require(not isinstance(head.cv2[0], P2OffsetRegression), variant, "P2 offset regression must be disabled")

    if not hasattr(network, "args"):
        network.args = get_cfg()
    criterion = network.init_criterion()
    _require(isinstance(criterion.assigner, TaskAlignedAssigner), variant, "expected standard TaskAlignedAssigner")
    _require(not criterion.ghm_cls, variant, "GHM classification objective must be disabled")

    calibrator = head.ftsc_calibrator
    _require(calibrator is not None and calibrator.policy == "f5", variant, "expected F5 FTSC calibrator")
    _require(
        calibrator.evidence_names == ("position_gaussian", "dfl_distribution"),
        variant,
        f"unexpected evidence bank {calibrator.evidence_names}",
    )
    _require(
        set(calibrator.providers) == {"position_gaussian", "dfl_distribution"},
        variant,
        "unexpected evidence provider",
    )
    _require(calibrator.providers["position_gaussian"].alpha == 6.0, variant, "position_alpha must equal 6")
    dfl_provider = calibrator.providers["dfl_distribution"]
    _require(dfl_provider.detach, variant, "DFL evidence must be detached")
    _require(
        dfl_provider.entropy_tau == 1.0 and dfl_provider.variance_tau == 0.0,
        variant,
        "unexpected DFL evidence parameters",
    )
    _require(calibrator.dfl_apply_cls, variant, "DFL evidence must route to classification")
    _require(
        not calibrator.dfl_apply_box and not calibrator.dfl_apply_dfl,
        variant,
        "DFL evidence must not route to box/DFL loss",
    )
    _require(
        calibrator.apply_cls and calibrator.apply_box and calibrator.apply_dfl,
        variant,
        "Position routing changed",
    )
    _require(calibrator.per_gt_norm and calibrator.log_clip == 0.35, variant, "normalization/clip changed")
    _require(
        calibrator.warmup_epochs == 5 and calibrator.ramp_epochs == 10,
        variant,
        "warmup/ramp changed",
    )
    _require(calibrator.strength_reg_weight == 1e-4, variant, "strength regularization changed")
    _require(
        calibrator.strength_init == 1.0 and calibrator.strength_max == 2.0,
        variant,
        "strength initialization/bound changed",
    )
    _require(
        calibrator.classification_replacement == "bce" and calibrator.mal_loss is None,
        variant,
        "MAL must be disabled",
    )
    _require(not calibrator.um_enabled and calibrator.um_regularizer is None, variant, "UM must be disabled")
    _require(
        not calibrator.gt_mass_rebalance_cls and not calibrator.gt_mass_shuffle,
        variant,
        "GT-mass rebalance must be disabled",
    )
    _require(not calibrator.dfl_shuffle_within_gt, variant, "DFL shuffle must be disabled")
    _require(not calibrator.position_task_specific_strength, variant, "Position strength must remain shared")

    fixed_dfl_ids = {
        id(parameter)
        for module in network.modules()
        if isinstance(module, DFL)
        for parameter in module.parameters(recurse=True)
    }
    fixed_projection = {
        name: parameter for name, parameter in network.named_parameters() if id(parameter) in fixed_dfl_ids
    }
    _require(bool(fixed_projection), variant, "fixed DFL projection was not found")
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = network
    trainer.args = SimpleNamespace(freeze=None)
    trainer._freeze_layers([])
    _require(
        all(not parameter.requires_grad for parameter in fixed_projection.values()),
        variant,
        "fixed DFL projection is trainable",
    )

    position_strength = calibrator.strength_logits.get("position_gaussian")
    _require(position_strength is not None and position_strength.requires_grad, variant, "Position strength is frozen")
    dfl_effective = float(calibrator.strength("dfl_distribution", position_strength).detach().cpu().item())

    if variant == "F_learned":
        _require(not calibrator.fixed_strengths, variant, "learned control declares a fixed strength")
        dfl_strength = calibrator.strength_logits.get("dfl_distribution")
        _require(dfl_strength is not None and dfl_strength.requires_grad, variant, "DFL strength must be learned")
        dfl_mode, dfl_trainable = "learned", True
    else:
        _require(calibrator.fixed_strengths == {"dfl_distribution": 1.0}, variant, "expected fixed DFL strength 1")
        _require("dfl_distribution" not in calibrator.strength_logits, variant, "fixed DFL strength became a Parameter")
        _require(dfl_effective == 1.0, variant, f"effective DFL strength is {dfl_effective}, expected 1")
        dfl_mode, dfl_trainable = "fixed", False

    return {
        "fixed_dfl_parameters": tuple(sorted(fixed_projection)),
        "dfl_strength_mode": dfl_mode,
        "dfl_strength_trainable": dfl_trainable,
        "dfl_strength_effective": dfl_effective,
        "position_strength_trainable": True,
    }


def model_for(variant: str, pretrained: str):
    """Build one audit model, transfer identical pretrained weights, and preflight it."""
    workflow.local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    preflight_model(model, variant)
    return model


def preflight_variant(variant: str) -> dict[str, object]:
    """Construct the exact audit YAML without loading weights, then verify it."""
    workflow.local_ultralytics()
    from ultralytics import YOLO

    return preflight_model(YOLO(VARIANTS[variant]), variant)


def _ftsc_metadata(head, preflight: dict[str, object]) -> dict[str, object]:
    calibrator = head.ftsc_calibrator
    reference = calibrator.strength_logits["position_gaussian"]
    metadata: dict[str, object] = {
        "ftsc/enabled": 1.0,
        "ftsc/policy_f5": 1.0,
        "ftsc/evidence_position": 1.0,
        "ftsc/evidence_dfl_distribution": 1.0,
        "ftsc/dfl_detach": float(calibrator.providers["dfl_distribution"].detach),
        "ftsc/dfl_apply_cls": float(calibrator.dfl_apply_cls),
        "ftsc/dfl_apply_box": float(calibrator.dfl_apply_box),
        "ftsc/dfl_apply_dfl": float(calibrator.dfl_apply_dfl),
        "ftsc/apply_cls": float(calibrator.apply_cls),
        "ftsc/apply_box": float(calibrator.apply_box),
        "ftsc/apply_dfl": float(calibrator.apply_dfl),
        "ftsc/per_gt_norm": float(calibrator.per_gt_norm),
        "ftsc/log_clip": calibrator.log_clip,
        "ftsc/warmup_epochs": float(calibrator.warmup_epochs),
        "ftsc/ramp_epochs": float(calibrator.ramp_epochs),
        "ftsc/strength_reg_weight": calibrator.strength_reg_weight,
        "ftsc/dfl_strength_mode": preflight["dfl_strength_mode"],
        "ftsc/dfl_strength_trainable": float(bool(preflight["dfl_strength_trainable"])),
        "ftsc/position_strength_trainable": float(bool(preflight["position_strength_trainable"])),
        "ftsc/final_strength_position_gaussian": float(
            calibrator.strength("position_gaussian", reference).detach().cpu().item()
        ),
        "ftsc/final_strength_dfl_distribution": float(
            calibrator.strength("dfl_distribution", reference).detach().cpu().item()
        ),
    }
    if preflight["dfl_strength_mode"] == "fixed":
        metadata["ftsc/dfl_strength_fixed_value"] = 1.0
    return metadata


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    """Evaluate best.pt on val/test and export DFL-strength audit provenance."""
    variant = getattr(args, "_active_variant", run_dir.parent.name)
    seed = int(getattr(args, "_active_seed", run_dir.name.removeprefix("seed_")))
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        try:
            cached = json.loads(output.read_text(encoding="utf-8"))
            if (
                cached.get("protocol/version") == PROTOCOL_VERSION
                and cached.get(f"suite/{SUITE_NAME}") == 1.0
                and cached.get("variant") == variant
                and int(cached.get("seed")) == seed
            ):
                return cached
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    workflow.local_ultralytics()
    from ultralytics import YOLO

    trained_model = YOLO(run_dir / "weights/best.pt")
    preflight = preflight_model(trained_model, variant)
    metrics: dict[str, object] = {}
    for split in ("val", "test"):
        result = trained_model.val(
            data=str(data_yaml),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            iou=0.5,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)

    _, head = corrected._network_and_head(trained_model)
    metrics.update(
        {
            "variant": variant,
            "seed": seed,
            "nms_iou": 0.5,
            "protocol/version": PROTOCOL_VERSION,
            "protocol/fitness_metric": FITNESS_METRIC,
            "protocol/best_checkpoint_metric": FITNESS_METRIC,
            "protocol/freeze_fix_active": 1.0,
            f"suite/{SUITE_NAME}": 1.0,
            **_ftsc_metadata(head, preflight),
            **corrected._best_validation_epochs(run_dir / "results.csv"),
        }
    )
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
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
    parser.add_argument(
        "--augmentation-policy",
        choices=("yolo_default", "mosaic_random_perspective_only"),
        default="yolo_default",
    )
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    args.fitness_metric = FITNESS_METRIC
    return args


def _configure_shared_workflow() -> None:
    workflow.EXPERIMENT = EXPERIMENT_SLUG
    workflow.HF_REPO = HF_REPO
    workflow.VARIANTS = VARIANTS
    workflow.model_for = model_for
    workflow.evaluate = evaluate


def main() -> None:
    _configure_shared_workflow()
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()

    for variant in args.variants:
        report = preflight_variant(variant)
        print(f"Preflight passed: {variant}: {report}", flush=True)

    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        for variant in args.variants:
            amp[variant] = workflow.smoke(variant, data_yaml, args)
    if args.smoke_only:
        return

    rows = []
    for seed in args.seeds:
        for variant in args.variants:
            args._active_variant = variant
            args._active_seed = seed
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            metrics = evaluate(run_dir, data_yaml, args)
            rows.append({"variant": variant, "seed": seed, "run_dir": str(run_dir), **metrics})
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)

    output = args.project / "ftsc_dfl_strength_audit_results.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
