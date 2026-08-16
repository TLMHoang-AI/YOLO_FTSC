#!/usr/bin/env python3
"""Train/evaluate assignment-preserving anchor-free FTSC policies on LEVIR-Ship."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_anchor_free"
HF_REPO = "duyle2408/levir-yolov8n-p2-ftsc-anchor-free"

VARIANTS = {
    "ftsc_af_y0_baseline": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y0_baseline.yaml",
    "ftsc_af_y1_e4_position": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y1_e4_position.yaml",
    "ftsc_af_y2_f5_position": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y2_f5_position.yaml",
    "ftsc_af_y3_f5_position_cls": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y3_f5_position_cls.yaml",
    "ftsc_af_y4_f5_position_dflcls": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml",
    "ftsc_af_v11_a1_dflcls_only": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_v11_a1_dflcls_only.yaml",
    "ftsc_af_v11_a2_position_task_strengths": CONFIG_ROOT
    / "yolov8n_p2_levir_ftsc_af_v11_a2_position_task_strengths.yaml",
    "ftsc_af_v11_a3_dflcls_shuffled_within_gt": CONFIG_ROOT
    / "yolov8n_p2_levir_ftsc_af_v11_a3_dflcls_shuffled_within_gt.yaml",
}
V11_VARIANTS = (
    "ftsc_af_v11_a1_dflcls_only",
    "ftsc_af_v11_a2_position_task_strengths",
    "ftsc_af_v11_a3_dflcls_shuffled_within_gt",
)
SCREEN_VARIANTS = V11_VARIANTS

workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS

_base_model_for = workflow.model_for


def model_for(variant: str, pretrained: str):
    """Build a model and fail early if its FTSC policy does not match the requested variant."""
    model = _base_model_for(variant, pretrained)
    from ultralytics.nn.modules import Detect

    head = model.model.model[-1]
    if not isinstance(head, Detect) or head.stride.tolist() != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"{variant}: expected P2-P5 Detect strides, got {type(head).__name__} {head.stride}")
    calibrator = head.ftsc_calibrator
    if variant == "ftsc_af_y0_baseline":
        if calibrator is not None:
            raise ValueError(f"{variant}: baseline unexpectedly owns an FTSC calibrator")
        return model

    expected_policy = "e4" if "_e4_" in variant else "f5"
    if calibrator is None or calibrator.policy != expected_policy:
        raise ValueError(f"{variant}: expected {expected_policy} FTSC calibrator, got {calibrator}")
    if variant.endswith("position_cls") and (calibrator.apply_box or calibrator.apply_dfl):
        raise ValueError(f"{variant}: cls-only control unexpectedly calibrates localization losses")
    if variant.endswith("position_dflcls") and "dfl_distribution" not in calibrator.evidence_names:
        raise ValueError(f"{variant}: detached DFL evidence was not constructed")
    if variant == "ftsc_af_v11_a1_dflcls_only":
        if calibrator.evidence_names != ("dfl_distribution",):
            raise ValueError(f"{variant}: expected DFL-only evidence, got {calibrator.evidence_names}")
        if calibrator.position_tasks:
            raise ValueError(f"{variant}: Position routing must be disabled when its provider is removed")
        if not calibrator.dfl_apply_cls or calibrator.dfl_apply_box or calibrator.dfl_apply_dfl:
            raise ValueError(f"{variant}: expected detached DFL evidence routed to classification only")
    elif variant == "ftsc_af_v11_a2_position_task_strengths":
        expected_keys = {
            "position_gaussian_cls",
            "position_gaussian_box",
            "position_gaussian_dfl",
            "dfl_distribution",
        }
        if calibrator.evidence_names != ("position_gaussian", "dfl_distribution"):
            raise ValueError(f"{variant}: expected the unchanged Y4 evidence bank")
        if not calibrator.position_task_specific_strength or set(calibrator.strength_logits) != expected_keys:
            raise ValueError(f"{variant}: expected task-specific Position strengths {sorted(expected_keys)}")
    elif variant == "ftsc_af_v11_a3_dflcls_shuffled_within_gt":
        if calibrator.evidence_names != ("position_gaussian", "dfl_distribution"):
            raise ValueError(f"{variant}: expected the unchanged Y4 evidence bank")
        if not calibrator.dfl_shuffle_within_gt or not calibrator.providers["dfl_distribution"].detach:
            raise ValueError(f"{variant}: expected detached within-GT DFL shuffle")
    return model


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    """Evaluate best.pt on val/test with the experiment's fixed NMS IoU."""
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        try:
            metrics = json.loads(output.read_text(encoding="utf-8"))
            if metrics.get("nms_iou") == 0.5:
                return metrics
        except Exception:
            pass

    workflow.local_ultralytics()
    from ultralytics import YOLO

    metrics = {}
    trained_model = YOLO(run_dir / "weights/best.pt")
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
    metrics["nms_iou"] = 0.5
    calibrator = getattr(trained_model.model.model[-1], "ftsc_calibrator", None)
    metrics["ftsc/enabled"] = float(calibrator is not None)
    if calibrator is not None:
        metrics.update(
            {
                "ftsc/policy_f5": float(calibrator.policy == "f5"),
                "ftsc/evidence_position": float("position_gaussian" in calibrator.evidence_names),
                "ftsc/evidence_dfl_distribution": float("dfl_distribution" in calibrator.evidence_names),
                "ftsc/log_clip": calibrator.log_clip,
                "ftsc/per_gt_norm": float(calibrator.per_gt_norm),
                "ftsc/apply_cls": float(calibrator.apply_cls),
                "ftsc/apply_box": float(calibrator.apply_box),
                "ftsc/apply_dfl": float(calibrator.apply_dfl),
                "ftsc/dfl_apply_cls": float(calibrator.dfl_apply_cls),
                "ftsc/dfl_apply_box": float(calibrator.dfl_apply_box),
                "ftsc/dfl_apply_dfl": float(calibrator.dfl_apply_dfl),
                "ftsc/position_task_specific_strength": float(calibrator.position_task_specific_strength),
                "ftsc/dfl_shuffle_within_gt": float(calibrator.dfl_shuffle_within_gt),
                "ftsc/warmup_epochs": float(calibrator.warmup_epochs),
                "ftsc/ramp_epochs": float(calibrator.ramp_epochs),
                "ftsc/strength_reg_weight": calibrator.strength_reg_weight,
            }
        )
        if calibrator.dfl_shuffle_within_gt:
            metrics["ftsc/dfl_shuffle_seed"] = int(calibrator.dfl_shuffle_seed.detach().cpu().item())
            metrics["ftsc/final_dfl_shuffle_step"] = int(calibrator.dfl_shuffle_step.detach().cpu().item())
        if "position_gaussian" in calibrator.providers:
            metrics["ftsc/position_alpha"] = calibrator.providers["position_gaussian"].alpha
        if "dfl_distribution" in calibrator.providers:
            provider = calibrator.providers["dfl_distribution"]
            metrics["ftsc/dfl_entropy_tau"] = provider.entropy_tau
            metrics["ftsc/dfl_variance_tau"] = provider.variance_tau
            metrics["ftsc/dfl_detach"] = float(provider.detach)
        if calibrator.policy == "e4":
            for name in calibrator.evidence_names:
                metrics[f"ftsc/final_strength_{name}"] = calibrator.strength_init
        else:
            final_strengths = {}
            for key, logit in calibrator.strength_logits.items():
                value = float((calibrator.strength_max * logit.sigmoid()).detach().cpu().item())
                final_strengths[key] = value
                metrics[f"ftsc/final_strength_{key}"] = value
            position_keys = [key for key in final_strengths if key.startswith("position_gaussian_")]
            if position_keys:
                metrics["ftsc/final_strength_position_gaussian"] = sum(
                    final_strengths[key] for key in position_keys
                ) / len(position_keys)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(SCREEN_VARIANTS))
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
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    return args


def main() -> None:
    workflow.model_for = model_for
    workflow.evaluate = evaluate

    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        for variant in args.variants:
            args._active_variant = variant
            amp[variant] = workflow.smoke(variant, data_yaml, args)
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            args._active_variant = variant
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
