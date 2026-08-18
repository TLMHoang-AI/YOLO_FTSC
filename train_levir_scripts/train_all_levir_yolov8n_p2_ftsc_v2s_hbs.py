#!/usr/bin/env python3
"""Run the locked V2S-A factorial: FTSC Y0/Y4 x SET-HBS off/on."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_v2s_hbs_factorial"
HF_REPO = "duyle2408/levir-yolov8n-p2-ftsc-v2s-hbs"

VARIANTS = {
    "v2s_a0_y0": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y0_baseline.yaml",
    "v2s_a1_y4": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml",
    "v2s_a2_y0_hbs": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_v2s_a2_y0_hbs.yaml",
    "v2s_a3_y4_hbs": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_v2s_a3_y4_hbs.yaml",
}
FACTORS = {
    "v2s_a0_y0": (False, False),
    "v2s_a1_y4": (True, False),
    "v2s_a2_y0_hbs": (False, True),
    "v2s_a3_y4_hbs": (True, True),
}
FACTORIAL_METRICS = {
    "val_ap50": "val/metrics/mAP50(B)",
    "val_map50_95": "val/metrics/mAP50-95(B)",
    "val_ap75": "val/metrics/mAP75(B)",
    "val_precision": "val/metrics/precision(B)",
    "val_recall": "val/metrics/recall(B)",
}

workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS
_base_model_for = workflow.model_for


def model_for(variant: str, pretrained: str):
    """Build one factorial case and reject factor leakage before training."""
    model = _base_model_for(variant, pretrained)
    from ultralytics.nn.modules import Detect

    head = model.model.model[-1]
    expected_ftsc, expected_hbs = FACTORS[variant]
    if not isinstance(head, Detect) or head.stride.tolist() != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"{variant}: expected P2-P5 Detect strides, got {type(head).__name__} {head.stride}")
    if (head.ftsc_calibrator is not None) is not expected_ftsc:
        raise ValueError(f"{variant}: FTSC factor does not match the locked factorial design")
    if bool(head.hbs_enabled) is not expected_hbs:
        raise ValueError(f"{variant}: HBS factor does not match the locked factorial design")
    if expected_ftsc:
        calibrator = head.ftsc_calibrator
        if calibrator.policy != "f5" or calibrator.evidence_names != (
            "position_gaussian",
            "dfl_distribution",
        ):
            raise ValueError(f"{variant}: expected the canonical Y4/F5 FTSC policy")
    if expected_hbs:
        if [module.kernel_size for module in head.hbs_smoothers] != [3, 5, 5, 7]:
            raise ValueError(f"{variant}: HBS kernels do not implement SET Eq. (4)")
        if head.hbs_auxiliary_weight != 1.0 or any(module.reduction != 4 for module in head.hbs_smoothers):
            raise ValueError(f"{variant}: expected SET r=4 and auxiliary weight lambda=1")
    return model


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    """Strip the train-only HBS branch, then evaluate best.pt with fixed NMS IoU."""
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        try:
            metrics = json.loads(output.read_text(encoding="utf-8"))
            has_requested_splits = not getattr(args, "evaluate_test", False) or "test/metrics/mAP50(B)" in metrics
            if (
                metrics.get("suite/v2s_hbs_factorial") == 1.0
                and metrics.get("nms_iou") == 0.5
                and has_requested_splits
            ):
                return metrics
        except Exception:
            pass

    workflow.local_ultralytics()
    from ultralytics import YOLO

    trained_model = YOLO(run_dir / "weights/best.pt")
    head = trained_model.model.model[-1]
    hbs_configured = bool(getattr(head, "hbs_enabled", False))
    hbs_parameters = sum(
        parameter.numel()
        for module in getattr(head, "hbs_smoothers", [])
        for parameter in module.parameters()
    )
    hbs_metadata = {}
    if hbs_configured:
        for level, module in enumerate(head.hbs_smoothers, start=2):
            hbs_metadata[f"hbs/p{level}_kernel_size"] = float(module.kernel_size)
            hbs_metadata[f"hbs/p{level}_reduction"] = float(module.reduction)
    if hbs_configured:
        head.strip_hbs()
    inference_parameters = sum(parameter.numel() for parameter in trained_model.model.parameters())

    metrics: dict[str, float] = {
        "suite/v2s_hbs_factorial": 1.0,
        "nms_iou": 0.5,
        "hbs/configured": float(hbs_configured),
        "hbs/train_only_parameters": float(hbs_parameters),
        "hbs/active_inference_parameters": 0.0,
        "model/inference_parameters": float(inference_parameters),
        **hbs_metadata,
    }
    splits = ("val", "test") if getattr(args, "evaluate_test", False) else ("val",)
    for split in splits:
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

    calibrator = getattr(head, "ftsc_calibrator", None)
    metrics["ftsc/enabled"] = float(calibrator is not None)
    if calibrator is not None:
        metrics.update(
            {
                "ftsc/policy_f5": float(calibrator.policy == "f5"),
                "ftsc/evidence_position": float("position_gaussian" in calibrator.evidence_names),
                "ftsc/evidence_dfl_distribution": float("dfl_distribution" in calibrator.evidence_names),
                "ftsc/log_clip": calibrator.log_clip,
                "ftsc/per_gt_norm": float(calibrator.per_gt_norm),
            }
        )
        for key, logit in calibrator.strength_logits.items():
            metrics[f"ftsc/final_strength_{key}"] = float(
                (calibrator.strength_max * logit.sigmoid()).detach().cpu().item()
            )
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_factorial_effects(args: argparse.Namespace) -> None:
    """Write paired main effects and Y4 x HBS interaction for every complete seed."""
    rows = []
    for seed in args.seeds:
        case_metrics = {}
        for variant in VARIANTS:
            path = args.project / variant / f"seed_{seed}" / "evaluation_metrics.json"
            if not path.is_file():
                break
            case_metrics[variant] = json.loads(path.read_text(encoding="utf-8"))
        if len(case_metrics) != len(VARIANTS):
            continue
        for metric, key in FACTORIAL_METRICS.items():
            values = {variant: float(payload[key]) for variant, payload in case_metrics.items()}
            a0, a1 = values["v2s_a0_y0"], values["v2s_a1_y4"]
            a2, a3 = values["v2s_a2_y0_hbs"], values["v2s_a3_y4_hbs"]
            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "a0_y0": a0,
                    "a1_y4": a1,
                    "a2_y0_hbs": a2,
                    "a3_y4_hbs": a3,
                    "hbs_standalone_a2_minus_a0": a2 - a0,
                    "hbs_incremental_a3_minus_a1": a3 - a1,
                    "ftsc_without_hbs_a1_minus_a0": a1 - a0,
                    "ftsc_with_hbs_a3_minus_a2": a3 - a2,
                    "interaction": (a3 - a2) - (a1 - a0),
                }
            )
    if rows:
        workflow.write_csv(args.project / "factorial_effects.csv", rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
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
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test only after the validation case-selection rule is locked.",
    )
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
            amp[variant] = workflow.smoke(variant, data_yaml, args)
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            write_factorial_effects(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
