#!/usr/bin/env python3
"""Train/evaluate YOLOv8n-P2 FTSC ablations on LEVIR ship."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc"
HF_REPO = "duyle2408/levir-yolov8n-p2-ftsc"

VARIANTS = {
    "ftsc_y0_baseline": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y0_baseline.yaml",
    "ftsc_y1_detail_gate": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y1_detail_gate.yaml",
    "ftsc_y2_detail_gate_masked": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y2_detail_gate_masked.yaml",
}

workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS

_base_train_kwargs = workflow.train_kwargs
_base_model_for = workflow.model_for


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    """Return training kwargs with FTSC-specific toggles kept explicit."""
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    variant = getattr(args, "_active_variant", "")
    if variant == "ftsc_y2_detail_gate_masked":
        kwargs["p2_detail_rec_gain"] = float(getattr(args, "p2_detail_rec_gain", 0.1))
    return kwargs


def model_for(variant: str, pretrained: str):
    """Build model and verify FTSC topology before loading pretrained weights."""
    model = _base_model_for(variant, pretrained)
    from ultralytics.nn.modules import Detect, FTSCFeatureCalibrator, MaskedP2DetailReconstruction

    layers = model.model.model
    head = layers[-1]
    if not isinstance(head, Detect) or head.stride.tolist() != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError(f"{variant}: expected P2-P5 Detect strides, got {type(head).__name__} {head.stride}")
    if variant == "ftsc_y0_baseline":
        if head.f != [19, 22, 25, 28]:
            raise ValueError(f"{variant}: unexpected Detect inputs {head.f}")
    elif variant == "ftsc_y1_detail_gate":
        if not isinstance(layers[19], FTSCFeatureCalibrator) or head.f != [19, 22, 25, 28]:
            raise ValueError(f"{variant}: FTSC layer or Detect inputs did not resolve as specified")
    elif variant == "ftsc_y2_detail_gate_masked":
        if (
            not isinstance(layers[19], FTSCFeatureCalibrator)
            or not isinstance(layers[20], MaskedP2DetailReconstruction)
            or head.f != [20, 23, 26, 29]
        ):
            raise ValueError(f"{variant}: FTSC/masked-detail topology did not resolve as specified")
    return model


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    """Evaluate with explicit NMS IoU=0.5 and refresh stale metric files if needed."""
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
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
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
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


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
    parser.add_argument("--p2-detail-rec-gain", type=float, default=0.1)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    return args


def main() -> None:
    workflow.train_kwargs = train_kwargs
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
