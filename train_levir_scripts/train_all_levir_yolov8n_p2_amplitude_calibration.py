#!/usr/bin/env python3
"""Train, evaluate, and upload the YOLOv8n LEVIR FPN-only P2-only Amplitude Calibration experiments."""

import os
import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_amplitude_calibration"
HF_REPO = "duyle2408/levir-yolov8n-p2-amplitude-calibration-seed42"

# Import workflow and register local ultralytics
import train_all_levir_yolov8n_p2_routing as workflow
workflow.local_ultralytics()

from ultralytics.nn.modules.conv import P2AmplitudeCalibrator, LearnableGlobalScalar, AmplitudePerturbation

# Set variant configuration mapping
VARIANTS = {
    "global_scalar": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_global_scalar.yaml",
    "amplitude_calibrator": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_amplitude_calibrator.yaml",
    "amplitude_perturbation": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_amplitude_perturbation.yaml",
    "calibrator_perturbation": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_calibrator_perturbation.yaml",
}

# Override workflow VARIANTS so workflow functions find the configs
workflow.VARIANTS = VARIANTS

def verify_topology(model, variant):
    """Ensure that the model topology parsed from the YAML matches expectations."""
    if variant == "global_scalar":
        assert isinstance(model[19], LearnableGlobalScalar), f"Topology mismatch: expected LearnableGlobalScalar at layer 19 for {variant}"
        assert model[20].f == [19]
    elif variant == "amplitude_calibrator":
        assert isinstance(model[19], P2AmplitudeCalibrator), f"Topology mismatch: expected P2AmplitudeCalibrator at layer 19 for {variant}"
        assert model[20].f == [19]
    elif variant == "amplitude_perturbation":
        assert isinstance(model[19], AmplitudePerturbation), f"Topology mismatch: expected AmplitudePerturbation at layer 19 for {variant}"
        assert model[20].f == [19]
    elif variant == "calibrator_perturbation":
        assert isinstance(model[19], P2AmplitudeCalibrator), f"Topology mismatch: expected P2AmplitudeCalibrator at layer 19 for {variant}"
        assert isinstance(model[20], AmplitudePerturbation), f"Topology mismatch: expected AmplitudePerturbation at layer 20 for {variant}"
        assert model[21].f == [20]

def model_for(variant: str, pretrained: str):
    from ultralytics import YOLO
    model = YOLO(VARIANTS[variant])
    # Run topology verification
    verify_topology(model.model.model, variant)
    model.load(pretrained, smart_transfer=True)
    return model

# Register custom model loader in workflow
workflow.model_for = model_for

def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    """Evaluate YOLOv8 model with mandatory explicit NMS IoU = 0.5."""
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        try:
            metrics = json.loads(output.read_text(encoding="utf-8"))
            if metrics.get("nms_iou") == 0.5:
                return metrics
        except Exception:
            pass
            
    from ultralytics import YOLO

    metrics = {}
    for split in ("val", "test"):
        # Explicitly use NMS IoU 0.50
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
            iou=0.5
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
        
    metrics["nms_iou"] = 0.5
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics

# Register custom evaluation in workflow
workflow.evaluate = evaluate

def parse_args(argv=None):
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
    parser.add_argument("--patience", type=int, default=0)  # default patience=0 for direct comparability
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    return args

def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    
    # Initialize uploader first to fail-fast if HF_TOKEN is missing
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
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)

if __name__ == "__main__":
    main()
