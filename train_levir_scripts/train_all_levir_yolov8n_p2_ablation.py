#!/usr/bin/env python3
"""Train the YOLOv8n P2 with WIoU loss and our custom ablation variants."""

import os
from pathlib import Path
import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_ablation"

# We define the 3 ablation variants
workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-ablation"

workflow.VARIANTS = {
    "small_weight": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml",
    "partial_clip": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml",
    "small_weight_partial_clip": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml"
}

_baseline_train_kwargs = workflow.train_kwargs
_parse_args = workflow.parse_args

def wiou_train_kwargs(args, data_yaml, seed, amp):
    kwargs = _baseline_train_kwargs(args, data_yaml, seed, amp)
    kwargs.update(
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
    )
    os.environ["YOLO_VARIANT"] = getattr(args, "current_variant", "")
    return kwargs

def parse_args(argv=None):
    args = _parse_args(argv)
    args.runner = Path(__file__)
    return args

def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    
    args.no_upload = True
    
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        for variant in args.variants:
            args.current_variant = variant
            os.environ["YOLO_VARIANT"] = variant
            amp[variant] = workflow.smoke(variant, data_yaml, args)
            
    if args.smoke_only:
        return
        
    for seed in args.seeds:
        for variant in args.variants:
            args.current_variant = variant
            os.environ["YOLO_VARIANT"] = variant
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)

workflow.train_kwargs = wiou_train_kwargs
workflow.parse_args = parse_args
workflow.main = main

if __name__ == "__main__":
    workflow.main()
