#!/usr/bin/env python3
"""Train, evaluate, and upload the YOLOv8n LEVIR PAN-to-P3 P1-GER experiments."""

import os
from pathlib import Path
import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_pan_p3_p1ger"

# Set experiment name and destination repo
workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-pan-p3-p1ger-3seed"

# Set variant configuration mapping
workflow.VARIANTS = {
    "pan_p3_p1ger": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_pan_p3_p1ger.yaml"
}

_baseline_train_kwargs = workflow.train_kwargs
_parse_args = workflow.parse_args

def parse_args(argv=None):
    args = _parse_args(argv)
    args.runner = Path(__file__)
    return args

def main() -> None:
    args = parse_args()
    args.no_upload = True
    
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        for variant in args.variants:
            amp[variant] = workflow.smoke(variant, data_yaml, args)
            
    if args.smoke_only:
        return
        
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)

workflow.parse_args = parse_args
workflow.main = main

if __name__ == "__main__":
    workflow.main()
