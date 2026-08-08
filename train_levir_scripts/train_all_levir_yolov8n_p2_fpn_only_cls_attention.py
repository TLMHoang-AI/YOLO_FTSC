#!/usr/bin/env python3
"""Train, evaluate, and upload the YOLOv8n LEVIR FPN-only P2-only Classification-only Attention experiments."""

import os
from pathlib import Path
import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_fpn_only_cls_attention"

# Set experiment name and destination repo
workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-fpn-only-cls-attention-seed42"

# Set variant configuration mapping
workflow.VARIANTS = {
    "fpn_only_cbam_clsonly": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_clsonly.yaml",
    "fpn_only_kvca_clsonly": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_kvca_clsonly.yaml"
}

_baseline_train_kwargs = workflow.train_kwargs
_parse_args = workflow.parse_args

def parse_args(argv=None):
    args = _parse_args(argv)
    args.runner = Path(__file__)
    # Override seeds to only seed 42
    args.seeds = [42]
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
