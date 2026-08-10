#!/usr/bin/env python3
"""Train, evaluate, and upload the YOLOv8n LEVIR FPN-only P2-only Amplitude Calibration experiments."""

import os
import argparse
from pathlib import Path
import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_amplitude_calibration"

# Set experiment name and destination repo
workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-amplitude-calibration-seed42"

# Set variant configuration mapping
workflow.VARIANTS = {
    "amplitude_calibrator": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_amplitude_calibrator.yaml",
    "amplitude_perturbation": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_amplitude_perturbation.yaml",
    "calibrator_perturbation": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_calibrator_perturbation.yaml",
}

_baseline_train_kwargs = workflow.train_kwargs
_parse_args = workflow.parse_args

def parse_args(argv=None):
    args = _parse_args(argv)
    args.runner = Path(__file__)
    # Default to seed 42 to match user's prior FPN-only attention settings,
    # but allow CLI override.
    if not argv:
        args.seeds = [42]
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
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)

workflow.parse_args = parse_args
workflow.main = main

if __name__ == "__main__":
    workflow.main()
