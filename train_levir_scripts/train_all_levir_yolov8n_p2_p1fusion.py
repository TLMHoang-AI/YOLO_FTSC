#!/usr/bin/env python3
"""Train, evaluate, and upload the YOLOv8n LEVIR P1 Detail Fusion experiment."""

import os
from pathlib import Path
import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_p1fusion"

# Set experiment name and destination repo
workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-p1fusion-3seed"

# Set variant configuration mapping
workflow.VARIANTS = {
    "p1fusion": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_p1fusion.yaml"
}

_baseline_train_kwargs = workflow.train_kwargs
_parse_args = workflow.parse_args

def parse_args(argv=None):
    args = _parse_args(argv)
    args.runner = Path(__file__)
    return args

workflow.parse_args = parse_args

if __name__ == "__main__":
    workflow.main()
