#!/usr/bin/env python3
"""Train the YOLOv8n P2 with WIoU loss."""

from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_wiou"
VARIANT = "p2_wiou"
workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-wiou"
workflow.VARIANTS = {
    VARIANT: ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml"
}
_baseline_train_kwargs = workflow.train_kwargs
_parse_args = workflow.parse_args


def wiou_train_kwargs(args, data_yaml, seed, amp):
    kwargs = _baseline_train_kwargs(args, data_yaml, seed, amp)
    kwargs.update(
        bbox_iou_loss="wiou",
        wiou_monotonous=False,
    )
    return kwargs


def parse_args(argv=None):
    args = _parse_args(argv)
    args.runner = Path(__file__)
    return args


workflow.train_kwargs = wiou_train_kwargs
workflow.parse_args = parse_args


if __name__ == "__main__":
    workflow.main()
