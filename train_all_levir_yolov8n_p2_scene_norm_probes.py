#!/usr/bin/env python3
"""Train/evaluate/upload seed-42 P2 scene-normalization probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_all_levir_yolov8n_p2_feature_probes as workflow

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
_BASE_PARSE_ARGS = workflow.parse_args
VARIANTS = {
    "mean_center": CONFIG_ROOT / "yolov8n_p2_fpn_only_probe_mean_center.yaml",
    "location_rms": CONFIG_ROOT / "yolov8n_p2_fpn_only_probe_location_rms.yaml",
    "std_norm": CONFIG_ROOT / "yolov8n_p2_fpn_only_probe_std_norm.yaml",
    "global_add": CONFIG_ROOT / "yolov8n_p2_fpn_only_probe_global_add.yaml",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _BASE_PARSE_ARGS(argv)
    parser.project = ROOT / "runs/levir_yolov8n_p2_scene_norm_probes"
    parser.hf_repo_id = "duyle2408/levir-yolov8n-p2-scene-norm-probes-seed42"
    parser.variants = list(VARIANTS)
    return parser


def main() -> None:
    workflow.VARIANTS = VARIANTS
    workflow.parse_args = parse_args
    workflow.main()


if __name__ == "__main__":
    main()
