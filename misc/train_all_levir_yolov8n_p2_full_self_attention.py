#!/usr/bin/env python3
"""Train the seed-42 P2-only full-resolution self-attention screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import misc.train_all_levir_yolov8n_p2_patch_kvca as workflow


ROOT = Path(__file__).resolve().parent
VARIANT = "full_self_attention_p2"
CONFIG = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_full_self_attention.yaml"
workflow.VARIANTS = {VARIANT: CONFIG}
workflow.__file__ = __file__


def model_for(variant: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import FullSelfAttention

    model = YOLO(CONFIG)
    model.load(pretrained, smart_transfer=True)
    attention = model.model.model[19]
    head = model.model.model[20]
    if not isinstance(attention, FullSelfAttention) or attention.num_heads != 4:
        raise TypeError(f"Unexpected layer 19: {type(attention).__name__}")
    if head.f != [19] or head.nl != 1 or head.stride.tolist() != [4.0]:
        raise ValueError(f"Expected Detect([19]) at stride 4, got f={head.f}, stride={head.stride}")
    return model


def write_manifest(model, variant: str, run_dir: Path, args: argparse.Namespace) -> None:
    attention = model.model.model[19]
    head = model.model.model[20]
    payload = {
        "variant": variant,
        "seed": 42,
        "split_seed": 42,
        "config": CONFIG.name,
        "attention_class": type(attention).__name__,
        "connectivity": "full-resolution P2 self-attention",
        "num_heads": attention.num_heads,
        "channels": attention.c2,
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "amp": True,
        "nms_iou": 0.5,
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=[VARIANT], default=[VARIANT])
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_full_self_attention")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-full-self-attention-seed42")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


workflow.model_for = model_for
workflow.write_manifest = write_manifest
workflow.parse_args = parse_args


if __name__ == "__main__":
    workflow.main()
