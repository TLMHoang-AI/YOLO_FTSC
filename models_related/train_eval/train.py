#!/usr/bin/env python3
"""Train Ultralytics YOLO baselines or custom YAML models on CPU by default."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from misc_varroa.train_varroa.prepare_dataset import prepare_dataset


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def prefer_local_ultralytics() -> None:
    local_ultralytics = Path(__file__).resolve().parents[1] / "ultralytics"
    if (local_ultralytics / "ultralytics" / "__init__.py").is_file():
        sys.path.insert(0, str(local_ultralytics))


def load_model(args: argparse.Namespace):
    prefer_local_ultralytics()
    from ultralytics import YOLO

    if args.resume:
        return YOLO(args.resume)

    if args.model_yaml:
        model = YOLO(args.model_yaml)
        if args.pretrained:
            model.load(args.pretrained)
        return model

    return YOLO(args.weights)


def check_resume_checkpoint(path: str) -> None:
    """Fail early when a checkpoint cannot be resumed with optimizer state."""
    import torch

    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [key for key in ("epoch", "optimizer", "train_args") if ckpt.get(key) is None]
    if missing:
        raise RuntimeError(
            f"Checkpoint is not resumable: {ckpt_path} is missing {', '.join(missing)}. "
            "Use an incomplete-run weights/last.pt with optimizer state, or train from it as --weights without --resume."
        )


def train(args: argparse.Namespace) -> None:
    data_yaml = Path(args.data_yaml) if args.data_yaml else prepare_dataset(args.root, args.yolo_dir, limit=args.limit)
    if args.resume:
        check_resume_checkpoint(args.resume)
    model = load_model(args)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        bbox_iou_loss=args.bbox_iou_loss,
        wiou_monotonous=args.wiou_monotonous,
    )
    if args.resume:
        train_kwargs["resume"] = args.resume
    model.train(**train_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8/YOLOv10 Varroa models with Ultralytics.")
    parser.add_argument("--root", default=".", help="Repo root containing train/val/test folders.")
    parser.add_argument("--yolo-dir", default="yolo_related/datasets/varroa_yolo")
    parser.add_argument("--data-yaml", default=None, help="Use an existing Ultralytics dataset YAML.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images per split during conversion.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--weights", default="yolov8n.pt", help="Baseline pretrained model, e.g. yolov8n.pt.")
    model_group.add_argument("--model-yaml", default=None, help="Custom architecture YAML.")
    model_group.add_argument("--pretrained", default=None, help="Weights to partial-load into --model-yaml.")
    model_group.add_argument(
        "--resume",
        default=None,
        help="Resume an incomplete Ultralytics training run from weights/last.pt with optimizer state.",
    )

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--epochs", type=int, default=100)
    train_group.add_argument("--imgsz", type=int, default=640)
    train_group.add_argument("--batch-size", type=int, default=4)
    train_group.add_argument("--project", default="yolo_related/runs/train")
    train_group.add_argument("--name", default="yolov8n_varroa_cpu")
    train_group.add_argument("--patience", type=int, default=20)
    train_group.add_argument("--workers", type=int, default=2)
    train_group.add_argument("--device", default="cpu", help="Defaults to CPU because this project target has no GPU.")
    train_group.add_argument(
        "--bbox-iou-loss",
        default="ciou",
        choices=("ciou", "wiou", "iou", "giou", "diou", "eiou", "siou"),
        help="Bounding-box regression IoU loss. Use 'wiou' to enable Wise-IoU.",
    )
    train_group.add_argument(
        "--wiou-monotonous",
        action="store_true",
        help="Use monotonic Wise-IoU focusing. Default is non-monotonic when --bbox-iou-loss wiou is selected.",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
