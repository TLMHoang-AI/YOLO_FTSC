#!/usr/bin/env python3
"""Evaluate an Ultralytics YOLO checkpoint on the standalone Varroa YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from PIL import Image

from misc_varroa.train_varroa.prepare_dataset import (
    clamp_xyxy_boxes,
    label_path_for,
    prepare_dataset,
    read_original_boxes,
    resolve_repo_root,
)


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def prefer_local_ultralytics() -> None:
    local_ultralytics = Path(__file__).resolve().parents[1] / "ultralytics"
    if (local_ultralytics / "ultralytics" / "__init__.py").is_file():
        sys.path.insert(0, str(local_ultralytics))


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-12)


def match_counts(pred_boxes: list[list[float]], gt_boxes: list[list[float]], iou_threshold: float) -> tuple[int, int, int]:
    matched_gt: set[int] = set()
    true_positive = 0
    for pred in pred_boxes:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_idx = idx
                best_iou = iou
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            true_positive += 1
    false_positive = len(pred_boxes) - true_positive
    false_negative = len(gt_boxes) - true_positive
    return true_positive, false_positive, false_negative


def model_profile(model, imgsz: int) -> tuple[int, float]:
    """Return parameter count and GFLOPs for an Ultralytics model."""
    torch_model = getattr(model, "model", None)
    if torch_model is None:
        raise TypeError("Expected an Ultralytics PyTorch model with a .model attribute.")
    params = sum(p.numel() for p in torch_model.parameters())
    from ultralytics.utils.torch_utils import get_flops

    gflops = get_flops(torch_model, imgsz)
    return params, gflops


def run_eval(args: argparse.Namespace) -> None:
    prefer_local_ultralytics()
    from ultralytics import YOLO

    data_yaml = Path(args.data_yaml) if args.data_yaml else prepare_dataset(args.root, args.yolo_dir, limit=args.limit)
    model = YOLO(args.weights)
    params, gflops = model_profile(model, args.imgsz)

    val_metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        conf=args.map_conf,
        iou=args.nms_iou,
        project=args.project,
        name=f"{args.name}_val",
    )

    root = resolve_repo_root(args.root)
    split_root = root / args.split
    image_dir = split_root / "videos"
    label_dir = split_root / "labels"
    images = sorted(image_dir.rglob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    inference_seconds = 0.0

    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = clamp_xyxy_boxes(read_original_boxes(label_path_for(image_path, image_dir, label_dir)), width, height)
        start_time = time.perf_counter()
        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            device=args.device,
            verbose=False,
        )[0]
        inference_seconds += time.perf_counter() - start_time

        pred_boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []
        pred_scores = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else []
        tp, fp, fn = match_counts(pred_boxes, gt_boxes, args.match_iou)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        rows.append(
            {
                "path": str(image_path),
                "gt": len(gt_boxes),
                "pred": len(pred_boxes),
                "max_score": max(pred_scores) if pred_scores else "",
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fps = len(images) / inference_seconds if inference_seconds > 0 else ""
    avg_ms = (inference_seconds / len(images) * 1000.0) if images and inference_seconds > 0 else ""

    per_image_path = out_dir / f"{args.split}_per_image.csv"
    summary_path = out_dir / f"{args.split}_summary.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "gt", "pred", "max_score", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "weights": args.weights,
        "conf": args.conf,
        "match_iou": args.match_iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "params": params,
        "gflops": gflops,
        "fps": fps,
        "avg_inference_ms": avg_ms,
        **totals,
        "map50": getattr(val_metrics.box, "map50", ""),
        "map50_95": getattr(val_metrics.box, "map", ""),
    }
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(
        f"Eval {args.split}: mAP50 {summary['map50']} | mAP50-95 {summary['map50_95']} | "
        f"P {precision:.4f} R {recall:.4f} F1 {f1:.4f} | TP {totals['tp']} FP {totals['fp']} FN {totals['fn']} | "
        f"Params {params} GFLOPs {gflops} FPS {fps if fps == '' else f'{fps:.2f}'}"
    )
    print(f"Wrote {per_image_path} and {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO Varroa checkpoint on CPU by default.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--yolo-dir", default="yolo_related/datasets/varroa_yolo")
    parser.add_argument("--data-yaml", default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--map-conf", type=float, default=0.001)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--project", default="yolo_related/runs/eval")
    parser.add_argument("--name", default="varroa_eval")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional max images per split for smoke tests.")
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()
