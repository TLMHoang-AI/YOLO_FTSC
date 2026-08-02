#!/usr/bin/env python3
"""Reproduce GCTS v2 localization, selector, and gate diagnostics on a YOLO dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent


def local_ultralytics() -> None:
    package = ROOT / "models_related/ultralytics"
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)[:, None]
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)[None]
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def read_labels(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    if not rows:
        return np.empty((0, 4)), np.empty((0, 4))
    normalized = np.asarray([[float(value) for value in row[1:5]] for row in rows])
    cx, cy, bw, bh = normalized.T * np.asarray([width, height, width, height])[:, None]
    boxes = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 1)
    return boxes, normalized


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def diagnose(args: argparse.Namespace) -> dict:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import v10GCTSDetect

    model = YOLO(args.weights)
    head = model.model.model[-1]
    if not isinstance(head, v10GCTSDetect):
        raise TypeError(f"Expected v10GCTSDetect, found {type(head).__name__}")
    head.capture_diagnostics = True
    images = sorted(path for path in args.images.glob("*.png"))[: args.limit or None]
    stats: dict[str, list[float]] = defaultdict(list)
    buckets: dict[str, list[float]] = defaultdict(list)

    for image in images:
        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        height, width = result.orig_shape
        gt, normalized = read_labels(args.labels / f"{image.stem}.txt", width, height)
        pred = result.boxes.xyxy.detach().cpu().numpy()
        ious = box_iou(gt, pred)
        if ious.size:
            gt_indices, pred_indices = linear_sum_assignment(-ious)
            for gi, pi in zip(gt_indices, pred_indices):
                iou = float(ious[gi, pi])
                gt_cx = (gt[gi, 0] + gt[gi, 2]) / 2
                gt_cy = (gt[gi, 1] + gt[gi, 3]) / 2
                pred_cx = (pred[pi, 0] + pred[pi, 2]) / 2
                pred_cy = (pred[pi, 1] + pred[pi, 3]) / 2
                gt_w, gt_h = gt[gi, 2] - gt[gi, 0], gt[gi, 3] - gt[gi, 1]
                pred_w, pred_h = pred[pi, 2] - pred[pi, 0], pred[pi, 3] - pred[pi, 1]
                diagonal = float(np.hypot(gt_w, gt_h))
                bucket = "tiny_lt20" if diagonal < 20 else "large_ge20"
                buckets[f"{bucket}_iou"].append(iou)
                buckets[f"{bucket}_iou75"].append(float(iou >= 0.75))
                stats["matched_iou"].append(iou)
                stats["center_dx_p3"].append((pred_cx - gt_cx) / 8)
                stats["center_dy_p3"].append((pred_cy - gt_cy) / 8)
                stats["width_ratio"].append(pred_w / max(gt_w, 1e-9))
                stats["height_ratio"].append(pred_h / max(gt_h, 1e-9))

        routed = head.last_gcts
        if routed is None:
            raise RuntimeError("GCTS diagnostic capture did not receive routing tensors")
        alpha = routed["alpha"][0].detach().cpu()
        gate = routed["gate"][0, 0].detach().cpu()
        occupied = torch.zeros_like(gate, dtype=torch.bool)
        for center_x, center_y, box_w, box_h in normalized:
            gx = min(int(center_x * alpha.shape[-1]), alpha.shape[-1] - 1)
            gy = min(int(center_y * alpha.shape[-2]), alpha.shape[-2] - 1)
            frac_x = center_x * alpha.shape[-1] - gx
            frac_y = center_y * alpha.shape[-2] - gy
            x_hat = float(alpha[1, gy, gx] + alpha[3, gy, gx])
            y_hat = float(alpha[2, gy, gx] + alpha[3, gy, gx])
            stats["selector_x_error"].append(x_hat - frac_x)
            stats["selector_y_error"].append(y_hat - frac_y)
            stats["selector_coord_abs_error"].extend((abs(x_hat - frac_x), abs(y_hat - frac_y)))
            entropy = float(-(alpha[:, gy, gx] * alpha[:, gy, gx].clamp_min(1e-9).log()).sum() / np.log(4))
            stats["selector_entropy"].append(entropy)
            diagonal = np.hypot(box_w * width, box_h * height)
            stats["gate_tiny" if diagonal < 20 else "gate_large"].append(float(gate[gy, gx]))
            occupied[gy, gx] = True
        stats["gate_background"].extend(gate[~occupied].flatten().tolist())

    impulse = torch.nn.functional.pixel_unshuffle(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]), 2).flatten().tolist()
    report = {
        "weights": str(args.weights),
        "images": len(images),
        "candidate_order_tl_tr_bl_br": impulse,
        "metrics": {name: mean(values) for name, values in {**stats, **buckets}.items()},
    }
    if args.data:
        validation = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=args.device, plots=False)
        report["detection"] = {"ap50": float(validation.box.map50), "ap75": float(validation.box.map75), "map": float(validation.box.map)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/levir_gcts_v2/diagnostics.json")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(diagnose(parse_args()), indent=2))
