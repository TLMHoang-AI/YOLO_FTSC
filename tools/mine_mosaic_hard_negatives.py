#!/usr/bin/env python3
"""Mine a static hard-negative crop bank from a detector on the train split.

Ported verbatim from Duy's M5 implementation. This utility is not invoked by
the preparation/preflight path and must be run separately before future M5
training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_upstream_root = os.environ.get("MOSAIC_MINER_ULTRALYTICS_ROOT")
if _upstream_root:
    sys.path.insert(0, _upstream_root)


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def read_gt(path: Path, shape):
    h, w = shape[:2]; boxes = []
    if not path.exists(): return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5: continue
        _, cx, cy, bw, bh = map(float, fields[:5])
        boxes.append(np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf-min", type=float, default=0.20)
    parser.add_argument("--gt-iou-max", type=float, default=0.10)
    parser.add_argument("--context-expand", type=float, default=4.0)
    parser.add_argument("--min-crop", type=int, default=96)
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    bank = []
    for image_path in sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}):
        image = cv2.imread(str(image_path))
        if image is None: continue
        gt = read_gt(args.labels / f"{image_path.stem}.txt", image.shape)
        result = model.predict(str(image_path), conf=args.conf_min, verbose=False)[0]
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            if any(iou(box, truth) > args.gt_iou_max for truth in gt): continue
            side = max(float(box[2] - box[0]), float(box[3] - box[1]), float(args.min_crop)) * args.context_expand
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            x1 = int(np.clip(cx - side / 2, 0, image.shape[1] - 1)); y1 = int(np.clip(cy - side / 2, 0, image.shape[0] - 1))
            x2 = int(np.clip(cx + side / 2, x1 + 1, image.shape[1])); y2 = int(np.clip(cy + side / 2, y1 + 1, image.shape[0]))
            if any(iou(np.array([x1, y1, x2, y2]), truth) > args.gt_iou_max for truth in gt): continue
            bank.append({"image": str(image_path.resolve()), "crop_xyxy": [x1, y1, x2, y2], "fp_conf": float(conf)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(bank)} hard-negative crops to {args.output}")


if __name__ == "__main__": main()
