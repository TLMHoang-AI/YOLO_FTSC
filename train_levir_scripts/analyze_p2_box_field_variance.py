#!/usr/bin/env python3
"""Falsify P2 box-field ambiguity using one LEVIR YOLOv8n checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def load_gt(image: Path, width: int, height: int, ratio: float, pad: tuple[float, float]) -> torch.Tensor:
    rows = []
    for line in labels_for(image).read_text().splitlines():
        _, x, y, w, h = map(float, line.split()[:5])
        rows.append(((x - w / 2) * width, (y - h / 2) * height,
                     (x + w / 2) * width, (y + h / 2) * height))
    boxes = torch.tensor(rows, dtype=torch.float32)
    if boxes.numel():
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + pad[0]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + pad[1]
    return boxes.reshape(-1, 4)


def local_variances(boxes: torch.Tensor, scores: torch.Tensor, gt: torch.Tensor, size: int = 3) -> tuple[float, float, float]:
    """Return pixel, GT-normalized, and confidence-weighted normalized edge variance."""
    gh, gw = scores.shape
    cx, cy = float((gt[0] + gt[2]) / 8), float((gt[1] + gt[3]) / 8)  # P2 stride=4, cell center=.5
    ix, iy = round(cx - 0.5), round(cy - 0.5)
    radius = size // 2
    xs = torch.arange(max(0, ix - radius), min(gw, ix + radius + 1), device=boxes.device)
    ys = torch.arange(max(0, iy - radius), min(gh, iy + radius + 1), device=boxes.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    field = boxes[yy, xx].reshape(-1, 4)
    pixel = field.var(dim=0, correction=0).mean()
    scale = torch.tensor([gt[2] - gt[0], gt[3] - gt[1]] * 2, device=boxes.device).clamp_min(1)
    field = field / scale
    normalized = field.var(dim=0, correction=0).mean()
    weight = scores[yy, xx].reshape(-1).clamp_min(1e-8)
    weight = weight / weight.sum()
    mean = (weight[:, None] * field).sum(0)
    weighted = (weight[:, None] * (field - mean).square()).sum(0).mean()
    return float(pixel), float(normalized), float(weighted)


def bootstrap_delta(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, repeats: int = 4000) -> list[float]:
    delta = np.empty(repeats)
    for i in range(repeats):
        delta[i] = np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
    return np.quantile(delta, [0.025, 0.5, 0.975]).tolist()


def summarize(rows: list[dict], key: str) -> dict:
    bins = {name: np.array([r[key] for r in rows if r["iou_bin"] == name]) for name in ("<0.5", "0.5-0.75", ">=0.75")}
    out = {name: {"n": len(v), "median": float(np.median(v)), "mean": float(np.mean(v))} for name, v in bins.items()}
    if len(bins["<0.5"]) and len(bins[">=0.75"]):
        out["low_minus_high_median_bootstrap_95ci"] = bootstrap_delta(
            bins["<0.5"], bins[">=0.75"], np.random.default_rng(42))
        out["low_to_high_median_ratio"] = float(np.median(bins["<0.5"]) / max(np.median(bins[">=0.75"]), 1e-12))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/p2_box_field_seed42"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--neighborhood", type=int, choices=(3, 5), default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.checkpoint)
    net = model.model.cuda().eval()
    head = net.model[-1]
    if [float(x) for x in head.stride] != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"Expected P2-P5 strides, got {head.stride.tolist()}")
    images = sorted(args.images.glob("*"))[: args.limit]
    rows = []
    letterbox = LetterBox(new_shape=(512, 512), auto=False, stride=32)
    for index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        h0, w0 = original.shape[:2]
        ratio = min(512 / h0, 512 / w0)
        pad = ((512 - round(w0 * ratio)) / 2, (512 - round(h0 * ratio)) / 2)
        gt = load_gt(image_path, w0, h0, ratio, pad).cuda()
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).cuda().permute(2, 0, 1).float()[None] / 255
        with torch.inference_mode():
            decoded, raw = net(tensor)
            detections = non_max_suppression(decoded, conf_thres=0.001, iou_thres=0.7, nc=1, max_det=300)[0]
            all_boxes = head._get_decode_boxes(raw)[0]
            p2_count = raw["feats"][0].shape[2] * raw["feats"][0].shape[3]
            p2_boxes = all_boxes[:, :p2_count].T
            p2_boxes = torch.cat((p2_boxes[:, :2] - p2_boxes[:, 2:] / 2,
                                  p2_boxes[:, :2] + p2_boxes[:, 2:] / 2), 1)
            p2_scores = raw["scores"][0, 0, :p2_count].sigmoid().reshape(raw["feats"][0].shape[2:])
        ious = box_iou(gt, detections[:, :4]) if len(detections) else gt.new_zeros((len(gt), 0))
        best_iou = ious.max(1).values if ious.numel() else gt.new_zeros(len(gt))
        for gt_index, (box, iou) in enumerate(zip(gt, best_iou)):
            pixel_var, normalized_var, weighted_var = local_variances(
                p2_boxes.reshape(128, 128, 4), p2_scores, box, args.neighborhood)
            iou_value = float(iou)
            iou_bin = "<0.5" if iou_value < 0.5 else "0.5-0.75" if iou_value < 0.75 else ">=0.75"
            rows.append({"image": image_path.name, "gt_index": gt_index, "best_iou": iou_value,
                         "iou_bin": iou_bin, "pixel_variance": pixel_var,
                         "normalized_variance": normalized_var,
                         "confidence_weighted_normalized_variance": weighted_var})
        if index % 100 == 0:
            print(f"{index}/{len(images)} images, {len(rows)} GT")

    fields = ["image", "gt_index", "best_iou", "iou_bin", "pixel_variance", "normalized_variance",
              "confidence_weighted_normalized_variance"]
    with (args.output / "per_gt.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    if not rows:
        raise RuntimeError("No GT boxes found in selected images")
    summary = {key: summarize(rows, key) for key in (
        "pixel_variance", "normalized_variance", "confidence_weighted_normalized_variance")}
    summary["images"] = len(images); summary["gt"] = len(rows); summary["neighborhood"] = args.neighborhood
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
