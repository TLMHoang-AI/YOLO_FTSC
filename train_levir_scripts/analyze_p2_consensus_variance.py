#!/usr/bin/env python3
"""Compare fixed-patch and exact TAL-positive P2 box-field variance."""

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
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402


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


def normalized_stats(field: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    wh = (gt[2:] - gt[:2]).clamp_min(1)
    normalized = field / torch.cat((wh, wh))
    centered = normalized - normalized.mean(0, keepdim=True)
    variance = centered.square().mean()
    smooth_l1 = torch.nn.functional.smooth_l1_loss(centered, torch.zeros_like(centered))
    return float(variance), float(smooth_l1)


def fixed_field(boxes: torch.Tensor, gt: torch.Tensor, size: int) -> torch.Tensor:
    height, width = boxes.shape[:2]
    ix = round(float((gt[0] + gt[2]) / 8) - 0.5)
    iy = round(float((gt[1] + gt[3]) / 8) - 0.5)
    radius = size // 2
    return boxes[
        max(0, iy - radius):min(height, iy + radius + 1),
        max(0, ix - radius):min(width, ix + radius + 1),
    ].reshape(-1, 4)


def summarize(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows if row[key] is not None])
    result = {"n": int(values.size), "median": float(np.median(values)), "mean": float(values.mean())}
    for label in ("<0.5", "0.5-0.75", ">=0.75"):
        group = np.asarray([row[key] for row in rows if row["iou_bin"] == label and row[key] is not None])
        result[label] = {"n": int(group.size), "median": float(np.median(group)), "mean": float(group.mean())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.checkpoint)
    net = model.model.cuda().eval()
    head = net.model[-1]
    criterion = net.init_criterion()
    if [float(value) for value in head.stride] != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"Expected P2-P5, got {head.stride.tolist()}")

    rows = []
    letterbox = LetterBox(new_shape=(512, 512), auto=False, stride=32)
    images = sorted(args.images.glob("*"))
    for image_index, image_path in enumerate(images, 1):
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
            pred_dist = raw["boxes"].permute(0, 2, 1).contiguous()
            pred_scores = raw["scores"].permute(0, 2, 1).contiguous()
            anchor_points, stride_tensor = make_anchors(raw["feats"], criterion.stride, 0.5)
            pred_bboxes = criterion.bbox_decode(anchor_points, pred_dist, None, stride_tensor)
            gt_bboxes = gt[None]
            gt_labels = torch.zeros((1, len(gt), 1), device=gt.device)
            mask_gt = torch.ones((1, len(gt), 1), device=gt.device, dtype=torch.bool)
            _, _, _, fg_mask, target_gt_idx = criterion.assigner(
                pred_scores.sigmoid(), pred_bboxes * stride_tensor, anchor_points * stride_tensor,
                gt_labels, gt_bboxes, mask_gt,
            )
            n_p2 = raw["feats"][0].shape[2] * raw["feats"][0].shape[3]
            p2_boxes = pred_bboxes[0, :n_p2] * 4
            p2_map = p2_boxes.reshape(*raw["feats"][0].shape[2:], 4)
            p2_fg = fg_mask[0, :n_p2].bool()
            p2_gt_idx = target_gt_idx[0, :n_p2]
        ious = box_iou(gt, detections[:, :4]) if len(detections) else gt.new_zeros((len(gt), 0))
        best_iou = ious.max(1).values if ious.numel() else gt.new_zeros(len(gt))
        for gt_index, (gt_box, iou) in enumerate(zip(gt, best_iou)):
            fixed3_var, fixed3_loss = normalized_stats(fixed_field(p2_map, gt_box, 3), gt_box)
            fixed5_var, fixed5_loss = normalized_stats(fixed_field(p2_map, gt_box, 5), gt_box)
            tal_mask = p2_fg & (p2_gt_idx == gt_index)
            tal_count = int(tal_mask.sum())
            tal_var, tal_loss = normalized_stats(p2_boxes[tal_mask], gt_box) if tal_count >= 2 else (None, None)
            iou_value = float(iou)
            iou_bin = "<0.5" if iou_value < 0.5 else "0.5-0.75" if iou_value < 0.75 else ">=0.75"
            rows.append({
                "image": image_path.name, "gt_index": gt_index, "best_iou": iou_value, "iou_bin": iou_bin,
                "fixed3_variance": fixed3_var, "fixed3_smooth_l1": fixed3_loss,
                "fixed5_variance": fixed5_var, "fixed5_smooth_l1": fixed5_loss,
                "tal_positive_count": tal_count, "tal_variance": tal_var, "tal_smooth_l1": tal_loss,
            })
        if image_index % 100 == 0:
            print(f"{image_index}/{len(images)} images, {len(rows)} GT")

    fields = list(rows[0])
    with (args.output / "per_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {key: summarize(rows, key) for key in (
        "fixed3_variance", "fixed5_variance", "tal_variance", "tal_smooth_l1")}
    summary.update(images=len(images), gt=len(rows), tal_coverage=sum(row["tal_positive_count"] >= 2 for row in rows) / len(rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
