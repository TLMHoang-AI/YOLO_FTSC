#!/usr/bin/env python3
"""Compare global score ordering for seed-42 channel-only and spatial-only P2 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
EXPECTED_IMAGES, EXPECTED_GT, BOOTSTRAP_REPEATS = 788, 696, 4000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def load_gt(image: Path, width: int, height: int, ratio: float = 1.0, pad: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    boxes = []
    for line in labels_for(image).read_text(encoding="utf-8").splitlines():
        _, x, y, w, h = map(float, line.split()[:5])
        boxes.append(((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height))
    gt = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if gt.numel():
        gt[:, [0, 2]] = gt[:, [0, 2]] * ratio + pad[0]
        gt[:, [1, 3]] = gt[:, [1, 3]] * ratio + pad[1]
    return gt


def assign_candidates(gt: torch.Tensor, boxes: torch.Tensor, centers: torch.Tensor, box_iou) -> list[torch.Tensor]:
    """Assign each center-inside-GT P2 anchor once to its highest-IoU eligible GT."""
    if not len(gt):
        return []
    inside = ((centers[None, :, 0] >= gt[:, None, 0]) & (centers[None, :, 0] <= gt[:, None, 2]) &
              (centers[None, :, 1] >= gt[:, None, 1]) & (centers[None, :, 1] <= gt[:, None, 3]))
    owner = box_iou(gt, boxes).masked_fill(~inside, -1).argmax(0)
    owned = inside.any(0)
    return [torch.where(owned & (owner == index))[0] for index in range(len(gt))]


def raw_rows(net, image_path: Path, device: str, imgsz: int, box_iou) -> tuple[list[dict], torch.Tensor]:
    from ultralytics.data.augment import LetterBox

    original = cv2.imread(str(image_path))
    if original is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = original.shape[:2]
    ratio = min(imgsz / height, imgsz / width)
    pad = ((imgsz - round(width * ratio)) / 2, (imgsz - round(height * ratio)) / 2)
    gt = load_gt(image_path, width, height, ratio, pad).to(device)
    image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=original)
    tensor = torch.from_numpy(image[..., ::-1].copy()).to(device).permute(2, 0, 1).float()[None] / 255
    head = net.model[-1]
    with torch.inference_mode():
        _, raw = net(tensor)
        feature = raw["feats"][0]
        count, stride = feature.shape[2] * feature.shape[3], float(head.stride[0])
        decoded = head._get_decode_boxes(raw)[0, :, :count].T
        boxes = torch.cat((decoded[:, :2] - decoded[:, 2:] / 2, decoded[:, :2] + decoded[:, 2:] / 2), 1)
        scores = raw["scores"][0, 0, :count].sigmoid()
        ys, xs = torch.meshgrid(torch.arange(feature.shape[2], device=device), torch.arange(feature.shape[3], device=device), indexing="ij")
        centers = torch.stack(((xs.reshape(-1) + 0.5) * stride, (ys.reshape(-1) + 0.5) * stride), 1)
        candidates = assign_candidates(gt, boxes, centers, box_iou)
    rows = []
    for gt_index, (gt_box, indices) in enumerate(zip(gt, candidates)):
        row = {"image": image_path.name, "gt_index": gt_index, "candidate_count": int(len(indices))}
        if len(indices):
            ious, candidate_scores = box_iou(gt_box[None], boxes[indices])[0], scores[indices]
            top = int(candidate_scores.argmax())
            top_score = float(candidate_scores[top])
            row.update(iou_topscore=float(ious[top]), top_score=top_score, score_half_top_multiplicity=int((candidate_scores >= 0.5 * top_score).sum()))
        else:
            row.update(iou_topscore=None, top_score=None, score_half_top_multiplicity=None)
        rows.append(row)
    return rows, load_gt(image_path, width, height)


def greedy_matches(boxes: torch.Tensor, scores: torch.Tensor, gt: torch.Tensor, threshold: float, box_iou) -> list[bool]:
    matches = [False] * len(boxes)
    available = torch.ones(len(gt), dtype=torch.bool)
    for index in scores.argsort(descending=True).tolist():
        if not available.any():
            break
        ious = box_iou(boxes[index:index + 1], gt)[0]
        ious[~available] = -1
        best_iou, best = ious.max(0)
        if float(best_iou) >= threshold:
            matches[index] = True
            available[best] = False
    return matches


def prediction_rows(wrapper, image_path: Path, gt: torch.Tensor, args: argparse.Namespace, box_iou) -> list[dict]:
    result = wrapper.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, iou=0.5, device=args.device, verbose=False)[0]
    boxes, scores = result.boxes.xyxy.cpu(), result.boxes.conf.cpu()
    matched_50, matched_75 = greedy_matches(boxes, scores, gt, 0.5, box_iou), greedy_matches(boxes, scores, gt, 0.75, box_iou)
    return [
        {"image": image_path.name, "prediction_index": index, "confidence": float(scores[index]),
         "x1": float(boxes[index, 0]), "y1": float(boxes[index, 1]), "x2": float(boxes[index, 2]), "y2": float(boxes[index, 3]),
         "tp_iou_0.5": matched_50[index], "tp_iou_0.75": matched_75[index]}
        for index in range(len(boxes))
    ]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {str(q): None for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return {str(q): float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def raw_summary(rows: list[dict]) -> dict:
    ious = [row["iou_topscore"] for row in rows if row["iou_topscore"] is not None]
    multiplicities = [row["score_half_top_multiplicity"] for row in rows if row["score_half_top_multiplicity"] is not None]
    return {
        "gt": len(rows), "without_candidates": sum(row["candidate_count"] == 0 for row in rows),
        "iou_topscore_quantiles": quantiles(ious),
        "iou_topscore_rate_ge_0.5": float(np.mean(np.asarray(ious) >= 0.5)) if ious else None,
        "iou_topscore_rate_ge_0.75": float(np.mean(np.asarray(ious) >= 0.75)) if ious else None,
        "score_half_top_multiplicity_quantiles": quantiles(multiplicities),
    }


def add_global_pr(rows: list[dict], total_gt: int, threshold: float) -> dict:
    ordered = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    true_positive = false_positive = 0
    curve = []
    key, label = f"tp_iou_{threshold}", str(threshold).replace(".", "_")
    for rank, row in enumerate(ordered, 1):
        true_positive += int(row[key])
        false_positive += int(not row[key])
        precision, recall = true_positive / (true_positive + false_positive), true_positive / total_gt
        row.update({f"global_rank_iou_{label}": rank, f"precision_iou_{label}": precision, f"recall_iou_{label}": recall})
        curve.append({"confidence": row["confidence"], "precision": precision, "recall": recall})
    precisions, recalls = [point["precision"] for point in curve], [point["recall"] for point in curve]
    return {
        "curve": curve,
        "precision_at_recall": {str(target): max((p for p, r in zip(precisions, recalls) if r >= target), default=None) for target in (0.25, 0.5, 0.75)},
        "recall_at_precision": {str(target): max((r for p, r in zip(precisions, recalls) if p >= target), default=None) for target in (0.5, 0.75, 0.9)},
        "tp_confidence_quantiles": quantiles([row["confidence"] for row in rows if row[key]]),
    }


def image_metrics(raw: list[dict], predictions: list[dict], gt_by_image: dict[str, int]) -> dict[str, dict[str, float]]:
    predictions_by_image: dict[str, list[dict]] = {}
    for row in predictions:
        predictions_by_image.setdefault(row["image"], []).append(row)
    values = {}
    for image, gt_count in gt_by_image.items():
        selected = [row for row in raw if row["image"] == image and row["iou_topscore"] is not None]
        values[image] = {
            "raw_iou_topscore": float(np.mean([row["iou_topscore"] for row in selected])) if selected else math.nan,
            "raw_iou_topscore_rate_ge_0.5": float(np.mean([row["iou_topscore"] >= 0.5 for row in selected])) if selected else math.nan,
            "raw_iou_topscore_rate_ge_0.75": float(np.mean([row["iou_topscore"] >= 0.75 for row in selected])) if selected else math.nan,
            "raw_half_top_multiplicity": float(np.mean([row["score_half_top_multiplicity"] for row in selected])) if selected else math.nan,
            "post_nms_recall_iou_0.5": sum(row["tp_iou_0.5"] for row in predictions_by_image.get(image, [])) / gt_count if gt_count else math.nan,
            "post_nms_recall_iou_0.75": sum(row["tp_iou_0.75"] for row in predictions_by_image.get(image, [])) / gt_count if gt_count else math.nan,
        }
    return values


def paired_image_bootstrap(channel: dict[str, dict[str, float]], spatial: dict[str, dict[str, float]]) -> dict:
    if channel.keys() != spatial.keys():
        raise RuntimeError("Channel and spatial image keys differ")
    rng, output = np.random.default_rng(42), {}
    for metric in next(iter(channel.values())):
        deltas = np.asarray([channel[name][metric] - spatial[name][metric] for name in channel], dtype=float)
        deltas = deltas[np.isfinite(deltas)]
        if not len(deltas):
            output[metric] = {"images": 0, "mean_delta": None, "bootstrap_95ci": None}
            continue
        samples = np.mean(rng.choice(deltas, size=(BOOTSTRAP_REPEATS, len(deltas)), replace=True), axis=1)
        output[metric] = {"images": int(len(deltas)), "mean_delta": float(deltas.mean()), "bootstrap_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])]}
    return output


def inspect_model(name: str, checkpoint: Path, images: list[Path], args: argparse.Namespace, box_iou) -> tuple[list[dict], list[dict], dict[str, int]]:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ChannelAttention, SpatialAttention

    wrapper = YOLO(checkpoint)
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if train_args.get("seed") != 42:
        raise RuntimeError(f"{name}: expected checkpoint seed 42, got {train_args.get('seed')!r}")
    trained_imgsz = train_args.get("imgsz")
    if trained_imgsz not in (args.imgsz, [args.imgsz], (args.imgsz,)):
        raise RuntimeError(f"{name}: expected checkpoint imgsz {args.imgsz}, got {trained_imgsz!r}")
    net = wrapper.model.to(args.device).eval()
    expected = ChannelAttention if name == "channel" else SpatialAttention
    if not isinstance(net.model[19], expected) or net.model[-1].f != [19] or net.model[-1].stride.tolist() != [4.0]:
        raise RuntimeError(f"{name}: expected {expected.__name__} immediately before one-level P2 Detect [4.0]")
    raw, predictions, gt_by_image = [], [], {}
    for index, image in enumerate(images, 1):
        model_raw, original_gt = raw_rows(net, image, args.device, args.imgsz, box_iou)
        raw.extend({"model": name, **row} for row in model_raw)
        gt_by_image[image.name] = len(original_gt)
        predictions.extend({"model": name, **row} for row in prediction_rows(wrapper, image, original_gt, args, box_iou))
        if index % 100 == 0:
            print(f"{name}: {index}/{len(images)} images")
    del net, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return raw, predictions, gt_by_image


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Fixed seed-42 test images directory")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/p2_channel_score_stability_seed42")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.001, help="Low post-NMS confidence threshold retained for PR curves")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    local_ultralytics()
    from ultralytics.utils.metrics import box_iou

    for path in (args.channel_checkpoint, args.spatial_checkpoint, args.images):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest_path = args.images.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("seed") != 42 or manifest.get("splits", {}).get("test", {}).get("images") != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected seed-42 manifest with {EXPECTED_IMAGES} test images: {manifest_path}")
    images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(f"Expected {EXPECTED_IMAGES} test images, found {len(images)}")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = {"channel": args.channel_checkpoint, "spatial": args.spatial_checkpoint}
    all_raw, all_predictions, gt_counts, image_values = [], [], {}, {}
    for name, checkpoint in checkpoints.items():
        raw, predictions, gt_by_image = inspect_model(name, checkpoint, images, args, box_iou)
        all_raw.extend(raw)
        all_predictions.extend(predictions)
        gt_counts[name] = len(raw)
        image_values[name] = image_metrics(raw, predictions, gt_by_image)
    if set(gt_counts.values()) != {EXPECTED_GT}:
        raise RuntimeError(f"Expected {EXPECTED_GT} GT for each checkpoint, got {gt_counts}")
    write_csv(args.output / "per_gt.csv", all_raw)
    pr = {}
    for name in checkpoints:
        rows = [row for row in all_predictions if row["model"] == name]
        pr[name] = {f"iou_{threshold}": add_global_pr(rows, EXPECTED_GT, threshold) for threshold in (0.5, 0.75)}
    write_csv(args.output / "per_prediction.csv", all_predictions)
    summary = {
        "protocol": {
            "seed": 42, "split": "test", "images": EXPECTED_IMAGES, "gt": EXPECTED_GT, "imgsz": args.imgsz,
            "raw_candidate_rule": "decoded P2 anchor center inside GT; overlapping candidates assigned once to highest-IoU GT",
            "post_nms": {"iou": 0.5, "confidence": args.conf, "matching": "per-image greedy one-to-one at IoU 0.5 and 0.75"},
            "ordering": "post-NMS predictions globally sorted descending by confidence for PR", "bootstrap": {"unit": "paired image", "repeats": BOOTSTRAP_REPEATS, "seed": 42},
        },
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "raw_p2": {name: raw_summary([row for row in all_raw if row["model"] == name]) for name in checkpoints},
        "post_nms": pr,
        "paired_channel_minus_spatial": paired_image_bootstrap(image_values["channel"], image_values["spatial"]),
        "files": {"per_gt": "per_gt.csv", "per_prediction": "per_prediction.csv"},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "paired_channel_minus_spatial": summary["paired_channel_minus_spatial"]}, indent=2))


if __name__ == "__main__":
    main()
