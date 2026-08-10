#!/usr/bin/env python3
"""Measure raw-P2 foreground/background score separation for the matched attention controls."""

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
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

from train_levir_scripts.analyze_p2_cbam_ranking import labels_for, load_gt  # noqa: E402


MODELS = ("plain", "channel_only", "spatial_only", "full_cbam")
EMPTY_TOPK = (1, 5, 10, 50)
METRICS = ("s_pos", "s_bg", "margin", "bg_wins")


def inspect_model(name: str, checkpoint: Path, images: list[Path], args: argparse.Namespace) -> list[dict]:
    wrapper = YOLO(checkpoint)
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if train_args.get("seed") != 42:
        raise RuntimeError(f"{name}: expected training seed 42, got {train_args.get('seed')!r}")
    net = wrapper.model.to(args.device).eval()
    head = net.model[-1]
    if head.stride.tolist() != [4.0]:
        raise RuntimeError(f"{name}: expected P2-only Detect stride [4.0], got {head.stride.tolist()}")
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    rows = []
    for index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Could not read {image_path}")
        height, width = original.shape[:2]
        ratio = min(args.imgsz / height, args.imgsz / width)
        pad = ((args.imgsz - round(width * ratio)) / 2, (args.imgsz - round(height * ratio)) / 2)
        gt, _ = load_gt(image_path, width, height, ratio, pad)
        gt = gt.to(args.device)
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(args.device).permute(2, 0, 1).float()[None] / 255
        with torch.inference_mode():
            _, raw = net(tensor)
            count = raw["feats"][0].shape[2] * raw["feats"][0].shape[3]
            decoded = head._get_decode_boxes(raw)[0, :, :count].T
            boxes = torch.cat((decoded[:, :2] - decoded[:, 2:] / 2, decoded[:, :2] + decoded[:, 2:] / 2), 1)
            scores = raw["scores"][0, 0, :count].sigmoid()
        row = {"model": name, "image": image_path.name, "has_gt": bool(len(gt))}
        if not len(gt):
            ranked = scores.sort(descending=True).values
            row.update({f"empty_top{k}_mean": float(ranked[:k].mean()) for k in EMPTY_TOPK})
            row["empty_max"] = float(ranked[0])
        else:
            max_iou = box_iou(gt, boxes).amax(dim=0)
            positive = scores[max_iou >= args.positive_iou]
            background = scores[max_iou <= args.background_iou]
            if not len(positive) or not len(background):
                raise RuntimeError(f"{name}/{image_path.name}: empty positive or hard-background candidate set")
            s_pos, s_bg = float(positive.max()), float(background.max())
            row.update(s_pos=s_pos, s_bg=s_bg, margin=s_pos - s_bg, bg_wins=float(s_bg > s_pos),
                       positive_candidates=int(len(positive)), background_candidates=int(len(background)))
        rows.append(row)
        if index % 100 == 0:
            print(f"{name}: {index}/{len(images)}")
    del net, wrapper
    torch.cuda.empty_cache()
    return rows


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "q90": float(np.quantile(array, 0.9)), "q95": float(np.quantile(array, 0.95)),
            "q99": float(np.quantile(array, 0.99))}


def summarize(rows: list[dict]) -> dict:
    empty = [row for row in rows if not row["has_gt"]]
    positive = [row for row in rows if row["has_gt"]]
    return {
        "empty_images": {metric: describe([row[metric] for row in empty])
                         for metric in ("empty_max", *(f"empty_top{k}_mean" for k in EMPTY_TOPK))},
        "positive_images": {metric: describe([row[metric] for row in positive]) for metric in METRICS},
    }


def paired_bootstrap(rows: dict[str, list[dict]], left: str, right: str, repeats: int) -> dict:
    rng = np.random.default_rng(42)
    left_rows = {row["image"]: row for row in rows[left]}
    right_rows = {row["image"]: row for row in rows[right]}
    if left_rows.keys() != right_rows.keys():
        raise RuntimeError(f"Image keys differ: {left} vs {right}")
    result = {}
    metrics = ("empty_max", *(f"empty_top{k}_mean" for k in EMPTY_TOPK), *METRICS)
    for metric in metrics:
        pairs = [(left_rows[key].get(metric), right_rows[key].get(metric)) for key in left_rows]
        delta = np.asarray([a - b for a, b in pairs if a is not None and b is not None], dtype=np.float64)
        samples = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
        result[metric] = {"n": len(delta), "mean_delta": float(delta.mean()),
                          "bootstrap_mean_95ci": np.quantile(samples, [0.025, 0.975]).tolist()}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in MODELS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--positive-iou", type=float, default=0.10)
    parser.add_argument("--background-iou", type=float, default=0.01)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.background_iou < args.positive_iou <= 1:
        raise ValueError("Require 0 <= background_iou < positive_iou <= 1")
    checkpoints = {name: getattr(args, name) for name in MODELS}
    images = sorted(args.images.glob("*.png"))
    if len(images) != 788:
        raise RuntimeError(f"Expected 788 test images, found {len(images)}")
    rows = {name: inspect_model(name, checkpoint, images, args) for name, checkpoint in checkpoints.items()}
    counts = {(sum(not row["has_gt"] for row in model_rows), sum(row["has_gt"] for row in model_rows))
              for model_rows in rows.values()}
    if counts != {(374, 414)}:
        raise RuntimeError(f"Expected 374 empty and 414 positive images, got {counts}")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for model_rows in rows.values() for row in model_rows for key in row})
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in MODELS:
            writer.writerows(rows[name])
    summary = {
        "protocol": {"stage": "raw decoded P2 before threshold/NMS", "seed": 42, "split": "test",
                     "images": 788, "empty_images": 374, "positive_images": 414,
                     "positive_candidate": f"max IoU with any GT >= {args.positive_iou}",
                     "hard_background_candidate": f"max IoU with every GT <= {args.background_iou}",
                     "s_pos": "maximum confidence among positive candidates",
                     "s_bg": "maximum confidence among hard-background candidates",
                     "uncertainty": "paired image bootstrap; not training-seed uncertainty"},
        "models": {name: summarize(model_rows) for name, model_rows in rows.items()},
        "paired": {f"{name}_minus_plain": paired_bootstrap(rows, name, "plain", args.bootstrap_repeats)
                   for name in MODELS if name != "plain"},
        "channel_minus_spatial": paired_bootstrap(rows, "channel_only", "spatial_only", args.bootstrap_repeats),
        "channel_minus_full_cbam": paired_bootstrap(rows, "channel_only", "full_cbam", args.bootstrap_repeats),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: summary["models"][name] for name in MODELS}, indent=2))


if __name__ == "__main__":
    main()
