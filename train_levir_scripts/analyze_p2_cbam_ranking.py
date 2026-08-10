#!/usr/bin/env python3
"""Compare pre-NMS P2 localization/confidence ranking across three seed-42 checkpoints."""

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
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.utils.metrics import box_iou  # noqa: E402

MODEL_ORDER = ("baseline_p2_p3", "cbam_p2", "baseline_p2_p5")
EXPECTED_LEVELS = {"baseline_p2_p3": 2, "cbam_p2": 1, "baseline_p2_p5": 4}
METRICS = ("iou_best", "iou_topscore", "rank_gap", "confidence_iou_spearman", "best_iou_confidence", "best_iou_score_rank")
GROUPS = ("all", "tiny", "small", "large")


def labels_for(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def load_gt(image: Path, width: int, height: int, ratio: float, pad: tuple[float, float]) -> tuple[torch.Tensor, list[float]]:
    boxes, areas = [], []
    for line in labels_for(image).read_text().splitlines():
        _, x, y, w, h = map(float, line.split()[:5])
        boxes.append(((x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height))
        areas.append(w * width * h * height)
    tensor = torch.tensor(boxes, dtype=torch.float32)
    if tensor.numel():
        tensor[:, [0, 2]] = tensor[:, [0, 2]] * ratio + pad[0]
        tensor[:, [1, 3]] = tensor[:, [1, 3]] * ratio + pad[1]
    return tensor.reshape(-1, 4), areas


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if len(a) < 2:
        return math.nan
    ra, rb = average_ranks(a.detach().cpu().numpy()), average_ranks(b.detach().cpu().numpy())
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def assign_candidates(gt: torch.Tensor, boxes: torch.Tensor, centers: torch.Tensor) -> list[torch.Tensor]:
    """Assign each center-inside-GT P2 anchor once, preferring the GT with highest decoded IoU."""
    if not len(gt):
        return []
    inside = ((centers[None, :, 0] >= gt[:, None, 0]) & (centers[None, :, 0] <= gt[:, None, 2]) &
              (centers[None, :, 1] >= gt[:, None, 1]) & (centers[None, :, 1] <= gt[:, None, 3]))
    ious = box_iou(gt, boxes)
    eligible_iou = ious.masked_fill(~inside, -1)
    owner = eligible_iou.argmax(0)
    has_owner = inside.any(0)
    return [torch.where(has_owner & (owner == index))[0] for index in range(len(gt))]


def gt_group(area: float) -> str:
    return "tiny" if area < 100 else "small" if area <= 400 else "large"


def inspect_model(name: str, checkpoint: Path, images: list[Path], args: argparse.Namespace) -> list[dict]:
    wrapper = YOLO(checkpoint)
    train_args = (getattr(wrapper, "ckpt", None) or {}).get("train_args", {})
    if train_args.get("seed") != 42:
        raise RuntimeError(f"{name}: expected checkpoint training seed 42, got {train_args.get('seed')!r}")
    trained_imgsz = train_args.get("imgsz")
    if trained_imgsz not in (args.imgsz, [args.imgsz], (args.imgsz,)):
        raise RuntimeError(f"{name}: expected training imgsz {args.imgsz}, got {trained_imgsz!r}")
    net = wrapper.model.to(args.device).eval()
    head = net.model[-1]
    strides = [float(value) for value in head.stride]
    if not strides or strides[0] != 4.0:
        raise RuntimeError(f"{name}: expected P2 stride 4 first, got {strides}")
    expected = EXPECTED_LEVELS[name]
    if len(strides) != expected:
        raise RuntimeError(f"{name}: expected {expected} Detect levels, got strides {strides}")

    rows = []
    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)
    for image_index, image_path in enumerate(images, 1):
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = original.shape[:2]
        ratio = min(args.imgsz / height, args.imgsz / width)
        pad = ((args.imgsz - round(width * ratio)) / 2, (args.imgsz - round(height * ratio)) / 2)
        gt_cpu, areas = load_gt(image_path, width, height, ratio, pad)
        gt = gt_cpu.to(args.device)
        image = letterbox(image=original)
        tensor = torch.from_numpy(image[..., ::-1].copy()).to(args.device).permute(2, 0, 1).float()[None] / 255

        with torch.inference_mode():
            _, raw = net(tensor)
            p2_height, p2_width = raw["feats"][0].shape[2:]
            p2_count = p2_height * p2_width
            decoded = head._get_decode_boxes(raw)[0, :, :p2_count].T
            boxes = torch.cat((decoded[:, :2] - decoded[:, 2:] / 2, decoded[:, :2] + decoded[:, 2:] / 2), 1)
            scores = raw["scores"][0, 0, :p2_count].sigmoid()
            ys, xs = torch.meshgrid(
                torch.arange(p2_height, device=args.device), torch.arange(p2_width, device=args.device), indexing="ij")
            centers = torch.stack(((xs.reshape(-1) + 0.5) * 4, (ys.reshape(-1) + 0.5) * 4), 1)
            candidates = assign_candidates(gt, boxes, centers)

        for gt_index, (gt_box, area, indices) in enumerate(zip(gt, areas, candidates)):
            row = {"model": name, "image": image_path.name, "gt_index": gt_index,
                   "area_px2": area, "size_group": gt_group(area), "candidate_count": int(len(indices))}
            if not len(indices):
                row.update({metric: math.nan for metric in METRICS})
            else:
                candidate_ious = box_iou(gt_box[None], boxes[indices])[0]
                candidate_scores = scores[indices]
                best_index = int(candidate_ious.argmax())
                top_index = int(candidate_scores.argmax())
                iou_best = float(candidate_ious[best_index])
                iou_topscore = float(candidate_ious[top_index])
                row.update(
                    iou_best=iou_best,
                    iou_topscore=iou_topscore,
                    rank_gap=max(0.0, iou_best - iou_topscore),
                    confidence_iou_spearman=spearman(candidate_scores, candidate_ious),
                    best_iou_confidence=float(candidate_scores[best_index]),
                    best_iou_score_rank=int((candidate_scores > candidate_scores[best_index]).sum()) + 1,
                )
                if row["iou_best"] + 1e-7 < row["iou_topscore"] or row["rank_gap"] < 0:
                    raise AssertionError(f"Invalid ranking metrics for {image_path.name} GT {gt_index}")
            rows.append(row)
        if image_index % 100 == 0:
            print(f"{name}: {image_index}/{len(images)} images, {len(rows)} GT")
    del net, wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def finite_values(rows: list[dict], metric: str, group: str) -> np.ndarray:
    values = [row[metric] for row in rows if group == "all" or row["size_group"] == group]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)


def descriptive_summary(rows: list[dict]) -> dict:
    result = {}
    for group in GROUPS:
        selected = rows if group == "all" else [row for row in rows if row["size_group"] == group]
        result[group] = {"gt": len(selected), "without_candidates": sum(row["candidate_count"] == 0 for row in selected)}
        for metric in METRICS:
            values = finite_values(selected, metric, "all")
            result[group][metric] = {"n": int(len(values)), "mean": float(values.mean()) if len(values) else None,
                                      "median": float(np.median(values)) if len(values) else None}
    return result


def paired_delta(rows_by_model: dict[str, list[dict]], left: str, right: str, repeats: int) -> dict:
    left_rows = {(row["image"], row["gt_index"]): row for row in rows_by_model[left]}
    right_rows = {(row["image"], row["gt_index"]): row for row in rows_by_model[right]}
    if left_rows.keys() != right_rows.keys():
        raise RuntimeError(f"GT keys differ between {left} and {right}")
    rng = np.random.default_rng(42)
    result = {}
    for group in GROUPS:
        keys = [key for key, row in left_rows.items() if group == "all" or row["size_group"] == group]
        result[group] = {}
        for metric in METRICS:
            paired = np.asarray(
                [(left_rows[key][metric], right_rows[key][metric]) for key in keys], dtype=np.float64
            ).reshape(-1, 2)
            paired = paired[np.isfinite(paired).all(1)]
            delta = paired[:, 0] - paired[:, 1] if len(paired) else np.asarray([])
            if len(delta):
                samples = np.empty(repeats)
                for index in range(repeats):
                    samples[index] = np.median(rng.choice(delta, len(delta), replace=True))
                ci = np.quantile(samples, [0.025, 0.975]).tolist()
                result[group][metric] = {"n": int(len(delta)), "mean_delta": float(delta.mean()),
                                          "median_delta": float(np.median(delta)), "median_delta_bootstrap_95ci": ci}
            else:
                result[group][metric] = {"n": 0, "mean_delta": None, "median_delta": None,
                                          "median_delta_bootstrap_95ci": None}
    return result


def decision_gate(primary: dict, reference: dict) -> dict:
    p, r = primary["all"], reference["all"]
    value = lambda summary, metric: summary[metric]["median_delta"]
    required = ("iou_best", "iou_topscore", "rank_gap", "confidence_iou_spearman")
    if any(value(summary, metric) is None for summary in (p, r) for metric in required):
        return {"verdict": "inconclusive_insufficient_candidates",
                "recommendation": "Run the full test split before making an architecture decision.",
                "note": "Directional evidence only: the three checkpoints do not share identical Detect topology."}
    ranking = (value(p, "iou_best") >= -0.01 and value(p, "iou_topscore") < 0 and
               value(p, "rank_gap") > 0 and value(p, "confidence_iou_spearman") < 0)
    same_reference_pattern = (value(r, "iou_topscore") < 0 and value(r, "rank_gap") > 0 and
                              value(r, "confidence_iou_spearman") < 0)
    regression = value(p, "iou_best") < -0.01 and value(r, "iou_best") < -0.01
    if ranking and same_reference_pattern:
        verdict = "ranking_misalignment"
        recommendation = "Proceed to a detached localization-guided CBAM experiment."
    elif regression:
        verdict = "regression_or_tal_degradation"
        recommendation = "Do not add a geometry gate yet; investigate training/TAL semantics."
    else:
        verdict = "inconclusive_topology_confound"
        recommendation = "Do not change the architecture from this seed-42 diagnostic alone."
    return {"verdict": verdict, "recommendation": recommendation,
            "note": "Directional evidence only: the three checkpoints do not share identical Detect topology."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-p2-p3", type=Path, required=True)
    parser.add_argument("--cbam-p2", type=Path, required=True)
    parser.add_argument("--baseline-p2-p5", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Fixed seed-42 test images directory")
    parser.add_argument("--output", type=Path, default=Path("diagnostics/p2_cbam_ranking_seed42"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = {"baseline_p2_p3": args.baseline_p2_p3, "cbam_p2": args.cbam_p2,
                   "baseline_p2_p5": args.baseline_p2_p5}
    for path in (*checkpoints.values(), args.images):
        if not path.exists():
            raise FileNotFoundError(path)
    images = sorted(path for path in args.images.iterdir()
                    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise RuntimeError(f"No images found in {args.images}")
    manifest_path = args.images.parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("seed") != 42:
        raise RuntimeError(f"Expected fixed split seed 42 in {manifest_path}, got {manifest.get('seed')!r}")
    if manifest.get("splits", {}).get("test", {}).get("images") != 788:
        raise RuntimeError(f"Expected 788 test images in {manifest_path}")
    if args.limit is None and len(images) != 788:
        raise RuntimeError(f"Expected 788 fixed-split test images, found {len(images)}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows_by_model = {name: inspect_model(name, checkpoints[name], images, args) for name in MODEL_ORDER}
    gt_counts = {name: len(rows) for name, rows in rows_by_model.items()}
    if len(set(gt_counts.values())) != 1:
        raise RuntimeError(f"GT counts differ: {gt_counts}")
    if args.limit is None and next(iter(gt_counts.values())) != 696:
        raise RuntimeError(f"Expected 696 fixed-split GT, got {gt_counts}")

    fields = ["model", "image", "gt_index", "area_px2", "size_group", "candidate_count", *METRICS]
    with (args.output / "per_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in MODEL_ORDER:
            writer.writerows(rows_by_model[name])

    primary = paired_delta(rows_by_model, "cbam_p2", "baseline_p2_p3", args.bootstrap_repeats)
    reference = paired_delta(rows_by_model, "cbam_p2", "baseline_p2_p5", args.bootstrap_repeats)
    summary = {
        "protocol": {"seed": 42, "imgsz": args.imgsz, "images": len(images), "gt": gt_counts,
                     "split_manifest": str(manifest_path),
                     "candidate_rule": "P2 anchor center inside GT; overlapping anchors assigned to highest-IoU GT",
                     "prediction_stage": "decoded P2 boxes and sigmoid class scores before threshold and NMS",
                     "checkpoints": {name: str(path) for name, path in checkpoints.items()}},
        "models": {name: descriptive_summary(rows) for name, rows in rows_by_model.items()},
        "paired_delta_cbam_minus_p2_p3": primary,
        "paired_delta_cbam_minus_p2_p5": reference,
        "decision": decision_gate(primary, reference),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
