#!/usr/bin/env python3
"""Audit LEVIR P2 baseline vs DBSS P2-aware without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download


BASE_REPO = "duyle2408/levir-ship-yolo-p2"
AWARE_REPO = "duyle2408/levir_dbss_p2_aware"
METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "AP75": "metrics/mAP75(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
SCENE_RE = re.compile(r"^(.*)_(-?\d+)_(-?\d+)$")
SIZE_NAMES = ("very_tiny", "tiny", "larger")


def download(repo: str, name: str) -> Path:
    return Path(hf_hub_download(repo, name, repo_type="dataset"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scene(stem: str) -> str:
    match = SCENE_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot parse scene from {stem}")
    return match.group(1)


def paired_tables(out: Path) -> list[dict[str, object]]:
    base = {int(row["seed"]): row for row in read_csv(download(BASE_REPO, "summary_runs.csv"))}
    aware_rows = read_csv(download(AWARE_REPO, "summary_runs.csv"))
    aware = {
        int(row["seed"]): row
        for row in aware_rows
        if row["mechanism"] == "dbss_p2_aware" and row["model"] == "yolov8n"
    }
    if set(base) != {42, 43, 44} or set(aware) != {42, 43, 44}:
        raise RuntimeError(f"Expected paired seeds 42/43/44, got baseline={sorted(base)}, aware={sorted(aware)}")
    rows = []
    for seed_id in (42, 43, 44):
        for split in ("val", "test"):
            record: dict[str, object] = {"seed": seed_id, "split": split}
            for label, key in METRICS.items():
                b, a = float(base[seed_id][f"{split}/{key}"]), float(aware[seed_id][f"{split}/{key}"])
                record[f"base_{label}"] = b
                record[f"aware_{label}"] = a
                record[f"delta_{label}"] = a - b
            rows.append(record)
    write_csv(out / "paired_by_seed.csv", rows)
    summary = []
    for split in ("val", "test"):
        group = [row for row in rows if row["split"] == split]
        for label in METRICS:
            values = [float(row[f"delta_{label}"]) for row in group]
            summary.append({
                "split": split,
                "metric": label,
                "delta_mean": statistics.fmean(values),
                "delta_std": statistics.stdev(values),
                "aware_wins": sum(value > 0 for value in values),
                "ties": sum(value == 0 for value in values),
                "aware_losses": sum(value < 0 for value in values),
            })
    write_csv(out / "paired_summary.csv", summary)
    return rows


def epoch_curves(out: Path) -> None:
    combined = []
    columns = {
        "metrics/mAP50-95(B)": "val_mAP50-95",
        "loss_dbss_pos": "loss_dbss_pos",
        "dbss_delta_q_pos": "delta_q_pos",
        "dbss_displacement_ratio": "displacement_ratio",
        "p2_positive_fraction": "f_P2",
    }
    for seed_id in (42, 43, 44):
        remote = f"runs/yolov8n/dbss_p2_aware/seed_{seed_id}/results.csv"
        for row in read_csv(download(AWARE_REPO, remote)):
            combined.append({"seed": seed_id, "epoch": int(row["epoch"]), **{
                output: float(row[source]) for source, output in columns.items()
            }})
    write_csv(out / "aware_epoch_curves.csv", combined)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(columns), 1, figsize=(9, 14), sharex=True)
    for axis, key in zip(axes, columns.values()):
        for seed_id in (42, 43, 44):
            group = [row for row in combined if row["seed"] == seed_id]
            axis.plot([row["epoch"] for row in group], [row[key] for row in group], label=f"seed {seed_id}")
        axis.set_ylabel(key)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("epoch")
    figure.tight_layout()
    figure.savefig(out / "aware_epoch_curves.png", dpi=160)
    plt.close(figure)


def label_records(root: Path, split: str) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]]]:
    images, boxes = [], {}
    for image in sorted((root / "images" / split).glob("*.png")):
        label = root / "labels" / split / f"{image.stem}.txt"
        rows = []
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                _, x, y, w, h = map(float, line.split())
                rows.append(np.array([x, y, w, h], dtype=np.float32))
        boxes[image.stem] = rows
        images.append({"split": split, "image": image.name, "scene": scene(image.stem), "objects": len(rows)})
    return images, boxes


def dataset_audit(data_yaml: Path, out: Path) -> tuple[Path, dict[str, dict[str, list[np.ndarray]]]]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    if (out / "background_statistics.csv").is_file():
        return root, {split: label_records(root, split)[1] for split in ("train", "val", "test")}
    all_images, all_boxes, scene_counts, object_rows, background_rows = [], {}, [], [], []
    scene_sets = {}
    for split in ("train", "val", "test"):
        images, boxes = label_records(root, split)
        all_images.extend(images); all_boxes[split] = boxes
        scene_sets[split] = {str(row["scene"]) for row in images}
        counts = defaultdict(int)
        for row in images:
            counts[str(row["scene"])] += 1
        scene_counts.extend({"split": split, "scene": key, "crops": value} for key, value in sorted(counts.items()))
        for stem, records in boxes.items():
            for box in records:
                w, h = float(box[2] * 512), float(box[3] * 512)
                object_rows.append({"split": split, "image": f"{stem}.png", "scene": scene(stem),
                                    "w_px": w, "h_px": h, "sqrt_area_px": math.sqrt(w * h)})
        for row in images:
            image = cv2.imread(str(root / "images" / split / str(row["image"])), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx, gy = np.diff(gray, axis=1), np.diff(gray, axis=0)
            histogram = np.histogram(gray, bins=32, range=(0, 256))[0].astype(float)
            probability = histogram / histogram.sum()
            entropy = float(-(probability[probability > 0] * np.log2(probability[probability > 0])).sum())
            black = np.all(image <= 5, axis=2)
            background_rows.append({**row, "black_ratio": float(black.mean()), "gray_mean": float(gray.mean()),
                                    "gray_std": float(gray.std()), "gradient_mean": float((np.abs(gx).mean() + np.abs(gy).mean()) / 2),
                                    "gradient_std": float((gx.std() + gy.std()) / 2), "entropy32": entropy})
    write_csv(out / "image_counts.csv", all_images)
    write_csv(out / "objects.csv", object_rows)
    write_csv(out / "scene_crop_counts.csv", scene_counts)
    write_csv(out / "background_statistics.csv", background_rows)
    overlap = [{"set": split, "scenes": len(scene_sets[split])} for split in scene_sets]
    overlap += [
        {"set": "train_val_overlap", "scenes": len(scene_sets["train"] & scene_sets["val"])},
        {"set": "train_test_overlap", "scenes": len(scene_sets["train"] & scene_sets["test"])},
        {"set": "val_test_overlap", "scenes": len(scene_sets["val"] & scene_sets["test"])},
        {"set": "three_way_overlap", "scenes": len(set.intersection(*scene_sets.values()))},
        {"set": "union", "scenes": len(set.union(*scene_sets.values()))},
    ]
    write_csv(out / "scene_overlap.csv", overlap)
    split_summary = []
    for split in ("train", "val", "test"):
        images = [row for row in all_images if row["split"] == split]
        objects = [row for row in object_rows if row["split"] == split]
        split_summary.append({
            "split": split, "images": len(images), "objects": len(objects),
            "objects_per_image": len(objects) / len(images),
            "empty_images": sum(int(row["objects"]) == 0 for row in images),
            "empty_ratio": sum(int(row["objects"]) == 0 for row in images) / len(images),
            "mean_w_px": statistics.fmean(float(row["w_px"]) for row in objects),
            "mean_h_px": statistics.fmean(float(row["h_px"]) for row in objects),
            "mean_sqrt_area_px": statistics.fmean(float(row["sqrt_area_px"]) for row in objects),
        })
    expected_images = {"train": 2320, "val": 788, "test": 788}
    actual_images = {row["split"]: row["images"] for row in split_summary}
    if actual_images != expected_images:
        raise RuntimeError(f"Unexpected split membership: {actual_images}")
    write_csv(out / "split_summary.csv", split_summary)
    return root, all_boxes


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    return np.array([(x - w / 2) * 512, (y - h / 2) * 512, (x + w / 2) * 512, (y + h / 2) * 512])


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)[:, None]
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)[None]
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def size_name(sqrt_area: float) -> str:
    return "very_tiny" if sqrt_area < 16 else "tiny" if sqrt_area < 24 else "larger"


def subgroup_diagnostics(model, root: Path, boxes: dict[str, list[np.ndarray]], split: str, seed_id: int, variant: str) -> list[dict[str, object]]:
    paths = sorted((root / "images" / split).glob("*.png"))
    counters = defaultdict(lambda: defaultdict(float))
    results = model.predict(paths, stream=True, imgsz=512, batch=8, conf=0.001, iou=0.7, device=0, verbose=False)
    for path, result in zip(paths, results):
        gt_rows = boxes[path.stem]
        gt = np.array([xywh_to_xyxy(row) for row in gt_rows]) if gt_rows else np.empty((0, 4))
        pred = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        image_group = "empty" if len(gt) == 0 else "one_object" if len(gt) == 1 else "multiple_objects"
        scene_id = scene(path.stem)
        for threshold in (0.5, 0.75):
            matched_gt, matched_pred = set(), set()
            if len(gt) and len(pred):
                ious = box_iou(pred, gt)
                for pred_index in np.argsort(-scores):
                    gt_index = int(ious[pred_index].argmax())
                    if ious[pred_index, gt_index] >= threshold and gt_index not in matched_gt:
                        matched_gt.add(gt_index); matched_pred.add(int(pred_index))
            for group in (image_group, f"scene:{scene_id}"):
                counters[(threshold, group)]["tp"] += len(matched_gt)
                counters[(threshold, group)]["fp"] += len(pred) - len(matched_pred)
                counters[(threshold, group)]["fn"] += len(gt) - len(matched_gt)
                counters[(threshold, group)]["images"] += 1
            for gt_index, row in enumerate(gt_rows):
                group = size_name(math.sqrt(float(row[2] * row[3])) * 512)
                counters[(threshold, group)]["tp"] += gt_index in matched_gt
                counters[(threshold, group)]["fn"] += gt_index not in matched_gt
                counters[(threshold, group)]["objects"] += 1
            for pred_index, pred_box in enumerate(pred):
                if pred_index not in matched_pred:
                    group = size_name(math.sqrt(max(0.0, float((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])))))
                    counters[(threshold, group)]["fp"] += 1
    rows = []
    for (threshold, group), values in counters.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        rows.append({"variant": variant, "seed": seed_id, "split": split, "iou": threshold, "group": group,
                     **values, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                     "fp_per_image": fp / max(values["images"], 1)})
    return rows


def assignment_context(criterion, preds, batch):
    from ultralytics.utils.tal import make_anchors
    pred_dist = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    anchors, strides = make_anchors(preds["feats"], criterion.stride, 0.5)
    size = torch.tensor(preds["feats"][0].shape[2:], device=criterion.device, dtype=pred_scores.dtype) * criterion.stride[0]
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(targets, pred_scores.shape[0], scale_tensor=size[[1, 0, 1, 0]])
    labels, gt_boxes = targets.split((1, 4), 2)
    mask_gt = gt_boxes.sum(2, keepdim=True).gt_(0)
    pred_boxes = criterion.bbox_decode(anchors, pred_dist, preds.get("dfl_residual"), strides)
    _, target_boxes, target_scores, fg, _ = criterion.assigner(
        pred_scores.detach().sigmoid(), (pred_boxes.detach() * strides).type(gt_boxes.dtype),
        anchors * strides, labels, gt_boxes, mask_gt,
    )
    return pred_boxes * strides, target_boxes, target_scores, fg.bool(), preds["feats"][0].shape[2] * preds["feats"][0].shape[3]


def checkpoint_diagnostics(model, root: Path, boxes: dict[str, list[np.ndarray]], split: str, seed_id: int, variant: str) -> dict[str, object]:
    from ultralytics.nn.modules import DBSS
    module = next((item for item in model.model.modules() if isinstance(item, DBSS)), None)
    network = model.model
    network.train()
    device = next(network.parameters()).device
    for item in network.modules():
        if isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
            item.eval()
    criterion = network.init_criterion()
    sums = defaultdict(float); batches = 0
    paths = sorted((root / "images" / split).glob("*.png"))
    for start in range(0, len(paths), 8):
        group = paths[start:start + 8]
        images, batch_idx, classes, labels = [], [], [], []
        for index, path in enumerate(group):
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            images.append(torch.from_numpy(image).permute(2, 0, 1))
            for row in boxes[path.stem]:
                batch_idx.append(index); classes.append(0); labels.append(row)
        batch = {
            "img": torch.stack(images).to(device).float() / 255,
            "batch_idx": torch.tensor(batch_idx, device=device),
            "cls": torch.tensor(classes, device=device).reshape(-1, 1),
            "bboxes": torch.tensor(np.array(labels), device=device).reshape(-1, 4),
        }
        with torch.no_grad():
            preds = network(batch["img"])
            pred_boxes, target_boxes, scores, fg, n_p2 = assignment_context(criterion, preds, batch)
            p2_fg, p2_scores = fg[:, :n_p2], scores[:, :n_p2]
            sums["p2_positive"] += p2_fg.sum().item(); sums["total_positive"] += fg.sum().item()
            if p2_fg.any():
                sums["tal_score_sum"] += p2_scores.max(-1).values[p2_fg].sum().item()
                pboxes, tboxes = pred_boxes[:, :n_p2][p2_fg], target_boxes[:, :n_p2][p2_fg]
                intersection = torch.minimum(pboxes[:, 2:], tboxes[:, 2:]) - torch.maximum(pboxes[:, :2], tboxes[:, :2])
                intersection = intersection.clamp_min(0).prod(1)
                union = (pboxes[:, 2:] - pboxes[:, :2]).prod(1) + (tboxes[:, 2:] - tboxes[:, :2]).prod(1) - intersection
                sums["assigned_iou_sum"] += (intersection / union.clamp_min(1e-9)).sum().item()
            if module is not None and module.last_aux is not None:
                aux = module.last_aux
                pre_embedding, post_embedding = module._embed(aux["pre"]), module._embed(aux["post"])
                pre_ratios, post_ratios = [], []
                for index in range(pre_embedding.shape[0]):
                    pre_tokens = pre_embedding[index].flatten(1).T
                    post_tokens = post_embedding[index].flatten(1).T
                    candidates = torch.nn.functional.adaptive_avg_pool2d(
                        pre_embedding[index:index + 1], module.candidate_grid
                    ).flatten(2).squeeze(0).T
                    scores_for_bases = (
                        torch.nn.functional.normalize(candidates, dim=-1)
                        @ torch.nn.functional.normalize(pre_tokens, dim=-1).T
                    ).mean(1)
                    basis_indices = module._select_bases(scores_for_bases, torch.nn.functional.normalize(candidates, dim=-1))
                    bases = candidates[basis_indices]
                    pre_projection = module._project(pre_tokens, bases)
                    post_projection = module._project(post_tokens, bases)
                    pre_ratios.append((pre_tokens - pre_projection).square().sum(-1) / pre_tokens.square().sum(-1).clamp_min(1e-6))
                    post_ratios.append((post_tokens - post_projection).square().sum(-1) / post_tokens.square().sum(-1).clamp_min(1e-6))
                q_pre, q_post = torch.stack(pre_ratios), torch.stack(post_ratios)
                neg = ~p2_fg
                sums["q_pos_pre_sum"] += q_pre[p2_fg].sum().item(); sums["q_pos_post_sum"] += q_post[p2_fg].sum().item()
                sums["q_neg_pre_sum"] += q_pre[neg].sum().item(); sums["q_neg_post_sum"] += q_post[neg].sum().item()
                sums["negatives"] += neg.sum().item(); sums["displacement"] += aux["displacement_ratio"].item()
            batches += 1
    positive, negative = sums["p2_positive"], sums["negatives"]
    result = {"variant": variant, "seed": seed_id, "split": split,
              "f_P2": positive / max(sums["total_positive"], 1),
              "mean_tal_score_P2": sums["tal_score_sum"] / max(positive, 1),
              "mean_assigned_iou_P2": sums["assigned_iou_sum"] / max(positive, 1)}
    if module is not None:
        result.update({"q_pos_pre": sums["q_pos_pre_sum"] / max(positive, 1),
                       "q_pos_post": sums["q_pos_post_sum"] / max(positive, 1),
                       "q_neg_pre": sums["q_neg_pre_sum"] / max(negative, 1),
                       "q_neg_post": sums["q_neg_post_sum"] / max(negative, 1),
                       "displacement_ratio": sums["displacement"] / batches})
        result["delta_q_pos"] = result["q_pos_post"] - result["q_pos_pre"]
        result["G_pre"] = result["q_pos_pre"] - result["q_neg_pre"]
        result["G_post"] = result["q_pos_post"] - result["q_neg_post"]
    return result


def checkpoint_path(repo: str, remote: str) -> Path:
    return download(repo, remote)


def gpu_diagnostics(root: Path, boxes_by_split, out: Path) -> None:
    remote = Path("/marimo/yolo_code/models_related/ultralytics")
    local = remote if remote.is_dir() else Path(__file__).resolve().parents[1] / "models_related/ultralytics"
    sys.path.insert(0, str(local))
    from ultralytics import YOLO
    import ultralytics.nn.modules.head as head
    from ultralytics.nn.modules import DWConv

    if not hasattr(head, "P2OffsetRegression"):
        class P2OffsetRegression(torch.nn.Module):
            """Checkpoint compatibility shim; matches the training revision."""

            def __init__(self, old_head, reg_max: int, rho: float = 0.5):
                super().__init__()
                self.reg_max, self.rho = reg_max, rho
                self.stem = torch.nn.Sequential(old_head[0], old_head[1])
                channels = old_head[1].conv.out_channels
                self.offset = torch.nn.Sequential(DWConv(channels, channels, 3), torch.nn.Conv2d(channels, 8, 1))
                self.sides = torch.nn.ModuleList(torch.nn.Conv2d(channels, reg_max, 1) for _ in range(4))

            def forward(self, x):
                import torch.nn.functional as functional
                feature = self.stem(x)
                batch, _, height, width = feature.shape
                offset = self.rho * self.offset(feature).tanh().reshape(batch, 4, 2, height, width)
                yy, xx = torch.meshgrid(torch.arange(height, device=x.device, dtype=x.dtype),
                                        torch.arange(width, device=x.device, dtype=x.dtype), indexing="ij")
                base = torch.stack((2 * (xx + 0.5) / width - 1, 2 * (yy + 0.5) / height - 1), dim=-1)
                grids = base[None, None] + torch.stack((2 * offset[:, :, 0] / width, 2 * offset[:, :, 1] / height), dim=-1)
                sampled = functional.grid_sample(
                    feature[:, None].expand(-1, 4, -1, -1, -1).reshape(4 * batch, -1, height, width),
                    grids.reshape(4 * batch, height, width, 2), mode="bilinear", padding_mode="border",
                    align_corners=False,
                ).reshape(batch, 4, -1, height, width)
                return torch.cat([self.sides[index](sampled[:, index]) for index in range(4)], dim=1)

        P2OffsetRegression.__module__ = head.__name__
        setattr(head, "P2OffsetRegression", P2OffsetRegression)
    diagnostics, subgroups = [], []
    for seed_id in (42, 43, 44):
        specs = {
            "baseline": (BASE_REPO, f"train/yolov8n_p2_baseline_seed{seed_id}/weights/best.pt"),
            "aware": (AWARE_REPO, f"runs/yolov8n/dbss_p2_aware/seed_{seed_id}/weights/best.pt"),
        }
        for variant, (repo, remote) in specs.items():
            model = YOLO(checkpoint_path(repo, remote))
            model.model.to("cuda")
            strides = [float(value) for value in model.model.stride.tolist()]
            if strides != [4.0, 8.0, 16.0, 32.0]:
                raise RuntimeError(f"{variant} seed {seed_id}: unexpected strides {strides}")
            for split in ("val", "test"):
                diagnostics.append(checkpoint_diagnostics(model, root, boxes_by_split[split], split, seed_id, variant))
                model.model.eval()
                subgroups.extend(subgroup_diagnostics(model, root, boxes_by_split[split], split, seed_id, variant))
    write_csv(out / "checkpoint_diagnostics.csv", diagnostics)
    write_csv(out / "subgroup_metrics.csv", subgroups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/levir_dbss_generalization"))
    parser.add_argument("--skip-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paired_tables(args.output)
    epoch_curves(args.output)
    root, boxes = dataset_audit(args.data, args.output)
    if not args.skip_gpu:
        gpu_diagnostics(root, boxes, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
