#!/usr/bin/env python3
"""Plot qualitative cases comparing SA-YOLO against trained YOLOv8n."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "models_related" / "ultralytics"

DGFE_REPO = "duyle2408/varroa-yolo-under-2m-wiou"
DGFE_FILE = "train_local_detail_dgfe_sweep/yolov8_varroa_local_detail_p3only_dgfe_ultra_n_seed43/weights/best.pt"
BASELINE_REPO = "duyle2408/varroa-yolo-baselines-part1-full"
BASELINE_FILE = "train/yolov8n_seed43/weights/best.pt"


def prefer_local_ultralytics() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            del sys.modules[name]


prefer_local_ultralytics()

from ultralytics import YOLO  # noqa: E402


@dataclass
class Pred:
    box: list[float] | None
    conf: float
    iou: float


@dataclass
class Case:
    title: str
    image: Path
    gt_boxes: list[list[float]]
    dgfe: Pred
    baseline: Pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgfe-repo", default=DGFE_REPO)
    parser.add_argument("--dgfe-file", default=DGFE_FILE)
    parser.add_argument("--baseline-repo", default=BASELINE_REPO)
    parser.add_argument("--baseline-file", default=BASELINE_FILE)
    parser.add_argument("--dgfe-weight", type=Path)
    parser.add_argument("--baseline-weight", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "varroa_yolo" / "varroa.yaml")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "visualize" / "dgfe_vs_trained_yolov8n_seed43_4cases.png")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def download_weight(repo_id: str, filename: str, local_path: Path | None) -> Path:
    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required unless a local weight path is provided") from exc

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))
    except Exception as exc:
        try:
            from huggingface_hub import list_repo_files

            checkpoints = [
                path
                for path in list_repo_files(repo_id, repo_type="dataset")
                if path.endswith((".pt", ".pth", ".onnx", ".engine"))
            ]
        except Exception:
            checkpoints = []
        hint = f" Available checkpoint-like files: {checkpoints[:20]}" if checkpoints else " No checkpoint-like files were listed in the repo."
        raise RuntimeError(f"Could not download {repo_id}:{filename}.{hint}") from exc


def read_dataset(data_yaml: Path, split: str) -> tuple[Path, list[Path]]:
    data_yaml = data_yaml.expanduser().resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    image_dir = Path(data[split])
    if not image_dir.is_absolute():
        image_dir = root / image_dir

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return root, images


def label_path(root: Path, split: str, image: Path) -> Path:
    return root / "labels" / split / f"{image.stem}.txt"


def yolo_labels_to_xyxy(path: Path, size: tuple[int, int]) -> list[list[float]]:
    width, height = size
    boxes = []
    if not path.is_file():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        boxes.append([x1, y1, x2, y2])
    return boxes


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def best_prediction(model: YOLO, image: Path, gt_boxes: list[list[float]], args: argparse.Namespace) -> Pred:
    result = model.predict(
        source=str(image),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return Pred(None, 0.0, 0.0)

    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    best = Pred(None, 0.0, 0.0)
    for box, conf in zip(boxes, confs):
        iou = max((box_iou(box, gt) for gt in gt_boxes), default=0.0)
        if iou > best.iou or (iou == best.iou and conf > best.conf):
            best = Pred([float(v) for v in box], float(conf), float(iou))
    return best


def add_case(cases: dict[str, list[Case]], key: str, case: Case) -> None:
    limits = {"dgfe_only": 2, "better_iou": 2}
    if len(cases[key]) < limits[key]:
        cases[key].append(case)


def complete(cases: dict[str, list[Case]]) -> bool:
    return len(cases["dgfe_only"]) >= 2 and len(cases["better_iou"]) >= 2


def find_cases(args: argparse.Namespace, dgfe_model: YOLO, baseline_model: YOLO) -> list[Case]:
    root, images = read_dataset(args.data, args.split)
    cases: dict[str, list[Case]] = {"dgfe_only": [], "better_iou": []}

    for index, image in enumerate(images, start=1):
        if args.max_images is not None and index > args.max_images:
            break

        with Image.open(image) as im:
            gt_boxes = yolo_labels_to_xyxy(label_path(root, args.split, image), im.size)
        if not gt_boxes:
            continue

        dgfe = best_prediction(dgfe_model, image, gt_boxes, args)
        baseline = best_prediction(baseline_model, image, gt_boxes, args)
        dgfe_correct = dgfe.iou >= args.iou_thres
        baseline_correct = baseline.iou >= args.iou_thres

        if dgfe.box is not None and baseline.box is None:
            add_case(cases, "dgfe_only", Case("SA-YOLO predicts, YOLOv8n misses", image, gt_boxes, dgfe, baseline))
        elif dgfe_correct and baseline_correct and dgfe.iou > baseline.iou:
            add_case(cases, "better_iou", Case("Both correct, SA-YOLO localizes better", image, gt_boxes, dgfe, baseline))

        if complete(cases):
            break

    if not complete(cases):
        counts = {key: len(value) for key, value in cases.items()}
        raise RuntimeError(f"Found too few cases with conf={args.conf}, iou_thres={args.iou_thres}: {counts}")

    return [*cases["dgfe_only"][:2], *cases["better_iou"][:2]]


def draw_box(ax: Any, box: list[float], color: str, label: str, label_pos: str = "top") -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
    if label_pos == "bottom":
        ax.text(x1, y2 + 3, label, color=color, fontsize=8, weight="bold", va="top")
    else:
        ax.text(x1, max(0, y1 - 3), label, color=color, fontsize=8, weight="bold", va="bottom")


def plot_cases(cases: list[Case], out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10, 9), constrained_layout=True)
    for index, case in enumerate(cases):
        row = index // 2
        col = (index % 2) * 2
        gt_ax, pred_ax = axes[row, col], axes[row, col + 1]
        with Image.open(case.image) as im:
            image = im.convert("RGB")

        gt_ax.imshow(image)
        gt_ax.axis("off")
        gt_ax.set_title("Ground truth", fontsize=12)
        for gt in case.gt_boxes:
            draw_box(gt_ax, gt, "lime", "GT")

        pred_ax.imshow(image)
        pred_ax.axis("off")
        pred_ax.set_title("Prediction", fontsize=12)
        if case.dgfe.box is not None:
            draw_box(pred_ax, case.dgfe.box, "yellow", "SA-YOLO")
        if case.baseline.box is not None:
            draw_box(pred_ax, case.baseline.box, "cyan", "YOLOv8n", label_pos="bottom")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dgfe_weight = download_weight(args.dgfe_repo, args.dgfe_file, args.dgfe_weight)
    baseline_weight = download_weight(args.baseline_repo, args.baseline_file, args.baseline_weight)

    print(f"SA-YOLO weight: {dgfe_weight}")
    print(f"Baseline weight: {baseline_weight}")
    print(f"Ultralytics: {sys.modules['ultralytics'].__file__}")

    dgfe_model = YOLO(str(dgfe_weight))
    baseline_model = YOLO(str(baseline_weight))
    cases = find_cases(args, dgfe_model, baseline_model)
    plot_cases(cases, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
