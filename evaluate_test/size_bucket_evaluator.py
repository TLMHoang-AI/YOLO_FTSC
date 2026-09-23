"""TinyBenchmark-compatible, class-agnostic size evaluation on native YOLO test images."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

AREA_LABELS = ("all", "tiny", "tiny1", "tiny2", "tiny3", "small", "medium", "reasonable")
AREA_RANGES = (
    (1**2, 1e5**2),
    (1**2, 20**2),
    (1**2, 8**2),
    (8**2, 12**2),
    (12**2, 20**2),
    (20**2, 32**2),
    (32**2, 96**2),
    (32**2, 1e5**2),
)
IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(6))
BUCKET_LABELS = ("Tiny1", "Tiny2", "Tiny3", "Small", "Medium")
SIZE_METRIC_KEYS = (
    "test_size/available",
    "test_size/AP50",
    "test_size/AP75",
    "test_size/mAP50-75",
    *(f"test_size/AP50-{name}" for name in BUCKET_LABELS),
    *(f"test_size/AP-{name}" for name in BUCKET_LABELS),
    "test_size/protocol",
    "test_size/dataset",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "vendor/tinyperson_cocoeval.py"


def resolve_evaluator_path() -> Path:
    if not EVALUATOR_PATH.is_file():
        raise FileNotFoundError(f"TinyBenchmark evaluator is missing: {EVALUATOR_PATH}")
    return EVALUATOR_PATH


def _resolve_test_images(data_yaml: Path) -> list[Path]:
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    test = Path(config["test"])
    if not test.is_absolute():
        test = root / test
    if test.is_file():
        images = [Path(line.strip()) for line in test.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [path if path.is_absolute() else (test.parent / path).resolve() for path in images]
    return sorted(
        path for path in test.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
        parts[image_index] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image_path.with_suffix(".txt")


def _read_yolo_labels(label_path: Path, width: int, height: int) -> list[list[float]]:
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        _, cx, cy, box_width, box_height = map(float, values)
        x = (cx - box_width / 2.0) * width
        y = (cy - box_height / 2.0) * height
        width_px, height_px = box_width * width, box_height * height
        if width_px > 0 and height_px > 0:
            boxes.append([x, y, width_px, height_px])
    return boxes


def _coco_ground_truth(image_paths: list[Path]) -> dict[str, Any]:
    """Emit every native class as category 1 for the secondary localization diagnostic."""
    from PIL import Image

    images, annotations = [], []
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": str(image_path), "width": width, "height": height})
        for x, y, box_width, box_height in _read_yolo_labels(_label_path(image_path), width, height):
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            })
            annotation_id += 1
    return {
        "info": {"description": "Native YOLO test split with TinyBenchmark area buckets"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "object"}],
    }


def _predictions(
    model: Any,
    image_paths: list[Path],
    *,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Run partner-matched prediction settings and collapse all classes to category 1."""
    detections = []
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        iou=0.5,
        verbose=False,
        stream=True,
    )
    for image_id, result in enumerate(results, 1):
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
        scores = boxes.conf.detach().cpu().tolist() if boxes is not None else []
        for (x1, y1, x2, y2), score in zip(xyxy, scores):
            detections.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })
    return detections


def _precision_ap(precision: Any, iou_index: int, area_index: int) -> float:
    import numpy as np

    values = np.asarray(precision[iou_index, :, :, area_index, -1])
    values = values[values > -1]
    return float(values.mean()) if values.size else -1.0


def _evaluate_coco(gt_path: Path, prediction_path: Path) -> dict[str, float]:
    import numpy as np
    from pycocotools.coco import COCO

    evaluator_path = resolve_evaluator_path()
    spec = importlib.util.spec_from_file_location("tinyperson_cocoeval_native", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous_standard = module.Params.EVAL_STRANDARD
    module.Params.EVAL_STRANDARD = "tiny"
    old_linspace = module.np.linspace
    had_np_float = hasattr(module.np, "float")
    if not had_np_float:
        module.np.float = float

    def compatible_linspace(start, stop, num=50, *args, **kwargs):
        if isinstance(num, (float, np.floating)):
            num = int(num)
        return old_linspace(start, stop, num, *args, **kwargs)

    module.np.linspace = compatible_linspace
    try:
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(prediction_path))
        evaluator = module.COCOeval(coco_gt, coco_dt, "bbox", True, True)
        evaluator.params.iouThrs = np.asarray(IOU_THRESHOLDS, dtype=float)
        evaluator.evaluate()
        evaluator.accumulate()
        precision = evaluator.eval["precision"]
        area_indices = {name: evaluator.params.areaRngLbl.index(name) for name in AREA_LABELS}
        all_ap = [_precision_ap(precision, index, area_indices["all"]) for index in range(6)]
        valid_all = [value for value in all_ap if value > -1]
        metrics = {
            "test_size/available": 1.0,
            "test_size/AP50": all_ap[0],
            "test_size/AP75": all_ap[-1],
            "test_size/mAP50-75": float(np.mean(valid_all)) if valid_all else -1.0,
        }
        for name in BUCKET_LABELS:
            area_index = area_indices[name.lower()]
            bucket_ap = [_precision_ap(precision, index, area_index) for index in range(6)]
            valid_bucket = [value for value in bucket_ap if value > -1]
            metrics[f"test_size/AP50-{name}"] = bucket_ap[0]
            metrics[f"test_size/AP-{name}"] = float(np.mean(valid_bucket)) if valid_bucket else -1.0
        return metrics
    finally:
        module.np.linspace = old_linspace
        if not had_np_float:
            del module.np.float
        module.Params.EVAL_STRANDARD = previous_standard


def evaluate_native_test_size_buckets(
    run_dir: Path,
    data_yaml: Path,
    *,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
) -> dict[str, float | str]:
    from ultralytics import YOLO

    image_paths = _resolve_test_images(data_yaml)
    if not image_paths:
        raise FileNotFoundError(f"No test images resolved from {data_yaml}")
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    gt_path = evaluation_dir / "test_size_ground_truth.json"
    prediction_path = evaluation_dir / "test_size_predictions.json"
    gt_path.write_text(json.dumps(_coco_ground_truth(image_paths), indent=2) + "\n", encoding="utf-8")
    predictions = _predictions(
        YOLO(run_dir / "weights/best.pt"), image_paths,
        imgsz=imgsz, batch=batch, device=device, workers=workers,
    )
    prediction_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    metrics = _evaluate_coco(gt_path, prediction_path)
    metrics["test_size/protocol"] = (
        "TinyBenchmark area buckets on native YOLO test images; "
        "class-agnostic; IoU=0.50:0.05:0.75; maxDets=200"
    )
    metrics["test_size/dataset"] = "visdrone"
    return metrics


__all__ = [
    "AREA_LABELS", "AREA_RANGES", "BUCKET_LABELS", "EVALUATOR_PATH", "IOU_THRESHOLDS", "SIZE_METRIC_KEYS",
    "evaluate_native_test_size_buckets", "resolve_evaluator_path",
]
