#!/usr/bin/env python3
"""Shared additive detection evaluation for native LEVIR-Ship and Varroa splits.

This module deliberately keeps three protocols separate: Ultralytics detector
metrics (``{split}/metrics/*``), standard COCO area metrics (``test_coco/*``),
and the existing TinyBenchmark evaluator (``test_size/*``).
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = (PROJECT_ROOT / "models_related" / "ultralytics").resolve()


def ensure_local_ultralytics() -> str:
    """Bind evaluation to this repository's fork, never pip Ultralytics."""
    if str(LOCAL_ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(LOCAL_ULTRALYTICS))
    loaded = sys.modules.get("ultralytics")
    if loaded is None:
        loaded = importlib.import_module("ultralytics")
    location = Path(getattr(loaded, "__file__", "")).resolve()
    if LOCAL_ULTRALYTICS not in location.parents:
        raise RuntimeError(f"external Ultralytics is loaded from {location}; expected under {LOCAL_ULTRALYTICS}")
    return str(location)


def resolve_device(device: str, torch: Any) -> tuple[str, int | None]:
    requested = str(device).strip().lower()
    if requested == "cpu" or not torch.cuda.is_available(): return "cpu", None
    index = int(requested.split(":", 1)[1]) if requested.startswith("cuda:") else int(requested) if requested.isdigit() else 0
    if index >= torch.cuda.device_count(): raise ValueError(f"requested CUDA device {index}, only {torch.cuda.device_count()} available")
    return f"cuda:{index}", index


def actual_precision(model: Any) -> str:
    """Prefer validator/backend state; dtype is only a final compatibility fallback."""
    for owner in (getattr(model, "validator", None), getattr(model, "predictor", None)):
        backend = getattr(owner, "model", None)
        if getattr(backend, "fp16", None) is True or getattr(getattr(owner, "args", None), "half", None) is True: return "FP16"
        if getattr(backend, "fp16", None) is False or getattr(getattr(owner, "args", None), "half", None) is False: return "FP32"
    return "FP16" if str(next(model.model.parameters()).dtype).endswith("float16") else "FP32"

COCO_AREA_RANGES = {"all": (0.0, 1e10), "small": (0.0, 32.0**2), "medium": (32.0**2, 96.0**2), "large": (96.0**2, 1e10)}
COCO_IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(10))
MAP50_75_IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(6))
TINYBENCHMARK_PROTOCOL = "Tiny1=1-8px; Tiny2=8-12px; Tiny3=12-20px; Small=20-32px; Medium=32-96px; IoU=0.50:0.05:0.75"


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_divide(2.0 * precision * recall, precision + recall)


def map50_75(result: Any) -> float:
    """Partner-compatible mean AP at exactly IoU .50, .55, ..., .75."""
    all_ap = result.box.all_ap
    if getattr(all_ap, "ndim", 0) != 2 or all_ap.shape[1] < len(MAP50_75_IOU_THRESHOLDS):
        raise ValueError(f"expected all_ap[:, :6], got {getattr(all_ap, 'shape', None)}")
    return float(all_ap[:, :6].mean())


def standard_metrics(result: Any, split: str) -> dict[str, float]:
    raw = result.results_dict
    p = float(raw["metrics/precision(B)"])
    r = float(raw["metrics/recall(B)"])
    return {
        f"{split}/metrics/precision(B)": p, f"{split}/metrics/recall(B)": r,
        f"{split}/metrics/F1(B)": f1_score(p, r),
        f"{split}/metrics/mAP50(B)": float(raw["metrics/mAP50(B)"]),
        f"{split}/metrics/mAP75(B)": float(result.box.map75),
        f"{split}/metrics/mAP50-95(B)": float(raw["metrics/mAP50-95(B)"]),
        f"{split}/metrics/mAP50-75(B)": map50_75(result),
    }


def speed_metrics(speed: dict[str, Any]) -> dict[str, float]:
    preprocess = float(speed.get("preprocess", 0.0)); inference = float(speed.get("inference", 0.0)); postprocess = float(speed.get("postprocess", 0.0))
    total = preprocess + inference + postprocess
    return {
        "test_speed/preprocess_ms_per_image": preprocess, "test_speed/inference_ms_per_image": inference,
        "test_speed/postprocess_ms_per_image": postprocess, "test_speed/total_ms_per_image": total,
        "test_speed/inference_only_fps": safe_divide(1000.0, inference), "test_speed/end_to_end_fps": safe_divide(1000.0, total),
    }


def benchmark_provenance(*, device: str, imgsz: int, batch: int, workers: int, precision: str, split: str) -> dict[str, str | int | bool]:
    import torch
    resolved, index = resolve_device(device, torch)
    return {
        "benchmark/framework": "Ultralytics", "benchmark/backend": "PyTorch", "benchmark/checkpoint_format": "PyTorch .pt", "benchmark/device": resolved,
        "benchmark/gpu_name": torch.cuda.get_device_name(index) if index is not None else "CPU",
        "benchmark/cuda_version": str(torch.version.cuda or "none"), "benchmark/pytorch_version": str(torch.__version__),
        "benchmark/python_version": platform.python_version(), "benchmark/imgsz": int(imgsz),
        "benchmark/batch": int(batch), "benchmark/workers_requested": int(workers), "benchmark/precision": precision,
        "benchmark/half": precision == "FP16", "benchmark/split": split,
    }


def model_complexity(model: Any, imgsz: int) -> dict[str, float | int]:
    ensure_local_ultralytics()
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    inner = model.model
    params = int(get_num_params(inner))
    return {"model/parameters": params, "model/parameters_M": params / 1e6,
            "model/GFLOPs": float(get_flops(inner, imgsz=imgsz) or 0.0), "model/GFLOPs_imgsz": int(imgsz)}


def _test_images(data_yaml: Path) -> list[Path]:
    from evaluate_test.size_bucket_evaluator import _resolve_test_images
    return _resolve_test_images(data_yaml)


def _yolo_label_path(image: Path) -> Path:
    from evaluate_test.size_bucket_evaluator import _label_path
    return _label_path(image)


def _coco_ground_truth(images: list[Path]) -> dict[str, Any]:
    from PIL import Image
    records: list[dict[str, Any]] = []; annotations: list[dict[str, Any]] = []; categories: set[int] = set(); ann_id = 1
    for image_id, image_path in enumerate(images, 1):
        with Image.open(image_path) as picture: width, height = picture.size
        records.append({"id": image_id, "file_name": str(image_path), "width": width, "height": height})
        label = _yolo_label_path(image_path)
        if not label.is_file(): continue
        for line in label.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5: continue
            cls, cx, cy, bw, bh = map(float, values); bw *= width; bh *= height
            if bw <= 0 or bh <= 0: continue
            x, y = (cx * width - bw / 2), (cy * height - bh / 2); category_id = int(cls) + 1; categories.add(category_id)
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": category_id, "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0}); ann_id += 1
    return {"images": records, "annotations": annotations, "categories": [{"id": category, "name": str(category - 1)} for category in sorted(categories)]}


def _coco_predictions(model: Any, images: list[Path], *, imgsz: int, batch: int, device: str, workers: int, nms_iou: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for image_id, result in enumerate(model.predict(source=[str(path) for path in images], stream=True, imgsz=imgsz, batch=batch, device=device, workers=workers, conf=0.001, iou=nms_iou, verbose=False, save=False), 1):
        boxes = result.boxes
        if boxes is None: continue
        for box, score, cls in zip(boxes.xyxy.detach().cpu().tolist(), boxes.conf.detach().cpu().tolist(), boxes.cls.detach().cpu().tolist(), strict=True):
            x1, y1, x2, y2 = box
            output.append({"image_id": image_id, "category_id": int(cls) + 1, "bbox": [float(x1), float(y1), float(max(0, x2-x1)), float(max(0, y2-y1))], "score": float(score)})
    return output


def coco_area_metrics(model: Any, data_yaml: Path, *, imgsz: int, batch: int, device: str, workers: int, nms_iou: float) -> dict[str, float | str]:
    """Standard pycocotools bbox AP on original native test images and labels."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import tempfile
    images = _test_images(data_yaml)
    if not images: raise FileNotFoundError(f"No native test images in {data_yaml}")
    ground_truth, predictions = _coco_ground_truth(images), _coco_predictions(model, images, imgsz=imgsz, batch=batch, device=device, workers=workers, nms_iou=nms_iou)
    with tempfile.TemporaryDirectory(prefix="native_coco_") as temporary:
        gt_path = Path(temporary) / "gt.json"; gt_path.write_text(json.dumps(ground_truth), encoding="utf-8")
        coco_gt = COCO(str(gt_path))
        if predictions: coco_dt = coco_gt.loadRes(predictions)
        else:
            coco_dt = COCO(); coco_dt.dataset = {"images": ground_truth["images"], "categories": ground_truth["categories"], "annotations": []}; coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox"); evaluator.evaluate(); evaluator.accumulate()
        precision = evaluator.eval["precision"]
        def ap(iou: float | None, area: str) -> float:
            values = precision if iou is None else precision[[index for index, threshold in enumerate(evaluator.params.iouThrs) if abs(threshold-iou) < 1e-9]]
            values = values[:, :, :, evaluator.params.areaRngLbl.index(area), -1]; values = values[values > -1]
            return float(values.mean()) if values.size else -1.0
        return {"test_coco/mAP50-95": ap(None, "all"), "test_coco/AP50": ap(.50, "all"), "test_coco/AP75": ap(.75, "all"), "test_coco/AP_small": ap(None, "small"), "test_coco/AP_medium": ap(None, "medium"), "test_coco/AP_large": ap(None, "large"), "test_coco/protocol": "standard COCO bbox; area small<32^2, medium=32^2..96^2, large>96^2; IoU=.50:.05:.95"}


def evaluate_run(run_dir: Path, data_yaml: Path, *, dataset: str, imgsz: int, batch: int, device: str, workers: int, nms_iou: float = 0.5, include_size: bool = True) -> dict[str, Any]:
    """Evaluate ``best.pt`` and write a new additive extended artifact only."""
    if dataset not in {"levir", "varroa"}: raise ValueError("dataset must be levir or varroa")
    ultralytics_path = ensure_local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(run_dir / "weights/best.pt")
    # Must happen before val()/predict(), whose backend may fuse Conv+BN.
    complexity = model_complexity(model, imgsz)
    metrics: dict[str, Any] = {"evaluation/checkpoint": "weights/best.pt", "evaluation/dataset": dataset, "evaluation/nms_iou": nms_iou}
    for split in ("val", "test"):
        result = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device, workers=workers, plots=False, iou=nms_iou, project=str(run_dir / "evaluation_extended"), name=split, exist_ok=True)
        metrics.update(standard_metrics(result, split))
        if split == "test":
            # Ultralytics exposes timings in ms/image. Its validator preserves model dtype for this run.
            precision = actual_precision(model)
            metrics.update(speed_metrics(result.speed)); metrics.update(benchmark_provenance(device=device, imgsz=imgsz, batch=batch, workers=workers, precision=precision, split="test")); metrics["runtime/ultralytics_path"] = ultralytics_path
    metrics.update(coco_area_metrics(model, data_yaml, imgsz=imgsz, batch=batch, device=device, workers=workers, nms_iou=nms_iou)); metrics.update(complexity)
    if include_size:
        from evaluate_test.size_bucket_evaluator import evaluate_native_test_size_buckets
        metrics.update(evaluate_native_test_size_buckets(run_dir, data_yaml, imgsz=imgsz, batch=batch, device=device, workers=workers, dataset=dataset))
    (run_dir / "evaluation_metrics_extended.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def load_merged_metrics(run_dir: Path) -> dict[str, Any]:
    """Merge additive data, failing rather than replacing legacy values."""
    metrics: dict[str, Any] = {}
    for name in ("evaluation_metrics.json", "evaluation_metrics_extended.json"):
        path = run_dir / name
        if not path.is_file(): continue
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if key not in metrics: metrics[key] = value; continue
            old = metrics[key]
            if isinstance(old, (int, float)) and isinstance(value, (int, float)) and abs(float(old) - float(value)) <= 1e-9: continue
            if old != value: raise ValueError(f"extended evaluation conflicts with legacy metric {key!r}: {old!r} != {value!r}")
    return metrics


def same_benchmark_fingerprint(rows: list[dict[str, Any]]) -> bool:
    keys = ("benchmark/gpu_name", "benchmark/backend", "benchmark/device", "benchmark/imgsz", "benchmark/batch", "benchmark/precision")
    return len({tuple(row.get(key) for key in keys) for row in rows}) == 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--data-yaml", type=Path, required=True); parser.add_argument("--dataset", choices=("levir", "varroa"), required=True)
    parser.add_argument("--imgsz", type=int, required=True); parser.add_argument("--batch", type=int, required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--nms-iou", type=float, default=.5); parser.add_argument("--no-size", action="store_true")
    args = parser.parse_args(argv); print(json.dumps(evaluate_run(args.run_dir, args.data_yaml, dataset=args.dataset, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=args.workers, nms_iou=args.nms_iou, include_size=not args.no_size), indent=2, sort_keys=True))


if __name__ == "__main__": main()
