#!/usr/bin/env python3
"""Partner-style deployment inference and COCO scoring, isolated from native Ultralytics AP.

Source protocol: NguyenDE180300/small-obj-detection (main, 03e1f6baa13f14fb573334e0e7d2ec1f9fddf8c5),
inference_bundle/code/{benchmark_jetson.py,evaluate_saved_predictions.py}.
Every backend independently predicts the SAME original test images using
conf=0.001, NMS IoU=0.7, max_det=300, batch=1. Saved predictions feed a
single backend-independent COCOeval, plus the existing native TinyBenchmark
AP50-Small diagnostic. COCO AP must never be represented as Ultralytics AP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PARTNER_REPO = "NguyenDE180300/small-obj-detection"
PARTNER_COMMIT = "03e1f6baa13f14fb573334e0e7d2ec1f9fddf8c5"
PREDICT_CONF = 0.001
PREDICT_IOU = 0.7
PREDICT_MAX_DET = 300
PREDICT_BATCH = 1


def partner_coco_metrics(gt_path: Path, pred_path: Path) -> dict[str, float]:
    """Exact partner all-object COCO AP protocol (IoU .50:.05:.95, maxDets=300)."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(gt_path))
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {
            "images": ground_truth.dataset["images"],
            "categories": ground_truth.dataset["categories"],
            "annotations": [],
        }
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.params.iouThrs = np.linspace(0.50, 0.95, 10)
    evaluator.params.areaRng = [[0.0, 1e10]]
    evaluator.params.areaRngLbl = ["all"]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]

    def valid_mean(values: Any) -> float:
        valid = values[values > -1]
        if not valid.size:
            raise RuntimeError("Partner COCOeval has no valid annotated precision bucket")
        return float(valid.mean())

    return {
        "test_coco/AP50": valid_mean(precision[0]),
        "test_coco/mAP50-95": valid_mean(precision),
        "test_coco/AP75": valid_mean(precision[5]),
    }


def partner_native_metrics(
    args: Any, model_file: Path, backend: str, output_dir: Path, imgsz: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Independent backend predictions; one fixed COCO and TinyBenchmark scoring."""
    from PIL import Image
    from evaluate_test.standard_detection_metrics import (
        _coco_ground_truth as class_aware_ground_truth,
        ensure_local_ultralytics,
    )
    from evaluate_test.size_bucket_evaluator import (
        _resolve_test_images, _coco_ground_truth as class_agnostic_ground_truth,
        _evaluate_coco as tinybenchmark_metrics,
    )

    ensure_local_ultralytics()
    from ultralytics import YOLO

    from utils.ftsc_deployment_backend_eval_v1 import backend_device, require, sha256

    yaml_path = Path(args.data_yaml_template.format(seed=args.current_seed)).expanduser().resolve()
    require(yaml_path.is_file(), f"Missing frozen native test YAML: {yaml_path}")
    images = _resolve_test_images(yaml_path)
    require(bool(images) and all(p.is_file() for p in images), "Frozen native test images unavailable")

    output_dir.mkdir(parents=True, exist_ok=True)
    coco_gt_path = output_dir / "partner_test_coco_ground_truth.json"
    small_gt_path = output_dir / "partner_test_size_ground_truth.json"
    pred_path = output_dir / "partner_test_predictions.json"
    small_pred_path = output_dir / "partner_test_size_predictions.json"

    coco_gt = class_aware_ground_truth(images)
    coco_gt.setdefault("info", {"description": "Frozen native YOLO test converted to COCO"})
    coco_gt.setdefault("licenses", [])
    coco_gt_path.write_text(json.dumps(coco_gt) + "\n", encoding="utf-8")
    small_gt_path.write_text(json.dumps(class_agnostic_ground_truth(images)) + "\n", encoding="utf-8")

    model = YOLO(str(model_file), task="detect")
    predictions: list[dict[str, Any]] = []
    small_predictions: list[dict[str, Any]] = []
    for image_id, image_path in enumerate(images, 1):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            result = model.predict(
                source=rgb, imgsz=imgsz, batch=PREDICT_BATCH,
                device=backend_device(backend, args.device),
                conf=PREDICT_CONF, iou=PREDICT_IOU,
                max_det=PREDICT_MAX_DET, half=backend.endswith("16"),
                rect=False,  # same fixed letterbox tensor shape for .pt and static .engine
                verbose=False, save=False,
            )[0]
        boxes = result.boxes
        if boxes is None:
            continue
        coords = boxes.xyxy.detach().cpu().tolist()
        scores = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        for box, score, cls in zip(coords, scores, classes, strict=True):
            x1, y1, x2, y2 = map(float, box)
            base = {
                "image_id": image_id,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            }
            predictions.append({**base, "category_id": int(cls) + 1})
            small_predictions.append({**base, "category_id": 1})

    pred_path.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    small_pred_path.write_text(json.dumps(small_predictions) + "\n", encoding="utf-8")

    metrics = partner_coco_metrics(coco_gt_path, pred_path)
    metrics.update(tinybenchmark_metrics(small_gt_path, small_pred_path))
    metrics["evaluation/prediction_protocol"] = "partner_predict_conf001_iou07_maxdet300_batch1"
    metrics["evaluation/coco_protocol"] = "COCOeval_iou50_95_maxdet300"
    return metrics, {
        "data_yaml_sha256": sha256(yaml_path),
        "gt_sha256": sha256(coco_gt_path),
        "small_gt_sha256": sha256(small_gt_path),
        "predictions_sha256": sha256(pred_path),
        "small_predictions_sha256": sha256(small_pred_path),
    }
