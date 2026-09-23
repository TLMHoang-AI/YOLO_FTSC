"""Partner-compatible standard and size-bucket VisDrone checkpoint evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluate_test.size_bucket_evaluator import evaluate_native_test_size_buckets

STANDARD_METRIC_KEYS = ("val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95")


def _metric(result: Any, key: str) -> float:
    value = getattr(result, "results_dict", {}).get(key)
    if value is None:
        raise RuntimeError(f"Evaluator did not provide {key}")
    return float(value)


def evaluate_standard(
    model: Any,
    data_yaml: Path,
    run_dir: Path,
    *,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cuda",
    workers: int = 8,
) -> dict[str, float | str]:
    """Evaluate native ten-class val and official test-dev splits."""
    metrics: dict[str, float | str] = {"nms_iou": 0.5, "standard_eval/class_agnostic": False}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=imgsz, batch=batch,
            device=device, workers=workers, iou=0.5, plots=False,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics[f"{split}/AP50"] = _metric(result, "metrics/mAP50(B)")
        metrics[f"{split}/mAP50-95"] = _metric(result, "metrics/mAP50-95(B)")
    return metrics


def evaluate_checkpoint(
    run_dir: Path,
    data_yaml: Path,
    *,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cuda",
    workers: int = 8,
) -> dict[str, float | str]:
    from ultralytics import YOLO

    checkpoint = run_dir / "weights/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    metrics = evaluate_standard(
        YOLO(checkpoint), data_yaml, run_dir,
        imgsz=imgsz, batch=batch, device=device, workers=workers,
    )
    metrics.update(evaluate_native_test_size_buckets(
        run_dir, data_yaml, imgsz=imgsz, batch=batch, device=device, workers=workers,
    ))
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


__all__ = ["STANDARD_METRIC_KEYS", "evaluate_checkpoint", "evaluate_standard"]
