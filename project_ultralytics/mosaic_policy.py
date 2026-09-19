"""Source-faithful helpers used by the M5 hard-negative Mosaic policy.

Adapted from Duy's ``project_ultralytics.mosaic_policy``.  Only the helpers
required by M5 are carried here; detector and unrelated Mosaic policies stay
out of the FTSC fork.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MosaicProposal:
    donor_indices: tuple[int, ...]
    xc: int
    yc: int
    visibility: np.ndarray | None = None
    effective_count: float | None = None
    score: float | None = None


def resized_shape(shape: Iterable[int], imgsz: int) -> tuple[int, int]:
    """Return the long-side-preserving shape used by BaseDataset.load_image."""
    h, w = int(shape[0]), int(shape[1])
    ratio = float(imgsz) / max(h, w)
    return min(int(np.ceil(h * ratio)), imgsz), min(int(np.ceil(w * ratio)), imgsz)


def _xywh_to_xyxy(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    h, w = shape
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    return out


def _visible_fraction(boxes: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)
    x1, y1, x2, y2 = crop
    clipped = boxes.copy()
    clipped[:, 0] = np.maximum(clipped[:, 0], x1)
    clipped[:, 1] = np.maximum(clipped[:, 1], y1)
    clipped[:, 2] = np.minimum(clipped[:, 2], x2)
    clipped[:, 3] = np.minimum(clipped[:, 3], y2)
    before = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    after = np.maximum(clipped[:, 2] - clipped[:, 0], 0) * np.maximum(clipped[:, 3] - clipped[:, 1], 0)
    return (after / np.maximum(before, 1e-9)).astype(np.float32)


def mosaic_crops(shapes: list[tuple[int, int]], imgsz: int, xc: int, yc: int) -> list[tuple[int, int, int, int]]:
    crops = []
    for index, (h, w) in enumerate(shapes):
        if index == 0:
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
        elif index == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, imgsz * 2), yc
        elif index == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(imgsz * 2, yc + h)
        else:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, imgsz * 2), min(imgsz * 2, yc + h)
        pasted_w, pasted_h = max(x2a - x1a, 0), max(y2a - y1a, 0)
        if index == 0:
            x1b, y1b = w - pasted_w, h - pasted_h
        elif index == 1:
            x1b, y1b = 0, h - pasted_h
        elif index == 2:
            x1b, y1b = w - pasted_w, 0
        else:
            x1b, y1b = 0, 0
        crops.append((int(x1b), int(y1b), int(x1b + pasted_w), int(y1b + pasted_h)))
    return crops


def simulate_visible_boxes(metadata: list[dict], indices: list[int], imgsz: int, xc: int, yc: int) -> np.ndarray:
    """Simulate GT visibility for the anchor followed by three donors."""
    shapes = [resized_shape(metadata[i].get("shape", (imgsz, imgsz)), imgsz) for i in indices]
    crops = mosaic_crops(shapes, imgsz, xc, yc)
    values = []
    for index, crop, shape in zip(indices, crops, shapes, strict=True):
        boxes = _xywh_to_xyxy(np.asarray(metadata[index].get("bboxes", [])), shape)
        values.append(_visible_fraction(boxes, crop))
    return np.concatenate(values) if values else np.empty(0, dtype=np.float32)


def candidate_centers(imgsz: int, border: tuple[int, int], count: int) -> list[tuple[int, int]]:
    """Sample centers with the exact standard-Mosaic support and RNG order."""
    return [
        (int(random.uniform(-border[0], 2 * imgsz + border[0])), int(random.uniform(-border[1], 2 * imgsz + border[1])))
        for _ in range(max(int(count), 1))
    ]
