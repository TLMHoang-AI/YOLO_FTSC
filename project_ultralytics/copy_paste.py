"""Duy's object-detection Copy-Paste baseline, narrowed to CP2.

This is not Ultralytics' segmentation CopyPaste transform.  Source objects are
read from raw training images and appended as detection boxes/classes on the
final spatial canvas.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np

from ultralytics.utils.instance import Instances


@dataclass(frozen=True)
class ObjectRecord:
    image_index: int
    bbox_xyxy: tuple[float, float, float, float]
    class_id: int


class SmallObjectCopyPaste:
    """Source-faithful fixed, single-object hard Copy-Paste implementation."""

    def __init__(
        self,
        dataset,
        p: float = 0.5,
        unit: str = "single",
        copies: int = 1,
        placement: str = "random",
        max_overlap: float = 0.0,
        padding: float = 0.0,
        scale: float = 1.0,
        blend: str = "hard",
        max_trials: int = 30,
        allow_empty_target: bool = True,
        allow_same_source: bool = True,
        rng=None,
    ) -> None:
        if unit != "single":
            raise ValueError("The FTSC CP2 port supports only unit='single'")
        if placement not in {"random", "collision_aware"}:
            raise ValueError("placement must be 'random' or 'collision_aware'")
        if copies not in {1, 2}:
            raise ValueError("copies must be 1 or 2")
        if blend != "hard" or scale != 1.0 or padding != 0.0:
            raise ValueError("CP2 requires hard blend, scale=1.0, and padding=0.0")
        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        self.dataset = dataset
        self.p = float(p)
        self.unit = unit
        self.copies = int(copies)
        self.placement = placement
        self.max_overlap = float(max_overlap)
        self.padding = float(padding)
        self.scale = float(scale)
        self.blend = blend
        self.max_trials = int(max_trials)
        self.allow_empty_target = bool(allow_empty_target)
        self.allow_same_source = bool(allow_same_source)
        # Keep the default RNG implicit instead of storing the ``random``
        # module on the transform. PyTorch DataLoader workers started with
        # ``spawn`` must pickle the dataset/transform graph, and module objects
        # are not picklable. Resolving the module lazily preserves the existing
        # process-local, seed-controlled behavior.
        self.rng = rng
        self.object_pool: list[ObjectRecord] = []
        self._pool_built = False
        self.stats = {
            "applied_images": 0,
            "pasted_instances": 0,
            "source_single_count": 0,
            "rejected_boundary": 0,
            "rejected_collision": 0,
            "failed_trials": 0,
            "empty_target_seen": 0,
            "empty_target_augmented": 0,
            "pasted_width_sum": 0,
            "pasted_height_sum": 0,
            "pasted_area_sum": 0,
        }

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        xc, yc, width, height = boxes.T
        return np.stack((xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2), axis=1)

    def _build_pool(self) -> None:
        if self._pool_built:
            return
        for image_index, label in enumerate(getattr(self.dataset, "labels", [])):
            boxes = np.asarray(label.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
            if label.get("bbox_format", "xywh") == "xywh":
                boxes = self._xywh_to_xyxy(boxes)
            if label.get("normalized", False):
                h, w = label["shape"][:2]
                boxes[:, [0, 2]] *= w
                boxes[:, [1, 3]] *= h
            classes = np.asarray(label.get("cls", []), dtype=np.int64).reshape(-1)
            for box, class_id in zip(boxes, classes, strict=True):
                self.object_pool.append(ObjectRecord(image_index, tuple(map(float, box)), int(class_id)))
        self._pool_built = True

    def _load_raw(self, image_index: int) -> np.ndarray | None:
        paths = getattr(self.dataset, "im_files", [])
        if image_index >= len(paths):
            return None
        return cv2.imread(str(Path(paths[image_index])), cv2.IMREAD_COLOR)

    @staticmethod
    def _intersection_over_area(candidate: np.ndarray, existing: np.ndarray) -> float:
        if len(existing) == 0:
            return 0.0
        x1 = np.maximum(candidate[0], existing[:, 0])
        y1 = np.maximum(candidate[1], existing[:, 1])
        x2 = np.minimum(candidate[2], existing[:, 2])
        y2 = np.minimum(candidate[3], existing[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area = max(float((candidate[2] - candidate[0]) * (candidate[3] - candidate[1])), 1e-8)
        return float(np.max(intersection / area, initial=0.0))

    def _target_boxes(self, labels: dict[str, Any]) -> np.ndarray:
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return np.empty((0, 4), dtype=np.float32)
        instances.convert_bbox("xyxy")
        if instances.normalized:
            h, w = labels["img"].shape[:2]
            instances.denormalize(w, h)
        return np.asarray(instances.bboxes, dtype=np.float32).copy()

    def _choose_destination(self, patch_shape: tuple[int, int], canvas_shape: tuple[int, int], existing: np.ndarray):
        rng = self.rng if self.rng is not None else random
        patch_h, patch_w = patch_shape
        height, width = canvas_shape
        if patch_h <= 0 or patch_w <= 0 or patch_h > height or patch_w > width:
            self.stats["rejected_boundary"] += 1
            return None
        for _ in range(self.max_trials):
            x = rng.randint(0, width - patch_w)
            y = rng.randint(0, height - patch_h)
            box = np.array([x, y, x + patch_w, y + patch_h], dtype=np.float32)
            if self.placement == "collision_aware" and self._intersection_over_area(box, existing) > self.max_overlap:
                self.stats["rejected_collision"] += 1
                continue
            return x, y, box
        self.stats["failed_trials"] += 1
        return None

    @staticmethod
    def _crop_single(record: ObjectRecord, source_image: np.ndarray):
        x1, y1, x2, y2 = map(round, record.bbox_xyxy)
        height, width = source_image.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return source_image[y1:y2, x1:x2].copy()

    @staticmethod
    def _append_instances(labels: dict[str, Any], boxes: list[np.ndarray], classes: list[np.ndarray]) -> None:
        instances = labels["instances"]
        instances.convert_bbox("xyxy")
        height, width = labels["img"].shape[:2]
        if instances.normalized:
            instances.denormalize(width, height)
        new_boxes = np.concatenate(boxes, axis=0).astype(np.float32)
        segment_shape = instances.segments.shape[1:] if getattr(instances.segments, "ndim", 0) == 3 else (0, 2)
        new_segments = np.zeros((len(new_boxes), *segment_shape), dtype=np.float32)
        new_keypoints = None
        if instances.keypoints is not None:
            new_keypoints = np.zeros((len(new_boxes), *instances.keypoints.shape[1:]), dtype=np.float32)
        added = Instances(new_boxes, new_segments, new_keypoints, bbox_format="xyxy", normalized=False)
        labels["instances"] = Instances.concatenate([instances, added], axis=0)
        labels["cls"] = np.concatenate(
            [
                np.asarray(labels.get("cls", []), dtype=np.float32).reshape(-1, 1),
                np.concatenate(classes).astype(np.float32, copy=False).reshape(-1, 1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        rng = self.rng if self.rng is not None else random
        if self.p <= 0 or rng.random() >= self.p:
            return labels
        self._build_pool()
        if not self.object_pool:
            return labels
        target_image = labels["img"]
        target_index = labels.get("image_index")
        if target_index is None:
            target_path = labels.get("im_file")
            target_index = next(
                (i for i, path in enumerate(getattr(self.dataset, "im_files", [])) if str(path) == str(target_path)),
                None,
            )
        existing = self._target_boxes(labels)
        original_count = len(existing)
        if not original_count:
            self.stats["empty_target_seen"] += 1
            if not self.allow_empty_target:
                return labels
        source_record = rng.choice(self.object_pool)
        if not self.allow_same_source and target_index == source_record.image_index:
            alternatives = [record for record in self.object_pool if record.image_index != target_index]
            if not alternatives:
                return labels
            source_record = rng.choice(alternatives)
        source_image = self._load_raw(source_record.image_index)
        if source_image is None:
            return labels
        crop = self._crop_single(source_record, source_image)
        if crop is None:
            return labels
        pasted_boxes: list[np.ndarray] = []
        pasted_classes: list[np.ndarray] = []
        for _ in range(self.copies):
            destination = self._choose_destination(crop.shape[:2], target_image.shape[:2], existing)
            if destination is None:
                continue
            x, y, box = destination
            patch_h, patch_w = crop.shape[:2]
            target_image[y : y + patch_h, x : x + patch_w] = crop
            pasted_boxes.append(box.reshape(1, 4))
            pasted_classes.append(np.array([source_record.class_id], dtype=np.int64))
            existing = np.concatenate([existing, box.reshape(1, 4)], axis=0)
            self.stats["source_single_count"] += 1
            self.stats["pasted_width_sum"] += patch_w
            self.stats["pasted_height_sum"] += patch_h
            self.stats["pasted_area_sum"] += patch_w * patch_h
        if pasted_boxes:
            self._append_instances(labels, pasted_boxes, pasted_classes)
            self.stats["applied_images"] += 1
            self.stats["pasted_instances"] += sum(len(boxes) for boxes in pasted_boxes)
            if original_count == 0:
                self.stats["empty_target_augmented"] += 1
        return labels


def build_small_object_copy_paste(dataset, hyp) -> SmallObjectCopyPaste | None:
    """Build CP2 without routing through stock segmentation CopyPaste."""
    if not bool(getattr(hyp, "copy_paste_enabled", False)):
        return None
    mode = str(getattr(hyp, "copy_paste_mode", "single")).lower()
    if mode == "negative_canvas":
        from .negative_canvas_copy_paste import NegativeCanvasCopyPaste

        return NegativeCanvasCopyPaste(
            dataset=dataset,
            p=float(getattr(hyp, "negative_cp_p", 0.30)),
            target_policy=str(getattr(hyp, "negative_cp_target_policy", "empirical")),
            donor_policy=str(getattr(hyp, "negative_cp_donor_policy", "matched")),
            target_max_size=float(getattr(hyp, "negative_cp_target_max_size", 20.0)),
            deficit_gamma=float(getattr(hyp, "negative_cp_deficit_gamma", 0.5)),
            max_weight_ratio=float(getattr(hyp, "negative_cp_max_weight_ratio", 3.0)),
            matched_ratio_max=float(getattr(hyp, "negative_cp_matched_ratio_max", 1.5)),
            large_ratio_min=float(getattr(hyp, "negative_cp_large_ratio_min", 1.5)),
            large_ratio_max=float(getattr(hyp, "negative_cp_large_ratio_max", 2.5)),
            degradation=str(getattr(hyp, "negative_cp_degradation", "none")),
            blur_sigma=float(getattr(hyp, "negative_cp_blur_sigma", 0.5)),
            max_trials=int(getattr(hyp, "copy_paste_max_trials", 30)),
            debug_dir=getattr(hyp, "copy_paste_debug_dir", None),
            rng=getattr(hyp, "copy_paste_rng", None),
        )
    if mode not in {"single", "double", "cp2"}:
        raise ValueError("The FTSC suite exposes CP2 and negative_canvas Copy-Paste")
    copies = 2 if mode in {"double", "cp2"} else int(getattr(hyp, "copy_paste_copies", 1))
    return SmallObjectCopyPaste(
        dataset=dataset,
        p=float(getattr(hyp, "copy_paste_p", 0.5)),
        unit=str(getattr(hyp, "copy_paste_unit", "single")),
        copies=copies,
        placement=str(getattr(hyp, "copy_paste_placement", "random")),
        max_overlap=float(getattr(hyp, "copy_paste_max_overlap", 0.0)),
        padding=float(getattr(hyp, "copy_paste_padding", 0.0)),
        scale=float(getattr(hyp, "copy_paste_scale", 1.0)),
        blend=str(getattr(hyp, "copy_paste_blend", "hard")),
        max_trials=int(getattr(hyp, "copy_paste_max_trials", 30)),
        allow_empty_target=bool(getattr(hyp, "copy_paste_allow_empty_target", True)),
        allow_same_source=bool(getattr(hyp, "copy_paste_allow_same_source", True)),
        rng=getattr(hyp, "copy_paste_rng", None),
    )


def copy_paste_config(hyp) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(hyp, "copy_paste_enabled", False)),
        "mode": str(getattr(hyp, "copy_paste_mode", "single")),
        "p": float(getattr(hyp, "copy_paste_p", 0.5)),
        "unit": str(getattr(hyp, "copy_paste_unit", "single")),
        "copies": int(getattr(hyp, "copy_paste_copies", 1)),
        "placement": str(getattr(hyp, "copy_paste_placement", "random")),
        "max_overlap": float(getattr(hyp, "copy_paste_max_overlap", 0.0)),
        "scale": float(getattr(hyp, "copy_paste_scale", 1.0)),
        "padding": float(getattr(hyp, "copy_paste_padding", 0.0)),
        "blend": str(getattr(hyp, "copy_paste_blend", "hard")),
        "allow_empty_target": bool(getattr(hyp, "copy_paste_allow_empty_target", True)),
        "allow_same_source": bool(getattr(hyp, "copy_paste_allow_same_source", True)),
        "max_trials": int(getattr(hyp, "copy_paste_max_trials", 30)),
        "negative_cp_p": float(getattr(hyp, "negative_cp_p", 0.30)),
        "negative_cp_target_policy": str(getattr(hyp, "negative_cp_target_policy", "empirical")),
        "negative_cp_donor_policy": str(getattr(hyp, "negative_cp_donor_policy", "matched")),
        "negative_cp_target_max_size": float(getattr(hyp, "negative_cp_target_max_size", 20.0)),
        "negative_cp_deficit_gamma": float(getattr(hyp, "negative_cp_deficit_gamma", 0.5)),
        "negative_cp_max_weight_ratio": float(getattr(hyp, "negative_cp_max_weight_ratio", 3.0)),
        "negative_cp_matched_ratio_max": float(getattr(hyp, "negative_cp_matched_ratio_max", 1.5)),
        "negative_cp_large_ratio_min": float(getattr(hyp, "negative_cp_large_ratio_min", 1.5)),
        "negative_cp_large_ratio_max": float(getattr(hyp, "negative_cp_large_ratio_max", 2.5)),
        "negative_cp_degradation": str(getattr(hyp, "negative_cp_degradation", "none")),
        "negative_cp_blur_sigma": float(getattr(hyp, "negative_cp_blur_sigma", 0.5)),
        "negative_bank_required": False,
    }
