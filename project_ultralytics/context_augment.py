"""Corrected single-pass OACP R2 used by the FTSC augmentation suite.

This is a deliberately narrow adaptation of Duy's current OACP path.  The
pixel transform and RNG order are preserved, while adaptive OACP variants that
are not experiment factors in A0--A4 are intentionally excluded.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

import cv2
import numpy as np

OBJECT_MAX_SIZE = 32.0
SAFETY_EXPAND = 1.2
TRANSITION_SIGMA_RATIO = 0.06


def augmentation_config() -> dict[str, Any]:
    """Resolve and validate the source R2 contract from environment variables."""
    profile = os.environ.get("OACP_PROFILE", "").lower()
    if profile not in {"", "r2"}:
        raise ValueError(f"unknown OACP_PROFILE: {profile}")
    defaults = {
        "p": 0.40 if profile == "r2" else 0.20,
        "strength": (0.10, 0.25) if profile == "r2" else (0.20, 0.40),
        "scale": (0.80, 0.95) if profile == "r2" else (0.65, 0.85),
        "placement": "pre_transform" if profile == "r2" else "post_mosaic",
    }
    cfg = {
        "mode": os.environ.get("YOLO_CONTEXT_AUG", "none").lower(),
        "profile": profile or "default",
        "oacp_probability": float(os.environ.get("OACP_P", defaults["p"])),
        "oacp_strength": [
            float(os.environ.get("OACP_STRENGTH_MIN", defaults["strength"][0])),
            float(os.environ.get("OACP_STRENGTH_MAX", defaults["strength"][1])),
        ],
        "oacp_resolution_scale": [
            float(os.environ.get("OACP_SCALE_MIN", defaults["scale"][0])),
            float(os.environ.get("OACP_SCALE_MAX", defaults["scale"][1])),
        ],
        "protected_expand": float(os.environ.get("OACP_PROTECTED_EXPAND", 3.0)),
        "oacp_variant": os.environ.get("OACP_VARIANT", "current").lower(),
        "oacp_placement": os.environ.get("OACP_PLACEMENT", defaults["placement"]).lower(),
        "legacy_double_oacp": os.environ.get("YOLO_LEGACY_DOUBLE_OACP", "0").lower()
        in {"1", "true", "yes", "on"},
    }
    if cfg["oacp_variant"] != "current":
        raise ValueError("The FTSC suite ports only OACP variant='current'")
    if cfg["oacp_placement"] not in {"pre_transform", "post_mosaic"}:
        raise ValueError(f"unknown OACP_PLACEMENT: {cfg['oacp_placement']}")
    if profile == "r2":
        checks = (
            np.isclose(cfg["oacp_probability"], 0.40),
            np.allclose(cfg["oacp_strength"], [0.10, 0.25]),
            np.allclose(cfg["oacp_resolution_scale"], [0.80, 0.95]),
            np.isclose(cfg["protected_expand"], 3.0),
        )
        if not all(checks):
            raise ValueError("OACP_PROFILE=r2 requires p=.40, strength=.10-.25, scale=.80-.95, protected_expand=3")
    return cfg


def _boxes(labels: dict[str, Any], h: int, w: int) -> np.ndarray:
    instances = labels.get("instances")
    if instances is None and labels.get("bboxes") is not None:
        raw = np.asarray(labels["bboxes"], dtype=np.float32).reshape(-1, 4)
        if not len(raw):
            return np.empty((0, 4), dtype=np.float32)
        xc, yc, bw, bh = raw.T
        return np.stack(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h), axis=1)
    if instances is None or len(instances.bboxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(instances.bboxes, dtype=np.float32).copy()
    if getattr(instances, "bbox_format", "xywh") == "xywh":
        xc, yc, bw, bh = boxes.T
        boxes = np.stack((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2), axis=1)
    if getattr(instances, "normalized", False):
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
    return np.clip(boxes, [0, 0, 0, 0], [w, h, w, h])


def _mask_from_boxes(boxes: np.ndarray, h: int, w: int, expand: float) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w, half_h = (x2 - x1) * expand / 2, (y2 - y1) * expand / 2
        xa, xb = max(0, int(cx - half_w)), min(w, int(cx + half_w + 1))
        ya, yb = max(0, int(cy - half_h)), min(h, int(cy + half_h + 1))
        mask[ya:yb, xa:xb] = 1
    return mask


def protected_mask(boxes: np.ndarray, h: int, w: int, expand: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact current-OACP protected union and eligible tiny boxes."""
    sizes = (
        np.sqrt(np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1]))
        if len(boxes)
        else np.empty(0)
    )
    tiny = boxes[sizes < OBJECT_MAX_SIZE]
    protected = _mask_from_boxes(tiny, h, w, expand)
    protected |= _mask_from_boxes(boxes, h, w, SAFETY_EXPAND).astype(bool)
    return protected.astype(np.uint8), tiny


def _far_mask(protected: np.ndarray, h: int, w: int) -> np.ndarray:
    if not protected.any():
        return np.zeros((h, w), np.float32)
    sigma = max(3.0, min(h, w) * TRANSITION_SIGMA_RATIO)
    distance = cv2.distanceTransform((1 - protected).astype(np.uint8), cv2.DIST_L2, 3)
    return (1.0 - np.exp(-(distance * distance) / (2.0 * sigma * sigma))).astype(np.float32)


def _resize_degrade(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _record(labels: dict[str, Any], diagnostics: dict[str, Any]) -> None:
    path = os.environ.get("OACP_DIAGNOSTICS_PATH", "")
    if path:
        payload = {**diagnostics, "image": labels.get("im_file", "")}
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


class OACP:
    """Degrade far context while leaving the protected object context intact."""

    def __init__(self, p: float = 0.20) -> None:
        self.p = float(p)
        self.last_diagnostics: dict[str, Any] = {}

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        cfg = augmentation_config()
        image = labels.get("img")
        if image is None or image.ndim != 3:
            return labels
        h, w = image.shape[:2]
        boxes = _boxes(labels, h, w)
        protected, tiny = protected_mask(boxes, h, w, cfg["protected_expand"])
        diagnostics = {
            "variant": "current",
            "num_gt": int(len(boxes)),
            "num_eligible": int(len(tiny)),
            "protected_area_ratio": float(protected.mean()),
            "oacp_applied": False,
        }
        probability = self.p if self.p != 0.20 else cfg["oacp_probability"]
        diagnostics["oacp_probability_effective"] = float(probability)
        if not len(tiny):
            diagnostics["skip_reason"] = "no_eligible_tiny"
        elif random.random() >= probability:
            diagnostics["skip_reason"] = "probability_gate"
        elif protected.mean() > 0.55:
            diagnostics["skip_reason"] = "protected_coverage_gt_0.55"
        else:
            mask = _far_mask(protected, h, w)
            scale = random.uniform(*cfg["oacp_resolution_scale"])
            degraded = _resize_degrade(image, scale)
            base_strength = random.uniform(*cfg["oacp_strength"])
            attenuation = 1.0 - float(protected.mean())
            strength = float(base_strength * attenuation)
            out = image.astype(np.float32) * (1 - strength * mask[..., None])
            out += degraded.astype(np.float32) * (strength * mask[..., None])
            labels["img"] = np.clip(out, 0, 255).astype(image.dtype)
            diagnostics.update(
                oacp_applied=True,
                skip_reason="",
                sampled_scale=float(scale),
                base_strength=float(base_strength),
                effective_strength=strength,
                actual_perturbed_area_ratio=float((mask > 0).mean()),
            )
        self.last_diagnostics = diagnostics
        _record(labels, diagnostics)
        return labels


def build_context_augment(dataset) -> list[OACP]:
    """Build exactly one OACP transform when requested (no legacy double path)."""
    del dataset
    return [OACP()] if os.environ.get("YOLO_CONTEXT_AUG", "none").lower() == "oacp" else []
