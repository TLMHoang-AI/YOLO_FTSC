#!/usr/bin/env python3
import sys
from pathlib import Path

# Add local ultralytics to path
sys.path.insert(0, "/marimo/yolo_code/models_related/ultralytics")
from ultralytics import YOLO

try:
    print("=== Training FPN-Only P1-DRR with old partial clip (post-mosaic) ===")
    model = YOLO("/marimo/yolo_code/models_related/models_config/yolov8/levir/yolov8n_p2_levir_topdown_p1drr.yaml")
    model.train(
        data="/marimo/datasets/levir_ship_yolo_seed42/levir_ship.yaml",
        epochs=100,
        batch=8,
        project="/marimo/runs/levir_yolov8n_p2_topdown_seed42/p1drr_old_partial_clip",
        name="seed_42",
        device="cuda",
        workers=4,
        seed=42,
        deterministic=True,
        exist_ok=True
    )
except Exception as e:
    print("Error training:", e)
