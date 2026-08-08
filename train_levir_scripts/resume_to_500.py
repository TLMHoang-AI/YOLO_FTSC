#!/usr/bin/env python3
import sys
from pathlib import Path

# Add local ultralytics to path
sys.path.insert(0, "/marimo/yolo_code/models_related/ultralytics")
from ultralytics import YOLO

try:
    print("=== Resuming topdown_p1ger (A2) to 500 epochs with batch size 32 ===")
    # Load last.pt from the 200-epoch run
    last_pt = "/marimo/runs/levir_yolov8n_p2_topdown_seed42/topdown_p1ger_200/seed_42/weights/last.pt"
    model = YOLO(last_pt)
    
    # Train for 300 additional epochs
    model.train(
        data="/marimo/datasets/levir_ship_yolo_seed42/levir_ship.yaml",
        epochs=300,
        batch=32,
        project="/marimo/runs/levir_yolov8n_p2_topdown_seed42/topdown_p1ger_500",
        name="seed_42",
        device="cuda",
        workers=4,
        exist_ok=True
    )
except Exception as e:
    print("Error training topdown_p1ger to 500 epochs:", e)
