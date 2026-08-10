#!/usr/bin/env python3
import sys
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

# Download best.pt
repo_id = "duyle2408/levir-yolov8n-p2-topdown-3seed"
src_file = "runs/topdown_p1ger_200/seed_42/weights/best.pt"
dest_path = "./runs/topdown_p1ger_200_best.pt"

print("Downloading model weights from Hugging Face...")
try:
    downloaded = hf_hub_download(repo_id=repo_id, filename=src_file, repo_type="dataset")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(downloaded, dest_path)
    print("Downloaded to:", dest_path)
except Exception as e:
    print("Download failed:", e)
    sys.exit(1)

# Add local ultralytics to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models_related/ultralytics"))
from ultralytics import YOLO

model = YOLO(dest_path)
data_yaml = "./datasets/levir_ship_yolo_seed42/levir_ship.yaml"

print("\n=== Evaluating with NMS IoU threshold = 0.50 ===")
try:
    results = model.val(
        data=data_yaml,
        split="test",
        iou=0.50,
        imgsz=512,
        batch=8,
        device="cuda",
        verbose=False
    )
    mAP50 = results.results_dict.get("metrics/mAP50(B)", 0)
    precision = results.results_dict.get("metrics/precision(B)", 0)
    recall = results.results_dict.get("metrics/recall(B)", 0)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | mAP50: {mAP50:.4f}")
except Exception as e:
    print("Evaluation failed:", e)
