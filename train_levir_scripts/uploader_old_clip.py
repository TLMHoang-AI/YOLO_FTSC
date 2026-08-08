#!/usr/bin/env python3
import os
import time
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, "/marimo/yolo_code/train_levir_scripts")
sys.path.insert(0, "/marimo/yolo_code/models_related/ultralytics")
import train_all_levir_yolov8n_p2_routing as workflow

token_file = "/marimo/runs/levir_yolov8n_p2_topdown_seed42/.hf_token"
if not os.path.exists(token_file):
    print("HF token file not found!")
    sys.exit(1)

with open(token_file, "r") as f:
    token = f.read().strip()

from huggingface_hub import HfApi
api = HfApi(token=token)
repo_id = "duyle2408/levir-yolov8n-p2-topdown-3seed"

pid_file = "/marimo/runs/levir_yolov8n_p2_topdown_seed42/train_p1drr_old_partial_clip.pid"
if not os.path.exists(pid_file):
    print("PID file not found!")
    sys.exit(1)

with open(pid_file, "r") as f:
    pid = int(f.read().strip())

print(f"Watching training PID: {pid}")

def is_running(p):
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False

while is_running(pid):
    time.sleep(60)

print("Training finished! Evaluating and uploading...")

import argparse
args_eval = argparse.Namespace(
    dataset_root=Path("/marimo/datasets"),
    split_seed=42,
    imgsz=512,
    batch_size=8,
    device="cuda",
    workers=4
)
data_yaml = Path("/marimo/datasets/levir_ship_yolo_seed42/levir_ship.yaml")
local_dir = "/marimo/runs/levir_yolov8n_p2_topdown_seed42/p1drr_old_partial_clip/seed_42"

if os.path.exists(local_dir):
    try:
        workflow.evaluate(Path(local_dir), data_yaml, args_eval)
    except Exception as e:
        print("Evaluation error:", e)
        
    try:
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo="runs/p1drr_old_partial_clip/seed_42",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("Upload successful!")
    except Exception as e:
        print("Upload error:", e)
