#!/usr/bin/env python3
import os
import time
import sys
from pathlib import Path
import subprocess

# Add paths
sys.path.insert(0, "/marimo/yolo_code/train_levir_scripts")
sys.path.insert(0, "/marimo/yolo_code/models_related/ultralytics")
import train_all_levir_yolov8n_p2_routing as workflow

# Token verification
token = os.environ.get("HF_TOKEN")
if not token:
    token_file = "/marimo/runs/levir_yolov8n_p2_topdown_seed42/.hf_token"
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            token = f.read().strip()

if not token:
    print("HF_TOKEN not found in environment or fallback file!")
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi(token=token)
repo_id = "duyle2408/levir-yolov8n-p2-topdown-3seed"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

# 1. Upload already completed runs
completed_runs = [
    # (local_run_dir, variant, seed)
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/topdown_p1fusion_200/seed_42", "topdown_p1fusion_200", 42),
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/topdown_p1ger_200/seed_42", "topdown_p1ger_200", 42),
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/baseline_new/topdown_baseline/seed_42", "topdown_baseline", 42),
]

for local_dir, variant, seed in completed_runs:
    if os.path.exists(local_dir):
        print(f"Uploading completed run: {local_dir}...")
        try:
            api.upload_folder(
                folder_path=local_dir,
                path_in_repo=f"runs/{variant}/seed_{seed}",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print("Upload successful!")
        except Exception as e:
            print(f"Error uploading {local_dir}: {e}")

# 2. Wait for ongoing training runs to finish
pids = {}
for name in ["train_p1reg_only", "train_p1drr_partial_clip"]:
    pid_file = f"/marimo/runs/levir_yolov8n_p2_topdown_seed42/{name}.pid"
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            pids[name] = int(f.read().strip())

print("Tracking PIDs:", pids)

def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

# Loop until all PIDs finish
while any(is_running(pid) for pid in pids.values()):
    print("Training is still running. Sleeping for 60 seconds...")
    time.sleep(60)

print("All ongoing training runs finished!")

# 3. Perform evaluations on the newly completed runs and upload them!
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

post_completed_runs = [
    # (local_run_dir, variant, seed)
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/p1reg_only/topdown_p1reg_only/seed_42", "topdown_p1reg_only", 42),
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/p1drr_partial_clip/topdown_p1drr/seed_42", "topdown_p1drr_partial_clip", 42),
]

for local_dir, variant, seed in post_completed_runs:
    if os.path.exists(local_dir):
        print(f"Evaluating {variant}...")
        try:
            workflow.evaluate(Path(local_dir), data_yaml, args_eval)
        except Exception as e:
            print(f"Evaluation error for {variant}: {e}")
            
        print(f"Uploading run: {local_dir}...")
        try:
            api.upload_folder(
                folder_path=local_dir,
                path_in_repo=f"runs/{variant}/seed_{seed}",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print("Upload successful!")
        except Exception as e:
            print(f"Error uploading {local_dir}: {e}")

# Upload updated summary files as metadata
summary_paths = [
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/baseline_new/summary_runs.csv", "summary_runs_baseline.csv"),
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/p1reg_only/summary_runs.csv", "summary_runs_p1reg_only.csv"),
    ("/marimo/runs/levir_yolov8n_p2_topdown_seed42/p1drr_partial_clip/summary_runs.csv", "summary_runs_p1drr_partial_clip.csv"),
]
for local, remote in summary_paths:
    if os.path.exists(local):
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=f"summary/{remote}",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"Uploaded summary: {remote}")
        except Exception as e:
            print(f"Error uploading summary {remote}: {e}")

print("Uploader queue script completed successfully!")
