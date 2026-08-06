#!/usr/bin/env python3
import os
import csv
import json
import urllib.request
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Step 1: Download Baseline checkpoint
baseline_dir = ROOT / "runs/yolov8n_p2_baseline_seed42/weights"
baseline_weight = baseline_dir / "best.pt"

if not baseline_weight.exists():
    print("Downloading Baseline P2 seed 42 checkpoint from Hugging Face...")
    os.makedirs(baseline_dir, exist_ok=True)
    url = "https://huggingface.co/datasets/duyle2408/levir-ship-yolo-p2/resolve/main/train/yolov8n_p2_baseline_seed42/weights/best.pt"
    try:
        urllib.request.urlretrieve(url, baseline_weight)
        print("Download successful!")
    except Exception as e:
        print(f"Failed to download checkpoint: {e}")
        exit(1)
else:
    print("Baseline checkpoint already exists locally.")

# Step 2: Run variance diagnostics for Baseline if per_gt.csv doesn't exist
baseline_diag_dir = ROOT / "diagnostics/yolov8n_p2_baseline_seed42"
baseline_csv = baseline_diag_dir / "per_gt.csv"

if not baseline_csv.exists():
    print("Running variance diagnostics for Baseline P2...")
    os.makedirs(baseline_diag_dir, exist_ok=True)
    res = subprocess.run([
        "python3",
        "train_levir_scripts/analyze_p2_consensus_variance.py",
        "--checkpoint", str(baseline_weight),
        "--images", "datasets/levir_ship_yolo_seed42/images/test",
        "--output", str(baseline_diag_dir)
    ], cwd=str(ROOT), capture_output=True, text=True)
    print("Variance exit code:", res.returncode)
    if res.returncode != 0:
        print("Error running variance analysis:")
        print(res.stderr)
        exit(1)
else:
    print("Baseline variance CSV already exists.")

# Step 3: Load per_gt.csv files
g4_csv = ROOT / "diagnostics/p2_consensus_g4_seed42_test_variance/per_gt.csv"
if not g4_csv.exists():
    print(f"G4 variance CSV not found at: {g4_csv}")
    exit(1)

def load_per_gt(csv_path):
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["image"], int(row["gt_index"]))
            # Extract iou, bin and variances
            # Note baseline has variance, g4 has fixed3_variance, fixed5_variance
            iou = float(row["best_iou"])
            v3 = float(row.get("variance") or row.get("fixed3_variance") or 0.0)
            v5 = float(row.get("fixed5_variance") or 0.0)
            
            # Map iou to bin
            if iou < 0.5:
                bin_name = "<0.5"
            elif iou < 0.75:
                bin_name = "0.5-0.75"
            else:
                bin_name = ">=0.75"
                
            data[key] = {
                "iou": iou,
                "bin": bin_name,
                "v3": v3,
                "v5": v5
            }
    return data

base_data = load_per_gt(baseline_csv)
g4_data = load_per_gt(g4_csv)

# Step 4: Perform paired merge and calculations
keys = set(base_data.keys()) & set(g4_data.keys())
print(f"\nPaired Ground Truth count: {len(keys)}")

# 1. Transition Matrix
bins = ["<0.5", "0.5-0.75", ">=0.75"]
matrix = {b1: {b2: 0 for b2 in bins} for b1 in bins}

# 2. Delta metrics
paired_by_base_bin = {b: [] for b in bins}

for key in keys:
    b = base_data[key]
    g = g4_data[key]
    
    # Update transition matrix
    matrix[b["bin"]][g["bin"]] += 1
    
    # Calculate deltas
    delta_iou = g["iou"] - b["iou"]
    delta_v3 = g["v3"] - b["v3"]
    delta_v5 = g["v5"] - b["v5"]
    
    paired_by_base_bin[b["bin"]].append({
        "delta_iou": delta_iou,
        "delta_v3": delta_v3,
        "delta_v5": delta_v5,
        "base_iou": b["iou"],
        "g4_iou": g["iou"],
        "base_v3": b["v3"],
        "g4_v3": g["v3"]
    })

# Print Transition Table
print("\n=== TRANSITION MATRIX ===")
print("| Baseline → G4 | <0.5 | 0.5–0.75 | ≥0.75 |")
print("| ------------- | -----: | ---------: | ------: |")
for b in bins:
    row_str = f"| {b:<13} |"
    for g in bins:
        row_str += f" {matrix[b][g]:>5} |"
    print(row_str)

# Print Paired Table
print("\n=== PAIRED COMPARISON TABLE ===")
print("| Baseline group | Median ΔIoU | Median Δvariance 3×3 | Median Δvariance 5×5 |")
print("| -------------- | ----------: | -------------------: | -------------------: |")
for b in bins:
    items = paired_by_base_bin[b]
    if items:
        med_diou = np.median([x["delta_iou"] for x in items])
        med_dv3 = np.median([x["delta_v3"] for x in items])
        med_dv5 = np.median([x["delta_v5"] for x in items])
        print(f"| {b:<14} | {med_diou:>10.6f} | {med_dv3:>20.8f} | {med_dv5:>20.8f} |")
    else:
        print(f"| {b:<14} |          - |                    - |                    - |")

# Detailed breakdown of mid-tier (0.5-0.75) transition
mid_tier_items = paired_by_base_bin["0.5-0.75"]
transitioned = [x for x in mid_tier_items if x["g4_iou"] >= 0.75]
not_transitioned = [x for x in mid_tier_items if x["g4_iou"] < 0.75]

print("\n=== MID-TIER (0.5-0.75) BREAKDOWN ===")
print(f"Total mid-tier GTs: {len(mid_tier_items)}")
print(f"Transitioned to >=0.75: {len(transitioned)} ({len(transitioned)/len(mid_tier_items)*100:.1f}%)")
if transitioned:
    print(f"  - Median delta IoU: {np.median([x['delta_iou'] for x in transitioned]):.6f}")
    print(f"  - Median Baseline variance 3x3: {np.median([x['base_v3'] for x in transitioned]):.8f}")
    print(f"  - Median G4 variance 3x3: {np.median([x['g4_v3'] for x in transitioned]):.8f}")
    print(f"  - Median delta variance 3x3: {np.median([x['delta_v3'] for x in transitioned]):.8f}")

print(f"Not transitioned: {len(not_transitioned)} ({len(not_transitioned)/len(mid_tier_items)*100:.1f}%)")
if not_transitioned:
    print(f"  - Median delta IoU: {np.median([x['delta_iou'] for x in not_transitioned]):.6f}")
    print(f"  - Median Baseline variance 3x3: {np.median([x['base_v3'] for x in not_transitioned]):.8f}")
    print(f"  - Median G4 variance 3x3: {np.median([x['g4_v3'] for x in not_transitioned]):.8f}")
    print(f"  - Median delta variance 3x3: {np.median([x['delta_v3'] for x in not_transitioned]):.8f}")
