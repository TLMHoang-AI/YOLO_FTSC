#!/usr/bin/env python3
import os
import csv
import urllib.request
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Paths to the four CSV files
baseline_csv = ROOT / "diagnostics/yolov8n_p2_baseline_seed42/per_gt.csv"
g4_csv = ROOT / "diagnostics/p2_consensus_g4_seed42_test_variance/per_gt.csv"
wiou_csv = ROOT / "diagnostics/p2_wiou_seed42_test_variance/per_gt.csv"
wiou_g4_csv = ROOT / "diagnostics/p2_consensus_g4_wiou_seed42_test_variance/per_gt.csv"

# Download and analyze baseline if missing
if not baseline_csv.exists():
    baseline_dir = ROOT / "runs/yolov8n_p2_baseline_seed42/weights"
    baseline_weight = baseline_dir / "best.pt"
    if not baseline_weight.exists():
        print("Downloading Baseline P2 checkpoint...")
        os.makedirs(baseline_dir, exist_ok=True)
        url = "https://huggingface.co/datasets/duyle2408/levir-ship-yolo-p2/resolve/main/train/yolov8n_p2_baseline_seed42/weights/best.pt"
        urllib.request.urlretrieve(url, baseline_weight)
    
    print("Running variance diagnostics for Baseline P2...")
    os.makedirs(baseline_csv.parent, exist_ok=True)
    subprocess.run([
        "python3", "train_levir_scripts/analyze_p2_consensus_variance.py",
        "--checkpoint", str(baseline_weight),
        "--images", "datasets/levir_ship_yolo_seed42/images/test",
        "--output", str(baseline_csv.parent)
    ], cwd=str(ROOT))

def load_per_gt(csv_path):
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["image"], int(row["gt_index"]))
            iou = float(row["best_iou"])
            v3 = float(row.get("variance") or row.get("fixed3_variance") or 0.0)
            v5 = float(row.get("fixed5_variance") or 0.0)
            
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
wiou_data = load_per_gt(wiou_csv)
wiou_g4_data = load_per_gt(wiou_g4_csv)

keys = set(base_data.keys()) & set(g4_data.keys()) & set(wiou_data.keys()) & set(wiou_g4_data.keys())
print(f"\nPaired Ground Truth count (all four): {len(keys)}")

# Bins
bins = ["<0.5", "0.5-0.75", ">=0.75"]

# Transition Matrices
matrix_g4 = {b1: {b2: 0 for b2 in bins} for b1 in bins}
matrix_wiou = {b1: {b2: 0 for b2 in bins} for b1 in bins}
matrix_wiou_g4 = {b1: {b2: 0 for b2 in bins} for b1 in bins}

paired_by_base_bin = {b: [] for b in bins}

for key in keys:
    b = base_data[key]
    g = g4_data[key]
    w = wiou_data[key]
    wg = wiou_g4_data[key]
    
    matrix_g4[b["bin"]][g["bin"]] += 1
    matrix_wiou[b["bin"]][w["bin"]] += 1
    matrix_wiou_g4[b["bin"]][wg["bin"]] += 1
    
    paired_by_base_bin[b["bin"]].append({
        "base_iou": b["iou"],
        "g4_iou": g["iou"],
        "wiou_iou": w["iou"],
        "wiou_g4_iou": wg["iou"],
        "base_v3": b["v3"],
        "g4_v3": g["v3"],
        "wiou_v3": w["v3"],
        "wiou_g4_v3": wg["v3"],
    })

# Print Transition Matrices
print("\n=== TRANSITION MATRIX: BASELINE -> G4 ===")
for b in bins:
    print(f"| {b:<13} | {matrix_g4[b]['<0.5']:>5} | {matrix_g4[b]['0.5-0.75']:>8} | {matrix_g4[b]['>=0.75']:>5} |")

print("\n=== TRANSITION MATRIX: BASELINE -> WIoU ===")
for b in bins:
    print(f"| {b:<15} | {matrix_wiou[b]['<0.5']:>5} | {matrix_wiou[b]['0.5-0.75']:>8} | {matrix_wiou[b]['>=0.75']:>5} |")

print("\n=== TRANSITION MATRIX: BASELINE -> WIoU + G4 ===")
for b in bins:
    print(f"| {b:<15} | {matrix_wiou_g4[b]['<0.5']:>5} | {matrix_wiou_g4[b]['0.5-0.75']:>8} | {matrix_wiou_g4[b]['>=0.75']:>5} |")

# Print Paired Comparison Table
print("\n=== PAIRED COMPARISON: MEDIAN DELTAS ===")
print("| Baseline Group | G4 Med ΔIoU | WIoU Med ΔIoU | WIoU+G4 Med ΔIoU | G4 Med Δvar 3x3 | WIoU Med Δvar 3x3 | WIoU+G4 Med Δvar 3x3 |")
print("| -------------- | ----------: | ------------: | ---------------: | --------------: | ----------------: | -------------------: |")
for b in bins:
    items = paired_by_base_bin[b]
    med_diou_g4 = np.median([x["g4_iou"] - x["base_iou"] for x in items])
    med_diou_w = np.median([x["wiou_iou"] - x["base_iou"] for x in items])
    med_diou_wg = np.median([x["wiou_g4_iou"] - x["base_iou"] for x in items])
    
    med_dv3_g4 = np.median([x["g4_v3"] - x["base_v3"] for x in items])
    med_dv3_w = np.median([x["wiou_v3"] - x["base_v3"] for x in items])
    med_dv3_wg = np.median([x["wiou_g4_v3"] - x["base_v3"] for x in items])
    print(f"| {b:<14} | {med_diou_g4:>11.6f} | {med_diou_w:>13.6f} | {med_diou_wg:>16.6f} | {med_dv3_g4:>15.8f} | {med_dv3_w:>17.8f} | {med_dv3_wg:>20.8f} |")

# Detailed Breakdown of Mid-tier
mid_tier = paired_by_base_bin["0.5-0.75"]
print(f"\n=== MID-TIER (0.5-0.75) BREAKDOWN (Total: {len(mid_tier)}) ===")
trans_g4 = [x for x in mid_tier if x["g4_iou"] >= 0.75]
trans_w = [x for x in mid_tier if x["wiou_iou"] >= 0.75]
trans_wg = [x for x in mid_tier if x["wiou_g4_iou"] >= 0.75]
print(f"G4 transitioned:      {len(trans_g4)} ({len(trans_g4)/len(mid_tier)*100:.1f}%) | Med ΔIoU: {np.median([x['g4_iou'] - x['base_iou'] for x in trans_g4]):.6f} | Med Δvar: {np.median([x['g4_v3'] - x['base_v3'] for x in trans_g4]):.8f}")
print(f"WIoU transitioned:    {len(trans_w)} ({len(trans_w)/len(mid_tier)*100:.1f}%) | Med ΔIoU: {np.median([x['wiou_iou'] - x['base_iou'] for x in trans_w]):.6f} | Med Δvar: {np.median([x['wiou_v3'] - x['base_v3'] for x in trans_w]):.8f}")
print(f"WIoU+G4 transitioned: {len(trans_wg)} ({len(trans_wg)/len(mid_tier)*100:.1f}%) | Med ΔIoU: {np.median([x['wiou_g4_iou'] - x['base_iou'] for x in trans_wg]):.6f} | Med Δvar: {np.median([x['wiou_g4_v3'] - x['base_v3'] for x in trans_wg]):.8f}")
