#!/usr/bin/env python3
"""Diagnose Channel-KVCA beta parameter and run beta=0 intervention."""

import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO

def main():
    model_path = "/marimo/yolo_code/runs/levir_yolov8n_p2_channel_kvca/channel_kvca/seed_42/weights/best.pt"
    data_yaml = "/marimo/datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    
    # Load model
    model = YOLO(model_path)
    attention_module = model.model.model[19]
    
    # 1. Print learned beta value
    learned_beta = getattr(attention_module, "beta", None)
    if learned_beta is not None:
        print(f"Learned beta: {learned_beta.item():.6f}")
    else:
        print("Beta parameter not found in module!")
        
    # 2. Evaluate original model
    print("\n--- Evaluating with Original Learned Beta ---")
    val_orig = model.val(data=data_yaml, split="test", imgsz=512, batch=8, device=0, plots=False, iou=0.5, verbose=False)
    metrics_orig = val_orig.results_dict
    
    # 3. Set beta = 0.0
    if learned_beta is not None:
        attention_module.beta.data.fill_(0.0)
        print("\nSet beta to 0.0 successfully.")
    
    # 4. Evaluate with beta = 0.0
    print("\n--- Evaluating with Beta = 0.0 ---")
    val_zero = model.val(data=data_yaml, split="test", imgsz=512, batch=8, device=0, plots=False, iou=0.5, verbose=False)
    metrics_zero = val_zero.results_dict
    
    # Print comparison
    print("\n" + "="*80)
    print("CHANNEL-KVCA BETA DIAGNOSTIC COMPARISON")
    print("="*80)
    print(f"Beta value    | AP50     | AP75     | mAP50-95   | Precision  | Recall")
    print("-"*80)
    print(f"Original ({learned_beta.item():.4f})  | {metrics_orig['metrics/mAP50(B)']:.4f}   | {val_orig.box.map75:.4f}   | {metrics_orig['metrics/mAP50-95(B)']:.4f}     | {metrics_orig['metrics/precision(B)']:.4f}    | {metrics_orig['metrics/recall(B)']:.4f}")
    print(f"Beta = 0.0000  | {metrics_zero['metrics/mAP50(B)']:.4f}   | {val_zero.box.map75:.4f}   | {metrics_zero['metrics/mAP50-95(B)']:.4f}     | {metrics_zero['metrics/precision(B)']:.4f}    | {metrics_zero['metrics/recall(B)']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
