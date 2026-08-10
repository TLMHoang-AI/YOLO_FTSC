#!/usr/bin/env python3
"""Diagnose ChannelAttention mechanism via gate interventions."""

import sys
import os
import json
import torch
import random
import numpy as np
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/marimo/yolo_code/runs/levir_yolov8n_p2_channel_descriptor/gap/seed_42/weights/best.pt")
    parser.add_argument("--data-yaml", type=str, default="/marimo/datasets/levir_ship_yolo_seed42/levir_ship.yaml")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model_path)
    
    # Locate ChannelAttention layer
    attention_module = model.model.model[19]
    print("Found attention module:", type(attention_module))

    # Phase 1: Collect original gates for all images
    all_gates = []
    
    def collect_gates_hook(module, act_gate):
        # act_gate: [B, C, 1, 1]
        gates_cpu = act_gate.detach().cpu()
        for b in range(gates_cpu.shape[0]):
            all_gates.append(gates_cpu[b].squeeze()) # shape [C]
        return act_gate

    attention_module.override_gate_fn = collect_gates_hook

    print("Running initial validation to collect gate weights...")
    model.val(data=args.data_yaml, split="test", imgsz=512, batch=8, device=0, plots=False, iou=0.5, verbose=False)
    
    # Remove hook
    attention_module.override_gate_fn = None
    
    N_images = len(all_gates)
    C = all_gates[0].shape[0]
    print(f"Collected gates for {N_images} images. Channel count: {C}")
    
    # Convert to a single tensor of shape [N_images, C]
    all_gates_tensor = torch.stack(all_gates) # [N_images, C]
    global_mean = all_gates_tensor.mean().item()
    print(f"Collected gates global mean: {global_mean:.6f}")
    
    # Pre-calculate intervention gates
    interventions = {}
    
    # 1. Original (for check)
    interventions["Original"] = all_gates_tensor
    
    # 2. Static-channel
    static_gate = all_gates_tensor.mean(dim=0) # [C]
    interventions["Static-channel"] = static_gate.unsqueeze(0).repeat(N_images, 1) # [N_images, C]
    
    # 3. Dynamic-scalar
    dynamic_scalar_gate = all_gates_tensor.mean(dim=1, keepdim=True).repeat(1, C) # [N_images, C]
    interventions["Dynamic-scalar"] = dynamic_scalar_gate
    
    # 4. Global-scalar
    interventions["Global-scalar"] = torch.full_like(all_gates_tensor, global_mean)
    
    # 5. Cross-image shuffle
    random.seed(42)
    shuffled_indices = list(range(N_images))
    random.shuffle(shuffled_indices)
    interventions["Cross-image shuffle"] = all_gates_tensor[shuffled_indices]
    
    # 6. Channel shuffle
    channel_shuffled = all_gates_tensor.clone()
    for i in range(N_images):
        perm = torch.randperm(C)
        channel_shuffled[i] = channel_shuffled[i][perm]
    interventions["Channel shuffle"] = channel_shuffled
    
    # 7. Identity (g = 1.0)
    interventions["Identity (g=1.0)"] = torch.ones_like(all_gates_tensor)
    
    # 8. Fixed Scalar Sweeps
    for val in (0.25, 0.4, 0.5, 0.6, 0.75):
        interventions[f"Sweep g={val}"] = torch.full_like(all_gates_tensor, val)

    # Phase 2: Run validation with each intervention
    results = {}
    
    for name, gate_tensor in interventions.items():
        print(f"\n--- Running evaluation with intervention: {name} ---")
        current_image_idx = 0
        
        def override_hook(module, act_gate):
            nonlocal current_image_idx
            B = act_gate.shape[0]
            batch_override = []
            for b in range(B):
                idx = current_image_idx + b
                # Clamp to handle potential padding/extra samples at the end of dataloader
                idx = min(idx, N_images - 1)
                g = gate_tensor[idx]
                batch_override.append(g.view(1, -1, 1, 1))
            current_image_idx += B
            override_tensor = torch.cat(batch_override, dim=0).to(act_gate.device, dtype=act_gate.dtype)
            return override_tensor

        attention_module.override_gate_fn = override_hook
        
        val_res = model.val(data=args.data_yaml, split="test", imgsz=512, batch=8, device=0, plots=False, iou=0.5, verbose=False)
        
        # Save metrics
        metrics = val_res.results_dict
        results[name] = {
            "AP50": metrics["metrics/mAP50(B)"],
            "AP75": val_res.box.map75,
            "mAP50-95": metrics["metrics/mAP50-95(B)"],
            "Precision": metrics["metrics/precision(B)"],
            "Recall": metrics["metrics/recall(B)"]
        }
        print(f"Results for {name}: AP50={results[name]['AP50']:.4f}, AP75={results[name]['AP75']:.4f}, mAP50-95={results[name]['mAP50-95']:.4f}")
        
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE OF GATE INTERVENTIONS")
    print("="*80)
    print(f"{'Intervention':<25} | {'AP50':<8} | {'AP75':<8} | {'mAP50-95':<10} | {'Precision':<10} | {'Recall':<8}")
    print("-"*80)
    for name, r in results.items():
        print(f"{name:<25} | {r['AP50']:.4f}   | {r['AP75']:.4f}   | {r['mAP50-95']:.4f}     | {r['Precision']:.4f}    | {r['Recall']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
