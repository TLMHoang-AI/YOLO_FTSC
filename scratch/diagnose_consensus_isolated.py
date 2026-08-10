#!/usr/bin/env python3
"""Isolated, rigorous validation of Channel Consensus on LEVIR-Ship."""

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.utils import check_det_dataset

class ZeroDiagConv1x1(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self, x):
        # Enforce zero-diagonal weight constraint
        w = self.conv.weight
        diag_mask = 1.0 - torch.eye(x.shape[1], device=x.device).view(x.shape[1], x.shape[1], 1, 1)
        self.conv.weight.data.copy_(w.data * diag_mask)
        return self.conv(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collect_features_and_masks(checkpoint_path, dataloader, device, limit=500):
    # Load fresh YOLO inside collection to avoid hook residue
    model = YOLO(checkpoint_path)
    model.model.to(device)
    model.model.eval()
    
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach())
        
    # Hook layer 18 (P2 output)
    hook = model.model.model[18].register_forward_hook(hook_fn)
    
    p2_maps = []
    object_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            _ = model.model(images)
            p2 = features[-1]
            B, C, H, W = p2.shape
            
            for b in range(B):
                mask = torch.zeros((H, W), device=device)
                bbox_idx = (batch["batch_idx"] == b).nonzero(as_tuple=True)[0]
                bboxes = batch["bboxes"][bbox_idx] # Shape [N, 4] -> cx, cy, bw, bh (normalized)
                
                for box in bboxes:
                    cx, cy, bw, bh = box
                    # Convert cx, cy, bw, bh to x1, y1, x2, y2
                    x1 = cx - bw / 2.0
                    x2 = cx + bw / 2.0
                    y1 = cy - bh / 2.0
                    y2 = cy + bh / 2.0
                    
                    # Scale to feature map resolution H, W
                    col_start = max(0, int(x1 * W))
                    col_end = min(W, int(x2 * W) + 1)
                    row_start = max(0, int(y1 * H))
                    row_end = min(H, int(y2 * H) + 1)
                    
                    mask[row_start:row_end, col_start:col_end] = 1.0
                    
                p2_maps.append(p2[b])
                object_masks.append(mask)
                
            features.clear()
            if len(p2_maps) >= limit:
                break
                
    hook.remove()
    return torch.stack(p2_maps), torch.stack(object_masks).unsqueeze(1)

def train_zero_diag_recon(p2_maps, device, seed, epochs=30):
    set_seed(seed)
    C = p2_maps.shape[1]
    reconstructor = ZeroDiagConv1x1(channels=C).to(device)
    optimizer = optim.Adam(reconstructor.parameters(), lr=0.01)
    
    N = p2_maps.shape[0]
    indices = torch.randperm(N)
    train_idx = indices[:int(0.8*N)]
    
    for epoch in range(epochs):
        reconstructor.train()
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(shuffled), 16):
            batch_indices = shuffled[start:start+16]
            x_batch = p2_maps[batch_indices].to(device)
            
            optimizer.zero_grad()
            pred = reconstructor(x_batch)
            # Reconstruct the target tensor directly (diag=0 ensures self-reconstruction is impossible)
            loss = torch.mean(torch.abs(x_batch - pred))
            loss.backward()
            optimizer.step()
            
    return reconstructor

def compute_irreducibility(reconstructor, p2_maps, object_masks, device):
    reconstructor.eval()
    C = p2_maps.shape[1]
    e_obj = torch.zeros(C, device=device)
    e_bg = torch.zeros(C, device=device)
    
    with torch.no_grad():
        pred = reconstructor(p2_maps)
        error_map = torch.abs(p2_maps - pred) # [N, C, H, W]
        
        for c in range(C):
            obj_sum = torch.sum(error_map[:, c:c+1] * object_masks)
            obj_count = torch.sum(object_masks)
            bg_sum = torch.sum(error_map[:, c:c+1] * (1.0 - object_masks))
            bg_count = torch.sum(1.0 - object_masks)
            
            e_obj[c] = obj_sum / (obj_count + 1e-6)
            e_bg[c] = bg_sum / (bg_count + 1e-6)
            
    return e_obj - e_bg

def run_isolated_eval(checkpoint_path, data_yaml, muted_channels, scale, device):
    # Fully clean process isolation: load a new YOLO instance
    model = YOLO(checkpoint_path)
    model.model.to(device)
    
    def hook_fn(module, input, output):
        modified = output.clone()
        modified[:, muted_channels] *= scale
        return modified
        
    hook = model.model.model[18].register_forward_hook(hook_fn)
    
    # Run evaluation
    res = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
    hook.remove()
    
    return res.results_dict["metrics/mAP50(B)"], res.box.map75, res.results_dict["metrics/mAP50-95(B)"]

def main():
    checkpoint_path = ROOT / "runs/plain_p2_only/seed_42/weights/best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(hf_hub_download(
            repo_id="duyle2408/levir-yolov8n-p2-asymmetric-screen-seed42",
            repo_type="dataset",
            filename="runs/plain_p2_only/seed_42/weights/best.pt",
            local_dir=str(ROOT)
        ))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_yaml = ROOT / "datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    data_dict = check_det_dataset(data_yaml)
    
    validator = DetectionValidator(args=dict(data=str(data_yaml), imgsz=512, batch=8, device=device, rect=False))
    validator.data = data_dict
    
    # Load model just to get stride for dataloader
    temp_model = YOLO(checkpoint_path)
    validator.stride = temp_model.model.stride
    del temp_model
    
    train_loader = validator.get_dataloader(data_dict["train"], batch_size=8)
    test_loader = validator.get_dataloader(data_dict["test"], batch_size=8)
    
    # Collect features (increased limit to 800 for full split stability check)
    print("Collecting P2 maps for TRAIN set (limit 800)...")
    train_p2, train_obj = collect_features_and_masks(checkpoint_path, train_loader, device, limit=800)
    print("Collecting P2 maps for TEST set (limit 500)...")
    test_p2, test_obj = collect_features_and_masks(checkpoint_path, test_loader, device, limit=500)
    
    C = train_p2.shape[1]
    K = 6
    
    # 1. Test Seed Stability of 1x1 Zero-Diag Reconstructor
    print("\n=== RECONSTRUCTOR SEED STABILITY (5 SEEDS) ===")
    rankings = []
    for seed in range(5):
        recon = train_zero_diag_recon(train_p2, device, seed=seed)
        I_tr = compute_irreducibility(recon, train_p2, train_obj, device)
        sorted_chans = torch.argsort(I_tr, descending=True).tolist()
        rankings.append(sorted_chans)
        print(f"Seed {seed} | Top-6 Hard Channels: {sorted_chans[:K]}")
        
    # Calculate Jaccard similarity between all seed pairs
    jaccards = []
    for i in range(5):
        for j in range(i+1, 5):
            s1 = set(rankings[i][:K])
            s2 = set(rankings[j][:K])
            jaccards.append(len(s1.intersection(s2)) / len(s1.union(s2)))
    print(f"Mean Jaccard similarity of Top-6 hard channels across 5 seeds: {sum(jaccards)/len(jaccards):.4f}")
    
    # Use average ranking across seeds to determine train-hard set
    freqs = torch.zeros(C)
    for r in rankings:
        for rank, c in enumerate(r):
            # Give higher score to channels ranked at the top
            freqs[c] += (C - rank)
            
    sorted_train_consolidated = torch.argsort(freqs, descending=True).tolist()
    train_hard_channels = sorted_train_consolidated[:K]
    easy_channels = sorted_train_consolidated[-K:]
    print(f"Consolidated Train Hard Channels: {train_hard_channels}")
    print(f"Consolidated Train Easy Channels: {easy_channels}")
    
    # Test Stability against Test Set Reconstructor
    print("\n=== TRAIN VS TEST CROSS-SET STABILITY ===")
    test_recon = train_zero_diag_recon(test_p2, device, seed=42)
    I_te = compute_irreducibility(test_recon, test_p2, test_obj, device)
    test_hard_channels = torch.argsort(I_te, descending=True).tolist()[:K]
    print(f"Test Hard Channels: {test_hard_channels}")
    cross_jaccard = len(set(train_hard_channels).intersection(set(test_hard_channels))) / len(set(train_hard_channels).union(set(test_hard_channels)))
    print(f"Cross-Set Jaccard Similarity (Train vs Test): {cross_jaccard:.4f}")
    
    # Compute RMS and Variance
    rms = torch.sqrt(torch.mean(train_p2 ** 2, dim=(0, 2, 3)))
    variance = torch.var(train_p2, dim=(0, 2, 3))
    rms_high = torch.argsort(rms, descending=True).tolist()[:K]
    rms_low = torch.argsort(rms, descending=False).tolist()[:K]
    var_high = torch.argsort(variance, descending=True).tolist()[:K]
    
    # 2. RUN ISOLATED CONSTRUCT VALIDATION
    print("\n=== ISOLATED CONVERSION / CONFOUND COMPARISON ===")
    
    configs = {
        "Baseline": ([], 1.0),
        "Train-Hard (Oracle-Free)": (train_hard_channels, 0.0),
        "Test-Hard (Oracle)": (test_hard_channels, 0.0),
        "Easy (Predictable)": (easy_channels, 0.0),
        "Highest RMS": (rms_high, 0.0),
        "Highest Variance": (var_high, 0.0),
        "Lowest RMS": (rms_low, 0.0),
    }
    
    isolated_results = {}
    for name, (chans, scale) in configs.items():
        ap50, ap75, map50_95 = run_isolated_eval(checkpoint_path, data_yaml, chans, scale, device)
        isolated_results[name] = (ap50, ap75, map50_95)
        print(f"Evaluated {name:<25} | Test mAP50-95: {map50_95:.4f}")
        
    # Evaluate Random Channel Control (5 fresh runs)
    rand_ap50s, rand_ap75s, rand_maps = [], [], []
    for r_idx in range(5):
        set_seed(r_idx + 100)
        rand_chans = random.sample(range(32), K)
        ap50, ap75, map50_95 = run_isolated_eval(checkpoint_path, data_yaml, rand_chans, 0.0, device)
        rand_ap50s.append(ap50)
        rand_ap75s.append(ap75)
        rand_maps.append(map50_95)
    isolated_results["Random (5-trial Avg)"] = (sum(rand_ap50s)/5, sum(rand_ap75s)/5, sum(rand_maps)/5)
    
    print("\n--- Corrected Isolated Results Table ---")
    print(f"{'Configuration':<25} | {'Test AP50':<10} | {'Test AP75':<10} | {'Test mAP50-95':<12}")
    print("-" * 65)
    for name, metrics in isolated_results.items():
        print(f"{name:<25} | {metrics[0]:.4f}     | {metrics[1]:.4f}     | {metrics[2]:.4f}")
        
    # 3. RUN ISOLATED SOFT SWEEP
    print("\n=== ISOLATED SOFT SUPPRESSION SWEEP ===")
    sweep_scales = [0.0, 0.25, 0.50, 0.75, 1.0]
    sweep_results = {}
    for scale in sweep_scales:
        ap50, ap75, map50_95 = run_isolated_eval(checkpoint_path, data_yaml, train_hard_channels, scale, device)
        sweep_results[scale] = (ap50, ap75, map50_95)
        print(f"Sweep scale {scale:<5} | Test mAP50-95: {map50_95:.4f}")
        
    print(f"\n{'Scale (lambda)':<15} | {'Test AP50':<10} | {'Test AP75':<10} | {'Test mAP50-95':<12}")
    print("-" * 55)
    for scale, metrics in sweep_results.items():
        print(f"{scale:<15} | {metrics[0]:.4f}     | {metrics[1]:.4f}     | {metrics[2]:.4f}")

if __name__ == "__main__":
    main()
