#!/usr/bin/env python3
"""Rigorous validation of the Channel Irreducibility / Consensus hypothesis."""

import sys
import os
import random
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

class Reconstructor(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
    def forward(self, x, mask):
        return self.conv(x * mask)

def collect_features_and_masks(model, dataloader, device, limit=200):
    features = []
    p2_maps = []
    object_masks = []
    
    def hook_fn(module, input, output):
        features.append(output.detach())
        
    hook = model.model.model[18].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            _ = model.model(images)
            p2 = features[-1]
            B, C, H, W = p2.shape
            
            for b in range(B):
                mask = torch.zeros((H, W), device=device)
                bbox_idx = (batch["batch_idx"] == b).nonzero(as_tuple=True)[0]
                bboxes = batch["bboxes"][bbox_idx]
                
                for box in bboxes:
                    x1, y1, x2, y2 = box
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

def train_reconstructor(p2_maps, device, epochs=30):
    C = p2_maps.shape[1]
    reconstructor = Reconstructor(channels=C).to(device)
    optimizer = optim.Adam(reconstructor.parameters(), lr=0.01)
    
    N = p2_maps.shape[0]
    indices = torch.randperm(N)
    train_idx = indices[:int(0.8*N)]
    val_idx = indices[int(0.8*N):]
    
    for epoch in range(epochs):
        reconstructor.train()
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(shuffled), 16):
            batch_indices = shuffled[start:start+16]
            x_batch = p2_maps[batch_indices].to(device)
            B_b = x_batch.shape[0]
            
            mask = torch.ones((B_b, C, 1, 1), device=device)
            for b in range(B_b):
                masked_channels = torch.randperm(C)[:8]
                mask[b, masked_channels, 0, 0] = 0.0
                
            optimizer.zero_grad()
            pred = reconstructor(x_batch, mask)
            loss = torch.sum((1.0 - mask) * torch.abs(x_batch - pred)) / torch.sum(1.0 - mask)
            loss.backward()
            optimizer.step()
            
    return reconstructor

def compute_irreducibility(reconstructor, p2_maps, object_masks, device):
    reconstructor.eval()
    N_total, C, H, W = p2_maps.shape
    e_obj = torch.zeros(C, device=device)
    e_bg = torch.zeros(C, device=device)
    
    with torch.no_grad():
        for c in range(C):
            mask = torch.ones((N_total, C, 1, 1), device=device)
            mask[:, c, 0, 0] = 0.0
            pred = reconstructor(p2_maps, mask)
            error_map = torch.abs(p2_maps[:, c:c+1] - pred[:, c:c+1])
            
            obj_sum = torch.sum(error_map * object_masks)
            obj_count = torch.sum(object_masks)
            bg_sum = torch.sum(error_map * (1.0 - object_masks))
            bg_count = torch.sum(1.0 - object_masks)
            
            e_obj[c] = obj_sum / (obj_count + 1e-6)
            e_bg[c] = bg_sum / (bg_count + 1e-6)
            
    return e_obj - e_bg, e_obj, e_bg

def main():
    checkpoint_path = ROOT / "runs/plain_p2_only/seed_42/weights/best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(hf_hub_download(
            repo_id="duyle2408/levir-yolov8n-p2-asymmetric-screen-seed42",
            repo_type="dataset",
            filename="runs/plain_p2_only/seed_42/weights/best.pt",
            local_dir=str(ROOT)
        ))
    
    model = YOLO(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    
    data_yaml = ROOT / "datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    data_dict = check_det_dataset(data_yaml)
    
    validator = DetectionValidator(args=dict(data=str(data_yaml), imgsz=512, batch=8, device=device, rect=False))
    validator.data = data_dict
    validator.stride = model.model.stride
    
    # Dataloaders
    train_loader = validator.get_dataloader(data_dict["train"], batch_size=8)
    test_loader = validator.get_dataloader(data_dict["test"], batch_size=8)
    
    # 1. Collect features
    print("Collecting P2 maps for TRAIN set...")
    train_p2, train_obj = collect_features_and_masks(model, train_loader, device, limit=200)
    print("Collecting P2 maps for TEST set...")
    test_p2, test_obj = collect_features_and_masks(model, test_loader, device, limit=200)
    
    # 2. Train Reconstructors
    print("Training TRAIN Reconstructor...")
    train_recon = train_reconstructor(train_p2, device)
    print("Training TEST Reconstructor...")
    test_recon = train_reconstructor(test_p2, device)
    
    # 3. Compute rankings
    I_train, e_obj_tr, e_bg_tr = compute_irreducibility(train_recon, train_p2, train_obj, device)
    I_test, e_obj_te, e_bg_te = compute_irreducibility(test_recon, test_p2, test_obj, device)
    
    # Sort rankings
    sorted_train = torch.argsort(I_train, descending=True)
    sorted_test = torch.argsort(I_test, descending=True)
    
    # Jaccard / Overlap of Top 6 hard channels
    K = 6
    top_train_set = set(sorted_train[:K].tolist())
    top_test_set = set(sorted_test[:K].tolist())
    jaccard = len(top_train_set.intersection(top_test_set)) / len(top_train_set.union(top_test_set))
    
    print("\n=== SET STABILITY ===")
    print(f"Train hard channels: {sorted_train[:K].tolist()}")
    print(f"Test hard channels:  {sorted_test[:K].tolist()}")
    print(f"Jaccard similarity of Top-{K} hard channels: {jaccard:.4f}")
    
    # 4. Amplitude/Variance Ranking
    # Compute RMS and Variance per channel on train set P2 maps
    # train_p2 shape: [N, C, H, W]
    rms = torch.sqrt(torch.mean(train_p2 ** 2, dim=(0, 2, 3)))
    variance = torch.var(train_p2, dim=(0, 2, 3))
    
    sorted_rms_high = torch.argsort(rms, descending=True)
    sorted_rms_low = torch.argsort(rms, descending=False)
    sorted_var_high = torch.argsort(variance, descending=True)
    
    # 5. Suppression / Confound comparison
    def get_mute_hook(muted_list, scale=0.0):
        def hook(module, input, output):
            modified = output.clone()
            modified[:, muted_list] *= scale
            return modified
        return hook
        
    print("\n=== CONVERSION / CONFOUND COMPARISON ===")
    
    configs = {
        "Train-Hard (Oracle-Free)": sorted_train[:K].tolist(),
        "Test-Hard (Oracle)": sorted_test[:K].tolist(),
        "Easy (Predictable)": sorted_train[-K:].tolist(),
        "Highest RMS": sorted_rms_high[:K].tolist(),
        "Highest Variance": sorted_var_high[:K].tolist(),
        "Lowest RMS": sorted_rms_low[:K].tolist(),
    }
    
    # Baseline eval
    res = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
    base_ap50 = res.results_dict["metrics/mAP50(B)"]
    base_ap50_95 = res.results_dict["metrics/mAP50-95(B)"]
    base_ap75 = res.box.map75
    
    results = {
        "Baseline": (base_ap50, base_ap75, base_ap50_95)
    }
    
    for name, channels in configs.items():
        h = model.model.model[18].register_forward_hook(get_mute_hook(channels, 0.0))
        r = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
        h.remove()
        results[name] = (r.results_dict["metrics/mAP50(B)"], r.box.map75, r.results_dict["metrics/mAP50-95(B)"])
        
    # Random channel baseline (average of 5 trials)
    rand_ap50s, rand_ap75s, rand_ap50_95s = [], [], []
    for _ in range(5):
        rand_chans = random.sample(range(32), K)
        h = model.model.model[18].register_forward_hook(get_mute_hook(rand_chans, 0.0))
        r = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
        h.remove()
        rand_ap50s.append(r.results_dict["metrics/mAP50(B)"])
        rand_ap75s.append(r.box.map75)
        rand_ap50_95s.append(r.results_dict["metrics/mAP50-95(B)"])
    results["Random (5-trial Avg)"] = (sum(rand_ap50s)/5, sum(rand_ap75s)/5, sum(rand_ap50_95s)/5)
    
    # Print comparison table
    print(f"\n{'Configuration':<25} | {'Test AP50':<10} | {'Test AP75':<10} | {'Test mAP50-95':<12}")
    print("-" * 65)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics[0]:.4f}     | {metrics[1]:.4f}     | {metrics[2]:.4f}")
        
    # 6. Soft Suppression Sweep on Train-Hard set
    print("\n=== SOFT SUPPRESSION SWEEP ===")
    sweep_scales = [0.0, 0.25, 0.50, 0.75, 1.0]
    sweep_results = {}
    for scale in sweep_scales:
        h = model.model.model[18].register_forward_hook(get_mute_hook(sorted_train[:K].tolist(), scale))
        r = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
        h.remove()
        sweep_results[scale] = (r.results_dict["metrics/mAP50(B)"], r.box.map75, r.results_dict["metrics/mAP50-95(B)"])
        
    print(f"\n{'Scale (lambda)':<15} | {'Test AP50':<10} | {'Test AP75':<10} | {'Test mAP50-95':<12}")
    print("-" * 55)
    for scale, metrics in sweep_results.items():
        print(f"{scale:<15} | {metrics[0]:.4f}     | {metrics[1]:.4f}     | {metrics[2]:.4f}")

if __name__ == "__main__":
    main()
