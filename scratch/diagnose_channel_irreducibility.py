#!/usr/bin/env python3
"""Diagnose channel irreducibility on LEVIR-Ship P2 feature maps."""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

class Reconstructor(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        
    def forward(self, x, mask):
        # x: [B, C, H, W]
        # mask: [B, C, 1, 1] containing 0 for masked channels, 1 for unmasked
        return self.conv(x * mask)

def main():
    # 1. Download plain baseline model if not present
    checkpoint_path = ROOT / "runs/plain_p2_only/seed_42/weights/best.pt"
    if not checkpoint_path.exists():
        print("Downloading plain baseline checkpoint from Hugging Face...")
        checkpoint_path = Path(hf_hub_download(
            repo_id="duyle2408/levir-yolov8n-p2-asymmetric-screen-seed42",
            repo_type="dataset",
            filename="runs/plain_p2_only/seed_42/weights/best.pt",
            local_dir=str(ROOT)
        ))
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # Load model
    model = YOLO(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    
    # Get test data loader
    data_yaml = ROOT / "datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    validator = DetectionValidator(args=dict(
        data=str(data_yaml), imgsz=512, batch=8, device=device, rect=False
    ))
    dataloader = validator.get_dataloader(ROOT / "datasets/levir_ship_yolo_seed42", batch=8)
    
    # 2. Collect P2 activations and GT boxes
    print("Collecting P2 activations and target masks...")
    p2_maps = []
    object_masks = []
    
    # Set hook on layer 18 (C2f output, which is index 18)
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach())
        
    hook = model.model.model[18].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            _ = model.model(images)
            
            p2 = features[-1] # [B, C, H, W] (H=128, W=128 for imgsz=512)
            B, C, H, W = p2.shape
            
            # Construct GT object masks on the 128x128 grid
            # batch["cls"] is class, batch["bboxes"] is xyxy normalized boxes
            for b in range(B):
                mask = torch.zeros((H, W), device=device)
                # Filter bboxes belonging to batch index b
                bbox_idx = (batch["batch_idx"] == b).nonzero(as_tuple=True)[0]
                bboxes = batch["bboxes"][bbox_idx] # [N_boxes, 4] normalized xyxy
                
                for box in bboxes:
                    x1, y1, x2, y2 = box
                    # Map to H, W
                    col_start = max(0, int(x1 * W))
                    col_end = min(W, int(x2 * W) + 1)
                    row_start = max(0, int(y1 * H))
                    row_end = min(H, int(y2 * H) + 1)
                    mask[row_start:row_end, col_start:col_end] = 1.0
                    
                p2_maps.append(p2[b])
                object_masks.append(mask)
                
            features.clear()
            if len(p2_maps) >= 200: # Collect from first 200 test images to make it fast
                break
                
    hook.remove()
    
    p2_maps = torch.stack(p2_maps) # [N, C, H, W]
    object_masks = torch.stack(object_masks).unsqueeze(1) # [N, 1, H, W]
    print(f"Collected {p2_maps.shape[0]} samples. Feature map shape: {p2_maps.shape}")
    
    # 3. Train the tiny reconstructor
    C = p2_maps.shape[1]
    reconstructor = Reconstructor(channels=C).to(device)
    optimizer = optim.Adam(reconstructor.parameters(), lr=0.01)
    
    # Split into train/val subsets of collected features
    N_total = p2_maps.shape[0]
    indices = torch.randperm(N_total)
    train_idx = indices[:int(0.8*N_total)]
    val_idx = indices[int(0.8*N_total):]
    
    print("Training 1x1 Conv Reconstructor...")
    for epoch in range(30):
        reconstructor.train()
        # Shuffle train indices
        shuffled = train_idx[torch.randperm(len(train_idx))]
        epoch_loss = 0.0
        
        # Batch size 16
        for start_idx in range(0, len(shuffled), 16):
            batch_indices = shuffled[start_idx:start_idx+16]
            x_batch = p2_maps[batch_indices].to(device)
            B_b = x_batch.shape[0]
            
            # Generate random binary channel mask (mask 8 random channels out of 32 per batch sample)
            mask = torch.ones((B_b, C, 1, 1), device=device)
            for b in range(B_b):
                masked_channels = torch.randperm(C)[:8]
                mask[b, masked_channels, 0, 0] = 0.0
                
            optimizer.zero_grad()
            pred = reconstructor(x_batch, mask)
            
            # Compute loss only on masked channels
            loss = torch.sum((1.0 - mask) * torch.abs(x_batch - pred)) / torch.sum(1.0 - mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * B_b
            
        # Eval loss
        reconstructor.eval()
        with torch.no_grad():
            x_val = p2_maps[val_idx].to(device)
            B_v = x_val.shape[0]
            mask_val = torch.ones((B_v, C, 1, 1), device=device)
            for b in range(B_v):
                masked_channels = torch.randperm(C)[:8]
                mask_val[b, masked_channels, 0, 0] = 0.0
            pred_val = reconstructor(x_val, mask_val)
            val_loss = torch.sum((1.0 - mask_val) * torch.abs(x_val - pred_val)) / torch.sum(1.0 - mask_val)
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_loss/len(train_idx):.6f} | Val Loss: {val_loss.item():.6f}")

    # 4. Measure Irreducibility per channel
    reconstructor.eval()
    print("Measuring channel-wise irreducibility...")
    e_obj = torch.zeros(C, device=device)
    e_bg = torch.zeros(C, device=device)
    
    with torch.no_grad():
        for c in range(C):
            # Mask only channel c
            mask = torch.ones((N_total, C, 1, 1), device=device)
            mask[:, c, 0, 0] = 0.0
            
            # Reconstruct
            pred = reconstructor(p2_maps, mask)
            error_map = torch.abs(p2_maps[:, c:c+1] - pred[:, c:c+1]) # [N, 1, H, W]
            
            # Compute errors on object vs background regions
            # error_map * object_masks
            obj_sum = torch.sum(error_map * object_masks)
            obj_count = torch.sum(object_masks)
            
            bg_sum = torch.sum(error_map * (1.0 - object_masks))
            bg_count = torch.sum(1.0 - object_masks)
            
            e_obj[c] = obj_sum / (obj_count + 1e-6)
            e_bg[c] = bg_sum / (bg_count + 1e-6)

    # Compute Irreducibility
    I_c = e_obj - e_bg
    
    # Print channel stats
    print("\nChannel Irreducibility Statistics (Top 10 Hardest):")
    sorted_hard = torch.argsort(I_c, descending=True)
    for rank, c in enumerate(sorted_hard):
        print(f"Rank {rank+1:02d} | Channel {c:02d} | I_c: {I_c[c].item():.6f} | Obj Error: {e_obj[c].item():.6f} | BG Error: {e_bg[c].item():.6f}")

    # 5. Intervention / Kill-test
    # We select 6 channels (20% of 32)
    K = 6
    hard_channels = sorted_hard[:K].tolist()
    easy_channels = sorted_hard[-K:].tolist()
    
    print(f"\nPerforming Kill-tests on Test Set (NMS IoU = 0.5):")
    print(f"Muting hard channels: {hard_channels}")
    print(f"Muting easy channels: {easy_channels}")
    
    # Save baseline test accuracy first
    print("\nRunning Baseline Evaluation...")
    baseline_res = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
    baseline_ap50 = baseline_res.results_dict["metrics/mAP50(B)"]
    baseline_ap50_95 = baseline_res.results_dict["metrics/mAP50-95(B)"]
    baseline_ap75 = baseline_res.box.map75
    
    # Mute function hook
    def get_mute_hook(muted_list):
        def mute_hook_fn(module, input, output):
            # output is [B, C, H, W]
            modified = output.clone()
            modified[:, muted_list] = 0.0
            return modified
        return mute_hook_fn
        
    # Hard mute test
    hook_hard = model.model.model[18].register_forward_hook(get_mute_hook(hard_channels))
    hard_res = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
    hook_hard.remove()
    
    hard_ap50 = hard_res.results_dict["metrics/mAP50(B)"]
    hard_ap50_95 = hard_res.results_dict["metrics/mAP50-95(B)"]
    hard_ap75 = hard_res.box.map75
    
    # Easy mute test
    hook_easy = model.model.model[18].register_forward_hook(get_mute_hook(easy_channels))
    easy_res = model.val(data=str(data_yaml), split="test", imgsz=512, batch=8, device=device, plots=False, iou=0.5, verbose=False)
    hook_easy.remove()
    
    easy_ap50 = easy_res.results_dict["metrics/mAP50(B)"]
    easy_ap50_95 = easy_res.results_dict["metrics/mAP50-95(B)"]
    easy_ap75 = easy_res.box.map75
    
    print("\n--- Kill-test Results Table ---")
    print(f"{'Configuration':<25} | {'Test AP50':<10} | {'Test AP75':<10} | {'Test mAP50-95':<12}")
    print("-" * 65)
    print(f"{'Baseline':<25} | {baseline_ap50:.4f}     | {baseline_ap75:.4f}     | {baseline_ap50_95:.4f}")
    print(f"{'Mute Hard Channels':<25} | {hard_ap50:.4f}     | {hard_ap75:.4f}     | {hard_ap50_95:.4f} (Delta: {hard_ap50_95 - baseline_ap50_95:.4f})")
    print(f"{'Mute Easy Channels':<25} | {easy_ap50:.4f}     | {easy_ap75:.4f}     | {easy_ap50_95:.4f} (Delta: {easy_ap50_95 - baseline_ap50_95:.4f})")

if __name__ == "__main__":
    main()
