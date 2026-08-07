import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add ultralytics path
sys.path.insert(0, str(Path.cwd() / "models_related/ultralytics"))
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator

def main():
    model_path = "./diagnostics/hf_yolov8n_p2/train/yolov8n_p2_baseline_seed42/weights/best.pt"
    data_yaml = "./datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = YOLO(model_path)
    model = yolo_model.model.to(device).eval()
    
    print("Loading dataset...")
    data_dict = check_det_dataset(data_yaml)
    validator = DetectionValidator()
    validator.data = data_dict
    dataloader = validator.get_dataloader(data_dict["test"], batch_size=1)
    
    p1_features = None
    p2_features = None
    
    def hook_p1(module, input, output):
        nonlocal p1_features
        p1_features = output
        
    def hook_p2(module, input, output):
        nonlocal p2_features
        p2_features = output
        
    hook1 = model.model[0].register_forward_hook(hook_p1)
    hook2 = model.model[18].register_forward_hook(hook_p2)
    
    # We visualize one of the dormant targets that originally had ratio < 0.9 (e.g. Case 2 from GF1_WFV2_E123.6_N29.3_20190910_L2A0004239231_2048_2560.png)
    target_img_name = 'GF1_WFV2_E123.6_N29.3_20190910_L2A0004239231_2048_2560.png'
    target_box = [168, 270, 183, 283]
    
    print("Generating P1 fusion comparison plots...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"]
            im_file = batch.get("im_file", ["Unknown"])[0]
            name = Path(im_file).name
            if name != target_img_name:
                continue
                
            _ = model(imgs.float().to(device))
            
            # Downsample P1 and repeat
            p1_down = nn.MaxPool2d(2, 2)(p1_features)
            p1_down_rep = p1_down.repeat(1, 2, 1, 1)
            
            feat_mean_orig = torch.mean(p2_features, dim=1).cpu().numpy()[0]
            feat_mean_p1_down = torch.mean(p1_down_rep, dim=1).cpu().numpy()[0]
            
            img_rgb = imgs[0].numpy().transpose(1, 2, 0)
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            H, W = img_rgb.shape[:2]
            x1, y1, x2, y2 = target_box
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 1. Raw image crop
            axes[0].imshow(img_rgb)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
            axes[0].add_patch(rect)
            axes[0].set_title("Raw Image (Red Box: Dormant Target)")
            axes[0].axis("off")
            
            # 2. Original P2 (diluted)
            orig_heatmap = cv2.resize(feat_mean_orig, (W, H), interpolation=cv2.INTER_LINEAR)
            orig_heatmap = (orig_heatmap - orig_heatmap.min()) / (orig_heatmap.max() - orig_heatmap.min() + 1e-8)
            axes[1].imshow(orig_heatmap, cmap="jet")
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
            axes[1].add_patch(rect)
            axes[1].set_title("Original P2 Feature (Target Suppressed)")
            axes[1].axis("off")
            
            # 3. P1 Downsampled detail
            p1_heatmap = cv2.resize(feat_mean_p1_down, (W, H), interpolation=cv2.INTER_LINEAR)
            p1_heatmap = (p1_heatmap - p1_heatmap.min()) / (p1_heatmap.max() - p1_heatmap.min() + 1e-8)
            axes[2].imshow(p1_heatmap, cmap="jet")
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
            axes[2].add_patch(rect)
            axes[2].set_title("P1 Downsampled Feature (Target Activated!)")
            axes[2].axis("off")
            
            plt.tight_layout()
            out_file = "docs/reports/p1_fusion_activation_comparison.png"
            fig.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved: {out_file}")
            break
            
    hook1.remove()
    hook2.remove()

if __name__ == "__main__":
    main()
