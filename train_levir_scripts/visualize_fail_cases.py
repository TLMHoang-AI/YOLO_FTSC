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

def local_bg_subtraction(X_tensor, kernel_size=3):
    pad = kernel_size // 2
    avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    x_avg = avg_pool(X_tensor)
    return torch.clamp(X_tensor - x_avg, min=0.0)

def deterministic_dbss(X_tensor, num_bases=12, grid_size=16, ridge_lambda=1e-3):
    B, C, H, W = X_tensor.shape
    tokens = X_tensor.squeeze(0).permute(1, 2, 0).reshape(-1, C)
    
    grid_y = torch.linspace(0, H - 1, grid_size, dtype=torch.long)
    grid_x = torch.linspace(0, W - 1, grid_size, dtype=torch.long)
    grid_y_mesh, grid_x_mesh = torch.meshgrid(grid_y, grid_x, indexing="ij")
    bg_indices = grid_y_mesh * W + grid_x_mesh
    bg_tokens = tokens[bg_indices.flatten()]
    
    U_bg, S_bg, Vh_bg = torch.linalg.svd(bg_tokens, full_matrices=False)
    bases = Vh_bg[:num_bases]
    
    base_gram = bases @ bases.T
    rhs = bases @ tokens.T
    identity = torch.eye(num_bases, device=X_tensor.device)
    gram = base_gram + ridge_lambda * identity
    coefficients = torch.linalg.solve(gram, rhs)
    
    bg_reconstructed = (bases.T @ coefficients).T
    residual = tokens - bg_reconstructed
    residual = residual.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)
    return torch.clamp(residual, min=0.0)

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
    
    p2_features = None
    def hook_fn(module, input, output):
        nonlocal p2_features
        p2_features = output
        
    hook = model.model[18].register_forward_hook(hook_fn)
    
    fail_cases = [
        ('GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png', [87, 377, 93, 388]),
        ('GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png', [39, 277, 47, 284]),
        ('GF1_WFV2_E123.6_N29.3_20190910_L2A0004239231_2048_2560.png', [168, 270, 183, 283]),
        ('GF6_WFV_E133.6_N33.6_20200305_L1A1119973496-2_7680_5120.png', [106, 272, 123, 290])
    ]
    
    saved_count = 0
    
    print("Generating fail case plots...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"]
            im_file = batch.get("im_file", ["Unknown"])[0]
            name = Path(im_file).name
            
            # Check if this image has any fail case
            matched_boxes = [box for img_name, box in fail_cases if img_name == name]
            if not matched_boxes:
                continue
                
            _ = model(imgs.float().to(device))
            
            dbss_feat = deterministic_dbss(p2_features, num_bases=12)
            enhanced_feat = local_bg_subtraction(dbss_feat, kernel_size=3)
            
            feat_mean_orig = torch.mean(p2_features, dim=1).cpu().numpy()[0]
            feat_mean_enhanced = torch.mean(enhanced_feat, dim=1).cpu().numpy()[0]
            
            img_rgb = imgs[0].numpy().transpose(1, 2, 0)
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            H, W = img_rgb.shape[:2]
            
            for box in matched_boxes:
                x1, y1, x2, y2 = box
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 1. Raw image crop
                axes[0].imshow(img_rgb)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
                axes[0].add_patch(rect)
                axes[0].set_title(f"Raw Image (Fail Case {saved_count})")
                axes[0].axis("off")
                
                # 2. Original P2
                orig_heatmap = cv2.resize(feat_mean_orig, (W, H), interpolation=cv2.INTER_LINEAR)
                orig_heatmap = (orig_heatmap - orig_heatmap.min()) / (orig_heatmap.max() - orig_heatmap.min() + 1e-8)
                axes[1].imshow(orig_heatmap, cmap="jet")
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
                axes[1].add_patch(rect)
                axes[1].set_title("Original P2 Heatmap")
                axes[1].axis("off")
                
                # 3. Enhanced P2
                enhanced_heatmap = cv2.resize(feat_mean_enhanced, (W, H), interpolation=cv2.INTER_LINEAR)
                enhanced_heatmap = (enhanced_heatmap - enhanced_heatmap.min()) / (enhanced_heatmap.max() - enhanced_heatmap.min() + 1e-8)
                axes[2].imshow(enhanced_heatmap, cmap="jet")
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
                axes[2].add_patch(rect)
                axes[2].set_title("Enhanced P2 Heatmap")
                axes[2].axis("off")
                
                plt.tight_layout()
                out_file = f"docs/reports/fail_case_{saved_count}.png"
                fig.savefig(out_file, dpi=300)
                plt.close()
                print(f"Saved: {out_file}")
                saved_count += 1
                
    hook.remove()

if __name__ == "__main__":
    main()
