import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add ultralytics path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models_related/ultralytics"))
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
    
    p2_features = None
    def hook_fn(module, input, output):
        nonlocal p2_features
        p2_features = output
        
    hook = model.model[18].register_forward_hook(hook_fn)
    
    all_candidates = []
    
    print("Analyzing P2 activations relative to background...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch["img"] # (1, 3, 512, 512)
            x = imgs.float() / 255.0
            
            # Forward pass
            _ = model(x.to(device))
            
            # P2 feature map shape: (1, 64, 128, 128)
            feat = p2_features.cpu().numpy()[0]
            feat_mean = np.mean(feat, axis=0)
            
            B, C, H, W = imgs.shape
            bboxes = batch["bboxes"].numpy()
            
            if len(bboxes) == 0:
                continue
                
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= W
            pixel_boxes[:, [1, 3]] *= H
            
            x_center, y_center, w, h = pixel_boxes[:, 0], pixel_boxes[:, 1], pixel_boxes[:, 2], pixel_boxes[:, 3]
            
            H_f, W_f = feat_mean.shape
            scale = 4.0
            
            img_rgb = imgs[0].numpy().transpose(1, 2, 0)
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            
            for i in range(len(bboxes)):
                max_side = max(w[i], h[i])
                if max_side < 20:
                    x1 = int(np.clip(x_center[i] - w[i]/2, 0, W-1))
                    y1 = int(np.clip(y_center[i] - h[i]/2, 0, H-1))
                    x2 = int(np.clip(x_center[i] + w[i]/2, 0, W-1))
                    y2 = int(np.clip(y_center[i] + h[i]/2, 0, H-1))
                    
                    fx1 = int(np.clip(x1 / scale, 0, W_f - 1))
                    fy1 = int(np.clip(y1 / scale, 0, H_f - 1))
                    fx2 = int(np.clip(x2 / scale, 0, W_f - 1))
                    fy2 = int(np.clip(y2 / scale, 0, H_f - 1))
                    
                    fw = fx2 - fx1
                    fh = fy2 - fy1
                    if fw < 2 or fh < 2:
                        continue
                        
                    crop_p2_gt = feat_mean[fy1:fy2, fx1:fx2]
                    mean_gt = np.mean(crop_p2_gt)
                    
                    # Find adjacent background in P2
                    bg_cropped = False
                    mean_bg = 0.0
                    directions = [(fw, 0), (-fw, 0), (0, fh), (0, -fh)]
                    for dx, dy in directions:
                        fbg_x1 = fx1 + dx
                        fbg_y1 = fy1 + dy
                        fbg_x2 = fbg_x1 + fw
                        fbg_y2 = fbg_y1 + fh
                        
                        if fbg_x1 >= 0 and fbg_x2 <= W_f and fbg_y1 >= 0 and fbg_y2 <= H_f:
                            overlap = False
                            for j in range(len(bboxes)):
                                g_x1 = int((x_center[j] - w[j]/2) / scale)
                                g_y1 = int((y_center[j] - h[j]/2) / scale)
                                g_x2 = int((x_center[j] + w[j]/2) / scale)
                                g_y2 = int((y_center[j] + h[j]/2) / scale)
                                if not (fbg_x2 <= g_x1 or fbg_x1 >= g_x2 or fbg_y2 <= g_y1 or fbg_y1 >= g_y2):
                                    overlap = True
                                    break
                            if not overlap:
                                crop_p2_bg = feat_mean[fbg_y1:fbg_y2, fbg_x1:fbg_x2]
                                mean_bg = np.mean(crop_p2_bg)
                                bg_cropped = True
                                break
                    
                    if bg_cropped:
                        # Ratio of object activation to background activation
                        ratio = mean_gt / (mean_bg + 1e-8)
                        all_candidates.append({
                            "img_rgb": img_rgb,
                            "feat_mean": feat_mean,
                            "box_coords": (x1, y1, x2, y2, w[i], h[i]),
                            "ratio": ratio,
                            "mean_gt": mean_gt,
                            "mean_bg": mean_bg
                        })
                        
    hook.remove()
    
    # Sort candidates by activation ratio (ascending)
    all_candidates.sort(key=lambda x: x["ratio"])
    
    print(f"Total candidates analyzed: {len(all_candidates)}")
    
    # Save top 3 lowest activation examples
    for saved_count in range(min(3, len(all_candidates))):
        cand = all_candidates[saved_count]
        img_rgb = cand["img_rgb"]
        feat_mean = cand["feat_mean"]
        x1, y1, x2, y2, ow, oh = cand["box_coords"]
        ratio = cand["ratio"]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left: Raw Image
        axes[0].imshow(img_rgb)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, f"{ow:.1f}x{oh:.1f}", color="red", fontsize=8, weight="bold")
        axes[0].set_title(f"Raw Image (Lowest Activation Candidate)")
        axes[0].axis("off")
        
        # Right: Heatmap
        heatmap = cv2.resize(feat_mean, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        im_heat = axes[1].imshow(heatmap, cmap="jet")
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
        axes[1].add_patch(rect)
        axes[1].set_title(f"P2 Heatmap (Act Ratio: {ratio:.4f})")
        axes[1].axis("off")
        fig.colorbar(im_heat, ax=axes[1], shrink=0.8)
        
        plt.tight_layout()
        out_file = f"docs/reports/low_activation_example_{saved_count}.png"
        fig.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved low activation case {saved_count} with ratio {ratio:.4f} to {out_file}")

if __name__ == "__main__":
    main()
