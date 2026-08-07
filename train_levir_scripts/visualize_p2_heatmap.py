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
    
    saved_count = 0
    max_to_save = 3
    
    print("Generating P2 heatmaps...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch["img"] # (1, 3, 512, 512)
            x = imgs.float() / 255.0
            
            # Forward pass
            _ = model(x.to(device))
            
            # P2 feature map shape: (1, 64, 128, 128)
            feat = p2_features.cpu().numpy()[0]
            # Channel mean
            feat_mean = np.mean(feat, axis=0)
            
            B, C, H, W = imgs.shape
            bboxes = batch["bboxes"].numpy()
            
            if len(bboxes) == 0:
                continue
                
            # Convert xywh to pixel coordinates
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= W
            pixel_boxes[:, [1, 3]] *= H
            
            x_center, y_center, w, h = pixel_boxes[:, 0], pixel_boxes[:, 1], pixel_boxes[:, 2], pixel_boxes[:, 3]
            
            # Filter to see if there is any small object (<20px)
            has_small_obj = False
            small_boxes_coords = []
            for i in range(len(bboxes)):
                max_side = max(w[i], h[i])
                if max_side < 20:
                    has_small_obj = True
                    x1 = int(np.clip(x_center[i] - w[i]/2, 0, W-1))
                    y1 = int(np.clip(y_center[i] - h[i]/2, 0, H-1))
                    x2 = int(np.clip(x_center[i] + w[i]/2, 0, W-1))
                    y2 = int(np.clip(y_center[i] + h[i]/2, 0, H-1))
                    small_boxes_coords.append((x1, y1, x2, y2, w[i], h[i]))
            
            if not has_small_obj:
                continue
                
            # Prepare image for plotting
            img_rgb = imgs[0].numpy().transpose(1, 2, 0)
            # Normalize to 0-1 range
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            
            # Create side-by-side plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left: Raw Image with GT boxes
            axes[0].imshow(img_rgb)
            for x1, y1, x2, y2, ow, oh in small_boxes_coords:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
                axes[0].add_patch(rect)
                axes[0].text(x1, y1 - 5, f"{ow:.1f}x{oh:.1f}", color="red", fontsize=8, weight="bold")
            axes[0].set_title(f"Raw Image (Small Ships < 20px in Red)")
            axes[0].axis("off")
            
            # Right: P2 Heatmap
            # Resize heatmap to match image size (512x512) for overlay
            heatmap = cv2.resize(feat_mean, (W, H), interpolation=cv2.INTER_LINEAR)
            # Normalize heatmap for visualization
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            im_heat = axes[1].imshow(heatmap, cmap="jet")
            for x1, y1, x2, y2, ow, oh in small_boxes_coords:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
                axes[1].add_patch(rect)
            axes[1].set_title("P2 Feature Map Heatmap")
            axes[1].axis("off")
            fig.colorbar(im_heat, ax=axes[1], shrink=0.8)
            
            plt.tight_layout()
            out_file = f"docs/reports/p2_heatmap_example_{saved_count}.png"
            fig.savefig(out_file, dpi=300)
            plt.close()
            print(f"Saved: {out_file}")
            
            saved_count += 1
            if saved_count >= max_to_save:
                break
                
    hook.remove()
    print("Heatmap generation complete.")

if __name__ == "__main__":
    main()
