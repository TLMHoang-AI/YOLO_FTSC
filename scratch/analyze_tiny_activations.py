import os
import sys
import torch
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

# Add ultralytics path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models_related/ultralytics"))
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator

def analyze_tiny_activations(model_path, data_yaml):
    print(f"\nAnalyzing target activations for: {model_path}")
    yolo_model = YOLO(model_path)
    model = yolo_model.model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    data_dict = check_det_dataset(data_yaml)
    validator = DetectionValidator()
    validator.data = data_dict
    dataloader = validator.get_dataloader(data_dict["test"], batch_size=1)

    # We will hook:
    # 1. Layer 0 (first conv layer output - representing P1 scale raw features)
    # 2. Layer 18 (P2 baseline neck output)
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

    tiny_stats = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch["img"]
            x = imgs.float().to(device)
            if x.max() > 1.0:
                x /= 255.0

            _ = model(x)
            
            B, C, H, W = imgs.shape
            bboxes = batch["bboxes"].numpy()
            if len(bboxes) == 0:
                continue

            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= W
            pixel_boxes[:, [1, 3]] *= H

            # Input gray image
            img_gray = imgs[0].float().mean(dim=0).cpu().numpy() / 255.0

            for i in range(len(bboxes)):
                xc, yc, w, h = pixel_boxes[i]
                area = w * h
                if area >= 100: # Only analyze tiny targets (<100 px^2)
                    continue

                x1 = int(np.clip(xc - w/2, 0, W-1))
                y1 = int(np.clip(yc - h/2, 0, H-1))
                x2 = int(np.clip(xc + w/2, 0, W-1))
                y2 = int(np.clip(yc + h/2, 0, H-1))

                # 1. Activation ratio at Input Space
                input_crop = img_gray[y1:y2, x1:x2]
                input_mean = input_crop.mean()
                # Adjacent background at input
                shift = int(w)
                if x2 + shift < W:
                    input_bg = img_gray[y1:y2, x2+shift:x2+2*shift].mean()
                elif x1 - 2*shift >= 0:
                    input_bg = img_gray[y1:y2, x1-2*shift:x1-shift].mean()
                else:
                    input_bg = input_mean
                input_ratio = input_mean / (input_bg + 1e-8)

                # 2. Activation ratio at Layer 0 (P1 features - stride 2)
                p1_feat = p1_features.cpu().numpy()[0]
                p1_mean = np.mean(p1_feat, axis=0)
                H_p1, W_p1 = p1_mean.shape
                p1_x1 = int(x1 * (W_p1 / W))
                p1_y1 = int(y1 * (H_p1 / H))
                p1_x2 = int(x2 * (W_p1 / W))
                p1_y2 = int(y2 * (H_p1 / H))
                p1_crop = p1_mean[p1_y1:max(p1_y2, p1_y1+1), p1_x1:max(p1_x2, p1_x1+1)]
                p1_val = p1_crop.mean()
                # Background
                p1_shift = int(w * (W_p1 / W))
                if p1_x2 + p1_shift < W_p1:
                    p1_bg = p1_mean[p1_y1:max(p1_y2, p1_y1+1), p1_x2+p1_shift:max(p1_x2+2*p1_shift, p1_x2+p1_shift+1)].mean()
                else:
                    p1_bg = p1_val
                p1_ratio = p1_val / (p1_bg + 1e-8)

                # 3. Activation ratio at Layer 18 (P2 baseline - stride 4)
                p2_feat = p2_features.cpu().numpy()[0]
                p2_mean = np.mean(p2_feat, axis=0)
                H_p2, W_p2 = p2_mean.shape
                p2_x1 = int(x1 * (W_p2 / W))
                p2_y1 = int(y1 * (H_p2 / H))
                p2_x2 = int(x2 * (W_p2 / W))
                p2_y2 = int(y2 * (H_p2 / H))
                p2_crop = p2_mean[p2_y1:max(p2_y2, p2_y1+1), p2_x1:max(p2_x2, p2_x1+1)]
                p2_val = p2_crop.mean()
                # Background
                p2_shift = int(w * (W_p2 / W))
                if p2_x2 + p2_shift < W_p2:
                    p2_bg = p2_mean[p2_y1:max(p2_y2, p2_y1+1), p2_x2+p2_shift:max(p2_x2+2*p2_shift, p2_x2+p2_shift+1)].mean()
                else:
                    p2_bg = p2_val
                p2_ratio = p2_val / (p2_bg + 1e-8)

                tiny_stats.append({
                    "area": area,
                    "input_ratio": input_ratio,
                    "p1_ratio": p1_ratio,
                    "p2_ratio": p2_ratio
                })

    hook1.remove()
    hook2.remove()

    print(f"Analyzed {len(tiny_stats)} tiny targets (<100 px^2):")
    for idx, stat in enumerate(tiny_stats):
        print(f"  Target {idx} (Area: {stat['area']:.1f} px^2):")
        print(f"    Input space activation ratio: {stat['input_ratio']:.4f}")
        print(f"    P1 feature activation ratio:  {stat['p1_ratio']:.4f}")
        print(f"    P2 feature activation ratio:  {stat['p2_ratio']:.4f}")

def main():
    data_yaml = "./datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    cache_dir = Path("./runs/checkpoint_cache")
    p2_base_local = cache_dir / "duyle2408_levir-ship-yolo-p2_train_yolov8n_p2_baseline_seed42_weights_best.pt"

    if p2_base_local.exists():
        analyze_tiny_activations(p2_base_local, data_yaml)

if __name__ == "__main__":
    main()
