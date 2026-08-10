import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

# Add ultralytics path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models_related/ultralytics"))
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator

def compute_laplacian_variance(crop):
    if crop.size == 0:
        return 0.0
    # Normalize to 0-255 uint8 if it's float
    if crop.dtype != np.uint8:
        crop = (crop * 255).astype(np.uint8)
    return cv2.Laplacian(crop, cv2.CV_64F).var()

def analyze_model_frequency(model_path, data_yaml, layer_index):
    print(f"\nAnalyzing model: {model_path} (Layer {layer_index})")
    yolo_model = YOLO(model_path)
    model = yolo_model.model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    data_dict = check_det_dataset(data_yaml)
    validator = DetectionValidator()
    validator.data = data_dict
    dataloader = validator.get_dataloader(data_dict["test"], batch_size=1)

    features = None
    def hook_fn(module, input, output):
        nonlocal features
        features = output

    hook = model.model[layer_index].register_forward_hook(hook_fn)

    gt_vars = []
    bg_vars = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 50: # Limit to 50 images for speed
                break
            imgs = batch["img"]
            x = imgs.float().to(device)
            if x.max() > 1.0:
                x /= 255.0

            _ = model(x)
            
            # features shape: (1, C, H_f, W_f)
            feat = features.cpu().numpy()[0]
            feat_mean = np.mean(feat, axis=0) # (H_f, W_f)
            # Normalize to 0-1
            f_min, f_max = feat_mean.min(), feat_mean.max()
            if f_max > f_min:
                feat_mean = (feat_mean - f_min) / (f_max - f_min)

            H_f, W_f = feat_mean.shape
            scale_x = W_f / imgs.shape[3]
            scale_y = H_f / imgs.shape[2]

            bboxes = batch["bboxes"].numpy()
            if len(bboxes) == 0:
                continue

            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= imgs.shape[3]
            pixel_boxes[:, [1, 3]] *= imgs.shape[2]

            for i in range(len(bboxes)):
                xc, yc, w, h = pixel_boxes[i]
                x1 = int(np.clip(xc - w/2, 0, imgs.shape[3]-1))
                y1 = int(np.clip(yc - h/2, 0, imgs.shape[2]-1))
                x2 = int(np.clip(xc + w/2, 0, imgs.shape[3]-1))
                y2 = int(np.clip(yc + h/2, 0, imgs.shape[2]-1))

                # Feature coordinates
                fx1 = int(np.clip(x1 * scale_x, 0, W_f - 1))
                fy1 = int(np.clip(y1 * scale_y, 0, H_f - 1))
                fx2 = int(np.clip(x2 * scale_x, 0, W_f - 1))
                fy2 = int(np.clip(y2 * scale_y, 0, H_f - 1))

                if (fx2 - fx1) < 2 or (fy2 - fy1) < 2:
                    continue

                crop_gt = feat_mean[fy1:fy2, fx1:fx2]
                
                # Adjacent background (shift crop to the right/left by width of the crop)
                shift = int(w * scale_x)
                if fx2 + shift < W_f:
                    crop_bg = feat_mean[fy1:fy2, fx1+shift:fx2+shift]
                elif fx1 - shift >= 0:
                    crop_bg = feat_mean[fy1:fy2, fx1-shift:fx2-shift]
                else:
                    continue

                gt_vars.append(compute_laplacian_variance(crop_gt))
                bg_vars.append(compute_laplacian_variance(crop_bg))

    hook.remove()
    print(f"  GT Avg Laplacian Var: {np.mean(gt_vars):.6f}")
    print(f"  BG Avg Laplacian Var: {np.mean(bg_vars):.6f}")
    print(f"  Ratio GT/BG:          {np.mean(gt_vars) / (np.mean(bg_vars) + 1e-8):.2f}x")

def main():
    # Setup dataset
    data_yaml = "./datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    cache_dir = Path("./runs/checkpoint_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Download/Get yolov8n_p2_baseline
    p2_base_local = cache_dir / "duyle2408_levir-ship-yolo-p2_train_yolov8n_p2_baseline_seed42_weights_best.pt"
    if not p2_base_local.exists():
        print("Downloading yolov8n_p2_baseline...")
        try:
            downloaded = hf_hub_download(repo_id="duyle2408/levir-ship-yolo-p2", filename="train/yolov8n_p2_baseline_seed42/weights/best.pt", repo_type="dataset")
            shutil.copy(downloaded, p2_base_local)
        except Exception as e:
            print(f"Failed to download: {e}")

    # 2. Download topdown_baseline (FPN-only baseline)
    td_base_local = cache_dir / "topdown_baseline_best.pt"
    if not td_base_local.exists():
        print("Downloading topdown_baseline...")
        try:
            downloaded = hf_hub_download(repo_id="duyle2408/levir-yolov8n-p2-topdown-3seed", filename="runs/topdown_baseline/seed_42/weights/best.pt", repo_type="dataset")
            shutil.copy(downloaded, td_base_local)
        except Exception as e:
            print(f"Failed to download: {e}")

    # 3. Download topdown_p1drr (DRR)
    td_drr_local = cache_dir / "topdown_p1drr_best.pt"
    if not td_drr_local.exists():
        print("Downloading topdown_p1drr...")
        try:
            # Let's check evaluate_all_huggingface_models.py to get exact filename: runs/topdown_p1drr_partial_clip/seed_42/weights/best.pt
            downloaded = hf_hub_download(repo_id="duyle2408/levir-yolov8n-p2-topdown-3seed", filename="runs/topdown_p1drr_partial_clip/seed_42/weights/best.pt", repo_type="dataset")
            shutil.copy(downloaded, td_drr_local)
        except Exception as e:
            print(f"Failed to download: {e}")

    # Run comparison
    # For yolov8n_p2_baseline, layer 18 is P2 (raw FPN output before head)
    if p2_base_local.exists():
        analyze_model_frequency(p2_base_local, data_yaml, 18)
    
    # For topdown_baseline, layer 18 is P2 FPN output
    if td_base_local.exists():
        analyze_model_frequency(td_base_local, data_yaml, 18)

    # For topdown_p1drr, layer 19 is P1DRR output (gated rescued P2 FPN)
    if td_drr_local.exists():
        analyze_model_frequency(td_drr_local, data_yaml, 19)

if __name__ == "__main__":
    main()
