import os
import sys
import torch
import torch.nn as nn
import numpy as np
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
    
    results = []
    
    print("Evaluating all small objects in the dataset...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"]
            _ = model(imgs.float().to(device))
            
            p1_down = nn.MaxPool2d(2, 2)(p1_features)
            p1_down_rep = p1_down.repeat(1, 2, 1, 1)
            p1_enhanced = local_bg_subtraction(p1_down_rep, kernel_size=3)
            
            feat_mean_orig = torch.mean(p2_features, dim=1).cpu().numpy()[0]
            feat_mean_enh = torch.mean(p1_enhanced, dim=1).cpu().numpy()[0]
            
            B, C, H, W = imgs.shape
            bboxes = batch["bboxes"].numpy()
            if len(bboxes) == 0:
                continue
                
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= W
            pixel_boxes[:, [1, 3]] *= H
            
            x_center, y_center, w, h = pixel_boxes[:, 0], pixel_boxes[:, 1], pixel_boxes[:, 2], pixel_boxes[:, 3]
            H_f, W_f = feat_mean_orig.shape
            scale = 4.0
            
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
                        
                    # Find adjacent background
                    bg_cropped = False
                    fbg_x1, fbg_y1, fbg_x2, fbg_y2 = 0, 0, 0, 0
                    for dx, dy in [(fw, 0), (-fw, 0), (0, fh), (0, -fh)]:
                        bx1, by1 = fx1 + dx, fy1 + dy
                        bx2, by2 = bx1 + fw, by1 + fh
                        if bx1 >= 0 and bx2 <= W_f and by1 >= 0 and by2 <= H_f:
                            overlap = False
                            for j in range(len(bboxes)):
                                g_x1 = int((x_center[j] - w[j]/2) / scale)
                                g_y1 = int((y_center[j] - h[j]/2) / scale)
                                g_x2 = int((x_center[j] + w[j]/2) / scale)
                                g_y2 = int((y_center[j] + h[j]/2) / scale)
                                if not (bx2 <= g_x1 or bx1 >= g_x2 or by2 <= g_y1 or by1 >= g_y2):
                                    overlap = True
                                    break
                            if not overlap:
                                fbg_x1, fbg_y1, fbg_x2, fbg_y2 = bx1, by1, bx2, by2
                                bg_cropped = True
                                break
                                
                    if bg_cropped:
                        r_orig = np.mean(feat_mean_orig[fy1:fy2, fx1:fx2]) / (np.mean(feat_mean_orig[fbg_y1:fbg_y2, fbg_x1:fbg_x2]) + 1e-8)
                        r_enh = np.mean(feat_mean_enh[fy1:fy2, fx1:fx2]) / (np.mean(feat_mean_enh[fbg_y1:fbg_y2, fbg_x1:fbg_x2]) + 1e-8)
                        results.append((r_orig, r_enh))
                        
    hook1.remove()
    hook2.remove()
    
    results = np.array(results)
    print(f"\nTotal small objects found: {len(results)}")
    
    # Baseline stats
    b_mean = np.mean(results[:, 0])
    b_min = np.min(results[:, 0])
    b_active = np.sum(results[:, 0] > 1.0)
    
    # Method stats
    m_mean = np.mean(results[:, 1])
    m_min = np.min(results[:, 1])
    m_active = np.sum(results[:, 1] > 1.0)
    
    print("\n=== DATASET-WIDE SMALL OBJECT COMPARISON (n={} targets) ===".format(len(results)))
    print("Method                  | Mean Ratio   | Min Ratio    | Active Ratio (>1.0)")
    print("-" * 75)
    print("Original (P2 Baseline)   | {:<12.4f} | {:<12.4f} | {:>3}/{:<3} ({:.2f}%)".format(b_mean, b_min, b_active, len(results), (b_active/len(results))*100))
    print("P1 MaxPool + Sub 3x3    | {:<12.4f} | {:<12.4f} | {:>3}/{:<3} ({:.2f}%)".format(m_mean, m_min, m_active, len(results), (m_active/len(results))*100))

if __name__ == "__main__":
    main()
