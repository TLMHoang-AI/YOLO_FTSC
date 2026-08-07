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

def local_min_max_stretch(X_tensor, kernel_size=3, eps=1e-5):
    pad = kernel_size // 2
    max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    min_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    x_max = max_pool(X_tensor)
    x_min = -min_pool(-X_tensor)
    return (X_tensor - x_min) / (x_max - x_min + eps)

def local_bg_subtraction(X_tensor, kernel_size=3):
    pad = kernel_size // 2
    avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    x_avg = avg_pool(X_tensor)
    return torch.clamp(X_tensor - x_avg, min=0.0)

def local_std_scaling(X_tensor, kernel_size=3, beta=2.0, eps=1e-5):
    avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    mu_r = avg_pool(X_tensor)
    E_x2 = avg_pool(X_tensor ** 2)
    var_r = torch.clamp(E_x2 - (mu_r ** 2), min=0.0)
    std_r = torch.sqrt(var_r + eps)
    B, C, H, W = X_tensor.shape
    global_std = torch.std(X_tensor.view(B, C, -1), dim=2, keepdim=True).unsqueeze(-1)
    return X_tensor * (1.0 + beta * (std_r / (global_std + eps)))

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
    
    # Coordinates of the 15 targets
    low_act_targets = {
        'GF1_WFV2_E118.9_N24.3_20200710_L2A0004922278_11264_10240.png': [[87, 377, 93, 388], [39, 277, 47, 284], [127, 213, 135, 220]],
        'GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_7168_5120.png': [[319, 464, 331, 478], [455, 391, 469, 405]],
        'GF6_WFV_E133.6_N33.6_20200305_L1A1119973496-2_7680_5120.png': [[106, 272, 123, 290]],
        'GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-3_11793_8704.png': [[192, 98, 207, 116]],
        'GF1_WFV3_E120.2_N22.2_20200710_L2A0004922264_6656_5120.png': [[476, 456, 490, 472]],
        'GF6_WFV_E132.4_N35.8_20200914_L1A1120035552-1_9216_13824.png': [[17, 33, 33, 48], [182, 133, 193, 151]],
        'GF1_WFV1_E110.0_N17.9_20200703_L2A0004902374_8192_8704.png': [[174, 500, 181, 514]],
        'GF1_WFV2_E123.6_N29.3_20190910_L2A0004239231_2048_2560.png': [[168, 270, 183, 283]],
        'GF6_WFV_E133.6_N33.6_20200305_L1A1119973496-1_11264_8704.png': [[185, 501, 201, 517]],
        'GF1_WFV3_E112.3_N21.4_20190806_L2A0004164428_13824_4096.png': [[250, 332, 268, 348]],
        'GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_6144_5632.png': [[278, 507, 293, 523]]
    }
    
    results = []
    
    print("Evaluating advanced enhancements...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"]
            im_file = batch.get("im_file", ["Unknown"])[0]
            name = Path(im_file).name
            if name not in low_act_targets:
                continue
                
            _ = model(imgs.float().to(device))
            
            # P1 downsampling
            p1_down = nn.MaxPool2d(2, 2)(p1_features)
            p1_down_avg = nn.AvgPool2d(2, 2)(p1_features)
            
            p1_down_rep = p1_down.repeat(1, 2, 1, 1)
            p1_down_avg_rep = p1_down_avg.repeat(1, 2, 1, 1)
            
            # Local pooling on P1 MaxPool
            p1_max_sub = local_bg_subtraction(p1_down_rep, kernel_size=3)
            p1_max_stretch = local_min_max_stretch(p1_down_rep, kernel_size=3)
            
            # Local pooling on P1 AvgPool
            p1_avg_sub = local_bg_subtraction(p1_down_avg_rep, kernel_size=3)
            
            # Prepare maps
            map_orig = torch.mean(p2_features, dim=1).cpu().numpy()[0]
            map_p1_max = torch.mean(p1_down_rep, dim=1).cpu().numpy()[0]
            map_p1_avg = torch.mean(p1_down_avg_rep, dim=1).cpu().numpy()[0]
            map_p1_max_sub = torch.mean(p1_max_sub, dim=1).cpu().numpy()[0]
            map_p1_max_stretch = torch.mean(p1_max_stretch, dim=1).cpu().numpy()[0]
            map_p1_avg_sub = torch.mean(p1_avg_sub, dim=1).cpu().numpy()[0]
            
            bboxes = batch["bboxes"].numpy()
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= imgs.shape[3]
            pixel_boxes[:, [1, 3]] *= imgs.shape[2]
            
            gts = low_act_targets[name]
            for gt in gts:
                x1, y1, x2, y2 = gt
                fx1, fy1, fx2, fy2 = int(x1/4.0), int(y1/4.0), int(x2/4.0), int(y2/4.0)
                fw, fh = fx2-fx1, fy2-fy1
                
                bg_cropped = False
                fbg_x1, fbg_y1, fbg_x2, fbg_y2 = 0, 0, 0, 0
                for dx, dy in [(fw, 0), (-fw, 0), (0, fh), (0, -fh)]:
                    bx1, by1 = fx1 + dx, fy1 + dy
                    bx2, by2 = bx1 + fw, by1 + fh
                    if bx1 >= 0 and bx2 <= 128 and by1 >= 0 and by2 <= 128:
                        overlap = False
                        for j in range(len(bboxes)):
                            gx1 = int((pixel_boxes[j,0] - pixel_boxes[j,2]/2)/4)
                            gy1 = int((pixel_boxes[j,1] - pixel_boxes[j,3]/2)/4)
                            gx2 = int((pixel_boxes[j,0] + pixel_boxes[j,2]/2)/4)
                            gy2 = int((pixel_boxes[j,1] + pixel_boxes[j,3]/2)/4)
                            if not (bx2 <= gx1 or bx1 >= gx2 or by2 <= gy1 or by1 >= gy2):
                                overlap = True
                                break
                        if not overlap:
                            fbg_x1, fbg_y1, fbg_x2, fbg_y2 = bx1, by1, bx2, by2
                            bg_cropped = True
                            break
                            
                if not bg_cropped:
                    continue
                    
                def get_ratio(feat_map):
                    t = np.mean(feat_map[fy1:fy2, fx1:fx2])
                    b = np.mean(feat_map[fbg_y1:fbg_y2, fbg_x1:fbg_x2])
                    return t / (b + 1e-8)
                
                results.append({
                    "name": name,
                    "gt": gt,
                    "ratio_orig": get_ratio(map_orig),
                    "ratio_p1_max": get_ratio(map_p1_max),
                    "ratio_p1_avg": get_ratio(map_p1_avg),
                    "ratio_p1_max_sub": get_ratio(map_p1_max_sub),
                    "ratio_p1_max_stretch": get_ratio(map_p1_max_stretch),
                    "ratio_p1_avg_sub": get_ratio(map_p1_avg_sub)
                })
                
    hook1.remove()
    hook2.remove()
    
    print("\n=== ADVANCED ENHANCEMENT METHOD COMPARISON (n=15 targets) ===")
    print(f"{'Method':<30} | {'Mean Ratio':<12} | {'Min Ratio':<12} | {'Active (>1.0)':<15}")
    print("-" * 80)
    
    keys = ["ratio_orig", "ratio_p1_max", "ratio_p1_avg", "ratio_p1_max_sub", "ratio_p1_max_stretch", "ratio_p1_avg_sub"]
    labels = ["Original", "P1 MaxPool", "P1 AvgPool", "P1 MaxPool + Subtraction 3x3", "P1 MaxPool + Stretching 3x3", "P1 AvgPool + Subtraction 3x3"]
    
    for key, label in zip(keys, labels):
        ratios = np.array([r[key] for r in results])
        mean_r = np.mean(ratios)
        min_r = np.min(ratios)
        active_count = np.sum(ratios > 1.0)
        print(f"{label:<30} | {mean_r:<12.4f} | {min_r:<12.4f} | {active_count:>2}/{len(ratios)}")

if __name__ == "__main__":
    main()
