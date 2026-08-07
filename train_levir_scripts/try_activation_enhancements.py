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

# --- Enhancement Methods ---

def local_min_max_stretch(X_tensor, kernel_size=3, eps=1e-5):
    # X_tensor shape: (1, C, H, W)
    pad = kernel_size // 2
    max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    min_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    
    x_max = max_pool(X_tensor)
    x_min = -min_pool(-X_tensor)
    
    X_stretched = (X_tensor - x_min) / (x_max - x_min + eps)
    return X_stretched

def local_bg_subtraction(X_tensor, kernel_size=3):
    pad = kernel_size // 2
    avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=pad)
    x_avg = avg_pool(X_tensor)
    return torch.clamp(X_tensor - x_avg, min=0.0)

def deterministic_dbss(X_tensor, num_bases=4, grid_size=16, ridge_lambda=1e-3):
    # X_tensor shape: (1, C, H, W)
    B, C, H, W = X_tensor.shape
    tokens = X_tensor.squeeze(0).permute(1, 2, 0).reshape(-1, C) # (H*W, C)
    
    # Sample candidate background tokens on a grid
    grid_y = torch.linspace(0, H - 1, grid_size, dtype=torch.long)
    grid_x = torch.linspace(0, W - 1, grid_size, dtype=torch.long)
    grid_y_mesh, grid_x_mesh = torch.meshgrid(grid_y, grid_x, indexing="ij")
    bg_indices = grid_y_mesh * W + grid_x_mesh
    bg_tokens = tokens[bg_indices.flatten()] # (grid_size*grid_size, C)
    
    # Compute SVD on background tokens to get basis vectors
    # bg_tokens: (M, C)
    U_bg, S_bg, Vh_bg = torch.linalg.svd(bg_tokens, full_matrices=False)
    # The columns of Vh_bg.T are the principal components (bases) in C-dimensional space
    bases = Vh_bg[:num_bases] # (num_bases, C)
    
    # Solve ridge regression to project all tokens onto the background bases
    # coefficients = inv(bases @ bases.T + lambda * I) @ bases @ tokens.T
    base_gram = bases @ bases.T
    rhs = bases @ tokens.T
    identity = torch.eye(num_bases, device=X_tensor.device)
    gram = base_gram + ridge_lambda * identity
    coefficients = torch.linalg.solve(gram, rhs) # (num_bases, H*W)
    
    # Reconstruct background components
    bg_reconstructed = (bases.T @ coefficients).T # (H*W, C)
    
    # Residual (suppressed background)
    residual = tokens - bg_reconstructed
    residual = residual.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
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
    
    # Coordinates of the 15 missed targets
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
    
    methods = {
        "Stretching 3x3": lambda x: local_min_max_stretch(x, kernel_size=3),
        "Stretching 5x5": lambda x: local_min_max_stretch(x, kernel_size=5),
        "Subtraction 3x3": lambda x: local_bg_subtraction(x, kernel_size=3),
        "Subtraction 5x5": lambda x: local_bg_subtraction(x, kernel_size=5),
        "Subtraction 7x7": lambda x: local_bg_subtraction(x, kernel_size=7),
        "DBSS (2 bases)": lambda x: deterministic_dbss(x, num_bases=2),
        "DBSS (4 bases)": lambda x: deterministic_dbss(x, num_bases=4),
        "DBSS (8 bases)": lambda x: deterministic_dbss(x, num_bases=8),
        "DBSS (12 bases)": lambda x: deterministic_dbss(x, num_bases=12),
        "DBSS (16 bases)": lambda x: deterministic_dbss(x, num_bases=16),
        "DBSS(12) + Subtraction 3x3": lambda x: local_bg_subtraction(deterministic_dbss(x, num_bases=12), kernel_size=3),
        "DBSS(12) + Stretching 3x3": lambda x: local_min_max_stretch(deterministic_dbss(x, num_bases=12), kernel_size=3),
        "DBSS(16) + Subtraction 3x3": lambda x: local_bg_subtraction(deterministic_dbss(x, num_bases=16), kernel_size=3),
        "DBSS(16) + Stretching 3x3": lambda x: local_min_max_stretch(deterministic_dbss(x, num_bases=16), kernel_size=3),
    }
    
    # Store ratios for each method
    method_ratios = {m: [] for m in methods}
    method_ratios["Original"] = []
    
    print("Evaluating enhancement methods...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"]
            im_file = batch.get("im_file", ["Unknown"])[0]
            name = Path(im_file).name
            if name not in low_act_targets:
                continue
                
            _ = model(imgs.float().to(device))
            
            # Compute maps for all methods
            enhanced_maps = {}
            for m_name, m_func in methods.items():
                enhanced_tensor = m_func(p2_features)
                # Mean channel activation
                enhanced_maps[m_name] = torch.mean(enhanced_tensor, dim=1).cpu().numpy()[0]
                
            feat_mean_orig = torch.mean(p2_features, dim=1).cpu().numpy()[0]
            
            bboxes = batch["bboxes"].numpy()
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= imgs.shape[3]
            pixel_boxes[:, [1, 3]] *= imgs.shape[2]
            
            gts = low_act_targets[name]
            for gt in gts:
                x1, y1, x2, y2 = gt
                fx1, fy1, fx2, fy2 = int(x1/4.0), int(y1/4.0), int(x2/4.0), int(y2/4.0)
                fw, fh = fx2-fx1, fy2-fy1
                
                # Background bounding box setup
                bg_cropped = False
                fbg_x1, fbg_y1, fbg_x2, fbg_y2 = 0, 0, 0, 0
                for dx, dy in [(fw, 0), (-fw, 0), (0, fh), (0, -fh)]:
                    bx1 = fx1 + dx
                    by1 = fy1 + dy
                    bx2 = bx1 + fw
                    by2 = by1 + fh
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
                    
                # Compute Original Ratio
                target_act_orig = np.mean(feat_mean_orig[fy1:fy2, fx1:fx2])
                bg_act_orig = np.mean(feat_mean_orig[fbg_y1:fbg_y2, fbg_x1:fbg_x2])
                ratio_orig = target_act_orig / (bg_act_orig + 1e-8)
                method_ratios["Original"].append(ratio_orig)
                
                # Compute Ratios for each enhancement method
                for m_name in methods:
                    emap = enhanced_maps[m_name]
                    target_act = np.mean(emap[fy1:fy2, fx1:fx2])
                    bg_act = np.mean(emap[fbg_y1:fbg_y2, fbg_x1:fbg_x2])
                    ratio = target_act / (bg_act + 1e-8)
                    method_ratios[m_name].append(ratio)
                    
    hook.remove()
    
    print("\n=== MEAN ACTIVATION RATIO COMPARISON (n=15 targets) ===")
    print(f"{'Method':<20} | {'Mean Ratio':<12} | {'Min Ratio':<12} | {'Max Ratio':<12} | {'Active (>1.0)':<15}")
    print("-" * 75)
    for m_name, ratios in method_ratios.items():
        r_arr = np.array(ratios)
        mean_r = np.mean(r_arr)
        min_r = np.min(r_arr)
        max_r = np.max(r_arr)
        active_count = np.sum(r_arr > 1.0)
        print(f"{m_name:<20} | {mean_r:<12.4f} | {min_r:<12.4f} | {max_r:<12.4f} | {active_count:>2}/{len(ratios)}")

if __name__ == "__main__":
    main()
