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

def compute_local_ring_zscore(X_tensor, eps=1e-5):
    # X_tensor shape: (1, C, H, W)
    avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    # 9 * AvgPool3x3(X)
    sum_x = 9.0 * avg_pool(X_tensor)
    mu_r = (sum_x - X_tensor) / 8.0
    
    # 9 * AvgPool3x3(X^2)
    sum_x2 = 9.0 * avg_pool(X_tensor ** 2)
    E_x2 = (sum_x2 - X_tensor ** 2) / 8.0
    
    var_r = torch.clamp(E_x2 - (mu_r ** 2), min=0.0)
    
    # Z score
    Z = (X_tensor - mu_r) / torch.sqrt(var_r + eps)
    Z = torch.clamp(Z, -5.0, 5.0)
    return Z

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
    
    # Coordinates of the 15 missed low-activation targets
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
    
    print("Evaluating local ring Z-score for targets...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"]
            im_file = batch.get("im_file", ["Unknown"])[0]
            name = Path(im_file).name
            if name not in low_act_targets:
                continue
                
            _ = model(imgs.float().to(device))
            
            # Compute Z-score map using the formula
            Z_map = compute_local_ring_zscore(p2_features)
            Z_mean = torch.mean(Z_map, dim=1).cpu().numpy()[0] # (128, 128)
            
            feat_mean = torch.mean(p2_features, dim=1).cpu().numpy()[0]
            
            bboxes = batch["bboxes"].numpy()
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= imgs.shape[3]
            pixel_boxes[:, [1, 3]] *= imgs.shape[2]
            
            gts = low_act_targets[name]
            for gt in gts:
                x1, y1, x2, y2 = gt
                fx1, fy1, fx2, fy2 = int(x1/4.0), int(y1/4.0), int(x2/4.0), int(y2/4.0)
                fw, fh = fx2-fx1, fy2-fy1
                
                # Mean original activation inside target
                target_act_orig = np.mean(feat_mean[fy1:fy2, fx1:fx2])
                
                # Mean Z-score inside target
                target_z = np.mean(Z_mean[fy1:fy2, fx1:fx2])
                max_z = np.max(Z_mean[fy1:fy2, fx1:fx2])
                
                # Local background in both original and Z-score
                bg_cropped = False
                mean_bg_orig = 0.0
                mean_bg_z = 0.0
                for dx, dy in [(fw, 0), (-fw, 0), (0, fh), (0, -fh)]:
                    fbg_x1, fbg_y1 = fx1+dx, fy1+dy
                    fbg_x2, fbg_y2 = fbg_x1+fw, fbg_y1+fh
                    if fbg_x1 >= 0 and fbg_x2 <= 128 and fbg_y1 >= 0 and fbg_y2 <= 128:
                        overlap = False
                        for j in range(len(bboxes)):
                            gx1 = int((pixel_boxes[j,0] - pixel_boxes[j,2]/2)/4)
                            gy1 = int((pixel_boxes[j,1] - pixel_boxes[j,3]/2)/4)
                            gx2 = int((pixel_boxes[j,0] + pixel_boxes[j,2]/2)/4)
                            gy2 = int((pixel_boxes[j,1] + pixel_boxes[j,3]/2)/4)
                            if not (fbg_x2 <= gx1 or fbg_x1 >= gx2 or fbg_y2 <= gy1 or fbg_y1 >= gy2):
                                overlap = True
                                break
                        if not overlap:
                            mean_bg_orig = np.mean(feat_mean[fbg_y1:fbg_y2, fbg_x1:fbg_x2])
                            mean_bg_z = np.mean(Z_mean[fbg_y1:fbg_y2, fbg_x1:fbg_x2])
                            bg_cropped = True
                            break
                            
                ratio_orig = target_act_orig / (mean_bg_orig + 1e-8)
                ratio_z = target_z / (mean_bg_z + 1e-8)
                
                results.append({
                    "name": name,
                    "gt": gt,
                    "ratio_orig": ratio_orig,
                    "target_z": target_z,
                    "max_z": max_z,
                    "bg_z": mean_bg_z
                })
                
    hook.remove()
    
    print("\n--- RESULTS FOR LOCAL RING Z-SCORE ---")
    for res in results:
        print(f"Image: {res['name']} | GT: {res['gt']}")
        print(f"  Orig Act Ratio: {res['ratio_orig']:.4f}")
        print(f"  Target Z (Mean): {res['target_z']:.4f} | Target Z (Max): {res['max_z']:.4f} | Bg Z (Mean): {res['bg_z']:.4f}")
        print("-" * 50)
        
    mean_target_z = np.mean([r["target_z"] for r in results])
    mean_max_z = np.mean([r["max_z"] for r in results])
    print(f"\nAverage Target Z (Mean): {mean_target_z:.4f}")
    print(f"Average Target Z (Max):  {mean_max_z:.4f}")

if __name__ == "__main__":
    main()
