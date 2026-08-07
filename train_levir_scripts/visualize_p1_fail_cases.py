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
    print("Finding P1 MaxPool failed activation cases...")
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["img"]
            im_file = batch.get("im_file", ["Unknown"])[0]
            name = Path(im_file).name
            if name not in low_act_targets:
                continue
                
            _ = model(imgs.float().to(device))
            
            p1_down = nn.MaxPool2d(2, 2)(p1_features)
            p1_down_rep = p1_down.repeat(1, 2, 1, 1)
            
            feat_mean_orig = torch.mean(p2_features, dim=1).cpu().numpy()[0]
            feat_mean_p1 = torch.mean(p1_down_rep, dim=1).cpu().numpy()[0]
            
            img_rgb = imgs[0].numpy().transpose(1, 2, 0)
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            H, W = img_rgb.shape[:2]
            
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
                            
                if bg_cropped:
                    # Calculate P1 MaxPool ratio
                    t_val = np.mean(feat_mean_p1[fy1:fy2, fx1:fx2])
                    bg_val = np.mean(feat_mean_p1[fbg_y1:fbg_y2, fbg_x1:fbg_x2])
                    ratio = t_val / (bg_val + 1e-8)
                    
                    if ratio <= 1.0:
                        results.append({
                            "name": name,
                            "gt": gt,
                            "ratio": ratio,
                            "img_rgb": img_rgb,
                            "feat_mean_orig": feat_mean_orig,
                            "feat_mean_p1": feat_mean_p1
                        })
                        
    hook1.remove()
    hook2.remove()
    
    # Save the plots of the failed cases
    for idx, cand in enumerate(results):
        name = cand["name"]
        x1, y1, x2, y2 = cand["gt"]
        ratio = cand["ratio"]
        img_rgb = cand["img_rgb"]
        feat_mean_orig = cand["feat_mean_orig"]
        feat_mean_p1 = cand["feat_mean_p1"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_rgb)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
        axes[0].add_patch(rect)
        axes[0].set_title(f"Raw Image (P1 Fail Case {idx})")
        axes[0].axis("off")
        
        orig_heatmap = cv2.resize(feat_mean_orig, (W, H), interpolation=cv2.INTER_LINEAR)
        orig_heatmap = (orig_heatmap - orig_heatmap.min()) / (orig_heatmap.max() - orig_heatmap.min() + 1e-8)
        axes[1].imshow(orig_heatmap, cmap="jet")
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
        axes[1].add_patch(rect)
        axes[1].set_title("Original P2 Heatmap")
        axes[1].axis("off")
        
        p1_heatmap = cv2.resize(feat_mean_p1, (W, H), interpolation=cv2.INTER_LINEAR)
        p1_heatmap = (p1_heatmap - p1_heatmap.min()) / (p1_heatmap.max() - p1_heatmap.min() + 1e-8)
        axes[2].imshow(p1_heatmap, cmap="jet")
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=2, linestyle="--")
        axes[2].add_patch(rect)
        axes[2].set_title(f"P1 MaxPool Heatmap (Ratio: {ratio:.4f})")
        axes[2].axis("off")
        
        plt.tight_layout()
        out_file = f"docs/reports/p1_fail_case_{idx}.png"
        fig.savefig(out_file, dpi=300)
        plt.close()
        print(f"Saved: {out_file} (Ratio: {ratio:.4f})")

if __name__ == "__main__":
    main()
