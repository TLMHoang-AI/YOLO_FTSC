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
from ultralytics.utils.ops import xywh2xyxy

def box_iou(box1, box2):
    # box1: (N, 4), box2: (M, 4) in xyxy format
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = torch.clamp(rb - lt, min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-8)

def analyze_missed_targets(model_path, data_yaml):
    print(f"\nAnalyzing missed targets (FN) for: {model_path}")
    yolo_model = YOLO(model_path)
    model = yolo_model.model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    data_dict = check_det_dataset(data_yaml)
    validator = DetectionValidator()
    validator.data = data_dict
    dataloader = validator.get_dataloader(data_dict["test"], batch_size=1)

    # Categories
    # Tiny: < 100 px^2, Small: 100-400 px^2, Med/Large: > 400 px^2
    fn_by_size = {"tiny": 0, "small": 0, "large": 0}
    total_gt_by_size = {"tiny": 0, "small": 0, "large": 0}
    
    # Dense ports verification: how many FNs are close to another GT
    fn_dense_count = 0
    total_fn = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch["img"]
            x = imgs.float().to(device)
            if x.max() > 1.0:
                x /= 255.0

            # Get predictions (using default conf=0.25, iou=0.45)
            preds = model(x)
            
            # Predict boxes
            # Predict returns list of Results objects in ultralytics YOLO
            # Let's extract the boxes
            pred_boxes = yolo_model(x, conf=0.25, iou=0.45, verbose=False)[0].boxes.xyxy.cpu().numpy()

            bboxes = batch["bboxes"].numpy() # normalized xywh
            if len(bboxes) == 0:
                continue

            B, C, H, W = imgs.shape
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= W
            pixel_boxes[:, [1, 3]] *= H
            
            # Convert GT to xyxy
            gt_xyxy = xywh2xyxy(pixel_boxes)

            # Classify GT sizes
            for box in pixel_boxes:
                area = box[2] * box[3]
                if area < 100:
                    total_gt_by_size["tiny"] += 1
                elif area < 400:
                    total_gt_by_size["small"] += 1
                else:
                    total_gt_by_size["large"] += 1

            # Match GTs with Predictions
            if len(pred_boxes) == 0:
                # All GTs are FNs
                for idx, box in enumerate(pixel_boxes):
                    area = box[2] * box[3]
                    size_cat = "tiny" if area < 100 else ("small" if area < 400 else "large")
                    fn_by_size[size_cat] += 1
                    total_fn += 1
                    
                    # Proximity check
                    dists = np.sqrt((pixel_boxes[:, 0] - box[0])**2 + (pixel_boxes[:, 1] - box[1])**2)
                    # if there is another GT within 40px
                    if np.sum(dists < 40) > 1:
                        fn_dense_count += 1
                continue

            # Compute IoU between GT and Preds
            iou_matrix = box_iou(torch.tensor(gt_xyxy), torch.tensor(pred_boxes)).numpy()
            
            # For each GT, check if it has a match with IoU >= 0.3
            for idx, box in enumerate(pixel_boxes):
                area = box[2] * box[3]
                size_cat = "tiny" if area < 100 else ("small" if area < 400 else "large")
                
                max_iou = iou_matrix[idx].max() if iou_matrix.shape[1] > 0 else 0
                if max_iou < 0.3:
                    # Missed target (FN)
                    fn_by_size[size_cat] += 1
                    total_fn += 1
                    
                    # Proximity check: Is this FN close to another GT?
                    dists = np.sqrt((pixel_boxes[:, 0] - box[0])**2 + (pixel_boxes[:, 1] - box[1])**2)
                    if np.sum(dists < 40) > 1:
                        fn_dense_count += 1

    print("  Total GTs analyzed by size:", total_gt_by_size)
    print("  FNs by size:", fn_by_size)
    print("  FN Recall rates:")
    for cat in ["tiny", "small", "large"]:
        t = total_gt_by_size[cat]
        fn = fn_by_size[cat]
        rec = (t - fn) / (t + 1e-8) * 100
        print(f"    {cat.upper()}: {rec:.2f}% (Missed {fn}/{t})")
    
    print(f"  Proximity check: {fn_dense_count}/{total_fn} FNs ({fn_dense_count/(total_fn+1e-8)*100:.2f}%) are in close proximity (<40px) to other targets (Dense ports/groups).")

def main():
    data_yaml = "./datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    cache_dir = Path("./runs/checkpoint_cache")

    p2_base_local = cache_dir / "duyle2408_levir-ship-yolo-p2_train_yolov8n_p2_baseline_seed42_weights_best.pt"
    td_drr_local = cache_dir / "topdown_p1drr_best.pt"

    if p2_base_local.exists():
        analyze_missed_targets(p2_base_local, data_yaml)
    if td_drr_local.exists():
        analyze_missed_targets(td_drr_local, data_yaml)

if __name__ == "__main__":
    main()
