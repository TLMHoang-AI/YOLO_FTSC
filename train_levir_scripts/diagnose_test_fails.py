#!/usr/bin/env python3
import os
import sys
import numpy as np
from pathlib import Path

# Add local ultralytics to path
sys.path.insert(0, "/marimo/yolo_code/models_related/ultralytics")
from ultralytics import YOLO

def box_iou(box1, box2):
    # box1: [N, 4], box2: [M, 4]
    # format: [x1, y1, x2, y2]
    lt = np.maximum(box1[:, None, :2], box2[:, :2])
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-8)

def main():
    model_path = "/marimo/runs/levir_yolov8n_p2_topdown_seed42/topdown_p1ger_200/seed_42/weights/best.pt"
    img_dir = "/marimo/datasets/levir_ship_yolo_seed42/images/test"
    label_dir = "/marimo/datasets/levir_ship_yolo_seed42/labels/test"
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found!")
        return

    print("Loading model...")
    model = YOLO(model_path)
    
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(img_files)} test images.")
    
    total_gt_count = 0
    total_pred_count = 0
    
    # Categorization buckets
    # tiny (< 100 px^2), small (100-400 px^2), medium (> 400 px^2)
    categories = ["tiny (<100 px^2)", "small (100-400 px^2)", "medium (>400 px^2)"]
    gt_by_size = {c: 0 for c in categories}
    fn_by_size = {c: 0 for c in categories}
    tp_by_size = {c: 0 for c in categories}
    
    fp_count = 0
    fp_by_image = {}
    fn_by_image = {}
    
    for img_path in img_files:
        name = os.path.basename(img_path)
        lbl_path = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
        
        # Load GT boxes
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        # convert from norm xywh to absolute xyxy on 512x512
                        x1 = (x - w/2) * 512
                        y1 = (y - h/2) * 512
                        x2 = (x + w/2) * 512
                        y2 = (y + h/2) * 512
                        gt_boxes.append([x1, y1, x2, y2])
        gt_boxes = np.array(gt_boxes)
        
        # Run prediction
        res = model.predict(img_path, conf=0.25, verbose=False)[0]
        pred_boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.zeros((0, 4))
        
        total_gt_count += len(gt_boxes)
        total_pred_count += len(pred_boxes)
        
        # Classify GT sizes
        gt_cats = []
        for box in gt_boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            area = w * h
            if area < 100:
                cat = "tiny (<100 px^2)"
            elif area < 400:
                cat = "small (100-400 px^2)"
            else:
                cat = "medium (>400 px^2)"
            gt_by_size[cat] += 1
            gt_cats.append(cat)
            
        matched_gt = set()
        matched_pred = set()
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = box_iou(pred_boxes, gt_boxes)
            for p_idx in range(len(pred_boxes)):
                best_gt_idx = np.argmax(ious[p_idx])
                if ious[p_idx, best_gt_idx] >= 0.3:
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(p_idx)
                    tp_by_size[gt_cats[best_gt_idx]] += 1
                    
        # False Negatives (Missed)
        fns = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
        for idx in fns:
            fn_by_size[gt_cats[idx]] += 1
            
        # False Positives (False Alarms)
        fps = len(pred_boxes) - len(matched_pred)
        fp_count += fps
        
        if fps > 0:
            fp_by_image[name] = fps
        if len(fns) > 0:
            fn_by_image[name] = len(fns)
            
    print("\n================ DIAGNOSTIC REPORT ================")
    print(f"Total Ground-Truth Objects: {total_gt_count}")
    print(f"Total Predicted Objects   : {total_pred_count}")
    print(f"Total False Positives (FP): {fp_count}")
    print(f"Total False Negatives (FN): {sum(fn_by_size.values())}")
    
    print("\n--- Breakdown by Object Size ---")
    for c in categories:
        total = gt_by_size[c]
        tp = tp_by_size[c]
        fn = fn_by_size[c]
        recall = tp / (total + 1e-8)
        print(f"Category: {c:<22} | GT: {total:<4} | TP: {tp:<4} | FN (Missed): {fn:<4} | Recall: {recall:6.2%}")
        
    print("\n--- Top 10 Images with Most Missed Objects (False Negatives) ---")
    sorted_fns = sorted(fn_by_image.items(), key=lambda x: x[1], reverse=True)[:10]
    for idx, (img_name, count) in enumerate(sorted_fns, 1):
        print(f"{idx:<2}. {img_name:<80} | FNs: {count}")
        
    print("\n--- Top 10 Images with Most False Alarms (False Positives) ---")
    sorted_fps = sorted(fp_by_image.items(), key=lambda x: x[1], reverse=True)[:10]
    for idx, (img_name, count) in enumerate(sorted_fps, 1):
        print(f"{idx:<2}. {img_name:<80} | FPs: {count}")

if __name__ == "__main__":
    main()
