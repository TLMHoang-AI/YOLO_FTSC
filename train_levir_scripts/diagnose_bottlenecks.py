#!/usr/bin/env python3
"""Run the four bottleneck diagnostics probes on LEVIR-Ship using the trained YOLOv8n-P2 model."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import dist2bbox, non_max_suppression, xywh2xyxy, box_iou

def translate_image(img, dx, dy):
    """Translate image with replication padding on empty edges."""
    B, C, H, W = img.shape
    shifted = torch.zeros_like(img)
    
    src_y_start = max(0, -dy)
    src_y_end = H - max(0, dy)
    src_x_start = max(0, -dx)
    src_x_end = W - max(0, dx)
    
    dst_y_start = max(0, dy)
    dst_y_end = H - max(0, -dy)
    dst_x_start = max(0, dx)
    dst_x_end = W - max(0, -dx)
    
    shifted[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
    
    if dy > 0:
        shifted[:, :, :dy, :] = shifted[:, :, dy:dy+1, :]
    elif dy < 0:
        shifted[:, :, dy:, :] = shifted[:, :, dy-1:dy, :]
    if dx > 0:
        shifted[:, :, :, :dx] = shifted[:, :, :, dx:dx+1]
    elif dx < 0:
        shifted[:, :, :, dx:] = shifted[:, :, :, dx-1:dx]
    return shifted

def calculate_iou_single(box1, box2):
    """Calculate IoU between two 1D box tensors [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / max(union, 1e-8)

def main():
    checkpoint_path = ROOT.parent / "runs/levir_yolov8n_p2_psd/psd_none/seed_42/weights/best.pt"
    dataset_yaml = ROOT.parent / "datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        sys.exit(1)
        
    print(f"Loading model from: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    model.model.eval().to("cuda")
    
    # Load dataset
    print(f"Loading dataset from: {dataset_yaml}")
    data_dict = check_det_dataset(dataset_yaml)
    
    # Initialize DetectionValidator to leverage its dataloader building
    validator = DetectionValidator(args=model.overrides)
    validator.stride = model.model.stride
    validator.data = data_dict
    
    # We evaluate on the test split
    dataloader = validator.get_dataloader(data_dict["test"], batch_size=8)
    
    # Pre-allocate storage for metrics
    # Probe 1: Equivariance
    equiv_metrics = [] # dicts of {dc, dw, dh, iou_eq}
    
    # Probe 2: Oracle Candidate Gap
    oracle_gaps = []
    
    # Probe 3: DFL Uncertainty
    dfl_stats = {
        "l": {"entropy": [], "variance": [], "error": []},
        "t": {"entropy": [], "variance": [], "error": []},
        "r": {"entropy": [], "variance": [], "error": []},
        "b": {"entropy": [], "variance": [], "error": []},
    }
    
    # Probe 4: Edge decomposition
    edge_errors = {
        "cx": [], "cy": [], "w": [], "h": []
    }
    
    proj_weight = torch.arange(16, dtype=torch.float, device="cuda")
    
    print("Running diagnostics over test dataset...")
    for idx, batch in enumerate(dataloader):
        img = batch["img"].to("cuda", non_blocking=True).float() / 255.0
        B, C, H, W = img.shape
        
        # Get target boxes per image in batch
        batch_idx = batch["batch_idx"]
        bboxes_xywh = batch["bboxes"] # shape [N, 4], normalized xywh
        cls_ids = batch["cls"]
        
        # Denormalize targets to pixels
        # Ultralytics scales normalized targets by the image size during preprocessing
        # Let's decode GT boxes to pixel xyxy
        gt_boxes_batch = []
        for b in range(B):
            b_mask = batch_idx == b
            gt_bboxes = bboxes_xywh[b_mask]
            # Convert normalized xywh to pixel xyxy
            if gt_bboxes.numel() > 0:
                pixel_xywh = gt_bboxes.clone()
                pixel_xywh[:, [0, 2]] *= W
                pixel_xywh[:, [1, 3]] *= H
                pixel_xyxy = xywh2xyxy(pixel_xywh)
                gt_boxes_batch.append(pixel_xyxy.to("cuda"))
            else:
                gt_boxes_batch.append(torch.zeros((0, 4), device="cuda"))

        # Run inference once to get raw output
        with torch.no_grad():
            preds_tuple = model.model(img)
            decoded_preds = preds_tuple[0] # shape (B, 5, 21760)
            raw_logits = preds_tuple[1]    # dict containing "boxes", "scores", "feats"
            
            pred_distri = raw_logits["boxes"].permute(0, 2, 1).contiguous() # (B, 21760, 64)
            pred_scores = raw_logits["scores"].permute(0, 2, 1).contiguous().sigmoid() # (B, 21760, 1)
            
            # Post-NMS predictions
            post_nms_list = non_max_suppression(decoded_preds, conf_thres=0.001, iou_thres=0.7)
            
        # 1. One-Pixel Translation Equivariance Test
        shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in shifts:
            img_shift = translate_image(img, dx, dy)
            with torch.no_grad():
                preds_shift_tuple = model.model(img_shift)
                boxes_shift_list = non_max_suppression(preds_shift_tuple[0], conf_thres=0.001, iou_thres=0.7)
                
            for b in range(B):
                gt_boxes = gt_boxes_batch[b]
                if gt_boxes.shape[0] == 0:
                    continue
                
                # Original boxes for batch element
                orig_boxes = post_nms_list[b] # (N, 6) -> x1, y1, x2, y2, conf, cls
                # Shifted boxes for batch element
                shift_boxes = boxes_shift_list[b].clone() # (M, 6)
                
                if orig_boxes.shape[0] == 0 or shift_boxes.shape[0] == 0:
                    continue
                
                # Shift-back the boxes from shifted image
                shift_boxes[:, 0] -= dx
                shift_boxes[:, 1] -= dy
                shift_boxes[:, 2] -= dx
                shift_boxes[:, 3] -= dy
                
                # Match boxes with GT
                for gt in gt_boxes:
                    # Find best match in orig
                    ious_orig = box_iou(gt.unsqueeze(0), orig_boxes[:, :4]).squeeze(0)
                    best_orig_idx = ious_orig.argmax().item()
                    
                    # Find best match in shift_back
                    ious_shift = box_iou(gt.unsqueeze(0), shift_boxes[:, :4]).squeeze(0)
                    best_shift_idx = ious_shift.argmax().item()
                    
                    if ious_orig[best_orig_idx] > 0.5 and ious_shift[best_shift_idx] > 0.5:
                        b_orig = orig_boxes[best_orig_idx, :4]
                        b_shift = shift_boxes[best_shift_idx, :4]
                        
                        # Compute metrics
                        c_orig = torch.tensor([(b_orig[0] + b_orig[2]) / 2.0, (b_orig[1] + b_orig[3]) / 2.0])
                        c_shift = torch.tensor([(b_shift[0] + b_shift[2]) / 2.0, (b_shift[1] + b_shift[3]) / 2.0])
                        
                        dc = torch.norm(c_orig - c_shift).item()
                        dw = abs((b_orig[2] - b_orig[0]) - (b_shift[2] - b_shift[0])).item()
                        dh = abs((b_orig[3] - b_orig[1]) - (b_shift[3] - b_shift[1])).item()
                        iou_eq = calculate_iou_single(b_orig.tolist(), b_shift.tolist())
                        
                        equiv_metrics.append({"dc": dc, "dw": dw, "dh": dh, "iou_eq": iou_eq})

        # 2. Raw-Candidate Oracle Gap + 3 & 4: Uncertainty & Edge errors
        # Anchor details
        from ultralytics.utils.ops import make_anchors
        anchor_points, stride_tensor = make_anchors(raw_logits["feats"], model.model.stride, 0.5)
        n_p2 = math.prod(raw_logits["feats"][0].shape[2:])
        p2_anchors = anchor_points[:n_p2]
        p2_strides = stride_tensor[:n_p2]
        
        # Decode P2 boxes
        p2_distri_raw = pred_distri[:, :n_p2] # (B, n_p2, 64)
        p2_scores_raw = pred_scores[:, :n_p2] # (B, n_p2, 1)
        
        # Decode xyxy candidates
        p2_boxes_xyxy = []
        for b in range(B):
            bboxes_decoded = dist2bbox(p2_distri_raw[b], p2_anchors, xywh=False) * p2_strides
            p2_boxes_xyxy.append(bboxes_decoded) # shape (n_p2, 4)
            
        for b in range(B):
            gt_boxes = gt_boxes_batch[b]
            if gt_boxes.shape[0] == 0:
                continue
                
            orig_boxes = post_nms_list[b]
            
            for gt in gt_boxes:
                # Find all P2 anchors inside GT box (with extra 2px padding neighborhood)
                # gt is [x1, y1, x2, y2]
                pad = 2.0
                in_mask = (
                    (p2_anchors[:, 0] * p2_strides[:, 0] >= gt[0] - pad) &
                    (p2_anchors[:, 0] * p2_strides[:, 0] <= gt[2] + pad) &
                    (p2_anchors[:, 1] * p2_strides[:, 1] >= gt[1] - pad) &
                    (p2_anchors[:, 1] * p2_strides[:, 1] <= gt[3] + pad)
                )
                
                indices = in_mask.nonzero(as_tuple=False).squeeze(-1)
                if indices.numel() == 0:
                    continue
                
                # Candidates
                cand_boxes = p2_boxes_xyxy[b][indices] # (K, 4)
                cand_scores = p2_scores_raw[b, indices].squeeze(-1) # (K,)
                
                # Compute IoUs with GT
                cand_ious = box_iou(gt.unsqueeze(0), cand_boxes).squeeze(0) # (K,)
                
                # Highest Score Candidate (b_score)
                best_score_idx = cand_scores.argmax().item()
                b_score = cand_boxes[best_score_idx]
                iou_score = cand_ious[best_score_idx].item()
                
                # Highest IoU Candidate (b_oracle)
                best_iou_idx = cand_ious.argmax().item()
                b_oracle = cand_boxes[best_iou_idx]
                iou_oracle = cand_ious[best_iou_idx].item()
                
                # Log oracle gap
                gap = iou_oracle - iou_score
                oracle_gaps.append(gap)
                
                # 3 & 4. DFL Uncertainty & Edge decomposition
                # We analyze the candidate that best matches the GT (b_oracle)
                best_anchor_idx = indices[best_iou_idx].item()
                
                # Logits for the 4 edges
                logits_edges = p2_distri_raw[b, best_anchor_idx].view(4, 16)
                pred_xyxy = b_oracle
                
                # True coordinates of the edges
                # gt is [x1_gt, y1_gt, x2_gt, y2_gt]
                # Convert prediction to dist form to align with DFL
                # edge sequence: l, t, r, b
                gt_l = (p2_anchors[best_anchor_idx, 0] * p2_strides[best_anchor_idx, 0] - gt[0]) / p2_strides[best_anchor_idx, 0]
                gt_t = (p2_anchors[best_anchor_idx, 1] * p2_strides[best_anchor_idx, 1] - gt[1]) / p2_strides[best_anchor_idx, 1]
                gt_r = (gt[2] - p2_anchors[best_anchor_idx, 0] * p2_strides[best_anchor_idx, 0]) / p2_strides[best_anchor_idx, 0]
                gt_b = (gt[3] - p2_anchors[best_anchor_idx, 1] * p2_strides[best_anchor_idx, 1]) / p2_strides[best_anchor_idx, 1]
                
                true_dists = [gt_l.item(), gt_t.item(), gt_r.item(), gt_b.item()]
                
                for s_idx, side in enumerate(["l", "t", "r", "b"]):
                    probs = F.softmax(logits_edges[s_idx], dim=-1)
                    
                    # Entropy
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                    
                    # Expected value
                    pred_dist_val = torch.sum(probs * proj_weight).item()
                    
                    # Variance
                    variance = torch.sum(probs * (proj_weight - pred_dist_val) ** 2).item()
                    
                    # Absolute error in pixels (multiply by stride)
                    abs_error = abs(pred_dist_val - true_dists[s_idx]) * float(p2_strides[best_anchor_idx, 0].item())
                    
                    dfl_stats[side]["entropy"].append(entropy)
                    dfl_stats[side]["variance"].append(variance)
                    dfl_stats[side]["error"].append(abs_error)
                    
                # 4. Edge decomposition
                # Signed edge errors in pixels
                el = (pred_xyxy[0] - gt[0]).item()
                er = (pred_xyxy[2] - gt[2]).item()
                et = (pred_xyxy[1] - gt[1]).item()
                eb = (pred_xyxy[3] - gt[3]).item()
                
                # Translation and scale errors
                ecx = (el + er) / 2.0
                ecy = (et + eb) / 2.0
                ew = er - el
                eh = eb - et
                
                edge_errors["cx"].append(ecx)
                edge_errors["cy"].append(ecy)
                edge_errors["w"].append(ew)
                edge_errors["h"].append(eh)

    print("\nProcessing diagnostic results...")
    
    # 1. Equivariance statistics
    if equiv_metrics:
        mean_dc = np.mean([x["dc"] for x in equiv_metrics])
        mean_dw = np.mean([x["dw"] for x in equiv_metrics])
        mean_dh = np.mean([x["dh"] for x in equiv_metrics])
        mean_iou_eq = np.mean([x["iou_eq"] for x in equiv_metrics])
    else:
        mean_dc, mean_dw, mean_dh, mean_iou_eq = 0, 0, 0, 0
        
    # 2. Oracle gap statistics
    if oracle_gaps:
        mean_gap = np.mean(oracle_gaps)
        max_gap = np.max(oracle_gaps)
    else:
        mean_gap, max_gap = 0, 0
        
    # 3. Spearman correlations
    correlations = {}
    for side in ["l", "t", "r", "b"]:
        errors = dfl_stats[side]["error"]
        entropies = dfl_stats[side]["entropy"]
        variances = dfl_stats[side]["variance"]
        
        if len(errors) > 1:
            corr_ent, _ = stats.spearmanr(entropies, errors)
            corr_var, _ = stats.spearmanr(variances, errors)
        else:
            corr_ent, corr_var = 0.0, 0.0
            
        correlations[side] = {"entropy": corr_ent, "variance": corr_var}
        
    # 4. Error decomposition statistics
    decomp_stats = {}
    for key in ["cx", "cy", "w", "h"]:
        vals = edge_errors[key]
        if vals:
            decomp_stats[key] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "mae": np.mean(np.abs(vals))
            }
        else:
            decomp_stats[key] = {"mean": 0, "std": 0, "mae": 0}
            
    # Formulate Report markdown
    report_content = f"""# LEVIR-Ship Bounding Box Bottleneck Diagnostics Report

Bản báo cáo này tổng hợp kết quả của 4 probe chẩn đoán (diagnostics) chạy trực tiếp trên checkpoint mô hình tốt nhất của baseline (`best.pt` - seed 42) chạy validation trên tập kiểm thử (test split) của LEVIR-Ship.

---

## 1. Probe 1: One-Pixel Translation Equivariance Test
Mục tiêu là kiểm chứng mô hình có đạt tính bất biến tịnh tiến sub-pixel hay không khi dịch chuyển ảnh đúng 1 pixel theo các hướng.

- **Độ lệch tâm trung bình ($\Delta c$):** `{mean_dc:.4f}` pixel
- **Độ lệch chiều rộng trung bình ($\Delta w$):** `{mean_dw:.4f}` pixel
- **Độ lệch chiều cao trung bình ($\Delta h$):** `{mean_dh:.4f}` pixel
- **Độ tương quan IoU giữa nguyên bản và dịch chuyển ($\\text{{IoU}}_{{eq}}$):** `{mean_iou_eq * 100:.2f}%`

*Nhận xét:* Nếu $\\text{{IoU}}_{{eq}}$ thấp và độ lệch tọa độ vượt quá 1-2 pixel, điều này chứng tỏ bộ lọc downsampling/pooling hiện tại đang làm mất phase-information nghiêm trọng ở các vật thể siêu nhỏ.

---

## 2. Probe 2: Raw-Candidate Oracle Gap
Mục tiêu là kiểm tra xem classifier chọn score đúng hay sai candidate có bounding box tốt nhất quanh GT.

- **Oracle Gap trung bình ($\\Delta_{\\text{{oracle}}}$):** `{mean_gap:.4f}` (Chênh lệch IoU tối đa của candidate tốt nhất so với candidate có score cao nhất)
- **Oracle Gap lớn nhất:** `{max_gap:.4f}`

*Nhận xét:* Nếu Oracle Gap lớn (ví dụ $> 0.15$), điều này khẳng định regression head đã tạo được bounding box tốt nhưng classification score đang ranking sai candidate.

---

## 3. Probe 3: DFL Uncertainty–Error Correlation
Mục tiêu là đo lường xem Entropy/Variance của phân phối DFL có đồng biến với sai số định vị thực tế của các cạnh hay không.

Hệ số tương quan Spearman giữa Độ bất định và Sai số định vị (pixels):

| Cạnh | Tương quan với Entropy | Tương quan với Variance |
| :--- | :---: | :---: |
| **Trái (l)** | `{correlations["l"]["entropy"]:.4f}` | `{correlations["l"]["variance"]:.4f}` |
| **Trên (t)** | `{correlations["t"]["entropy"]:.4f}` | `{correlations["t"]["variance"]:.4f}` |
| **Phải (r)** | `{correlations["r"]["entropy"]:.4f}` | `{correlations["r"]["variance"]:.4f}` |
| **Dưới (b)** | `{correlations["b"]["entropy"]:.4f}` | `{correlations["b"]["variance"]:.4f}` |

*Nhận xét:* Hệ số tương quan cao ($> 0.3$) chứng tỏ độ bất định của phân phối DFL có thể làm chỉ báo trực tiếp cho chất lượng box, mở ra hướng ứng dụng uncertainty-guided refinement.

---

## 4. Probe 4: Edge-Error Decomposition
Phân tích sai số định vị thành phần để xác định mô hình bị lệch tâm (translation) hay lệch kích cỡ (scale).

| Thành phần sai số | Giá trị trung bình (Signed Mean) | Sai lệch chuẩn (Std) | Sai số tuyệt đối trung bình (MAE) |
| :--- | :---: | :---: | :---: |
| **Lệch tâm X ($\\epsilon_{{cx}}$)** | `{decomp_stats["cx"]["mean"]:.4f}` px | `{decomp_stats["cx"]["std"]:.4f}` px | `{decomp_stats["cx"]["mae"]:.4f}` px |
| **Lệch tâm Y ($\\epsilon_{{cy}}$)** | `{decomp_stats["cy"]["mean"]:.4f}` px | `{decomp_stats["cy"]["std"]:.4f}` px | `{decomp_stats["cy"]["mae"]:.4f}` px |
| **Lệch Chiều rộng ($\\epsilon_w$)** | `{decomp_stats["w"]["mean"]:.4f}` px | `{decomp_stats["w"]["std"]:.4f}` px | `{decomp_stats["w"]["mae"]:.4f}` px |
| **Lệch Chiều cao ($\\epsilon_h$)** | `{decomp_stats["h"]["mean"]:.4f}` px | `{decomp_stats["h"]["std"]:.4f}` px | `{decomp_stats["h"]["mae"]:.4f}` px |

*Nhận xét:* 
- Nếu MAE lệch tâm X/Y lớn hơn MAE chiều rộng/cao, lỗi định vị chủ yếu do mô hình bị dịch chuyển box (translation).
- Nếu Signed Mean lệch chiều rộng/cao lệch xa khỏi 0, mô hình đang có xu hướng dự đoán box luôn to hơn hoặc nhỏ hơn GT một cách hệ thống.
"""
    
    output_path = ROOT.parent / "docs/reports/report_bottleneck_diagnostics.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_content, encoding="utf-8")
    print(f"\nDiagnostics report successfully written to: {output_path}")

if __name__ == "__main__":
    main()
