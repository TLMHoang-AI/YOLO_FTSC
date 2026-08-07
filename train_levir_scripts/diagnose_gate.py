#!/usr/bin/env python3
"""Run diagnostic gate analysis on frozen baseline and trained A2_v2 checkpoints."""

import os
import sys
from pathlib import Path
import torch
import numpy as np

# Insert ultralytics path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models_related/ultralytics"))
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops

def get_donut_mask(gt_mask, pad=2):
    """Create a ring/donut around the GT box mask."""
    import scipy.ndimage
    dilated = scipy.ndimage.binary_dilation(gt_mask, iterations=pad)
    return dilated & ~gt_mask

def run_diagnostics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Paths
    dataset_yaml = "/marimo/datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    baseline_path = "/marimo/yolo_code/diagnostics/hf_yolov8n_p2/train/yolov8n_p2_baseline_seed42/weights/best.pt"
    a2_path = "/marimo/runs/levir_yolov8n_p2_v2_seed42/p1ger/p1ger/seed_42/weights/best.pt"
    
    # Load dataset info
    dataset = check_det_dataset(dataset_yaml)
    val_loader = dataset["test"]  # test split is the validation set in our seed split
    
    # Let's import Ultralytics components
    from ultralytics.models.yolo.detect import DetectionValidator
    
    # We will load the baseline model to perform the frozen baseline analysis
    print("Loading baseline model...")
    model_base = YOLO(baseline_path).to(device)
    
    # Prepare diagnostic storage
    d_gt_list, d_ring_list, d_bg_list = [], [], []
    e1_gt_list, e1_ring_list, e1_bg_list = [], [], []
    e2_gt_list, e2_ring_list, e2_bg_list = [], [], []
    
    dormant_d_list = []
    
    # To get features, we register a forward hook on the layers
    # In YOLOv8 baseline:
    # Layer 2 is P1 backbone (stride 2)
    # Layer 18 is P2 neck (stride 4)
    # Let's inspect the model modules first to make sure we hook the right layers
    p1_feats = {}
    p2_feats = {}
    
    def hook_p1(module, input, output):
        p1_feats['val'] = output.detach()
        
    def hook_p2(module, input, output):
        p2_feats['val'] = output.detach()
        
    # Standard baseline model layers:
    # model_base.model.model[2] is Conv stride-2 (P1)
    # model_base.model.model[18] is C2f (P2 neck)
    h1 = model_base.model.model[2].register_forward_hook(hook_p1)
    h2 = model_base.model.model[18].register_forward_hook(hook_p2)
    
    # Load validation images and run prediction
    print("Running validation diagnostics...")
    
    # Let's use the validator to load validation batch by batch
    args = model_base.overrides.copy()
    args['data'] = dataset_yaml
    args['device'] = device
    validator = DetectionValidator(args=args)
    validator.data = dataset
    validator.stride = 32
    validator.device = device
    dataloader = validator.get_dataloader(dataset["test"], batch_size=1)
    
    # Setup downsampler for P1
    downsample = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    local_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 100:  # Diagnose first 100 images
            break
            
        # Run forward pass
        batch = validator.preprocess(batch)
        with torch.no_grad():
            preds = model_base.model(batch['img'])
            
        # Hooked outputs
        p1 = p1_feats['val']  # (1, C_p1, H_p1, W_p1)
        p2 = p2_feats['val']  # (1, C_p2, H_p2, W_p2)
        
        # Calculate Evidence exactly as in P1-GER v2
        p1_down = downsample(p1)
        avg_p1 = local_pool(p1_down)
        E1 = torch.mean(torch.abs(p1_down - avg_p1), dim=1, keepdim=True)
        
        avg_p2 = local_pool(p2)
        E2 = torch.mean(torch.abs(p2 - avg_p2), dim=1, keepdim=True)
        
        s1 = E1.mean((2, 3), keepdim=True).clamp_min(1e-6)
        s2 = E2.mean((2, 3), keepdim=True).clamp_min(1e-6)
        
        E1n = (E1 / s1).clamp(max=5.0)
        E2n = (E2 / s2).clamp(max=5.0)
        D = torch.clamp(E1n - E2n, min=0.0)
        
        # Convert to numpy
        E1n_np = E1n[0, 0].cpu().numpy()
        E2n_np = E2n[0, 0].cpu().numpy()
        D_np = D[0, 0].cpu().numpy()
        
        h_grid, w_grid = D_np.shape # (128, 128)
        
        # Ground truths in this image
        img_idx = batch['batch_idx'] == 0
        cls = batch['cls'][img_idx].cpu().numpy()
        bboxes = batch['bboxes'][img_idx].cpu().numpy() # normalized xyxy
        
        # Evaluate baseline predictions to find dormant ships (False Negatives)
        # Bounding boxes predicted by baseline model
        pred_results = validator.postprocess(preds, batch['img'], batch['img'])
        pred_boxes = pred_results[0].boxes.xyxy.cpu().numpy() if len(pred_results) > 0 else np.zeros((0, 4))
        
        for gt_box in bboxes:
            # Map box to 512x512 coordinates
            x1_512 = gt_box[0] * 512
            y1_512 = gt_box[1] * 512
            x2_512 = gt_box[2] * 512
            y2_512 = gt_box[3] * 512
            
            # Match with baseline predictions
            matched = False
            for pb in pred_boxes:
                # compute IoU
                inter_x1 = max(x1_512, pb[0])
                inter_y1 = max(y1_512, pb[1])
                inter_x2 = min(x2_512, pb[2])
                inter_y2 = min(y2_512, pb[3])
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                union_area = (x2_512 - x1_512)*(y2_512 - y1_512) + (pb[2] - pb[0])*(pb[3] - pb[1]) - inter_area
                iou = inter_area / (union_area + 1e-8)
                if iou >= 0.25:
                    matched = True
                    break
            
            # Map normalized coordinates to stride-4 grid coordinates
            gx1 = int(np.clip(gt_box[0] * w_grid, 0, w_grid - 1))
            gy1 = int(np.clip(gt_box[1] * h_grid, 0, h_grid - 1))
            gx2 = int(np.clip(gt_box[2] * w_grid, 0, w_grid - 1))
            gy2 = int(np.clip(gt_box[3] * h_grid, 0, h_grid - 1))
            
            # If coordinates collapse to single pixel, expand by 1
            if gx1 == gx2:
                gx2 = min(w_grid - 1, gx2 + 1)
            if gy1 == gy2:
                gy2 = min(h_grid - 1, gy2 + 1)
                
            # Create mask for this GT
            gt_mask = np.zeros((h_grid, w_grid), dtype=bool)
            gt_mask[gy1:gy2, gx1:gx2] = True
            
            # Create surrounding ring mask
            ring_mask = get_donut_mask(gt_mask, pad=2)
            
            # Create random background mask
            bg_mask = np.zeros((h_grid, w_grid), dtype=bool)
            # Find a random location that doesn't overlap with GT or ring
            overlap = True
            attempts = 0
            while overlap and attempts < 100:
                rx = np.random.randint(0, w_grid - (gx2 - gx1))
                ry = np.random.randint(0, h_grid - (gy2 - gy1))
                bg_mask_temp = np.zeros((h_grid, w_grid), dtype=bool)
                bg_mask_temp[ry:ry+(gy2-gy1), rx:rx+(gx2-gx1)] = True
                if not np.any(bg_mask_temp & gt_mask):
                    bg_mask = bg_mask_temp
                    overlap = False
                attempts += 1
                
            # Extract statistics
            d_gt = np.mean(D_np[gt_mask])
            d_ring = np.mean(D_np[ring_mask])
            d_bg = np.mean(D_np[bg_mask]) if not overlap else np.mean(D_np)
            
            e1_gt = np.mean(E1n_np[gt_mask])
            e1_ring = np.mean(E1n_np[ring_mask])
            e1_bg = np.mean(E1n_np[bg_mask]) if not overlap else np.mean(E1n_np)
            
            e2_gt = np.mean(E2n_np[gt_mask])
            e2_ring = np.mean(E2n_np[ring_mask])
            e2_bg = np.mean(E2n_np[bg_mask]) if not overlap else np.mean(E2n_np)
            
            d_gt_list.append(d_gt)
            d_ring_list.append(d_ring)
            d_bg_list.append(d_bg)
            
            e1_gt_list.append(e1_gt)
            e1_ring_list.append(e1_ring)
            e1_bg_list.append(e1_bg)
            
            e2_gt_list.append(e2_gt)
            e2_ring_list.append(e2_ring)
            e2_bg_list.append(e2_bg)
            
            if not matched:
                dormant_d_list.append(d_gt)
                
    h1.remove()
    h2.remove()
    
    print("\n=== Frozen Baseline Discrepancy Falsification Results ===")
    print(f"Total evaluated GT boxes: {len(d_gt_list)}")
    print(f"Total dormant (un-detected) ships: {len(dormant_d_list)}")
    print("\nMean Discrepancy D = ReLU(E1n - E2n):")
    print(f"  Inside GT boxes:          {np.mean(d_gt_list):.4f}")
    print(f"  Surrounding Ring (sea):   {np.mean(d_ring_list):.4f}")
    print(f"  Random Background:        {np.mean(d_bg_list):.4f}")
    print(f"  Specifically inside Dormant ships: {np.mean(dormant_d_list):.4f}")
    
    print("\nMean E1n (P1 Evidence normalized):")
    print(f"  Inside GT boxes:          {np.mean(e1_gt_list):.4f}")
    print(f"  Surrounding Ring (sea):   {np.mean(e1_ring_list):.4f}")
    print(f"  Random Background:        {np.mean(e1_bg_list):.4f}")
    
    print("\nMean E2n (P2 Evidence normalized):")
    print(f"  Inside GT boxes:          {np.mean(e2_gt_list):.4f}")
    print(f"  Surrounding Ring (sea):   {np.mean(e2_ring_list):.4f}")
    print(f"  Random Background:        {np.mean(e2_bg_list):.4f}")

    # 3. Analyze Gate activation on trained A2_v2 model if checkpoint exists
    if os.path.exists(a2_path):
        print("\nLoading trained A2_v2 model to evaluate gate activation...")
        model_a2 = YOLO(a2_path).to(device)
        
        gate_gt_list, gate_ring_list, gate_bg_list = [], [], []
        
        gate_vals = {}
        def hook_gate(module, input, output):
            # In block.py, G = torch.sigmoid(self.gate_conv(gate_input))
            # Let's capture G inside forward
            gate_vals['val'] = output.detach()
            
        # Locate the P1GER layer in model_a2
        # In yolov8n_p2_levir_p1ger.yaml: Layer 19 is P1GER
        ger_module = None
        for name, m in model_a2.model.named_modules():
            if m.__class__.__name__ == "P1GER":
                ger_module = m
                break
                
        if ger_module is not None:
            # We hook the sigmoid activation inside ger_module or the end of gate_conv forward
            # Wait, the sigmoid output G is returned or we can hook gate_conv and apply sigmoid ourselves
            # Let's hook the gate_conv output!
            h_gate = ger_module.gate_conv.register_forward_hook(hook_gate)
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 100:
                    break
                batch = validator.preprocess(batch)
                with torch.no_grad():
                    _ = model_a2.model(batch['img'])
                    
                gate_out = torch.sigmoid(gate_vals['val']) # (1, 1, H, W)
                gate_np = gate_out[0, 0].cpu().numpy()
                
                h_grid, w_grid = gate_np.shape
                img_idx = batch['batch_idx'] == 0
                bboxes = batch['bboxes'][img_idx].cpu().numpy()
                
                for gt_box in bboxes:
                    gx1 = int(np.clip(gt_box[0] * w_grid, 0, w_grid - 1))
                    gy1 = int(np.clip(gt_box[1] * h_grid, 0, h_grid - 1))
                    gx2 = int(np.clip(gt_box[2] * w_grid, 0, w_grid - 1))
                    gy2 = int(np.clip(gt_box[3] * h_grid, 0, h_grid - 1))
                    
                    if gx1 == gx2:
                        gx2 = min(w_grid - 1, gx2 + 1)
                    if gy1 == gy2:
                        gy2 = min(h_grid - 1, gy2 + 1)
                        
                    gt_mask = np.zeros((h_grid, w_grid), dtype=bool)
                    gt_mask[gy1:gy2, gx1:gx2] = True
                    ring_mask = get_donut_mask(gt_mask, pad=2)
                    
                    overlap = True
                    attempts = 0
                    bg_mask = np.zeros((h_grid, w_grid), dtype=bool)
                    while overlap and attempts < 100:
                        rx = np.random.randint(0, w_grid - (gx2 - gx1))
                        ry = np.random.randint(0, h_grid - (gy2 - gy1))
                        bg_mask_temp = np.zeros((h_grid, w_grid), dtype=bool)
                        bg_mask_temp[ry:ry+(gy2-gy1), rx:rx+(gx2-gx1)] = True
                        if not np.any(bg_mask_temp & gt_mask):
                            bg_mask = bg_mask_temp
                            overlap = False
                        attempts += 1
                        
                    gate_gt_list.append(np.mean(gate_np[gt_mask]))
                    gate_ring_list.append(np.mean(gate_np[ring_mask]))
                    gate_bg_list.append(np.mean(gate_np[bg_mask]) if not overlap else np.mean(gate_np))
                    
            h_gate.remove()
            
            print("\n=== Trained A2_v2 Gate Activation Stats ===")
            print(f"Mean Gate G inside GT boxes:        {np.mean(gate_gt_list):.4f}")
            print(f"Mean Gate G surrounding Ring (sea): {np.mean(gate_ring_list):.4f}")
            print(f"Mean Gate G random Background:      {np.mean(gate_bg_list):.4f}")
        else:
            print("P1GER module not found in A2 model.")
    else:
        print("\nA2_v2 model checkpoint not found at:", a2_path)

if __name__ == "__main__":
    run_diagnostics()
