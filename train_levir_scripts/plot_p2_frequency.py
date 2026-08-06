import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add ultralytics path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models_related/ultralytics"))
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator

def main():
    # Load model and dataset
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
    
    # intermediate output storage
    p2_features = None
    
    # Register forward hook on Layer 18 to get P2 feature maps
    def hook_fn(module, input, output):
        nonlocal p2_features
        p2_features = output
        
    hook = model.model[18].register_forward_hook(hook_fn)
    
    gt_p2_laps = []
    bg_p2_laps = []
    gt_p2_grads = []
    bg_p2_grads = []
    
    example_img = None
    example_gt_box = None
    example_p2_activation = None
    
    print("Analyzing P2 feature maps over test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch["img"] # (1, 3, 512, 512)
            x = imgs.float() / 255.0
            # Forward pass to trigger hook
            _ = model(x.to(device))
            
            # P2 features shape: (1, 64, 128, 128)
            feat = p2_features.cpu().numpy()[0] # (64, 128, 128)
            # Channel mean
            feat_mean = np.mean(feat, axis=0) # (128, 128)
            
            B, C, H, W = imgs.shape
            bboxes = batch["bboxes"].numpy() # normalized xywh
            
            if len(bboxes) == 0:
                continue
                
            # Convert xywh to pixel coordinates for 512x512
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= W
            pixel_boxes[:, [1, 3]] *= H
            
            x_center, y_center, w, h = pixel_boxes[:, 0], pixel_boxes[:, 1], pixel_boxes[:, 2], pixel_boxes[:, 3]
            x1 = np.clip(x_center - w/2, 0, W-1).astype(int)
            y1 = np.clip(y_center - h/2, 0, H-1).astype(int)
            x2 = np.clip(x_center + w/2, 0, W-1).astype(int)
            y2 = np.clip(y_center + h/2, 0, H-1).astype(int)
            
            img_gray = cv2.cvtColor(imgs[0].numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            
            # P2 feature map size is 1/4 of input image
            H_f, W_f = feat_mean.shape
            scale = 4.0
            
            for i in range(len(bboxes)):
                # Downscaled coordinates for P2 feature map
                fx1 = int(np.clip(x1[i] / scale, 0, W_f - 1))
                fy1 = int(np.clip(y1[i] / scale, 0, H_f - 1))
                fx2 = int(np.clip(x2[i] / scale, 0, W_f - 1))
                fy2 = int(np.clip(y2[i] / scale, 0, H_f - 1))
                
                fw = fx2 - fx1
                fh = fy2 - fy1
                
                if fw < 2 or fh < 2:
                    continue
                    
                crop_p2_gt = np.ascontiguousarray(feat_mean[fy1:fy2, fx1:fx2], dtype=np.float64)
                
                # Adjacent background in P2 feature map
                bg_cropped = False
                directions = [(fw, 0), (-fw, 0), (0, fh), (0, -fh)]
                for dx, dy in directions:
                    fbg_x1 = fx1 + dx
                    fbg_y1 = fy1 + dy
                    fbg_x2 = fbg_x1 + fw
                    fbg_y2 = fbg_y1 + fh
                    
                    if fbg_x1 >= 0 and fbg_x2 <= W_f and fbg_y1 >= 0 and fbg_y2 <= H_f:
                        # Check overlap with any GT box in P2 space
                        overlap = False
                        for j in range(len(bboxes)):
                            g_x1 = int(x1[j] / scale)
                            g_y1 = int(y1[j] / scale)
                            g_x2 = int(x2[j] / scale)
                            g_y2 = int(y2[j] / scale)
                            if not (fbg_x2 <= g_x1 or fbg_x1 >= g_x2 or fbg_y2 <= g_y1 or fbg_y1 >= g_y2):
                                overlap = True
                                break
                        if not overlap:
                            crop_p2_bg = np.ascontiguousarray(feat_mean[fbg_y1:fbg_y2, fbg_x1:fbg_x2], dtype=np.float64)
                            bg_cropped = True
                            break
                            
                if not bg_cropped:
                    continue
                    
                # Calculate Laplacian variance of P2 feature maps
                lap_gt = cv2.Laplacian(crop_p2_gt, cv2.CV_64F).var()
                lap_bg = cv2.Laplacian(crop_p2_bg, cv2.CV_64F).var()
                
                # Calculate Gradient magnitude of P2 feature maps
                sobelx_gt = cv2.Sobel(crop_p2_gt, cv2.CV_64F, 1, 0, ksize=1)
                sobely_gt = cv2.Sobel(crop_p2_gt, cv2.CV_64F, 0, 1, ksize=1)
                grad_gt = np.mean(np.sqrt(sobelx_gt**2 + sobely_gt**2))
                
                sobelx_bg = cv2.Sobel(crop_p2_bg, cv2.CV_64F, 1, 0, ksize=1)
                sobely_bg = cv2.Sobel(crop_p2_bg, cv2.CV_64F, 0, 1, ksize=1)
                grad_bg = np.mean(np.sqrt(sobelx_bg**2 + sobely_bg**2))
                
                gt_p2_laps.append(lap_gt)
                bg_p2_laps.append(lap_bg)
                gt_p2_grads.append(grad_gt)
                bg_p2_grads.append(grad_bg)
                
                # Save an example for visualization
                if example_img is None and w[i] > 20 and h[i] > 20:
                    example_img = img_gray[y1[i]:y2[i], x1[i]:x2[i]]
                    example_p2_activation = crop_p2_gt
                    
    # Remove hook
    hook.remove()
    
    print(f"Total P2 pairs analyzed: {len(gt_p2_laps)}")
    print(f"P2 Laplacian Var - GT: {np.mean(gt_p2_laps):.4f} vs BG: {np.mean(bg_p2_laps):.4f}")
    print(f"P2 Gradient Mag  - GT: {np.mean(gt_p2_grads):.4f} vs BG: {np.mean(bg_p2_grads):.4f}")
    
    # PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Example Ship Original Image vs P2 Feature Map
    if example_img is not None:
        axes[0, 0].imshow(example_img, cmap="gray")
        axes[0, 0].set_title("Original Ship Crop (Input Space)")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(example_p2_activation, cmap="viridis")
        axes[0, 1].set_title("Ship Activation in P2 Feature Map")
        axes[0, 1].axis("off")
        
        # 2D FFT Magnitude Spectrum
        f_orig = np.fft.fft2(example_img)
        fshift_orig = np.fft.fftshift(f_orig)
        magnitude_spectrum_orig = 20 * np.log(np.abs(fshift_orig) + 1e-8)
        
        f_p2 = np.fft.fft2(example_p2_activation)
        fshift_p2 = np.fft.fftshift(f_p2)
        magnitude_spectrum_p2 = 20 * np.log(np.abs(fshift_p2) + 1e-8)
        
        # Plot FFT
        fig_fft, axes_fft = plt.subplots(1, 2, figsize=(10, 5))
        axes_fft[0].imshow(magnitude_spectrum_orig, cmap="gray")
        axes_fft[0].set_title("2D FFT Magnitude Spectrum (Original)")
        axes_fft[0].axis("off")
        
        axes_fft[1].imshow(magnitude_spectrum_p2, cmap="gray")
        axes_fft[1].set_title("2D FFT Magnitude Spectrum (P2 Feature)")
        axes_fft[1].axis("off")
        
        fig_fft.tight_layout()
        fig_fft.savefig("docs/reports/p2_example_fft.png", dpi=300)
        plt.close(fig_fft)
        
    # 2. P2 Laplacian Variance Histogram
    axes[1, 0].hist(gt_p2_laps, bins=50, alpha=0.6, label="GT in P2", color="royalblue", density=True, range=(0, 0.5))
    axes[1, 0].hist(bg_p2_laps, bins=50, alpha=0.6, label="Adjacent BG in P2", color="orange", density=True, range=(0, 0.5))
    axes[1, 0].set_title("P2 Feature Map Laplacian Variance Distribution")
    axes[1, 0].set_xlabel("Variance")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()
    
    # 3. P2 Gradient Magnitude Histogram
    axes[1, 1].hist(gt_p2_grads, bins=50, alpha=0.6, label="GT in P2", color="royalblue", density=True, range=(0, 0.8))
    axes[1, 1].hist(bg_p2_grads, bins=50, alpha=0.6, label="Adjacent BG in P2", color="orange", density=True, range=(0, 0.8))
    axes[1, 1].set_title("P2 Feature Map Gradient Magnitude Distribution")
    axes[1, 1].set_xlabel("Mean Magnitude")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()
    
    fig.tight_layout()
    out_path = Path("docs/reports/p2_feature_frequency.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"P2 plots saved successfully to: {out_path}")
    print("Example FFT saved to docs/reports/p2_example_fft.png")

if __name__ == "__main__":
    main()
