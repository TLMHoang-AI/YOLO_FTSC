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

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / (nr + 1e-8)
    return radialprofile

def analyze_crop(crop, target_size=(32, 32)):
    # Crop is assumed float32
    # Resize to common size for fair frequency comparison
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Subtract mean to remove DC offset/mean intensity differences
    resized_zero_mean = resized - np.mean(resized)
    
    # 2D FFT
    f = np.fft.fft2(resized_zero_mean)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Radial profile
    center = (target_size[0] // 2, target_size[1] // 2)
    profile = radial_profile(magnitude_spectrum, center)
    
    # High frequency energy vs total energy
    # We define "high frequency" as radius > 8 (outer part of 32x32 grid)
    y, x = np.indices(magnitude_spectrum.shape)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    total_energy = np.sum(magnitude_spectrum)
    high_energy = np.sum(magnitude_spectrum[r > 8])
    
    ratio = high_energy / (total_energy + 1e-8)
    return magnitude_spectrum, profile, ratio

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
        
    # Layer 18 is the P2 neck/detect fusion output
    hook = model.model[18].register_forward_hook(hook_fn)
    
    # Storage for results
    results = {
        "small": {"raw_ratios": [], "p2_ratios": [], "raw_profiles": [], "p2_profiles": []},
        "large": {"raw_ratios": [], "p2_ratios": [], "raw_profiles": [], "p2_profiles": []}
    }
    
    example_crops = []
    
    print("Analyzing feature maps over test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            imgs = batch["img"] # (1, 3, 512, 512)
            x = imgs.float() / 255.0
            
            # Forward pass
            _ = model(x.to(device))
            
            # P2 shape: (1, 64, 128, 128)
            feat = p2_features.cpu().numpy()[0]
            # Channel mean representation
            feat_mean = np.mean(feat, axis=0)
            
            B, C, H, W = imgs.shape
            bboxes = batch["bboxes"].numpy() # normalized xywh
            
            if len(bboxes) == 0:
                continue
                
            pixel_boxes = bboxes.copy()
            pixel_boxes[:, [0, 2]] *= W
            pixel_boxes[:, [1, 3]] *= H
            
            x_center, y_center, w, h = pixel_boxes[:, 0], pixel_boxes[:, 1], pixel_boxes[:, 2], pixel_boxes[:, 3]
            x1 = np.clip(x_center - w/2, 0, W-1).astype(int)
            y1 = np.clip(y_center - h/2, 0, H-1).astype(int)
            x2 = np.clip(x_center + w/2, 0, W-1).astype(int)
            y2 = np.clip(y_center + h/2, 0, H-1).astype(int)
            
            img_gray = cv2.cvtColor(imgs[0].numpy().transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
            # Normalize gray image to 0-1 float
            img_gray_f = img_gray.astype(np.float32) / 255.0
            
            H_f, W_f = feat_mean.shape
            scale = 4.0 # 512 -> 128
            
            for i in range(len(bboxes)):
                # Get the size of the object
                obj_w = w[i]
                obj_h = h[i]
                max_side = max(obj_w, obj_h)
                
                # We crop raw image
                crop_raw = img_gray_f[y1[i]:y2[i], x1[i]:x2[i]]
                
                # Downscaled coordinates for P2
                fx1 = int(np.clip(x1[i] / scale, 0, W_f - 1))
                fy1 = int(np.clip(y1[i] / scale, 0, H_f - 1))
                fx2 = int(np.clip(x2[i] / scale, 0, W_f - 1))
                fy2 = int(np.clip(y2[i] / scale, 0, H_f - 1))
                
                # Skip if crop too small on P2 map to avoid degenerate FFT
                if (fx2 - fx1) < 2 or (fy2 - fy1) < 2:
                    continue
                    
                crop_p2 = feat_mean[fy1:fy2, fx1:fx2]
                
                # Perform analysis
                raw_mag, raw_prof, raw_ratio = analyze_crop(crop_raw)
                p2_mag, p2_prof, p2_ratio = analyze_crop(crop_p2)
                
                group = "small" if max_side < 20 else "large"
                results[group]["raw_ratios"].append(raw_ratio)
                results[group]["p2_ratios"].append(p2_ratio)
                results[group]["raw_profiles"].append(raw_prof)
                results[group]["p2_profiles"].append(p2_prof)
                
                # Keep examples for plotting
                if len(example_crops) < 3 and group == "small":
                    example_crops.append({
                        "raw": crop_raw,
                        "p2": crop_p2,
                        "raw_mag": raw_mag,
                        "p2_mag": p2_mag,
                        "size": (obj_w, obj_h)
                    })
                    
    hook.remove()
    
    print("\n--- STATISTICS ---")
    for group in ["small", "large"]:
        raw_r = np.array(results[group]["raw_ratios"])
        p2_r = np.array(results[group]["p2_ratios"])
        dilution = (raw_r - p2_r) / (raw_r + 1e-8) * 100
        
        print(f"Group: {group.upper()} (Count: {len(raw_r)})")
        print(f"  Raw High-Freq Ratio: {np.mean(raw_r):.4f} ± {np.std(raw_r):.4f}")
        print(f"  P2 High-Freq Ratio:  {np.mean(p2_r):.4f} ± {np.std(p2_r):.4f}")
        print(f"  Mean Dilution:       {np.mean(dilution):.2f}% ± {np.std(dilution):.2f}%")
        print("-" * 30)
        
    # PLOTTING
    # Plot 1: Radial profiles
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, group in enumerate(["small", "large"]):
        raw_profs = np.array(results[group]["raw_profiles"])
        p2_profs = np.array(results[group]["p2_profiles"])
        
        # Normalize profiles for comparison
        raw_mean = np.mean(raw_profs / (np.sum(raw_profs, axis=1, keepdims=True) + 1e-8), axis=0)
        p2_mean = np.mean(p2_profs / (np.sum(p2_profs, axis=1, keepdims=True) + 1e-8), axis=0)
        
        freqs = np.arange(len(raw_mean))
        
        axes[idx].plot(freqs, raw_mean, label="Raw Image Profile", color="royalblue", lw=2)
        axes[idx].plot(freqs, p2_mean, label="P2 Feature Profile", color="orange", lw=2)
        axes[idx].set_title(f"Average Radial Profile - {group.capitalize()} Objects (<20px)" if group == "small" else f"Average Radial Profile - {group.capitalize()} Objects (>=20px)")
        axes[idx].set_xlabel("Frequency (Radius)")
        axes[idx].set_ylabel("Normalized Energy")
        axes[idx].legend()
        axes[idx].grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig("docs/reports/small_object_frequency_radial.png", dpi=300)
    plt.close()
    
    # Plot 2: Example visualization
    if len(example_crops) > 0:
        fig, axes = plt.subplots(len(example_crops), 4, figsize=(16, 4 * len(example_crops)))
        if len(example_crops) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, eg in enumerate(example_crops):
            axes[i, 0].imshow(eg["raw"], cmap="gray")
            axes[i, 0].set_title(f"Raw Crop ({eg['size'][0]:.1f}x{eg['size'][1]:.1f})")
            axes[i, 0].axis("off")
            
            # log transform spectrum for visualization
            raw_mag_log = 20 * np.log(eg["raw_mag"] + 1e-8)
            axes[i, 1].imshow(raw_mag_log, cmap="viridis")
            axes[i, 1].set_title("Raw FFT Magnitude")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(eg["p2"], cmap="plasma")
            axes[i, 2].set_title("P2 Feature Crop")
            axes[i, 2].axis("off")
            
            p2_mag_log = 20 * np.log(eg["p2_mag"] + 1e-8)
            axes[i, 3].imshow(p2_mag_log, cmap="viridis")
            axes[i, 3].set_title("P2 FFT Magnitude")
            axes[i, 3].axis("off")
            
        plt.tight_layout()
        plt.savefig("docs/reports/small_object_fft_examples.png", dpi=300)
        plt.close()
        
    print("Plots saved to docs/reports/small_object_frequency_radial.png and docs/reports/small_object_fft_examples.png")

if __name__ == "__main__":
    main()
