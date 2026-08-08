#!/usr/bin/env python3
import sys
import os
import shutil
import json
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO

# Import split utility from workflow
sys.path.insert(0, str(ROOT / "train_levir_scripts"))
import train_all_levir_yolov8n_p2_routing as workflow

def ensure_dataset_split(seed):
    data_yaml = Path(f"/marimo/datasets/levir_ship_yolo_seed{seed}/levir_ship.yaml")
    if not data_yaml.exists():
        print(f"Generating split for seed {seed}...")
        workflow.create_fixed_split(Path("/marimo/yolo_code/LevirShipData"), data_yaml.parent, seed)
    return data_yaml

# Map configuration records
# (Family, Model/Config, HF Repo, File Path in Repo, Seed, Report Target)
models_to_eval = [
    # 1. yolov10n/dbss/hit
    ("YOLO DBSS/HIT", "yolov10n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/dbss/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov10n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/dbss/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov10n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/dbss/seed_44/weights/best.pt", 44, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov10n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/hit/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov10n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/hit/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov10n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/hit/seed_44/weights/best.pt", 44, "yolo_report"),
    
    # 2. yolov5n/dbss/hit
    ("YOLO DBSS/HIT", "yolov5n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/dbss/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov5n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/dbss/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov5n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/dbss/seed_44/weights/best.pt", 44, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov5n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/hit/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov5n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/hit/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov5n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/hit/seed_44/weights/best.pt", 44, "yolo_report"),

    # 3. yolov8n/dbss/hit
    ("YOLO DBSS/HIT", "yolov8n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/dbss/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov8n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/dbss/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov8n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/dbss/seed_44/weights/best.pt", 44, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov8n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/hit/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov8n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/hit/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLO DBSS/HIT", "yolov8n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/hit/seed_44/weights/best.pt", 44, "yolo_report"),

    # 4. dbss_p2_aware & routed
    ("YOLOv8n DBSS P2", "dbss_p2_aware", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_aware/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLOv8n DBSS P2", "dbss_p2_aware", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_aware/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLOv8n DBSS P2", "dbss_p2_aware", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_aware/seed_44/weights/best.pt", 44, "yolo_report"),
    ("YOLOv8n DBSS P2", "dbss_p2_routed", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_routed/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLOv8n DBSS P2", "dbss_p2_routed", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_routed/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLOv8n DBSS P2", "dbss_p2_routed", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_routed/seed_44/weights/best.pt", 44, "yolo_report"),

    # 5. yolov8n baseline (P2)
    ("YOLO P2", "yolov8n baseline", "duyle2408/levir-ship-yolo-p2", "train/yolov8n_p2_baseline_seed42/weights/best.pt", 42, "yolo_report"),
    ("YOLO P2", "yolov8n baseline", "duyle2408/levir-ship-yolo-p2", "train/yolov8n_p2_baseline_seed43/weights/best.pt", 43, "yolo_report"),
    ("YOLO P2", "yolov8n baseline", "duyle2408/levir-ship-yolo-p2", "train/yolov8n_p2_baseline_seed44/weights/best.pt", 44, "yolo_report"),
    ("YOLO P2", "yolov8n offset", "duyle2408/levir-ship-yolo-p2", "train/yolov8n_p2_offset_seed42/weights/best.pt", 42, "yolo_report"),

    # 6. dbss_pre_p2 & gcts_backbone
    ("YOLOv8n P2 routing", "dbss_pre_p2", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/dbss_pre_p2/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLOv8n P2 routing", "dbss_pre_p2", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/dbss_pre_p2/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLOv8n P2 routing", "dbss_pre_p2", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/dbss_pre_p2/seed_44/weights/best.pt", 44, "yolo_report"),
    ("YOLOv8n P2 routing", "gcts_backbone_p2_p3", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/gcts_backbone_p2_p3/seed_42/weights/best.pt", 42, "yolo_report"),
    ("YOLOv8n P2 routing", "gcts_backbone_p2_p3", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/gcts_backbone_p2_p3/seed_43/weights/best.pt", 43, "yolo_report"),
    ("YOLOv8n P2 routing", "gcts_backbone_p2_p3", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/gcts_backbone_p2_p3/seed_44/weights/best.pt", 44, "yolo_report"),

    # 7. baselines
    ("YOLO baseline", "yolo11n", "duyle2408/levir-ship-yolo-baselines", "train/yolo11n_seed42/weights/best.pt", 42, "yolo_report"),
    ("YOLO baseline", "yolo11n", "duyle2408/levir-ship-yolo-baselines", "train/yolo11n_seed43/weights/best.pt", 43, "yolo_report"),
    ("YOLO baseline", "yolo11n", "duyle2408/levir-ship-yolo-baselines", "train/yolo11n_seed44/weights/best.pt", 44, "yolo_report"),
    ("YOLO baseline", "yolov10n", "duyle2408/levir-ship-yolo-baselines", "train/yolov10n_seed42/weights/best.pt", 42, "yolo_report"),
    ("YOLO baseline", "yolov10n", "duyle2408/levir-ship-yolo-baselines", "train/yolov10n_seed43/weights/best.pt", 43, "yolo_report"),
    ("YOLO baseline", "yolov10n", "duyle2408/levir-ship-yolo-baselines", "train/yolov10n_seed44/weights/best.pt", 44, "yolo_report"),
    ("YOLO baseline", "yolov5nu", "duyle2408/levir-ship-yolo-baselines", "train/yolov5nu_seed42/weights/best.pt", 42, "yolo_report"),
    ("YOLO baseline", "yolov5nu", "duyle2408/levir-ship-yolo-baselines", "train/yolov5nu_seed43/weights/best.pt", 43, "yolo_report"),
    ("YOLO baseline", "yolov5nu", "duyle2408/levir-ship-yolo-baselines", "train/yolov5nu_seed44/weights/best.pt", 44, "yolo_report"),
    ("YOLO baseline", "yolov8n", "duyle2408/levir-ship-yolo-baselines", "train/yolov8n_seed42/weights/best.pt", 42, "yolo_report"),
    ("YOLO baseline", "yolov8n", "duyle2408/levir-ship-yolo-baselines", "train/yolov8n_seed43/weights/best.pt", 43, "yolo_report"),
    ("YOLO baseline", "yolov8n", "duyle2408/levir-ship-yolo-baselines", "train/yolov8n_seed44/weights/best.pt", 44, "yolo_report"),
    ("YOLO baseline", "yolov9t", "duyle2408/levir-ship-yolo-baselines", "train/yolov9t_seed42/weights/best.pt", 42, "yolo_report"),
    ("YOLO baseline", "yolov9t", "duyle2408/levir-ship-yolo-baselines", "train/yolov9t_seed43/weights/best.pt", 43, "yolo_report"),
    ("YOLO baseline", "yolov9t", "duyle2408/levir-ship-yolo-baselines", "train/yolov9t_seed44/weights/best.pt", 44, "yolo_report"),

    # 8. GCTS v1
    ("YOLOv10n GCTS v1", "bilinear_w01", "duyle2408/levir-yolov10n-gcts-ablation", "train/bilinear_w01/weights/best.pt", 42, "yolo_report"),
    ("YOLOv10n GCTS v1", "bilinear_w01", "duyle2408/levir-yolov10n-gcts-v1-seed43", "train/bilinear_w01/weights/best.pt", 43, "yolo_report"),
    ("YOLOv10n GCTS v1", "bilinear_w01", "duyle2408/levir-yolov10n-gcts-v1-seed44", "train/bilinear_w01/weights/best.pt", 44, "yolo_report"),
    ("YOLOv10n GCTS v1", "bilinear_w02", "duyle2408/levir-yolov10n-gcts-ablation", "train/bilinear_w02/weights/best.pt", 42, "yolo_report"),
    ("YOLOv10n GCTS v1", "bilinear_w02", "duyle2408/levir-yolov10n-gcts-v1-seed43", "train/bilinear_w02/weights/best.pt", 43, "yolo_report"),
    ("YOLOv10n GCTS v1", "onehot_w01", "duyle2408/levir-yolov10n-gcts-ablation", "train/onehot_w01/weights/best.pt", 42, "yolo_report"),
    ("YOLOv10n GCTS v1", "onehot_w01", "duyle2408/levir-yolov10n-gcts-v1-seed43", "train/onehot_w01/weights/best.pt", 43, "yolo_report"),
    ("YOLOv10n GCTS v1", "onehot_w02", "duyle2408/levir-yolov10n-gcts-ablation", "train/onehot_w02/weights/best.pt", 42, "yolo_report"),
    ("YOLOv10n GCTS v1", "onehot_w02", "duyle2408/levir-yolov10n-gcts-v1-seed43", "train/onehot_w02/weights/best.pt", 43, "yolo_report"),

    # 9. GCTS v2
    ("YOLOv10n GCTS v2", "v2_e05", "duyle2408/levir-yolov10n-gcts-v2-ablation", "train/v2_e05/weights/best.pt", 42, "yolo_report"),
    ("YOLOv10n GCTS v2", "v2_e05_nogate", "duyle2408/levir-yolov10n-gcts-v2-ablation", "train/v2_e05_nogate/weights/best.pt", 42, "yolo_report"),
    ("YOLOv10n GCTS v2", "v2_e10", "duyle2408/levir-yolov10n-gcts-v2-ablation", "train/v2_e10/weights/best.pt", 42, "yolo_report"),

    # 10. NUDFL
    ("YOLOv10n P3 NUDFL", "baseline_p3_nudfl", "duyle2408/levir-yolov10n-p3-nudfl-ablation", "train/baseline_p3_nudfl/weights/best.pt", 42, "yolo_report"),
    ("YOLOv10n P3 NUDFL", "gcts_v2_e05_p3_nudfl", "duyle2408/levir-yolov10n-p3-nudfl-ablation", "train/gcts_v2_e05_p3_nudfl/weights/best.pt", 42, "yolo_report"),
    ("YOLOv8n P3 NUDFL", "yolov8n_baseline", "duyle2408/levir-yolov8n-p3-nudfl-ablation", "train/yolov8n_baseline/weights/best.pt", 42, "yolo_report"),
    ("YOLOv8n P3 NUDFL", "yolov8n_p3_nudfl", "duyle2408/levir-yolov8n-p3-nudfl-ablation", "train/yolov8n_p3_nudfl/weights/best.pt", 42, "yolo_report"),

    # 11. Top-down / FPN-Only family (from investigate_pooling)
    ("FPN-Only Family", "topdown_baseline", "duyle2408/levir-yolov8n-p2-topdown-3seed", "runs/topdown_baseline/seed_42/weights/best.pt", 42, "pooling_report"),
    ("FPN-Only Family", "topdown_p1drr_partial_clip", "duyle2408/levir-yolov8n-p2-topdown-3seed", "runs/topdown_p1drr_partial_clip/seed_42/weights/best.pt", 42, "pooling_report"),
    ("FPN-Only Family", "topdown_p1reg_only", "duyle2408/levir-yolov8n-p2-topdown-3seed", "runs/topdown_p1reg_only/seed_42/weights/best.pt", 42, "pooling_report"),
    ("FPN-Only Family", "p1drr_old_partial_clip", "duyle2408/levir-yolov8n-p2-topdown-3seed", "runs/p1drr_old_partial_clip/seed_42/weights/best.pt", 42, "pooling_report"),
    ("FPN-Only Family", "topdown_p1ger_200", "duyle2408/levir-yolov8n-p2-topdown-3seed", "runs/topdown_p1ger_200/seed_42/weights/best.pt", 42, "pooling_report"),
    ("FPN-Only Family", "topdown_p1ger_500", "duyle2408/levir-yolov8n-p2-topdown-3seed", "runs/topdown_p1ger_500/seed_42/weights/best.pt", 42, "pooling_report"),
    ("FPN-Only Family", "topdown_p1fusion_200", "duyle2408/levir-yolov8n-p2-topdown-3seed", "runs/topdown_p1fusion_200/seed_42/weights/best.pt", 42, "pooling_report"),
]

cache_dir = ROOT / "runs/checkpoint_cache"
os.makedirs(cache_dir, exist_ok=True)

results_data = []

for idx, (family, config, repo, file_path, seed, target) in enumerate(models_to_eval, 1):
    print(f"\n[{idx}/{len(models_to_eval)}] Processing {family} | {config} | Seed {seed}...")
    
    # 1. Ensure split dataset exists
    try:
        data_yaml = ensure_dataset_split(seed)
    except Exception as e:
        print(f"Error creating dataset split for seed {seed}: {e}")
        continue
        
    # 2. Download model
    local_model_path = cache_dir / f"{repo.replace('/', '_')}_{file_path.replace('/', '_')}"
    if not local_model_path.exists():
        print(f"Downloading from HF repo: {repo}...")
        try:
            downloaded = hf_hub_download(repo_id=repo, filename=file_path, repo_type="dataset")
            shutil.copy(downloaded, local_model_path)
        except Exception as e:
            print(f"Download failed: {e}")
            continue
            
    # 3. Load and evaluate
    try:
        model = YOLO(local_model_path)
        print("Running validation split...")
        val_res = model.val(
            data=str(data_yaml),
            split="val",
            iou=0.50,
            imgsz=512,
            batch=8,
            device="cuda",
            verbose=False
        )
        print("Running test split...")
        test_res = model.val(
            data=str(data_yaml),
            split="test",
            iou=0.50,
            imgsz=512,
            batch=8,
            device="cuda",
            verbose=False
        )
        
        # Save metrics
        metrics = {
            "family": family,
            "config": config,
            "seed": seed,
            "target": target,
            "val/precision(B)": val_res.results_dict.get("metrics/precision(B)", 0),
            "val/recall(B)": val_res.results_dict.get("metrics/recall(B)", 0),
            "val/metrics/mAP50(B)": val_res.results_dict.get("metrics/mAP50(B)", 0),
            "val/metrics/mAP75(B)": float(val_res.box.map75),
            "val/metrics/mAP50-95(B)": val_res.results_dict.get("metrics/mAP50-95(B)", 0),
            "test/precision(B)": test_res.results_dict.get("metrics/precision(B)", 0),
            "test/recall(B)": test_res.results_dict.get("metrics/recall(B)", 0),
            "test/metrics/mAP50(B)": test_res.results_dict.get("metrics/mAP50(B)", 0),
            "test/metrics/mAP75(B)": float(test_res.box.map75),
            "test/metrics/mAP50-95(B)": test_res.results_dict.get("metrics/mAP50-95(B)", 0),
        }
        results_data.append(metrics)
        print(f"Success! Val mAP50: {metrics['val/metrics/mAP50(B)']:.4f} | Test mAP50: {metrics['test/metrics/mAP50(B)']:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

# Save all results to a JSON file
with open(ROOT / "runs/eval_nms_05_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

print("\n=== ALL EVALUATIONS COMPLETED ===")
