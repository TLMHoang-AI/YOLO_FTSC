#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path

ROOT = Path("/marimo/yolo_code")
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from ultralytics import YOLO

models_to_eval = [
    ("YOLO DBSS/HIT", "yolov10n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/dbss/seed_42/weights/best.pt"),
    ("YOLO DBSS/HIT", "yolov10n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov10n/hit/seed_42/weights/best.pt"),
    ("YOLO DBSS/HIT", "yolov5n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/dbss/seed_42/weights/best.pt"),
    ("YOLO DBSS/HIT", "yolov5n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov5n/hit/seed_42/weights/best.pt"),
    ("YOLO DBSS/HIT", "yolov8n + dbss", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/dbss/seed_42/weights/best.pt"),
    ("YOLO DBSS/HIT", "yolov8n + hit", "duyle2408/levir-yolo-dbss-hit-3seed", "runs/yolov8n/hit/seed_42/weights/best.pt"),
    ("YOLOv8n DBSS P2", "dbss_p2_aware", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_aware/seed_42/weights/best.pt"),
    ("YOLOv8n DBSS P2", "dbss_p2_routed", "duyle2408/levir_dbss_p2_aware", "runs/yolov8n/dbss_p2_routed/seed_42/weights/best.pt"),
    ("YOLO P2", "yolov8n baseline", "duyle2408/levir-ship-yolo-p2", "train/yolov8n_p2_baseline_seed42/weights/best.pt"),
    ("YOLO P2", "yolov8n offset", "duyle2408/levir-ship-yolo-p2", "train/yolov8n_p2_offset_seed42/weights/best.pt"),
    ("YOLOv8n P2 routing", "dbss_pre_p2", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/dbss_pre_p2/seed_42/weights/best.pt"),
    ("YOLOv8n P2 routing", "gcts_backbone_p2_p3", "duyle2408/levir-yolov8n-p2-routing-3seed", "runs/gcts_backbone_p2_p3/seed_42/weights/best.pt"),
    ("YOLO baseline", "yolo11n", "duyle2408/levir-ship-yolo-baselines", "train/yolo11n_seed42/weights/best.pt"),
    ("YOLO baseline", "yolov10n", "duyle2408/levir-ship-yolo-baselines", "train/yolov10n_seed42/weights/best.pt"),
    ("YOLO baseline", "yolov5nu", "duyle2408/levir-ship-yolo-baselines", "train/yolov5nu_seed42/weights/best.pt"),
    ("YOLO baseline", "yolov8n", "duyle2408/levir-ship-yolo-baselines", "train/yolov8n_seed42/weights/best.pt"),
    ("YOLO baseline", "yolov9t", "duyle2408/levir-ship-yolo-baselines", "train/yolov9t_seed42/weights/best.pt"),
    ("YOLOv10n GCTS v1", "bilinear_w01", "duyle2408/levir-yolov10n-gcts-ablation", "train/bilinear_w01/weights/best.pt"),
    ("YOLOv10n GCTS v1", "bilinear_w02", "duyle2408/levir-yolov10n-gcts-ablation", "train/bilinear_w02/weights/best.pt"),
    ("YOLOv10n GCTS v1", "onehot_w01", "duyle2408/levir-yolov10n-gcts-ablation", "train/onehot_w01/weights/best.pt"),
    ("YOLOv10n GCTS v1", "onehot_w02", "duyle2408/levir-yolov10n-gcts-ablation", "train/onehot_w02/weights/best.pt"),
    ("YOLOv10n GCTS v2", "v2_e05", "duyle2408/levir-yolov10n-gcts-v2-ablation", "train/v2_e05/weights/best.pt"),
    ("YOLOv10n GCTS v2", "v2_e05_nogate", "duyle2408/levir-yolov10n-gcts-v2-ablation", "train/v2_e05_nogate/weights/best.pt"),
    ("YOLOv10n GCTS v2", "v2_e10", "duyle2408/levir-yolov10n-gcts-v2-ablation", "train/v2_e10/weights/best.pt"),
    ("YOLOv10n P3 NUDFL", "baseline_p3_nudfl", "duyle2408/levir-yolov10n-p3-nudfl-ablation", "train/baseline_p3_nudfl/weights/best.pt"),
    ("YOLOv10n P3 NUDFL", "gcts_v2_e05_p3_nudfl", "duyle2408/levir-yolov10n-p3-nudfl-ablation", "train/gcts_v2_e05_p3_nudfl/weights/best.pt"),
    ("YOLOv8n P3 NUDFL", "yolov8n_baseline", "duyle2408/levir-yolov8n-p3-nudfl-ablation", "train/yolov8n_baseline/weights/best.pt"),
    ("YOLOv8n P3 NUDFL", "yolov8n_p3_nudfl", "duyle2408/levir-yolov8n-p3-nudfl-ablation", "train/yolov8n_p3_nudfl/weights/best.pt"),
]

cache_dir = ROOT / "runs/checkpoint_cache"

model_stats = {}
for family, config, repo, file_path in models_to_eval:
    local_model_path = cache_dir / f"{repo.replace('/', '_')}_{file_path.replace('/', '_')}"
    if local_model_path.exists():
        try:
            model = YOLO(local_model_path)
            model.info() # populates flops in model.model
            params = sum(p.numel() for p in model.model.parameters())
            flops = getattr(model.model, "flops", 0.0)
            model_stats[f"{family} | {config}"] = {
                "params": f"{params / 1e6:.2f}M",
                "flops": f"{flops:.2f}"
            }
        except Exception as e:
            model_stats[f"{family} | {config}"] = {"error": str(e)}
    else:
        model_stats[f"{family} | {config}"] = {"error": "not_cached"}

with open(ROOT / "runs/model_stats.json", "w") as f:
    json.dump(model_stats, f, indent=2)
print("STATS_DONE")
