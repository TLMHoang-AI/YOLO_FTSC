#!/usr/bin/env python3
import json
import numpy as np
import os

def format_val(vals):
    if len(vals) == 0:
        return "—"
    elif len(vals) == 1:
        return f"{vals[0]:.4f}"
    else:
        return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"

def main():
    # Load JSON results
    with open("runs/eval_nms_05_results.json", "r") as f:
        data = json.load(f)

    # 1. Process yolo_report (from report_yolo.md)
    yolo_records = [r for r in data if r["target"] == "yolo_report"]
    # Group by (family, config)
    groups = {}
    for r in yolo_records:
        key = (r["family"], r["config"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Build the report_yolo.md NMS 0.50 table
    # Columns: | Family | Model/config | Resolution | Seeds (n) | Test P | Test R | Test mAP50 | Test mAP75 | Test mAP50-95 | Source repo |
    # Order should match the order in report_yolo.md to make it easy to compare
    yolo_order = [
        ("YOLO DBSS/HIT", "yolov10n + dbss"),
        ("YOLO DBSS/HIT", "yolov10n + hit"),
        ("YOLO DBSS/HIT", "yolov5n + dbss"),
        ("YOLO DBSS/HIT", "yolov5n + hit"),
        ("YOLO DBSS/HIT", "yolov8n + dbss"),
        ("YOLO DBSS/HIT", "yolov8n + hit"),
        ("YOLOv8n DBSS P2", "dbss_p2_aware"),
        ("YOLOv8n DBSS P2", "dbss_p2_routed"),
        ("YOLO P2", "yolov8n baseline"),
        ("YOLO P2", "yolov8n offset"),
        ("YOLOv8n P2 routing", "dbss_pre_p2"),
        ("YOLOv8n P2 routing", "gcts_backbone_p2_p3"),
        ("YOLO baseline", "yolo11n"),
        ("YOLO baseline", "yolov10n"),
        ("YOLO baseline", "yolov5nu"),
        ("YOLO baseline", "yolov8n"),
        ("YOLO baseline", "yolov9t"),
        ("YOLOv10n GCTS v1", "bilinear_w01"),
        ("YOLOv10n GCTS v1", "bilinear_w02"),
        ("YOLOv10n GCTS v1", "onehot_w01"),
        ("YOLOv10n GCTS v1", "onehot_w02"),
        ("YOLOv10n GCTS v2", "v2_e05"),
        ("YOLOv10n GCTS v2", "v2_e05_nogate"),
        ("YOLOv10n GCTS v2", "v2_e10"),
        ("YOLOv10n P3 NUDFL", "baseline_p3_nudfl"),
        ("YOLOv10n P3 NUDFL", "gcts_v2_e05_p3_nudfl"),
        ("YOLOv8n P3 NUDFL", "yolov8n_baseline"),
        ("YOLOv8n P3 NUDFL", "yolov8n_p3_nudfl"),
    ]

    yolo_table_lines = [
        "| Family | Model/config | Resolution | Seeds (n) | Val P | Val R | Val mAP50 | Val mAP75 | Val mAP50-95 | Test P | Test R | Test mAP50 | Test mAP75 | Test mAP50-95 | Source repo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    # Map target source repo names
    repo_mapping = {
        "duyle2408/levir-yolo-dbss-hit-3seed": "[levir-yolo-dbss-hit-3seed](https://huggingface.co/datasets/duyle2408/levir-yolo-dbss-hit-3seed)",
        "duyle2408/levir_dbss_p2_aware": "[levir_dbss_p2_aware](https://huggingface.co/datasets/duyle2408/levir_dbss_p2_aware)",
        "duyle2408/levir-ship-yolo-p2": "[levir-ship-yolo-p2](https://huggingface.co/datasets/duyle2408/levir-ship-yolo-p2)",
        "duyle2408/levir-yolov8n-p2-routing-3seed": "[levir-yolov8n-p2-routing-3seed](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p2-routing-3seed)",
        "duyle2408/levir-ship-yolo-baselines": "[levir-ship-yolo-baselines](https://huggingface.co/datasets/duyle2408/levir-ship-yolo-baselines)",
        "duyle2408/levir-yolov10n-gcts-ablation": "[levir-yolov10n-gcts-ablation](https://huggingface.co/datasets/duyle2408/levir-yolov10n-gcts-ablation)",
        "duyle2408/levir-yolov10n-gcts-v1-seed43": "[levir-yolov10n-gcts-ablation](https://huggingface.co/datasets/duyle2408/levir-yolov10n-gcts-ablation)",
        "duyle2408/levir-yolov10n-gcts-v1-seed44": "[levir-yolov10n-gcts-ablation](https://huggingface.co/datasets/duyle2408/levir-yolov10n-gcts-ablation)",
        "duyle2408/levir-yolov10n-gcts-v2-ablation": "[levir-yolov10n-gcts-v2-ablation](https://huggingface.co/datasets/duyle2408/levir-yolov10n-gcts-v2-ablation)",
        "duyle2408/levir-yolov10n-p3-nudfl-ablation": "[levir-yolov10n-p3-nudfl-ablation](https://huggingface.co/datasets/duyle2408/levir-yolov10n-p3-nudfl-ablation)",
        "duyle2408/levir-yolov8n-p3-nudfl-ablation": "[levir-yolov8n-p3-nudfl-ablation](https://huggingface.co/datasets/duyle2408/levir-yolov8n-p3-nudfl-ablation)"
    }

    for family, config in yolo_order:
        runs = groups.get((family, config), [])
        if not runs:
            continue
        seeds = sorted(list(set(r["seed"] for r in runs)))
        n_seeds = f"{', '.join(map(str, seeds))} ({len(seeds)})"
        
        val_precisions = [r.get("val/precision(B)", 0) for r in runs]
        val_recalls = [r.get("val/recall(B)", 0) for r in runs]
        val_map50s = [r.get("val/metrics/mAP50(B)", 0) for r in runs]
        val_map75s = [r.get("val/metrics/mAP75(B)", 0) for r in runs]
        val_map50_95s = [r.get("val/metrics/mAP50-95(B)", 0) for r in runs]
        
        precisions = [r["test/precision(B)"] for r in runs]
        recalls = [r["test/recall(B)"] for r in runs]
        map50s = [r["test/metrics/mAP50(B)"] for r in runs]
        map75s = [r["test/metrics/mAP75(B)"] for r in runs]
        map50_95s = [r["test/metrics/mAP50-95(B)"] for r in runs]
        
        family_repo_mapping = {
            "YOLO DBSS/HIT": "duyle2408/levir-yolo-dbss-hit-3seed",
            "YOLOv8n DBSS P2": "duyle2408/levir_dbss_p2_aware",
            "YOLO P2": "duyle2408/levir-ship-yolo-p2",
            "YOLOv8n P2 routing": "duyle2408/levir-yolov8n-p2-routing-3seed",
            "YOLO baseline": "duyle2408/levir-ship-yolo-baselines",
            "YOLOv10n GCTS v1": "duyle2408/levir-yolov10n-gcts-ablation",
            "YOLOv10n GCTS v2": "duyle2408/levir-yolov10n-gcts-v2-ablation",
            "YOLOv10n P3 NUDFL": "duyle2408/levir-yolov10n-p3-nudfl-ablation",
            "YOLOv8n P3 NUDFL": "duyle2408/levir-yolov8n-p3-nudfl-ablation",
        }
        repo = family_repo_mapping.get(family, "duyle2408/levir-yolo-dbss-hit-3seed")
        repo_link = repo_mapping.get(repo, repo)
        
        line = f"| {family} | {config} | 512 | {n_seeds} | {format_val(val_precisions)} | {format_val(val_recalls)} | {format_val(val_map50s)} | {format_val(val_map75s)} | {format_val(val_map50_95s)} | {format_val(precisions)} | {format_val(recalls)} | {format_val(map50s)} | {format_val(map75s)} | {format_val(map50_95s)} | {repo_link} |"
        yolo_table_lines.append(line)

    # 2. Process pooling_report (from investigate_pooling.md)
    pooling_records = [r for r in data if r["target"] == "pooling_report"]
    # Group by config
    p_groups = {r["config"]: r for r in pooling_records}
    
    pooling_order = [
        ("A0_fpn: FPN-Only Baseline", "topdown_baseline", "100", "1.78M", "6.34"),
        ("A1_200: FPN-Only Plain Fusion", "topdown_p1fusion_200", "200", "1.78M", "6.36"),
        ("A2_200: FPN-Only P1-GER", "topdown_p1ger_200", "200", "1.78M", "6.40"),
        ("A3: Regression-Only Detail Injection", "topdown_p1reg_only", "100", "1.89M", "14.73"),
        ("A4: P1-DRR + Alternate Partial Clip", "topdown_p1drr_partial_clip", "100", "1.78M", "6.40"),
        ("A5: P1-DRR + Old Partial Clip (post-Mosaic)", "p1drr_old_partial_clip", "100", "1.78M", "6.40"),
        ("A2_500: FPN-Only P1-GER", "topdown_p1ger_500", "381", "1.78M", "6.40"),
    ]

    pooling_table_lines = [
        "| Cấu hình | Epochs | Tham số (Params) | GFLOPs (512x512) | Val mAP50 | Val Recall | Test mAP50 | Test Recall |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for label, config, epochs, params, gflops in pooling_order:
        r = p_groups.get(config)
        if not r:
            continue
        v_map50 = f"{r.get('val/metrics/mAP50(B)', 0):.4f}"
        v_rec = f"{r.get('val/recall(B)', 0):.4f}"
        map50 = f"{r['test/metrics/mAP50(B)']:.4f}"
        rec = f"{r['test/recall(B)']:.4f}"
        
        line = f"| **{label}** | {epochs} | {params} | {gflops} | {v_map50} | {v_rec} | {map50} | {rec} |"
        pooling_table_lines.append(line)

    # 3. Update report_yolo.md
    with open("docs/reports/report_yolo.md", "r") as f:
        yolo_content = f.read()
    
    # We append the new table section at the end of report_yolo.md
    new_yolo_section = "\n\n## Kết quả Test với NMS IoU = 0.50\n\n" + "\n".join(yolo_table_lines) + "\n"
    if "## Kết quả Test với NMS IoU = 0.50" in yolo_content:
        # Replace existing
        parts = yolo_content.split("## Kết quả Test với NMS IoU = 0.50")
        yolo_content = parts[0] + "## Kết quả Test với NMS IoU = 0.50\n\n" + "\n".join(yolo_table_lines) + "\n"
    else:
        yolo_content += new_yolo_section
        
    with open("docs/reports/report_yolo.md", "w") as f:
        f.write(yolo_content)
    print("Updated docs/reports/report_yolo.md")

    # 4. Update investigate_pooling.md
    with open("docs/reports/investigate_pooling.md", "r") as f:
        pool_content = f.read()

    new_pool_section = "\n\n### Kết quả Thực nghiệm FPN-Only & P1-DRR tại NMS IoU = 0.50 (Seed 42):\n\n" + "\n".join(pooling_table_lines) + "\n"
    if "### Kết quả Thực nghiệm FPN-Only & P1-DRR tại NMS IoU = 0.50" in pool_content:
        parts = pool_content.split("### Kết quả Thực nghiệm FPN-Only & P1-DRR tại NMS IoU = 0.50")
        # keep content after table if any
        subparts = parts[1].split("\n\n", 1)
        after_content = "\n\n" + subparts[1] if len(subparts) > 1 else ""
        pool_content = parts[0] + "### Kết quả Thực nghiệm FPN-Only & P1-DRR tại NMS IoU = 0.50 (Seed 42):\n\n" + "\n".join(pooling_table_lines) + after_content
    else:
        # Find where to insert: right before "### Phân tích Computational Cost & Trade-off:"
        if "### Phân tích Computational Cost & Trade-off:" in pool_content:
            parts = pool_content.split("### Phân tích Computational Cost & Trade-off:")
            pool_content = parts[0] + new_pool_section + "\n### Phân tích Computational Cost & Trade-off:" + parts[1]
        else:
            pool_content += new_pool_section

    with open("docs/reports/investigate_pooling.md", "w") as f:
        f.write(pool_content)
    print("Updated docs/reports/investigate_pooling.md")

if __name__ == "__main__":
    main()
