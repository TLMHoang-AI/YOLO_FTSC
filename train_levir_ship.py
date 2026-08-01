#!/usr/bin/env python3
"""Train the YOLOv8n-P2 DBSS/HIT core ablation matrix on LEVIR-Ship."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

from prepare_levir_ship import prepare


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "models_related/models_config/yolov8/levir"
VARIANTS = {
    "baseline": "yolov8n_p2_levir_baseline.yaml",
    "dbss_feature": "yolov8n_p2_levir_dbss_feature.yaml",
    "dbss_full": "yolov8n_p2_levir_dbss_full.yaml",
    "hit_no_transport": "yolov8n_p2_levir_hit_no_transport.yaml",
    "hit_full": "yolov8n_p2_levir_hit_full.yaml",
}


def upload_results(project: Path, repo_id: str | None = None) -> str | None:
    """Upload a completed matrix using HF_TOKEN without persisting the token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; skipping Hugging Face upload.")
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    namespace = api.whoami()["name"]
    repo_id = repo_id or f"{namespace}/levir-yolov8-p2-dbss-hit-ablation"
    missing = [
        str(project / variant / relative)
        for variant in VARIANTS
        for relative in ("weights/best.pt", "weights/last.pt", "results.csv")
        if not (project / variant / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError("Refusing incomplete Hugging Face upload:\n" + "\n".join(missing))
    configs_out = project / "configs"
    shutil.copytree(CONFIG_DIR, configs_out, dirs_exist_ok=True)
    (project / "hf_repo.txt").write_text(f"https://huggingface.co/datasets/{repo_id}\n", encoding="utf-8")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(project), repo_id=repo_id, repo_type="dataset")
    print(f"Uploaded results to https://huggingface.co/datasets/{repo_id}")
    return repo_id


def local_ultralytics() -> None:
    path = ROOT / "models_related/ultralytics"
    if (path / "ultralytics/__init__.py").is_file():
        sys.path.insert(0, str(path))


def train_variant(args: argparse.Namespace, variant: str, data_yaml: Path) -> dict[str, object]:
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(CONFIG_DIR / VARIANTS[variant])
    if args.pretrained:
        model.load(args.pretrained)
    try:
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, project=str(args.project), name=variant,
            seed=args.seed, deterministic=True, patience=args.patience,
        )
    except FloatingPointError:
        if not completed_variant(args.project, variant):
            raise
        print(f"Final validation failed for {variant}; retrying from saved best.pt")
    metrics = evaluate_checkpoint(args, variant, data_yaml)
    trained_model = getattr(getattr(model, "trainer", None), "model", model.model)
    metrics.update(getattr(trained_model, "mechanism_metrics", {}) or {})
    return {"variant": variant, **metrics}


def completed_variant(project: Path, variant: str) -> bool:
    run = project / variant
    return all((run / relative).is_file() for relative in ("weights/best.pt", "weights/last.pt", "results.csv"))


def evaluate_checkpoint(args: argparse.Namespace, variant: str, data_yaml: Path) -> dict[str, object]:
    local_ultralytics()
    from ultralytics import YOLO

    checkpoint = args.project / variant / "weights/best.pt"
    metrics = {}
    for split in ("val", "test"):
        result = YOLO(checkpoint).val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size, device=args.device,
            workers=args.workers, project=str(args.project / "evaluation"), name=f"{variant}_{split}", plots=False,
        )
        metrics.update({f"{split}/{key}": value for key, value in dict(result.results_dict or {}).items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--variant", choices=VARIANTS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-out", type=Path, default=ROOT / "datasets/levir_ship_yolo")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_ship")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hf-repo-id", default=None)
    parser.add_argument("--reuse-completed", action="store_true")
    args = parser.parse_args()

    data_yaml = prepare(args.data_root, args.dataset_out, args.seed, args.limit)
    variants = list(VARIANTS) if args.all else [args.variant]
    rows = []
    for variant in variants:
        if args.reuse_completed and completed_variant(args.project, variant):
            print(f"Reusing completed variant: {variant}")
            rows.append({"variant": variant, **evaluate_checkpoint(args, variant, data_yaml)})
        else:
            rows.append(train_variant(args, variant, data_yaml))
    args.project.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}, key=lambda key: (key != "variant", key))
    with (args.project / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    if args.all:
        upload_results(args.project, args.hf_repo_id)


if __name__ == "__main__":
    main()
