#!/usr/bin/env python3
"""Train the four YOLOv10n-GCTS ablations on LEVIR-Ship."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from prepare_levir_ship import prepare


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "models_related/ultralytics/ultralytics/cfg/models/v10"
VARIANTS = {
    "bilinear_w01": "yolov10n-gcts-bilinear-w01.yaml",
    "bilinear_w02": "yolov10n-gcts-bilinear-w02.yaml",
    "onehot_w01": "yolov10n-gcts-onehot-w01.yaml",
    "onehot_w02": "yolov10n-gcts-onehot-w02.yaml",
}
REQUIRED_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv")


def local_ultralytics() -> None:
    local = ROOT / "models_related/ultralytics"
    if (local / "ultralytics/__init__.py").is_file() and str(local) not in sys.path:
        sys.path.insert(0, str(local))


def completed(run_dir: Path) -> bool:
    return all((run_dir / relative).is_file() for relative in REQUIRED_ARTIFACTS)


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    local_ultralytics()
    from ultralytics import YOLO

    metrics = {}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, project=str(args.project / "evaluation"),
            name=f"{run_dir.name}_{split}", exist_ok=True, plots=False,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in (result.results_dict or {}).items()})
    return metrics


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}, key=lambda key: (key != "variant", key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def upload_run(run_dir: Path, config: Path, summary: Path, repo_id: str, token: str) -> None:
    if not completed(run_dir):
        raise FileNotFoundError(f"Refusing to upload incomplete run: {run_dir}")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_folder(folder_path=str(run_dir), repo_id=repo_id, repo_type="dataset", path_in_repo=f"train/{run_dir.name}")
    api.upload_file(path_or_fileobj=str(config), path_in_repo=f"configs/{config.name}", repo_id=repo_id, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(summary), path_in_repo="summary.csv", repo_id=repo_id, repo_type="dataset")
    print(f"Uploaded {run_dir.name} to https://huggingface.co/datasets/{repo_id}", flush=True)


def train_variant(variant: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    local_ultralytics()
    from ultralytics import YOLO

    run_dir = args.project / variant
    if completed(run_dir):
        print(f"Reusing completed run: {variant}", flush=True)
        return run_dir
    last = run_dir / "weights/last.pt"
    if last.is_file():
        print(f"Resuming partial run: {variant}", flush=True)
        YOLO(last).train(resume=True)
    else:
        model = YOLO(CONFIG_DIR / VARIANTS[variant])
        model.load(args.pretrained, smart_transfer=True)
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, patience=args.patience, seed=args.seed,
            deterministic=True, project=str(args.project), name=variant, exist_ok=True,
        )
    if not completed(run_dir):
        raise FileNotFoundError(f"Training ended without complete artifacts: {run_dir}")
    return run_dir


def run(args: argparse.Namespace) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be passed in the training process environment")
    local_ultralytics()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = args.hf_repo_id or f"{api.whoami()['name']}/levir-yolov10n-gcts-ablation"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    data_yaml = prepare(args.data_root, args.dataset_out, args.seed)
    variants = [args.variant] if args.variant else list(VARIANTS)
    rows = []
    for variant in variants:
        run_dir = train_variant(variant, data_yaml, args)
        rows.append({"variant": variant, **evaluate(run_dir, data_yaml, args)})
        summary = args.project / "summary.csv"
        write_summary(rows, summary)
        upload_run(run_dir, CONFIG_DIR / VARIANTS[variant], summary, repo_id, token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, help="Run one variant; default runs all four in order.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-out", type=Path, default=ROOT / "datasets/levir_gcts_seed42")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_gcts")
    parser.add_argument("--pretrained", default="yolov10n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-repo-id")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
