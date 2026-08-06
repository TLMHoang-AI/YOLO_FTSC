#!/usr/bin/env python3
"""Train five YOLO baselines on three random LEVIR-Ship splits."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from misc.prepare_levir_ship import prepare


ROOT = Path(__file__).resolve().parent
MODELS = ("yolov5nu.pt", "yolov8n.pt", "yolov9t.pt", "yolov10n.pt", "yolo11n.pt")
SEEDS = (42, 43, 44)


def local_ultralytics() -> None:
    local = ROOT.parent / "models_related/ultralytics"
    if (local / "ultralytics/__init__.py").is_file() and str(local) not in sys.path:
        sys.path.insert(0, str(local))


def completed(run_dir: Path) -> bool:
    return all((run_dir / relative).is_file() for relative in ("weights/best.pt", "weights/last.pt", "results.csv"))


def upload_run(run_dir: Path, repo_id: str | None, token: str) -> str:
    from huggingface_hub import HfApi

    if not completed(run_dir):
        raise FileNotFoundError(f"Refusing to upload incomplete run: {run_dir}")
    api = HfApi(token=token)
    if repo_id is None:
        repo_id = f"{api.whoami()['name']}/levir-ship-yolo-baselines"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir), repo_id=repo_id, repo_type="dataset",
        path_in_repo=f"train/{run_dir.name}",
    )
    print(f"Uploaded {run_dir.name} to https://huggingface.co/datasets/{repo_id}")
    return repo_id


def run(args: argparse.Namespace) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set before training")
    local_ultralytics()
    from ultralytics import YOLO

    repo_id = args.hf_repo_id
    for seed in SEEDS:
        data_yaml = prepare(args.data_root, args.dataset_root / f"levir_ship_yolo_seed{seed}", seed)
        for model_name in MODELS:
            run_name = f"{Path(model_name).stem}_seed{seed}"
            run_dir = args.project / run_name
            if completed(run_dir):
                print(f"Reusing completed run: {run_name}")
            else:
                last = run_dir / "weights/last.pt"
                model = YOLO(str(last) if last.is_file() else model_name)
                if last.is_file():
                    model.train(resume=True)
                else:
                    model.train(
                        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                        batch=args.batch_size, device=args.device, workers=args.workers,
                        patience=args.patience, seed=seed, deterministic=True,
                        project=str(args.project), name=run_name, exist_ok=True,
                    )
            repo_id = upload_run(run_dir, repo_id, token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / "runs/levir_ship_baselines")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hf-repo-id")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
