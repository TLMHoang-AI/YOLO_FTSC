#!/usr/bin/env python3
"""Train, evaluate, and upload DRENet on LEVIR-Ship dataset.

Style mirrors train_all_levir_yolov8n_p2_channel_descriptor.py.
DRENet uses a YOLOv5-style train.py — called via subprocess to avoid
namespace conflicts with the repo's own ultralytics code.

NMS IoU is explicitly set to 0.5 in all eval calls per AGENTS.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRENET = ROOT / "DRENet"
MODEL_CFG = DRENET / "models" / "DRENet.yaml"
HYP = DRENET / "data" / "hyp.scratch.yaml"

PUBLISHED_COUNTS = {"train": 2320, "val": 788, "test": 788}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.txt",
    "hyp.yaml",
    "opt.yaml",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)


# ---------------------------------------------------------------------------
# Split preparation
# ---------------------------------------------------------------------------

def prepare_split(args: argparse.Namespace) -> Path:
    """Validate the fixed seed=42 split and return its data yaml."""
    split_dir = args.dataset_root / "levir_ship_yolo_seed42"

    if not split_dir.exists():
        raise RuntimeError(f"Fixed split not found at {split_dir}.")

    for split, expected in PUBLISHED_COUNTS.items():
        imgs = list((split_dir / "images" / split).glob("*.png"))
        if len(imgs) != expected:
            raise ValueError(
                f"{split} has {len(imgs)} images, expected {expected}"
            )

    data_yaml = split_dir / "levir_ship.yaml"
    data_yaml.write_text(
        f"train: {split_dir}/images/train\n"
        f"val: {split_dir}/images/val\n"
        f"test: {split_dir}/images/test\n\n"
        f"nc: 1\nnames: ['ship']\n",
        encoding="utf-8",
    )
    return data_yaml



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def complete(run_dir: Path) -> bool:
    required_early = ("weights/best.pt", "weights/last.pt", "results.txt")
    return all((run_dir / p).is_file() for p in required_early)


def train(run_dir: Path, data_yaml: Path, seed: int, args: argparse.Namespace) -> None:
    if complete(run_dir):
        print(f"[train] {run_dir.name} already complete — skipping.")
        return

    cmd = [
        sys.executable, str(DRENET / "train.py"),
        "--data", str(data_yaml),
        "--cfg", str(MODEL_CFG),
        "--hyp", str(HYP),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.imgsz), str(args.imgsz),
        "--device", args.device,
        "--workers", str(args.workers),
        "--project", str(run_dir.parent),
        "--name", run_dir.name,
        "--seed", str(seed),
        "--exist-ok",
    ]
    print(f"[train] Running seed={seed}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(DRENET))
    if result.returncode != 0:
        raise RuntimeError(f"DRENet train.py exited with code {result.returncode}")

    missing = [p for p in ("weights/best.pt", "weights/last.pt", "results.txt") if not (run_dir / p).is_file()]
    if missing:
        raise RuntimeError(f"Training ended without required artifacts: {missing}")


# ---------------------------------------------------------------------------
# Evaluation  (NMS IoU MUST be 0.5 per AGENTS.md)
# ---------------------------------------------------------------------------

def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict:
    metrics: dict = {"checkpoint": "best.pt", "nms_iou": 0.5}
    best_pt = str(run_dir / "weights" / "best.pt")
    split_dir = Path(data_yaml).parent

    for split in ("val", "test"):
        split_yaml = run_dir / f"_eval_{split}.yaml"
        split_yaml.write_text(
            f"train: {split_dir}/images/train\n"
            f"val: {split_dir}/images/{split}\n"
            f"test: {split_dir}/images/{split}\n\n"
            f"nc: 1\nnames: ['ship']\n",
            encoding="utf-8",
        )
        eval_dir = run_dir / "evaluation" / split
        eval_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(DRENET / "test.py"),
            "--weights", best_pt,
            "--data", str(split_yaml),
            "--batch-size", str(args.batch_size),
            "--img-size", str(args.imgsz),
            "--device", args.device,
            "--iou-thres", "0.5",    # MANDATORY per AGENTS.md
            "--conf-thres", "0.001",
            "--project", str(eval_dir.parent),
            "--name", split,
            "--exist-ok",
        ]
        result = subprocess.run(cmd, cwd=str(DRENET), capture_output=True, text=True)
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        if result.returncode != 0:
            print(result.stderr[-1000:])
            raise RuntimeError(f"test.py eval on {split} failed: {result.returncode}")

        for line in result.stdout.splitlines():
            if line.strip().startswith("all"):
                parts = line.split()
                try:
                    metrics[f"{split}/metrics/precision"] = float(parts[3])
                    metrics[f"{split}/metrics/recall"] = float(parts[4])
                    metrics[f"{split}/metrics/mAP50(B)"] = float(parts[5])
                    metrics[f"{split}/metrics/mAP50-95(B)"] = float(parts[6])
                except (IndexError, ValueError):
                    pass
        split_yaml.unlink(missing_ok=True)

    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(run_dir: Path, seed: int, args: argparse.Namespace) -> None:
    import shutil
    shutil.copy2(MODEL_CFG, run_dir / "model_config.yaml")
    manifest = {
        "model": "DRENet",
        "train_seed": seed,
        "split_seed": 42,          # data split is always fixed at seed=42
        "model_cfg": MODEL_CFG.name,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "dataset": "LEVIR-Ship",
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Uploader
# ---------------------------------------------------------------------------

class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required before training starts")
        if not repo_id.strip():
            raise ValueError("--hf-repo-id is required")
        from huggingface_hub import HfApi
        self.repo_id, self.api = repo_id, HfApi(token=token)
        self.api.whoami()
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    @staticmethod
    def retry(op):
        for attempt in range(3):
            try:
                return op()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def upload_run(self, seed: int, run_dir: Path) -> None:
        missing = [p for p in REQUIRED if not (run_dir / p).is_file()]
        if missing:
            raise RuntimeError(f"Refusing incomplete upload, missing: {missing}")
        remote = f"runs/drenet/seed_{seed}"
        self.retry(lambda: self.api.upload_folder(
            folder_path=run_dir, path_in_repo=remote,
            repo_id=self.repo_id, repo_type="dataset",
        ))
        expected = {f"{remote}/{p}" for p in REQUIRED}
        uploaded = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        missing_remote = sorted(expected - uploaded)
        if missing_remote:
            raise RuntimeError(f"HF verification failed, missing: {missing_remote}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(
            json.dumps({"repo_id": self.repo_id, "seed": seed, "verified": sorted(expected)}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.retry(lambda: self.api.upload_file(
            path_or_fileobj=marker,
            path_in_repo=f"{remote}/{marker.name}",
            repo_id=self.repo_id, repo_type="dataset",
        ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_drenet")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-drenet-3seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()

    uploader = Uploader(args.hf_repo_id)

    # One fixed data split (seed=42); seeds 42/43/44 are training-only seeds
    data_yaml = prepare_split(args)

    for seed in args.seeds:
        print(f"\n{'='*60}\nTraining seed={seed} (split_seed=42)\n{'='*60}")
        run_dir = args.project / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train(run_dir, data_yaml, seed, args)
        metrics = evaluate(run_dir, data_yaml, args)
        write_manifest(run_dir, seed, args)

        missing = [p for p in REQUIRED if not (run_dir / p).is_file()]
        if missing:
            raise RuntimeError(f"Seed {seed}: post-eval artifacts incomplete: {missing}")

        val_map = metrics.get("val/metrics/mAP50(B)", "N/A")
        test_map = metrics.get("test/metrics/mAP50(B)", "N/A")
        print(f"\n[seed={seed}] val mAP50={val_map}  test mAP50={test_map}")
        uploader.upload_run(seed, run_dir)
        print(f"[seed={seed}] Upload complete ✓")


if __name__ == "__main__":
    main()
