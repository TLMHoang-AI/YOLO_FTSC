#!/usr/bin/env python3
"""Train, evaluate, resume, aggregate, and upload the LEVIR-Ship DBSS-P2-Aware matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

from misc.prepare_levir_ship import prepare


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG_ROOT = ROOT / "models_related/models_config"
MODELS = {
    "yolov8n": {
        "weights": "yolov8n.pt",
        "dbss_p2_routed": CONFIG_ROOT / "yolov8/levir/yolov8n_p2_levir_dbss_p2_routed.yaml",
        "dbss_p2_aware": CONFIG_ROOT / "yolov8/levir/yolov8n_p2_levir_dbss_p2_aware.yaml",
    },
}
MECHANISMS = ("dbss_p2_routed", "dbss_p2_aware")
REQUIRED_TRAIN_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv")


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def trained(run_dir: Path) -> bool:
    return all((run_dir / path).is_file() for path in REQUIRED_TRAIN_ARTIFACTS)


def ready_for_upload(run_dir: Path) -> bool:
    return trained(run_dir) and (run_dir / "evaluation_metrics.json").is_file()


def prepare_dataset(args: argparse.Namespace) -> Path:
    """Create one fixed split shared by every training RNG seed."""
    out = args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}"
    return prepare(args.data_root, out, args.split_seed, args.limit)


def seed_training(seed: int) -> None:
    """Seed model/training randomness without changing dataset membership."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    metrics_path = run_dir / "evaluation_metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in (result.results_dict or {}).items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def train(model_name: str, mechanism: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / model_name / mechanism / f"seed_{seed}"
    if trained(run_dir):
        print(f"Reusing completed training: {model_name}/{mechanism}/seed_{seed}", flush=True)
        return run_dir
    local_ultralytics()
    from ultralytics import YOLO

    seed_training(seed)

    last = run_dir / "weights/last.pt"
    if last.is_file():
        print(f"Resuming: {run_dir}", flush=True)
        YOLO(last).train(resume=True)
    else:
        spec = MODELS[model_name]
        model = YOLO(spec[mechanism])
        model.load(spec["weights"], smart_transfer=True)
        try:
            model.train(
                data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
                device=args.device, workers=args.workers, patience=args.patience, seed=seed, amp=args.amp,
                deterministic=True, project=str(args.project / model_name / mechanism),
                name=f"seed_{seed}", exist_ok=True,
            )
        except FloatingPointError:
            if not trained(run_dir):
                raise
            print(f"Final in-training validation failed; evaluating saved best checkpoint: {run_dir}", flush=True)
        diagnostics = getattr(getattr(model, "model", None), "mechanism_metrics", {}) or {}
        if diagnostics:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "mechanism_metrics.json").write_text(
                json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    if not trained(run_dir):
        raise FileNotFoundError(f"Training ended without required artifacts: {run_dir}")
    return run_dir


def collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for model_name in MODELS:
        for mechanism in MECHANISMS:
            for seed in args.seeds:
                run_dir = args.project / model_name / mechanism / f"seed_{seed}"
                metrics = run_dir / "evaluation_metrics.json"
                if ready_for_upload(run_dir):
                    rows.append({"model": model_name, "mechanism": mechanism, "seed": seed,
                                 **json.loads(metrics.read_text(encoding="utf-8"))})
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in ("model", "mechanism", "seed"), key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(args: argparse.Namespace) -> None:
    rows = collect_rows(args)
    write_csv(rows, args.project / "summary_runs.csv")
    aggregate = []
    for model_name in MODELS:
        for mechanism in MECHANISMS:
            group = [row for row in rows if row["model"] == model_name and row["mechanism"] == mechanism]
            if not group:
                continue
            output: dict[str, object] = {"model": model_name, "mechanism": mechanism, "runs": len(group)}
            for key in sorted(set.intersection(*(set(row) for row in group)) - {"model", "mechanism", "seed"}):
                values = [float(row[key]) for row in group]
                output[f"{key}/mean"] = statistics.fmean(values)
                output[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate.append(output)
    write_csv(aggregate, args.project / "summary_aggregate.csv")


class Uploader:
    def __init__(self, args: argparse.Namespace) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN must be set for training/upload; use --no-upload only for smoke tests")
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        namespace = self.api.whoami()["name"]
        self.repo_id = args.hf_repo_id or f"{namespace}/levir_dbss_p2_aware"
        self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=False, exist_ok=True)
        args.project.mkdir(parents=True, exist_ok=True)
        (args.project / "hf_repo.txt").write_text(f"https://huggingface.co/datasets/{self.repo_id}\n", encoding="utf-8")

    def retry(self, operation) -> None:
        for attempt in range(3):
            try:
                operation()
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def upload_run(self, run_dir: Path, model_name: str, mechanism: str, seed: int) -> None:
        if not ready_for_upload(run_dir):
            raise FileNotFoundError(f"Refusing incomplete upload: {run_dir}")
        self.retry(lambda: self.api.upload_folder(
            folder_path=str(run_dir), path_in_repo=f"runs/{model_name}/{mechanism}/seed_{seed}",
            repo_id=self.repo_id, repo_type="dataset",
        ))

    def upload_metadata(self, args: argparse.Namespace) -> None:
        files = [(args.project / name, name) for name in ("summary_runs.csv", "summary_aggregate.csv", "hf_repo.txt")]
        for model_name, spec in MODELS.items():
            for mechanism in MECHANISMS:
                files.append((spec[mechanism], f"configs/{model_name}/{mechanism}.yaml"))
        files.append((
            args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}/manifest.json",
            f"datasets/manifests/fixed_split_seed_{args.split_seed}.json",
        ))
        for local, remote in files:
            if local.is_file():
                self.retry(lambda local=local, remote=remote: self.api.upload_file(
                    path_or_fileobj=str(local), path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset"
                ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--mechanisms", nargs="+", choices=MECHANISMS, default=list(MECHANISMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_dbss_p2_aware")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    if args.upload_only and args.no_upload:
        raise ValueError("--upload-only and --no-upload are mutually exclusive")
    uploader = None if args.no_upload else Uploader(args)
    data_yaml = prepare_dataset(args)
    for seed in args.seeds:
        for model_name in args.models:
            for mechanism in args.mechanisms:
                run_dir = args.project / model_name / mechanism / f"seed_{seed}"
                if not args.upload_only:
                    run_dir = train(model_name, mechanism, seed, data_yaml, args)
                    evaluate(run_dir, data_yaml, args)
                    write_summaries(args)
                if uploader and ready_for_upload(run_dir):
                    uploader.upload_run(run_dir, model_name, mechanism, seed)
                    uploader.upload_metadata(args)
    write_summaries(args)
    if uploader:
        uploader.upload_metadata(args)


if __name__ == "__main__":
    main()
