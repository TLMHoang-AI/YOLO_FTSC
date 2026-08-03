#!/usr/bin/env python3
"""Train and evaluate YOLOv5n/8n/10n DBSS/HIT on fixed Varroa splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG_ROOT = ROOT / "models_related/models_config"
MODELS = {
    "yolov5n": {
        "weights": "yolov5nu.pt",
        "dbss": CONFIG_ROOT / "yolov5/levir/yolov5n_p2_levir_dbss_full.yaml",
        "hit": CONFIG_ROOT / "yolov5/levir/yolov5n_p2_levir_hit_full.yaml",
    },
    "yolov8n": {
        "weights": "yolov8n.pt",
        "dbss": CONFIG_ROOT / "yolov8/levir/yolov8n_p2_levir_dbss_full.yaml",
        "hit": CONFIG_ROOT / "yolov8/levir/yolov8n_p2_levir_hit_full.yaml",
    },
    "yolov10n": {
        "weights": "yolov10n.pt",
        "dbss": CONFIG_ROOT / "yolov10/levir/yolov10n_p2_levir_dbss_full.yaml",
        "hit": CONFIG_ROOT / "yolov10/levir/yolov10n_p2_levir_hit_full.yaml",
    },
}
EXPECTED_SPLITS = {"train": 2762, "val": 592, "test": 592}
REQUIRED_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv")


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_dataset(data_yaml: Path) -> None:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    for split, expected in EXPECTED_SPLITS.items():
        relative = payload.get(split)
        if not relative:
            raise ValueError(f"Missing {split!r} in {data_yaml}")
        image_dir = root / relative
        count = sum(path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} for path in image_dir.iterdir())
        if count != expected:
            raise ValueError(f"Varroa {split} has {count} images; expected {expected}")


def ensure_dataset(args: argparse.Namespace) -> None:
    try:
        validate_dataset(args.data_yaml)
        return
    except (FileNotFoundError, ValueError):
        from misc.prepare_dataset import prepare_dataset

        print(f"Preparing fixed Varroa split from {args.data_root}", flush=True)
        args.data_yaml = prepare_dataset(
            args.data_root,
            args.data_yaml.parent,
            gt_source="gt_one",
            only_positives=True,
            class_policy="map-3-to-1",
            seed=42,
        ).resolve()
        validate_dataset(args.data_yaml)


def complete(run_dir: Path) -> bool:
    return all((run_dir / relative).is_file() for relative in REQUIRED_ARTIFACTS)


def model_for(model_name: str, mechanism: str):
    local_ultralytics()
    from ultralytics import YOLO

    spec = MODELS[model_name]
    model = YOLO(spec[mechanism])
    model.load(spec["weights"], smart_transfer=True)
    return model


def train_kwargs(args: argparse.Namespace, seed: int, amp: bool) -> dict[str, object]:
    return {
        "data": str(args.data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": seed,
        "amp": amp,
        "deterministic": True,
        "plots": False,
    }


def archive_failed_amp(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    archived = run_dir.with_name(f"{run_dir.name}_amp_failed_{int(time.time())}")
    shutil.move(str(run_dir), str(archived))
    return archived


def train_one(
    model_name: str, mechanism: str, seed: int, amp: bool, args: argparse.Namespace
) -> Path:
    run_dir = args.project / model_name / mechanism / f"seed_{seed}"
    if complete(run_dir):
        print(f"Reusing completed run: {model_name}/{mechanism}/seed_{seed}", flush=True)
        return run_dir

    last = run_dir / "weights/last.pt"
    if last.is_file():
        local_ultralytics()
        from ultralytics import YOLO

        print(f"Resuming {run_dir}", flush=True)
        seed_everything(seed)
        YOLO(last).train(resume=True)
    else:
        kwargs = train_kwargs(args, seed, amp)
        kwargs.update(project=str(args.project / model_name / mechanism), name=f"seed_{seed}", exist_ok=True)
        seed_everything(seed)
        try:
            model_for(model_name, mechanism).train(**kwargs)
        except Exception as error:
            if not amp:
                raise
            archived = archive_failed_amp(run_dir)
            print(
                f"AMP failed for {model_name}/{mechanism}/seed_{seed}: {error!r}; "
                f"archived={archived}; retrying with amp=False",
                flush=True,
            )
            seed_everything(seed)
            kwargs["amp"] = False
            model_for(model_name, mechanism).train(**kwargs)
    if not complete(run_dir):
        raise FileNotFoundError(f"Training ended without required artifacts: {run_dir}")
    return run_dir


def smoke_amp(model_name: str, mechanism: str, args: argparse.Namespace) -> None:
    smoke_root = args.project / "_amp_smoke"
    kwargs = train_kwargs(args, args.seeds[0], True)
    kwargs.update(
        epochs=1,
        imgsz=min(args.imgsz, 256),
        batch=1,
        patience=0,
        workers=0,
        val=False,
        fraction=args.smoke_fraction,
        project=str(smoke_root / model_name),
        name=mechanism,
        exist_ok=True,
    )
    seed_everything(args.seeds[0])
    try:
        model_for(model_name, mechanism).train(**kwargs)
    except Exception as error:
        print(f"AMP smoke failed for {model_name}/{mechanism}: {error!r}; main run will use amp=False", flush=True)
        args.amp_overrides[(model_name, mechanism)] = False
    else:
        args.amp_overrides[(model_name, mechanism)] = True


def evaluate(run_dir: Path, args: argparse.Namespace) -> dict[str, float]:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, float] = {}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(args.data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_summaries(args: argparse.Namespace) -> None:
    rows = []
    for model_name in args.models:
        for mechanism in args.mechanisms:
            for seed in args.seeds:
                path = args.project / model_name / mechanism / f"seed_{seed}" / "evaluation_metrics.json"
                if path.is_file():
                    rows.append({"model": model_name, "mechanism": mechanism, "seed": seed, **json.loads(path.read_text())})
    if not rows:
        return

    def write(path: Path, values: list[dict[str, object]]) -> None:
        fields = sorted({key for row in values for key in row}, key=lambda key: (key not in {"model", "mechanism", "seed"}, key))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)

    args.project.mkdir(parents=True, exist_ok=True)
    write(args.project / "summary_runs.csv", rows)
    aggregate = []
    for model_name in args.models:
        for mechanism in args.mechanisms:
            group = [row for row in rows if row["model"] == model_name and row["mechanism"] == mechanism]
            if not group:
                continue
            record: dict[str, object] = {"model": model_name, "mechanism": mechanism, "runs": len(group)}
            metric_keys = set.intersection(*(set(row) for row in group)) - {"model", "mechanism", "seed"}
            for key in sorted(metric_keys):
                values = [float(row[key]) for row in group]
                record[f"{key}/mean"] = statistics.fmean(values)
                record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate.append(record)
    write(args.project / "summary_aggregate.csv", aggregate)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--mechanisms", nargs="+", choices=("dbss", "hit"), default=["dbss", "hit"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "datasets/varroa_yolo/varroa.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/varroa_dbss_hit")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_yaml = args.data_yaml.resolve()
    args.data_root = args.data_root.resolve()
    args.project = args.project.resolve()
    args.amp_overrides = {}
    ensure_dataset(args)
    if args.smoke_amp:
        for model_name in args.models:
            for mechanism in args.mechanisms:
                smoke_amp(model_name, mechanism, args)
    if args.smoke_only:
        return
    for seed in args.seeds:
        for model_name in args.models:
            for mechanism in args.mechanisms:
                amp = args.amp_overrides.get((model_name, mechanism), args.amp)
                run_dir = train_one(model_name, mechanism, seed, amp, args)
                evaluate(run_dir, args)
                write_summaries(args)


if __name__ == "__main__":
    main()
