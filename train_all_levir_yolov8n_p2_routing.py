#!/usr/bin/env python3
"""Train, evaluate, aggregate, and upload the YOLOv8n LEVIR P2-routing experiment."""

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
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
EXPERIMENT = "levir_yolov8n_p2_routing"
HF_REPO = "duyle2408/levir-yolov8n-p2-routing-3seed"
VARIANTS = {
    "dbss_pre_p2": CONFIG_ROOT / "yolov8n_p2_levir_dbss_pre_p2.yaml",
    "gcts_backbone_p2_p3": CONFIG_ROOT / "yolov8n_p2_levir_gcts_backbone.yaml",
}
REQUIRED_TRAIN_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv")
PUBLISHED_COUNTS = {"train": 2320, "val": 788, "test": 788}


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


def validate_split(data_yaml: Path) -> None:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    for split, expected in PUBLISHED_COUNTS.items():
        count = sum(path.suffix.lower() == ".png" for path in (root / payload[split]).iterdir())
        if count != expected:
            raise ValueError(f"LEVIR {split} has {count} images; expected {expected}")


def create_fixed_split(data_root: Path, output: Path, seed: int) -> Path:
    image_dir, label_dir = data_root / "All Images", data_root / "All Annotations"
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    if len(image_stems) != sum(PUBLISHED_COUNTS.values()) or image_stems != label_stems:
        raise ValueError("Expected 3,896 matching LEVIR PNG images and TXT annotations")
    stems = sorted(image_stems)
    random.Random(seed).shuffle(stems)
    for generated in (output / "images", output / "labels"):
        if generated.exists():
            shutil.rmtree(generated)
    start = 0
    manifest = {"seed": seed, "splits": {}}
    for split, count in PUBLISHED_COUNTS.items():
        selected = stems[start:start + count]
        start += count
        images_out, labels_out = output / "images" / split, output / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for stem in selected:
            for source, destination in (
                (image_dir / f"{stem}.png", images_out / f"{stem}.png"),
                (label_dir / f"{stem}.txt", labels_out / f"{stem}.txt"),
            ):
                if not destination.exists():
                    destination.symlink_to(source)
        manifest["splits"][split] = {"images": count}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    data_yaml = output / "levir_ship.yaml"
    data_yaml.write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n",
        encoding="utf-8",
    )
    return data_yaml


def prepare_fixed_split(args: argparse.Namespace) -> Path:
    data_yaml = args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}" / "levir_ship.yaml"
    try:
        validate_split(data_yaml)
    except (FileNotFoundError, KeyError, ValueError):
        data_yaml = create_fixed_split(args.data_root, data_yaml.parent, args.split_seed)
    validate_split(data_yaml)
    return data_yaml


def trained(run_dir: Path) -> bool:
    return all((run_dir / relative).is_file() for relative in REQUIRED_TRAIN_ARTIFACTS)


def complete(run_dir: Path) -> bool:
    return trained(run_dir) and (run_dir / "evaluation_metrics.json").is_file()


def remap_dbss_backbone(model, pretrained: str) -> None:
    """Restore official backbone weights after DBSS shifts target layers 1-9 by one index."""
    from ultralytics import YOLO

    source = YOLO(pretrained).model.state_dict()
    target = model.model.state_dict()
    transferred = {}
    for key, value in source.items():
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit() and 1 <= int(parts[1]) <= 9:
            parts[1] = str(int(parts[1]) + 1)
            destination = ".".join(parts)
            if destination in target and target[destination].shape == value.shape:
                transferred[destination] = value
    model.model.load_state_dict(transferred, strict=False)
    if not transferred:
        raise RuntimeError("DBSS backbone remap transferred no pretrained tensors")


def model_for(variant: str, pretrained: str):
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    if variant == "dbss_pre_p2":
        remap_dbss_backbone(model, pretrained)
    return model


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    return {
        "data": str(data_yaml), "epochs": args.epochs, "imgsz": args.imgsz,
        "batch": args.batch_size, "device": args.device, "workers": args.workers,
        "patience": args.patience, "seed": seed, "deterministic": True,
        "amp": amp, "plots": False,
    }


def smoke(variant: str, data_yaml: Path, args: argparse.Namespace) -> bool:
    seed = args.seeds[0]
    seed_everything(seed)
    kwargs = train_kwargs(args, data_yaml, seed, True)
    kwargs.update(
        epochs=1, imgsz=min(args.imgsz, 256), batch=1, workers=0, patience=0,
        val=False, fraction=args.smoke_fraction, project=str(args.project / "_smoke"),
        name=variant, exist_ok=True,
    )
    try:
        model_for(variant, args.pretrained).train(**kwargs)
    except Exception as error:
        print(f"AMP smoke failed for {variant}: {error!r}; using amp=False", flush=True)
        return False
    return True


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if trained(run_dir):
        print(f"Reusing trained run: {variant}/seed_{seed}", flush=True)
        return run_dir
    last = run_dir / "weights/last.pt"
    seed_everything(seed)
    if last.is_file():
        local_ultralytics()
        from ultralytics import YOLO

        print(f"Resuming {run_dir}", flush=True)
        YOLO(last).train(resume=True)
    else:
        kwargs = train_kwargs(args, data_yaml, seed, amp)
        kwargs.update(project=str(args.project / variant), name=f"seed_{seed}", exist_ok=True)
        try:
            model_for(variant, args.pretrained).train(**kwargs)
        except Exception as error:
            if not amp:
                raise
            archived = run_dir.with_name(f"{run_dir.name}_amp_failed_{int(time.time())}")
            if run_dir.exists():
                shutil.move(run_dir, archived)
            print(f"AMP failed for {variant}/seed_{seed}: {error!r}; retrying without AMP", flush=True)
            seed_everything(seed)
            kwargs["amp"] = False
            model_for(variant, args.pretrained).train(**kwargs)
    if not trained(run_dir):
        raise FileNotFoundError(f"Training ended without complete artifacts: {run_dir}")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float]:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    local_ultralytics()
    from ultralytics import YOLO

    metrics = {}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed", "runs"}, key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(args: argparse.Namespace) -> None:
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            path = args.project / variant / f"seed_{seed}" / "evaluation_metrics.json"
            if path.is_file():
                rows.append({"variant": variant, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})
    if not rows:
        return
    write_csv(args.project / "summary_runs.csv", rows)
    aggregate = []
    for variant in args.variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        record = {"variant": variant, "runs": len(group)}
        keys = set.intersection(*(set(row) for row in group)) - {"variant", "seed"}
        for key in sorted(keys):
            values = [float(row[key]) for row in group]
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(record)
    write_csv(args.project / "summary_aggregate.csv", aggregate)


class Uploader:
    def __init__(self, args: argparse.Namespace) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN must be configured before a full experiment launch")
        from huggingface_hub import HfApi

        self.api = HfApi(token=token)
        self.repo_id = args.hf_repo_id
        self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=False, exist_ok=True)

    @staticmethod
    def retry(operation) -> None:
        for attempt in range(3):
            try:
                operation()
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def upload_run(self, run_dir: Path, variant: str, seed: int) -> None:
        if not complete(run_dir):
            raise FileNotFoundError(f"Refusing to upload incomplete run: {run_dir}")
        self.retry(lambda: self.api.upload_folder(
            folder_path=run_dir, path_in_repo=f"runs/{variant}/seed_{seed}",
            repo_id=self.repo_id, repo_type="dataset",
        ))

    def upload_metadata(self, args: argparse.Namespace, data_yaml: Path) -> None:
        paths = [
            (VARIANTS[variant], f"configs/{VARIANTS[variant].name}") for variant in args.variants
        ] + [
            (getattr(args, "runner", Path(__file__)), f"code/{getattr(args, 'runner', Path(__file__)).name}"),
            (data_yaml.parent / "manifest.json", "dataset/fixed_split_seed_42.json"),
            (args.project / "summary_runs.csv", "summary_runs.csv"),
            (args.project / "summary_aggregate.csv", "summary_aggregate.csv"),
        ]
        for local, remote in paths:
            if local.is_file():
                self.retry(lambda local=local, remote=remote: self.api.upload_file(
                    path_or_fileobj=local, path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset",
                ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: smoke(variant, data_yaml, args) for variant in args.variants}
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, seed, data_yaml, amp[variant], args)
            evaluate(run_dir, data_yaml, args)
            write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
