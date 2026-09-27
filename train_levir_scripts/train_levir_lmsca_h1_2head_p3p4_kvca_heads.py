#!/usr/bin/env python3
"""Train/evaluate the canonical LMSCA H1 two-head model on LEVIR-Ship.

This runner owns only dataset orchestration. The architecture YAML is shared
with the Varroa LMSCA result and no FTSC modules or losses are enabled here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ULTRALYTICS_ROOT = PROJECT_ROOT / "models_related/ultralytics"
MODEL_CONFIG = PROJECT_ROOT / "models_related/models_config/yolov8/yolov8_lmsca_h1_2head_p3p4_kvca_heads.yaml"
EXPERIMENT = "levir_lmsca_h1_2head_p3p4_kvca_heads"
HF_REPO = "duyle2408/levir-lmsca-h1-2head-p3p4-kvca-heads"
SEEDS = (42, 43, 44)
SPLIT_SEED = 42
FITNESS_METRIC = "map50_95"
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "config.yaml", "experiment_manifest.json", "evaluation_metrics.json")


def local_ultralytics() -> None:
    for path in (PROJECT_ROOT, str(ULTRALYTICS_ROOT)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_split(args: argparse.Namespace) -> Path:
    from train_levir_scripts.train_all_levir_yolov8n_p2_routing import prepare_fixed_split, validate_split

    args.split_seed = SPLIT_SEED
    data_yaml = prepare_fixed_split(args)
    validate_split(data_yaml)
    return data_yaml


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int) -> dict[str, object]:
    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": seed,
        "deterministic": True,
        "amp": True,
        "optimizer": "auto",
        "cos_lr": False,
        "lrf": 0.01,
        "fitness_metric": FITNESS_METRIC,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "plots": False,
    }


def preflight_model() -> dict[str, object]:
    local_ultralytics()
    import torch
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(MODEL_CONFIG), ch=3, nc=1, verbose=False).eval()
    head = model.model[-1]
    if head.stride.tolist() != [8.0, 16.0]:
        raise AssertionError(f"Unexpected LMSCA Detect strides: {head.stride.tolist()}")
    if sum(parameter.numel() for parameter in model.parameters()) != 2_226_825:
        raise AssertionError("LMSCA parameter count changed")
    if len(getattr(head, "loc_cv", [])) or len(getattr(head, "cvq", [])):
        raise AssertionError("LMSCA must not instantiate inactive Detect experimental branches")
    if getattr(head, "ftsc_calibrator", None) is not None:
        raise AssertionError("LMSCA LEVIR integration must not enable FTSC")
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 128, 128))
    if not isinstance(output, tuple):
        raise AssertionError("Unexpected LMSCA synthetic inference output")
    return {
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "detect_from": list(head.f),
        "detect_stride": head.stride.tolist(),
        "p3_sr_ratio": model.model[28].sr_ratio,
        "p4_sr_ratio": model.model[29].sr_ratio,
        "ftsc_enabled": False,
        "synthetic_forward": "pass",
    }


def run_dir(args: argparse.Namespace, seed: int) -> Path:
    return args.project / f"seed_{seed}"


def complete(path: Path) -> bool:
    return all((path / relative).is_file() for relative in REQUIRED)


def train_one(args: argparse.Namespace, data_yaml: Path, seed: int) -> Path:
    local_ultralytics()
    from ultralytics import YOLO

    output = run_dir(args, seed)
    output.mkdir(parents=True, exist_ok=True)
    if complete(output):
        return output
    last = output / "weights/last.pt"
    if last.is_file():
        YOLO(str(last)).train(resume=True)
    else:
        seed_everything(seed)
        model = YOLO(str(MODEL_CONFIG))
        model.load(args.pretrained, smart_transfer=True)
        shutil.copy2(MODEL_CONFIG, output / "config.yaml")
        model.train(project=str(args.project), name=f"seed_{seed}", exist_ok=True, **train_kwargs(args, data_yaml, seed))
    if not all((output / relative).is_file() for relative in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")):
        raise RuntimeError(f"Incomplete LEVIR LMSCA training artifacts: {output}")
    return output


def evaluate_one(args: argparse.Namespace, data_yaml: Path, output: Path) -> dict[str, object]:
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, object] = {"checkpoint": "best.pt", "nms_iou": 0.5, "protocol/fitness_metric": FITNESS_METRIC}
    for split in ("val", "test"):
        result = YOLO(str(output / "weights/best.pt")).val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(output / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    path = output / "evaluation_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_manifest(args: argparse.Namespace, data_yaml: Path, output: Path, seed: int, preflight: dict[str, object]) -> None:
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(str(output / "weights/best.pt"))
    manifest = {
        "variant": EXPERIMENT,
        "seed": seed,
        "split_seed": SPLIT_SEED,
        "config": str(MODEL_CONFIG),
        "config_sha256": sha256(MODEL_CONFIG),
        "data_yaml": str(data_yaml),
        "training": {key: value for key, value in train_kwargs(args, data_yaml, seed).items() if key != "data"},
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "detect_from": list(model.model.model[-1].f),
        "detect_stride": model.model.model[-1].stride.tolist(),
        "preflight": preflight,
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summaries(args: argparse.Namespace) -> None:
    rows = []
    for seed in args.seeds:
        path = run_dir(args, seed) / "evaluation_metrics.json"
        if path.is_file():
            rows.append({"variant": EXPERIMENT, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})
    if not rows:
        return
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
    with (args.project / "summary_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    aggregate = {"variant": EXPERIMENT, "runs": len(rows)}
    for key in sorted(set.intersection(*(set(row) for row in rows)) - {"variant", "seed", "checkpoint"}):
        try:
            values = [float(row[key]) for row in rows]
        except (TypeError, ValueError):
            if all(row[key] == rows[0][key] for row in rows):
                aggregate[key] = rows[0][key]
            continue
        aggregate[f"{key}/mean"] = statistics.fmean(values)
        aggregate[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    with (args.project / "summary_aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump([aggregate], handle, indent=2, sort_keys=True); handle.write("\n")


class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required unless --skip-upload is set")
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    def upload_run(self, output: Path, seed: int) -> None:
        if not complete(output):
            raise RuntimeError(f"Refusing incomplete upload: {output}")
        self.api.upload_folder(
            folder_path=str(output), repo_id=self.repo_id, repo_type="dataset",
            path_in_repo=f"runs/{EXPERIMENT}/seed_{seed}",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.data_root, args.dataset_root, args.project = args.data_root.resolve(), args.dataset_root.resolve(), args.project.resolve()
    report = preflight_model()
    if args.preflight_only:
        print(json.dumps(report, indent=2, sort_keys=True)); return
    data_yaml = prepare_split(args)
    if args.prepare_only:
        print(data_yaml); return
    args.project.mkdir(parents=True, exist_ok=True)
    uploader = None if args.skip_upload else Uploader(args.hf_repo_id)
    for seed in args.seeds:
        output = train_one(args, data_yaml, seed)
        evaluate_one(args, data_yaml, output)
        write_manifest(args, data_yaml, output, seed, report)
        write_summaries(args)
        if uploader is not None:
            uploader.upload_run(output, seed)


if __name__ == "__main__":
    main()
