#!/usr/bin/env python3
"""Train, evaluate, and upload the seed-42 Patch-KVCA r0/r1 ablation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
VARIANTS = {
    "patch_kvca_r0": CONFIG_ROOT / "yolov8n_p2_fpn_only_patch_kvca_r0.yaml",
    "patch_kvca_r1": CONFIG_ROOT / "yolov8n_p2_fpn_only_patch_kvca_r1.yaml",
}
RADII = {"patch_kvca_r0": 0, "patch_kvca_r1": 1}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "experiment_manifest.json",
)


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_for(variant: str, pretrained: str):
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import PatchKVCompressedAttention

    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    attention = model.model.model[19]
    head = model.model.model[20]
    if not isinstance(attention, PatchKVCompressedAttention):
        raise TypeError(f"{variant}: layer 19 resolved to {type(attention).__name__}")
    if attention.patch_radius != RADII[variant] or attention.sr_ratio != 8 or attention.num_heads != 4:
        raise ValueError(f"{variant}: unexpected Patch-KVCA configuration")
    if head.f != [19] or head.nl != 1 or head.stride.tolist() != [4.0]:
        raise ValueError(f"{variant}: expected Detect([19]) at stride 4, got f={head.f}, stride={head.stride}")
    return model


def train_kwargs(args: argparse.Namespace, data_yaml: Path, variant: str, smoke: bool = False) -> dict:
    kwargs = {
        "data": str(data_yaml),
        "epochs": 1 if smoke else args.epochs,
        "imgsz": 256 if smoke else args.imgsz,
        "batch": 1 if smoke else args.batch_size,
        "device": args.device,
        "workers": 0 if smoke else args.workers,
        "patience": 0,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "plots": False,
        "project": str(args.project / ("_smoke" if smoke else variant)),
        "name": variant if smoke else "seed_42",
        "exist_ok": True,
    }
    if smoke:
        kwargs.update(fraction=0.01, val=False)
    return kwargs


def write_manifest(model, variant: str, run_dir: Path, args: argparse.Namespace) -> None:
    attention = model.model.model[19]
    head = model.model.model[20]
    payload = {
        "variant": variant,
        "seed": 42,
        "split_seed": 42,
        "config": VARIANTS[variant].name,
        "attention_class": type(attention).__name__,
        "patch_radius": attention.patch_radius,
        "sr_ratio": attention.sr_ratio,
        "num_heads": attention.num_heads,
        "channels": attention.c2,
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "amp": True,
        "nms_iou": 0.5,
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def train(variant: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / "seed_42"
    results = run_dir / "results.csv"
    complete = all((run_dir / path).is_file() for path in workflow.REQUIRED_TRAIN_ARTIFACTS)
    complete = complete and sum(1 for _ in results.open(encoding="utf-8")) - 1 == args.epochs
    if complete:
        print(f"Reusing completed training: {run_dir}", flush=True)
        if not (run_dir / "experiment_manifest.json").is_file():
            write_manifest(model_for(variant, args.pretrained), variant, run_dir, args)
        return run_dir
    seed_everything(42)
    model = model_for(variant, args.pretrained)
    model.train(**train_kwargs(args, data_yaml, variant))
    if not all((run_dir / path).is_file() for path in workflow.REQUIRED_TRAIN_ARTIFACTS):
        raise RuntimeError(f"Training ended without required artifacts: {run_dir}")
    write_manifest(model_for(variant, args.pretrained), variant, run_dir, args)
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        saved = json.loads(output.read_text())
        if saved.get("nms_iou") == 0.5 and saved.get("checkpoint") == "best.pt":
            return saved
    local_ultralytics()
    from ultralytics import YOLO

    metrics = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def upload(api, variant: str, run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"Refusing incomplete upload for {variant}: {missing}")
    remote_root = f"runs/{variant}/seed_42"

    def retry(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    retry(lambda: api.upload_folder(
        folder_path=run_dir, path_in_repo=remote_root,
        repo_id=args.hf_repo_id, repo_type="dataset",
    ))
    metadata = (
        (VARIANTS[variant], f"configs/{VARIANTS[variant].name}"),
        (Path(__file__), f"code/{Path(__file__).name}"),
        (data_yaml.parent / "manifest.json", "dataset/fixed_split_seed_42.json"),
        (args.project / "summary.json", "summary.json"),
    )
    for local, remote in metadata:
        retry(lambda local=local, remote=remote: api.upload_file(
            path_or_fileobj=local, path_in_repo=remote,
            repo_id=args.hf_repo_id, repo_type="dataset",
        ))
    files = set(retry(lambda: api.list_repo_files(args.hf_repo_id, repo_type="dataset")))
    expected = [f"{remote_root}/{path}" for path in REQUIRED]
    absent = [path for path in expected if path not in files]
    if absent:
        raise RuntimeError(f"HF verification failed: {absent}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "verified": expected}, indent=2) + "\n")
    retry(lambda: api.upload_file(
        path_or_fileobj=marker, path_in_repo=f"{remote_root}/{marker.name}",
        repo_id=args.hf_repo_id, repo_type="dataset",
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_patch_kvca")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-patch-kvca-seed42")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (
        path.resolve() for path in (args.data_root, args.dataset_root, args.project)
    )
    data_yaml = workflow.prepare_fixed_split(argparse.Namespace(
        data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42
    ))
    for variant in args.variants:
        seed_everything(42)
        model_for(variant, args.pretrained).train(**train_kwargs(args, data_yaml, variant, smoke=True))
    if args.smoke_only:
        return

    api = None
    if not args.no_upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required unless --no-upload is set")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.whoami()
        api.create_repo(args.hf_repo_id, repo_type="dataset", exist_ok=True)
    summary_path = args.project / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for variant in args.variants:
        run_dir = train(variant, data_yaml, args)
        summary[variant] = {"seed": 42, **evaluate(run_dir, data_yaml, args)}
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if api:
            upload(api, variant, run_dir, data_yaml, args)


if __name__ == "__main__":
    main()
