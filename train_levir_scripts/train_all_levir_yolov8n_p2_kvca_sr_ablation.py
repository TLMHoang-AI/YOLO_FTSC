#!/usr/bin/env python3
"""Train the seed-42 shared-P2 KVCA group-weight SR-ratio ablation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT.parent / "models_related/ultralytics"
CONFIG_ROOT = ROOT.parent / "models_related/models_config/yolov8/levir"
VARIANTS = {
    "kvca_groupweight_sr4": CONFIG_ROOT / "yolov8n_p2_fpn_only_kvca_groupweight_sr4.yaml",
    "kvca_groupweight_sr2": CONFIG_ROOT / "yolov8n_p2_fpn_only_kvca_groupweight_sr2.yaml",
}
VARIANT_META = {
    "kvca_groupweight_sr4": {"num_heads": 4, "sr_ratio": 4, "mode": "group_weight", "placement": "shared"},
    "kvca_groupweight_sr2": {"num_heads": 4, "sr_ratio": 2, "mode": "group_weight", "placement": "shared"},
}
RUNNER_PATH = Path(__file__)
REQUIRED = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "evaluation_metrics.json")


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict:
    local_ultralytics()
    from ultralytics import YOLO

    output = run_dir / "evaluation_metrics.json"
    if output.is_file() and json.loads(output.read_text()).get("nms_iou") == 0.5:
        return json.loads(output.read_text())
    metrics = {"nms_iou": 0.5, "checkpoint": "best.pt"}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=512, batch=8, device=args.device,
            workers=args.workers, plots=False, iou=0.5, project=str(run_dir / "evaluation"),
            name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def upload(api, args: argparse.Namespace, variant: str, run_dir: Path, data_yaml: Path) -> None:
    if not all((run_dir / path).is_file() for path in REQUIRED):
        raise RuntimeError(f"Refusing to upload incomplete run: {run_dir}")
    remote = f"runs/{variant}/seed_42"

    def retry(fn):
        for attempt in range(3):
            try:
                return fn()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    retry(lambda: api.upload_folder(folder_path=run_dir, path_in_repo=remote,
                                    repo_id=args.hf_repo_id, repo_type="dataset"))
    metadata = (
        (VARIANTS[variant], f"configs/{VARIANTS[variant].name}"),
        (RUNNER_PATH, f"code/{RUNNER_PATH.name}"),
        (data_yaml.parent / "manifest.json", "dataset/fixed_split_seed_42.json"),
        (args.project / "summary.json", "summary.json"),
    )
    for local, target in metadata:
        if local.is_file():
            retry(lambda local=local, target=target: api.upload_file(
                path_or_fileobj=local, path_in_repo=target, repo_id=args.hf_repo_id, repo_type="dataset"))
    files = set(retry(lambda: api.list_repo_files(args.hf_repo_id, repo_type="dataset")))
    missing = [f"{remote}/{path}" for path in REQUIRED if f"{remote}/{path}" not in files]
    if missing:
        raise RuntimeError(f"HF verification failed: {missing}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_root": remote,
                                  "verified": list(REQUIRED)}, indent=2) + "\n")
    retry(lambda: api.upload_file(path_or_fileobj=marker, path_in_repo=f"{remote}/{marker.name}",
                                  repo_id=args.hf_repo_id, repo_type="dataset"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / "runs/levir_yolov8n_p2_kvca_sr_ablation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required; upload is mandatory")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.whoami()
    api.create_repo(args.hf_repo_id, repo_type="dataset", exist_ok=True)
    args.data_root, args.dataset_root, args.project = (p.resolve() for p in (args.data_root, args.dataset_root, args.project))
    data_yaml = workflow.prepare_fixed_split(argparse.Namespace(
        data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42))
    local_ultralytics()
    from ultralytics import YOLO

    # Forward smoke both resolved graphs before any expensive training.
    for variant, config in VARIANTS.items():
        model = YOLO(config)
        head = model.model.model[-1]
        if head.stride.tolist() != [4.0] or head.nl != 1:
            raise RuntimeError(f"{variant}: expected P2-only stride [4], got {head.stride.tolist()}")
        model.model.to(args.device).eval()( __import__("torch").zeros(1, 3, 256, 256, device=args.device))

    summary_path = args.project / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for variant, config in VARIANTS.items():
        run_dir = args.project / variant / "seed_42"
        rows = run_dir / "results.csv"
        trained = all((run_dir / p).is_file() for p in workflow.REQUIRED_TRAIN_ARTIFACTS)
        trained = trained and sum(1 for _ in rows.open()) - 1 == 100
        if not trained:
            model = YOLO(config)
            model.load("yolov8n.pt", smart_transfer=True)
            model.train(data=str(data_yaml), epochs=100, imgsz=512, batch=8, device=args.device,
                        workers=args.workers, patience=0, seed=42, deterministic=True, amp=True,
                        plots=False, project=str(args.project / variant), name="seed_42", exist_ok=True)
        metrics = evaluate(run_dir, data_yaml, args)
        model = YOLO(run_dir / "weights/best.pt")
        summary[variant] = {"seed": 42, **VARIANT_META[variant],
                            "params": int(sum(p.numel() for p in model.model.parameters())), **metrics}
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        upload(api, args, variant, run_dir, data_yaml)


if __name__ == "__main__":
    main()
