#!/usr/bin/env python3
"""Run the seed-42 P2-only shared-attention 400-epoch screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow


ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT.parent / "models_related/ultralytics"
CBAM_CONFIG = ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_cbam_block.yaml"
KVCA_CONFIG = ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_kvca_block_groupweight.yaml"
KVCA_REPO = "duyle2408/levir-yolov8n-p2-fpn-only-attention-seed42"
KVCA_LAST = "shared/fpn_only_kvca_block/seed_42/weights/last.pt"
VARIANTS = ("fpn_only_kvca_block_continue", "fpn_only_cbam_block_new")
HF_REPO = "duyle2408/levir-yolov8n-p2-shared-attention-seed43-44-100"


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def train_kwargs(args, data_yaml, epochs, project, name, seed):
    return {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": 512,
        "batch": 8,
        "device": args.device,
        "workers": args.workers,
        "patience": 0,
        "seed": seed,
        "deterministic": True,
        "amp": True,
        "plots": False,
        "project": str(project),
        "name": name,
        "exist_ok": True,
    }


def train_variant(variant, seed, args, data_yaml):
    local_ultralytics()
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    run_dir = args.project / variant / f"seed_{seed}"
    expected_epochs = 300 if variant == VARIANTS[0] and seed == 42 else args.epochs_new
    results = run_dir / "results.csv"
    complete = all((run_dir / path).is_file() for path in workflow.REQUIRED_TRAIN_ARTIFACTS)
    complete = complete and sum(1 for _ in results.open(encoding="utf-8")) - 1 == expected_epochs
    if complete:
        print(f"Reusing complete training artifacts: {run_dir}", flush=True)
        return run_dir
    if variant == VARIANTS[0] and seed == 42:
        source = hf_hub_download(KVCA_REPO, KVCA_LAST, repo_type="dataset", token=args.hf_token)
        model, epochs = YOLO(source), 300
        origin = {
            "mode": "weights_continue",
            "source": f"hf://datasets/{KVCA_REPO}/{KVCA_LAST}",
            "completed_epochs_before": 100,
            "additional_epochs": 300,
            "effective_total_epochs": 400,
            "exact_optimizer_resume": False,
            "reason": "Published last.pt is stripped: epoch=-1, optimizer=None, ema=None.",
        }
    else:
        config = KVCA_CONFIG if variant == VARIANTS[0] else CBAM_CONFIG
        epochs = args.epochs_new
        model = YOLO(config)
        model.load("yolov8n.pt", smart_transfer=True)
        origin = {
            "mode": "new_training",
            "source": str(config),
            "completed_epochs_before": 0,
            "additional_epochs": epochs,
            "effective_total_epochs": epochs,
        }
    kwargs = train_kwargs(args, data_yaml, epochs, args.project / variant, f"seed_{seed}", seed)
    model.train(**kwargs)
    if not all((run_dir / path).is_file() for path in workflow.REQUIRED_TRAIN_ARTIFACTS):
        raise RuntimeError(f"Training ended without required artifacts: {run_dir}")
    (run_dir / "continuation_manifest.json").write_text(json.dumps(origin, indent=2) + "\n")
    return run_dir


def evaluate_checkpoints(run_dir, args, data_yaml):
    local_ultralytics()
    from ultralytics import YOLO

    all_metrics = {}
    for checkpoint in ("best", "last"):
        output = run_dir / f"evaluation_metrics_{checkpoint}.json"
        if output.is_file():
            saved = json.loads(output.read_text())
            if saved.get("nms_iou") == 0.5:
                all_metrics[checkpoint] = saved
                continue
        metrics = {"nms_iou": 0.5}
        for split in ("val", "test"):
            result = YOLO(run_dir / f"weights/{checkpoint}.pt").val(
                data=str(data_yaml), split=split, imgsz=512, batch=8, device=args.device,
                workers=args.workers, plots=False, iou=0.5, project=str(run_dir / "evaluation"),
                name=f"{checkpoint}_{split}", exist_ok=True,
            )
            metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
            metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
        output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        all_metrics[checkpoint] = metrics
    return all_metrics


def upload_run(api, args, variant, seed, run_dir, data_yaml):
    required = (
        "weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
        "evaluation_metrics_best.json", "evaluation_metrics_last.json", "continuation_manifest.json",
    )
    if not all((run_dir / name).is_file() for name in required):
        raise RuntimeError(f"Refusing to upload incomplete run: {run_dir}")
    remote_root = f"runs/{variant}/seed_{seed}"

    def retry(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    retry(lambda: api.upload_folder(folder_path=run_dir, path_in_repo=remote_root, repo_id=args.hf_repo_id, repo_type="dataset"))
    metadata = (
        (CBAM_CONFIG, f"configs/{CBAM_CONFIG.name}"),
        (KVCA_CONFIG, f"configs/{KVCA_CONFIG.name}"),
        (Path(__file__), f"code/{Path(__file__).name}"),
        (data_yaml.parent / "manifest.json", "dataset/fixed_split_seed_42.json"),
        (args.project / "summary.json", "summary.json"),
    )
    for local, remote in metadata:
        if local.is_file():
            retry(lambda local=local, remote=remote: api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=args.hf_repo_id, repo_type="dataset"))
    files = set(retry(lambda: api.list_repo_files(args.hf_repo_id, repo_type="dataset")))
    missing = [f"{remote_root}/{name}" for name in required if f"{remote_root}/{name}" not in files]
    if missing:
        raise RuntimeError(f"HF verification failed; missing: {missing}")
    marker = run_dir / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "remote_root": remote_root, "verified": list(required)}, indent=2) + "\n")
    retry(lambda: api.upload_file(path_or_fileobj=marker, path_in_repo=f"{remote_root}/{marker.name}", repo_id=args.hf_repo_id, repo_type="dataset"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / "runs/levir_yolov8n_p2_shared_attention_400")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--epochs-new", type=int, default=100)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required; upload is mandatory")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.whoami()
    api.create_repo(repo_id=args.hf_repo_id, repo_type="dataset", private=False, exist_ok=True)
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    split_args = argparse.Namespace(data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42)
    data_yaml = workflow.prepare_fixed_split(split_args)
    local_ultralytics()
    from ultralytics import YOLO

    smoke = YOLO(CBAM_CONFIG)
    head = smoke.model.model[-1]
    if head.stride.tolist() != [4.0]:
        raise RuntimeError(f"CBAM graph is not P2-only: {head.stride.tolist()}")
    if args.smoke_only:
        kwargs = train_kwargs(args, data_yaml, 1, args.project / "_smoke", "fpn_only_cbam_block_new", args.seeds[0])
        kwargs.update(fraction=0.01, batch=1, workers=0, imgsz=256, val=False)
        smoke.train(**kwargs)
        return
    summary_path = args.project / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for variant in VARIANTS:
        if "best" in summary.get(variant, {}):
            summary[variant] = {"seed_42": summary[variant]}
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train_variant(variant, seed, args, data_yaml)
            summary.setdefault(variant, {})[f"seed_{seed}"] = evaluate_checkpoints(run_dir, args, data_yaml)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            upload_run(api, args, variant, seed, run_dir, data_yaml)


if __name__ == "__main__":
    main()
