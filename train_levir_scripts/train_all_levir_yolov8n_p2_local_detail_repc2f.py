#!/usr/bin/env python3
"""Train and evaluate YOLOv8n P2-P5 with LocalDetailRepC2f on P2/P3."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "levir_yolov8n_p2_local_detail_repc2f"
HF_REPO = "duyle2408/levir-yolov8n-p2-local-detail-repc2f-seed42"
VARIANTS = {
    "local_detail_repc2f": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_local_detail_repc2f.yaml"
}

# Reuse the proven LEVIR split/train/evaluate/upload workflow with this experiment's identity.
workflow.EXPERIMENT = EXPERIMENT
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.variants = list(VARIANTS)
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        amp = {variant: workflow.smoke(variant, data_yaml, args) for variant in args.variants}
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
