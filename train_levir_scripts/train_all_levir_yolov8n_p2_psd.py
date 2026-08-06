#!/usr/bin/env python3
"""Train, evaluate, summarize, and upload the YOLOv8n P2 Positive-Support Dropout matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "levir_yolov8n_p2_psd"
HF_REPO = "duyle2408/levir-yolov8n-p2-psd-3seed"
VARIANTS = {
    "psd_dominant": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml",
    "psd_random": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml",
    "psd_none": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_baseline.yaml",
}

workflow.EXPERIMENT = EXPERIMENT
workflow.HF_REPO = HF_REPO
workflow.VARIANTS = VARIANTS
_base_train_kwargs = workflow.train_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    return parser.parse_args()


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool) -> dict[str, object]:
    kwargs = _base_train_kwargs(args, data_yaml, seed, amp)
    variant = getattr(args, "current_variant", "psd_dominant")
    
    # Determine the PSD mode based on the variant name
    mode = "dominant"
    if "random" in variant:
        mode = "random"
    elif "none" in variant:
        mode = "none"
        
    kwargs.update(
        positive_support_dropout=True,
        positive_support_mode=mode,
        positive_support_gain=0.25,
        positive_support_prob=0.5,
        positive_support_min_count=3,
        positive_support_aux_topk=3,
        positive_support_radius=2,
        positive_support_fill_kernel=3,
        positive_support_warmup_start=5,
        positive_support_warmup_end=15,
        p2_offset_regression=False,
        dfl_residual=False,
        box_consensus_gain=0.0,
        pc_dfl_gain=0.0,
    )
    return kwargs


workflow.train_kwargs = train_kwargs


def main() -> None:
    args = parse_args()
    args.variants = list(VARIANTS)
    args.runner = Path(__file__).resolve()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    
    amp = {}
    if not args.no_smoke:
        for variant in args.variants:
            args.current_variant = variant
            amp[variant] = workflow.smoke(variant, data_yaml, args, amp=args.amp)
    else:
        amp = {variant: args.amp for variant in args.variants}
        
    if args.smoke_only:
        return
        
    for seed in args.seeds:
        for variant in args.variants:
            args.current_variant = variant
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
