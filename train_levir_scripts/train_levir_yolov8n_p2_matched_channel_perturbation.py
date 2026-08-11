#!/usr/bin/env python3
"""Train/evaluate seed-42 YOLOv8n P2 MatchedChannelPerturbation."""

import json
from pathlib import Path

import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parent
EXPERIMENT_SLUG = "levir_yolov8n_p2_matched_channel_perturbation"

workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.HF_REPO = "duyle2408/levir-yolov8n-p2-matched-channel-perturbation-seed42"
workflow.VARIANTS = {
    "matched_channel_perturbation": ROOT.parent / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_matched_channel_perturbation.yaml",
}

_parse_args = workflow.parse_args


def parse_args(argv=None):
    args = _parse_args(argv)
    args.runner = Path(__file__)
    args.seeds = [42]
    return args


def patch_yaml_from_stats(yaml_path: Path, stats_path: Path) -> None:
    stats = json.loads(stats_path.read_text())
    line = f"  - [-1, 1, MatchedChannelPerturbation, [{stats['mu']:.8f}, {stats['sigma_delta']:.8f}, {stats['q01']:.8f}, {stats['q99']:.8f}]]"
    text = yaml_path.read_text()
    text = "\n".join(line if "MatchedChannelPerturbation" in old else old for old in text.splitlines()) + "\n"
    yaml_path.write_text(text)


def main() -> None:
    args = parse_args()
    stats_path = ROOT.parent / "scratch/gap_gate_train_stats.json"
    if stats_path.exists():
        patch_yaml_from_stats(workflow.VARIANTS["matched_channel_perturbation"], stats_path)

    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        for variant in args.variants:
            amp[variant] = workflow.smoke(variant, data_yaml, args)
    if args.smoke_only:
        return
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = workflow.train(variant, seed, data_yaml, amp[variant], args)
            workflow.evaluate(run_dir, data_yaml, args)
            workflow.write_summaries(args)


workflow.parse_args = parse_args
workflow.main = main

if __name__ == "__main__":
    workflow.main()
