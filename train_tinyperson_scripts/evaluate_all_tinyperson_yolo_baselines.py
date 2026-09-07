#!/usr/bin/env python3
"""Re-run the established TinyPerson validation and merged test flow on saved runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_tinyperson_scripts import train_all_tinyperson as workflow
from train_tinyperson_scripts.train_all_tinyperson_yolo_baselines import MODELS, SEEDS, selected_jobs, standard_ultralytics

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "TinyPerson" / "tiny_set")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/tinyperson_yolo_baselines")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--machine-count", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.machine_count < 1 or not 0 <= args.machine_index < args.machine_count:
        raise ValueError("machine-index must be in [0, machine-count)")
    args.data_root, args.dataset_root, args.project = args.data_root.resolve(), args.dataset_root.resolve(), args.project.resolve()
    test_out = workflow.prepare_test_set(args.data_root, args.dataset_root)
    standard_ultralytics()
    for model_name, seed in selected_jobs(args.models, args.seeds, args.machine_index, args.machine_count):
        seed_dir = args.dataset_root / f"tinyperson_seed_{seed}_corner_sw640_sh512"
        run_dir = args.project / model_name / f"seed_{seed}_corner_sw640_sh512"
        if not (run_dir / "weights/best.pt").is_file():
            raise FileNotFoundError(f"Missing checkpoint: {run_dir / 'weights/best.pt'}")
        workflow.evaluate(run_dir, seed_dir / "tinyperson.yaml", test_out, args.data_root, args)
        print(f"Evaluated {model_name}/seed_{seed}: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
