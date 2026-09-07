#!/usr/bin/env python3
"""Benchmark completed TinyPerson baseline checkpoints without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_tinyperson_scripts import train_all_tinyperson as workflow
from train_tinyperson_scripts.train_all_tinyperson_yolo_baselines import MODELS, SEEDS, Uploader, selected_jobs, standard_ultralytics

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=ROOT / 'datasets')
    p.add_argument('--project', type=Path, default=ROOT / 'runs/tinyperson_yolo_baselines')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--hf-repo-id', default='duyle2408/tinyperson-yolo-baselines')
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--machine-index', type=int, default=0)
    p.add_argument('--machine-count', type=int, default=1)
    p.add_argument('--force', action='store_true', help='Recompute TinyBenchmark metrics even when upload marker exists')
    p.add_argument('--complexity-only', action='store_true', help='Only measure params/GFLOPs for uploaded checkpoints')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 0 <= args.machine_index < args.machine_count:
        raise ValueError('machine-index must be in [0, machine-count)')
    args.data_root = args.data_root.resolve(); args.dataset_root = args.dataset_root.resolve(); args.project = args.project.resolve()
    test_out = workflow.prepare_test_set(args.data_root, args.dataset_root)
    uploader = Uploader(args.hf_repo_id)
    for model_name, seed in selected_jobs(args.models, args.seeds, args.machine_index, args.machine_count):
        run_dir = args.project / model_name / f'seed_{seed}_corner_sw640_sh512'
        metrics_path = run_dir / 'evaluation_metrics.json'
        has_tiny_metrics = metrics_path.is_file() and 'test_merged/AP-Tiny1' in json.loads(metrics_path.read_text())
        if not args.force and (not (run_dir / 'upload_complete.json').is_file() or has_tiny_metrics):
            print(f'Skip incomplete/non-uploaded run: {run_dir}', flush=True)
            continue
        seed_dir = args.dataset_root / f'tinyperson_seed_{seed}_corner_sw640_sh512'
        if not args.complexity_only:
            workflow.evaluate(run_dir, seed_dir / 'tinyperson.yaml', test_out, args.data_root, args)
        manifest_path = run_dir / 'experiment_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        YOLO = standard_ultralytics()
        model = YOLO(run_dir / 'weights/best.pt')
        manifest['params'] = sum(parameter.numel() for parameter in model.model.parameters())
        try:
            from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler
            gflops = float(get_flops(model.model, imgsz=args.imgsz))
            manifest['gflops'] = gflops if gflops > 0 else float(get_flops_with_torch_profiler(model.model, imgsz=args.imgsz))
        except Exception as exc:
            manifest['gflops'] = None
            manifest['gflops_error'] = f'{type(exc).__name__}: {exc}'
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        uploader.upload_and_verify(model_name, seed, run_dir)
        print(f'Benchmarked and uploaded {model_name}/seed_{seed}', flush=True)


if __name__ == '__main__':
    main()
