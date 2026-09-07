#!/usr/bin/env python3
"""Train the five standard Ultralytics YOLO baselines on TinyPerson.

The dataset preparation and TinyBenchmark evaluation are intentionally shared
with ``train_all_tinyperson.py`` so every model uses the same corner-window
split, merged-original test protocol, and NMS IoU=0.5. Jobs are processed
sequentially and can be partitioned across machines with ``--machine-index``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SEEDS = (42, 43, 44)
MODELS = {
    "yolov5": "yolov5nu.pt",
    "yolov8": "yolov8n.pt",
    "yolov9": "yolov9t.pt",
    "yolov10": "yolov10n.pt",
    "yolov11": "yolo11n.pt",
}
USING_LOCAL_FORK = False
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
)


def workflow():
    from train_tinyperson_scripts import train_all_tinyperson as module

    return module


def standard_ultralytics():
    """Load the main codebase's Ultralytics compatibility fork."""
    local_path = str(ROOT / "models_related/ultralytics")
    while local_path in sys.path:
        sys.path.remove(local_path)
    global USING_LOCAL_FORK
    USING_LOCAL_FORK = True
    sys.path.insert(0, local_path)
    from ultralytics import YOLO

    return YOLO


BASELINE_OVERRIDES = {
    "factorized_tal_target": False,
    "factorized_support_gain": 0.0,
    "scale_temper_target": False,
    "loc_assign": False,
    "ggcf_refine": False,
    "ggcf_assign_refined": False,
    "positive_support_dropout": False,
    "box_consensus_gain": 0.0,
    "quality_gain": 0.0,
    "boundary_contrast": 0.0,
    "evidence_aux_gain": 0.0,
}


def complete(run_dir: Path) -> bool:
    return all((run_dir / path).is_file() for path in REQUIRED)


def training_complete(run_dir: Path) -> bool:
    results = run_dir / "results.csv"
    return all((run_dir / path).is_file() for path in ("weights/best.pt", "weights/last.pt")) and results.is_file() and sum(1 for _ in results.open(encoding="utf-8")) > 1


def selected_jobs(models: list[str], seeds: list[int], machine_index: int, machine_count: int):
    jobs = [(model, seed) for seed in seeds for model in models]
    return [job for index, job in enumerate(jobs) if index % machine_count == machine_index]


class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for upload-required Marimo training")
        if not repo_id.strip():
            raise ValueError("--hf-repo-id is required")
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.api.whoami()
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    @staticmethod
    def retry(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def upload_and_verify(self, model_name: str, seed: int, run_dir: Path) -> None:
        missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
        if missing:
            raise RuntimeError(f"{model_name}/seed_{seed}: incomplete upload: {missing}")
        remote = f"runs/{model_name}/seed_{seed}"
        expected = {f"{remote}/{path}" for path in REQUIRED}
        remote_files = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        if not (run_dir / "upload_complete.json").is_file() or not expected.issubset(remote_files):
            self.retry(lambda: self.api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset"))
            remote_files = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        missing_remote = sorted(expected - remote_files)
        if missing_remote:
            raise RuntimeError(f"{model_name}/seed_{seed}: Hugging Face verification failed: {missing_remote}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(json.dumps({"repo_id": self.repo_id, "model": model_name, "seed": seed, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
        self.retry(lambda: self.api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/{marker.name}", repo_id=self.repo_id, repo_type="dataset"))


def train_one(model_name: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    module = workflow()
    run_dir = args.project / model_name / f"seed_{seed}_corner_sw640_sh512"
    if training_complete(run_dir):
        print(f"Reusing completed training: {run_dir}", flush=True)
        return run_dir
    module.seed_everything(seed)
    YOLO = standard_ultralytics()

    last = run_dir / "weights/last.pt"
    model = YOLO(str(last) if last.is_file() else MODELS[model_name])
    kwargs = dict(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch_size, device=args.device, workers=args.workers,
        patience=args.patience, seed=seed, deterministic=True, amp=True,
        plots=False, project=str(args.project / model_name),
        name=f"seed_{seed}_corner_sw640_sh512", exist_ok=True,
        **(BASELINE_OVERRIDES if USING_LOCAL_FORK else {}),
    )
    if last.is_file():
        model.train(resume=True)
    else:
        model.train(**kwargs)
    if not training_complete(run_dir):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    return run_dir


def write_metadata(model_name: str, seed: int, run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "model": model_name,
        "pretrained": MODELS[model_name],
        "seed": seed,
        "git_sha": git_sha,
        "command": sys.argv,
        "data_yaml": str(data_yaml),
        "split_protocol": "official TinyPerson corner windows, source-image grouped 90/10 split",
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "machine_index": args.machine_index,
        "machine_count": args.machine_count,
    }
    YOLO = standard_ultralytics()
    model = YOLO(run_dir / "weights/best.pt")
    manifest["params"] = sum(parameter.numel() for parameter in model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler
        gflops = float(get_flops(model.model, imgsz=args.imgsz))
        manifest["gflops"] = gflops if gflops > 0 else float(get_flops_with_torch_profiler(model.model, imgsz=args.imgsz))
    except Exception as exc:
        manifest["gflops"] = None
        manifest["gflops_error"] = f"{type(exc).__name__}: {exc}"
    (run_dir / "config.yaml").write_text(f"model: {MODELS[model_name]}\nname: {model_name}\n", encoding="utf-8")
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary(args: argparse.Namespace) -> None:
    rows = []
    for model_name, seed in selected_jobs(args.models, args.seeds, args.machine_index, args.machine_count):
        path = args.project / model_name / f"seed_{seed}_corner_sw640_sh512" / "evaluation_metrics.json"
        if path.is_file():
            rows.append({"model": model_name, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})
    if not rows:
        return
    args.project.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"model", "seed"}, key))
    with (args.project / f"summary_machine{args.machine_index}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for model_name in args.models:
        group = [row for row in rows if row["model"] == model_name]
        if not group:
            continue
        record = {"model": model_name, "runs": len(group)}
        for key in sorted(set.intersection(*(set(row) for row in group)) - {"model", "seed", "checkpoint"}):
            values = [float(row[key]) for row in group]
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregates.append(record)
    (args.project / f"summary_machine{args.machine_index}.json").write_text(json.dumps(aggregates, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "TinyPerson" / "tiny_set")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/tinyperson_yolo_baselines")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--hf-repo-id", default="duyle2408/tinyperson-yolo-baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--machine-index", type=int, default=0)
    parser.add_argument("--machine-count", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.machine_count < 1 or not 0 <= args.machine_index < args.machine_count:
        raise ValueError("machine-index must be in [0, machine-count)")
    args.data_root, args.dataset_root, args.project = args.data_root.resolve(), args.dataset_root.resolve(), args.project.resolve()
    module = workflow()
    test_out_dir = module.prepare_test_set(args.data_root, args.dataset_root)
    for seed in args.seeds:
        module.prepare_seed_dataset(args.data_root, args.dataset_root, test_out_dir, seed)
    if args.prepare_only:
        print("TinyPerson baseline datasets prepared.", flush=True)
        return

    # All actual runs are upload-required and therefore fail closed unless
    # launched by utils.marimo_ops launch with the shared workflow marker.
    from utils.marimo_ops import require_training_context
    require_training_context(hf_repo_id=args.hf_repo_id)
    uploader = Uploader(args.hf_repo_id)
    for model_name, seed in selected_jobs(args.models, args.seeds, args.machine_index, args.machine_count):
        seed_dir = args.dataset_root / f"tinyperson_seed_{seed}_corner_sw640_sh512"
        run_dir = train_one(model_name, seed, seed_dir / "tinyperson.yaml", args)
        standard_ultralytics()
        module.evaluate(run_dir, seed_dir / "tinyperson.yaml", test_out_dir, args.data_root, args)
        write_metadata(model_name, seed, run_dir, seed_dir / "tinyperson.yaml", args)
        if not complete(run_dir):
            raise RuntimeError(f"Required post-evaluation artifacts are incomplete: {run_dir}")
        uploader.upload_and_verify(model_name, seed, run_dir)
        write_summary(args)
    write_summary(args)
    print("TinyPerson YOLO baseline matrix complete.", flush=True)


if __name__ == "__main__":
    main()
