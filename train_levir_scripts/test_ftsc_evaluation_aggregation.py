"""No-training regression checks for LEVIR extended-metric aggregation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from train_levir_scripts import train_levir_ftsc_augmentation_suite as augmentation
from train_levir_scripts import train_levir_ftsc_musgd_cp_ablation_v1 as musgd


def _write_runs(runner, root: Path, gpu_names: tuple[str, str]) -> None:
    case = runner.CASES[0]
    for seed, gpu in zip((42, 43), gpu_names, strict=True):
        run = root / case / f"seed_{seed}"; run.mkdir(parents=True)
        payload = {"test/metrics/mAP50(B)": .5 + seed / 10000, "test_speed/inference_ms_per_image": 2.0,
                   "benchmark/gpu_name": gpu, "benchmark/backend": "PyTorch", "benchmark/device": "cuda:0",
                   "benchmark/imgsz": 512, "benchmark/batch": 8, "benchmark/precision": "FP16"}
        (run / "evaluation_metrics_extended.json").write_text(json.dumps(payload))


def _assert_aggregation(runner, tmp_path: Path) -> None:
    args = runner.parse_args(["--cases", runner.CASES[0], "--seeds", "42", "43", "--project", str(tmp_path / "same")])
    _write_runs(runner, args.project, ("GPU", "GPU")); runner.aggregate(args)
    row = next(csv.DictReader((args.project / "summary_aggregate.csv").open()))
    assert "test_speed/inference_ms_per_image/mean" in row and "test/metrics/mAP50(B)/mean" in row
    args.project = tmp_path / "mixed"; _write_runs(runner, args.project, ("GPU-0", "GPU-1")); runner.aggregate(args)
    row = next(csv.DictReader((args.project / "summary_aggregate.csv").open()))
    assert row["benchmark/speed_aggregate"] == "mixed hardware / non-comparable"
    assert "test_speed/inference_ms_per_image/mean" not in row
    assert "test/metrics/mAP50(B)/mean" in row


def test_augmentation_suite_speed_aggregation(tmp_path): _assert_aggregation(augmentation, tmp_path)
def test_musgd_suite_speed_aggregation(tmp_path): _assert_aggregation(musgd, tmp_path)
