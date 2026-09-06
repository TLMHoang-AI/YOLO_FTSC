"""Tests for the corrected B/F matched-control runner without training."""

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train_all_levir_yolov8n_p2_ftsc_corrected_controls as runner
import train_all_levir_yolov8n_p2_routing as workflow
from ultralytics.nn.tasks import DetectionModel


def test_variant_set_default_seed_and_ap50_protocol_are_locked():
    assert runner.VARIANTS == {
        "B_tal_baseline": runner.CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y0_baseline.yaml",
        "F_canonical_y4": runner.CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml",
    }
    args = runner.parse_args([])
    assert args.variants == ["B_tal_baseline", "F_canonical_y4"]
    assert args.seeds == [42]
    assert args.fitness_metric == "map50"
    assert runner.FITNESS_METRIC == "map50"
    assert runner.PROTOCOL_VERSION == "ftsc_corrected_ap50_v1"
    assert args.project.name == "levir_yolov8n_p2_ftsc_corrected_controls"
    assert workflow.train_kwargs(args, Path("data.yaml"), seed=42, amp=False)["fitness_metric"] == "map50"


def test_b_and_f_model_construction_invariants_and_freeze_preflight():
    baseline = DetectionModel(runner.VARIANTS["B_tal_baseline"], verbose=False)
    canonical = DetectionModel(runner.VARIANTS["F_canonical_y4"], verbose=False)

    baseline_report = runner.preflight_model(baseline, "B_tal_baseline")
    canonical_report = runner.preflight_model(canonical, "F_canonical_y4")

    assert baseline.model[-1].ftsc_calibrator is None
    calibrator = canonical.model[-1].ftsc_calibrator
    assert calibrator.evidence_names == ("position_gaussian", "dfl_distribution")
    assert calibrator.policy == "f5"
    assert calibrator.providers["dfl_distribution"].detach is True
    assert calibrator.dfl_apply_cls is True
    assert calibrator.dfl_apply_box is False
    assert calibrator.dfl_apply_dfl is False
    assert all(name.endswith(".dfl.conv.weight") for name in baseline_report["fixed_dfl_parameters"])
    assert all(name.endswith(".dfl.conv.weight") for name in canonical_report["fixed_dfl_parameters"])
    assert canonical_report["dfl_strength_trainable"] is True
    assert canonical_report["position_strength_trainable"] is True


def test_validation_epoch_diagnostics_use_validation_columns_only(tmp_path):
    results = tmp_path / "results.csv"
    with results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"])
        writer.writeheader()
        writer.writerows(
            [
                {"epoch": 0, "metrics/mAP50(B)": 0.82, "metrics/mAP50-95(B)": 0.31},
                {"epoch": 1, "metrics/mAP50(B)": 0.80, "metrics/mAP50-95(B)": 0.33},
            ]
        )

    assert runner._best_validation_epochs(results) == {
        "protocol/best_ap50_epoch": 0.0,
        "protocol/best_map50_95_epoch": 1.0,
    }


def test_evaluation_exports_protocol_strength_and_summary_metadata(tmp_path, monkeypatch):
    network = DetectionModel(runner.VARIANTS["F_canonical_y4"], verbose=False)

    class FakeResult:
        results_dict = {
            "metrics/precision(B)": 0.7,
            "metrics/recall(B)": 0.6,
            "metrics/mAP50(B)": 0.8,
            "metrics/mAP50-95(B)": 0.3,
        }
        box = SimpleNamespace(map75=0.1)

    class FakeYOLO:
        def __init__(self, _):
            self.model = network

        def val(self, **_):
            return FakeResult()

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    run_dir = tmp_path / "F_canonical_y4" / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n0,0.82,0.31\n1,0.80,0.33\n",
        encoding="utf-8",
    )
    args = runner.parse_args([])
    args.project = tmp_path
    args.variants = ["F_canonical_y4"]
    args._active_variant = "F_canonical_y4"
    args._active_seed = 42
    args.imgsz = 512
    args.batch_size = 1
    args.device = "cpu"
    args.workers = 0

    metrics = runner.evaluate(run_dir, Path("data.yaml"), args)

    assert metrics["variant"] == "F_canonical_y4"
    assert metrics["seed"] == 42
    assert metrics["nms_iou"] == 0.5
    assert metrics["protocol/version"] == "ftsc_corrected_ap50_v1"
    assert metrics["protocol/fitness_metric"] == "map50"
    assert metrics["protocol/best_checkpoint_metric"] == "map50"
    assert metrics["protocol/freeze_fix_active"] == 1.0
    assert metrics["suite/corrected_controls"] == 1.0
    assert metrics["ftsc/dfl_strength_trainable"] == 1.0
    assert metrics["ftsc/position_strength_trainable"] == 1.0
    assert metrics["ftsc/final_strength_dfl_distribution"] == pytest.approx(1.0)
    assert metrics["ftsc/final_strength_position_gaussian"] == pytest.approx(1.0)
    assert torch.isfinite(torch.tensor(metrics["ftsc/final_strength_dfl_distribution"]))

    workflow.write_summaries(args)
    with (tmp_path / "summary_aggregate.csv").open(encoding="utf-8") as handle:
        aggregate = next(csv.DictReader(handle))
    assert aggregate["protocol/version"] == "ftsc_corrected_ap50_v1"
    assert aggregate["protocol/fitness_metric"] == "map50"
    assert aggregate["runs"] == "1"


def test_baseline_metadata_marks_ftsc_disabled():
    baseline = DetectionModel(runner.VARIANTS["B_tal_baseline"], verbose=False)
    report = runner.preflight_model(baseline, "B_tal_baseline")
    assert runner._ftsc_metadata(baseline.model[-1], report) == {"ftsc/enabled": 0.0}
