"""Protocol tests for future anchor-free FTSC/LEVIR runs."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import ultralytics
import train_all_levir_yolov8n_p2_ftsc_anchor_free as ftsc_runner
import train_all_levir_yolov8n_p2_routing as shared_workflow


def test_ftsc_runner_locks_ap50_checkpoint_fitness():
    args = ftsc_runner.parse_args([])

    assert args.fitness_metric == "map50"
    assert ftsc_runner.FITNESS_METRIC == "map50"
    assert ftsc_runner.PROTOCOL_VERSION == "ftsc_corrected_ap50_v1"
    kwargs = shared_workflow.train_kwargs(args, Path("data.yaml"), seed=42, amp=False)
    assert kwargs["fitness_metric"] == "map50"


def test_shared_levir_workflow_preserves_framework_default_without_override():
    args = ftsc_runner.parse_args([])
    del args.fitness_metric

    kwargs = shared_workflow.train_kwargs(args, Path("data.yaml"), seed=42, amp=False)
    assert kwargs["fitness_metric"] == "map50_95"


def test_ftsc_evaluation_exports_corrected_protocol_metadata(tmp_path, monkeypatch):
    class FakeResult:
        results_dict = {"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.3}
        box = SimpleNamespace(map75=0.1)

    class FakeYOLO:
        def __init__(self, _):
            head = SimpleNamespace(ftsc_calibrator=None, ghm_enabled=False, quality_head=False)
            self.model = SimpleNamespace(model=[head])

        def val(self, **_):
            return FakeResult()

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    args = ftsc_runner.parse_args([])
    args.imgsz = 512
    args.batch_size = 1
    args.device = "cpu"
    args.workers = 0

    metrics = ftsc_runner.evaluate(tmp_path, Path("data.yaml"), args)

    assert metrics["protocol/fitness_metric"] == "map50"
    assert metrics["protocol/best_checkpoint_metric"] == "map50"
    assert metrics["protocol/freeze_fix_active"] == 1.0
    assert metrics["protocol/version"] == "ftsc_corrected_ap50_v1"


def test_summary_export_preserves_invariant_string_protocol_metadata(tmp_path):
    run_dir = tmp_path / "case" / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps({"test/metrics/mAP50(B)": 0.8, "protocol/fitness_metric": "map50"}), encoding="utf-8"
    )
    args = SimpleNamespace(project=tmp_path, variants=["case"], seeds=[42])

    shared_workflow.write_summaries(args)

    with (tmp_path / "summary_aggregate.csv").open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["protocol/fitness_metric"] == "map50"
