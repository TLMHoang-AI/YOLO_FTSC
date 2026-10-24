"""Focused, training-free contracts for TinyPerson/Varroa portability."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from train_tinyperson_scripts import train_all_tinyperson as tiny_native
from train_tinyperson_scripts import train_tinyperson_ftsc_cross_yolo_portability_v1 as tiny
from train_levir_scripts import train_varroa_ftsc_cross_yolo_portability_v1 as varroa
from utils import cross_yolo_portability as common


SUITES = (tiny, varroa)


@pytest.mark.parametrize("suite", SUITES)
def test_registry_yaml_graph_and_family_native_contracts(suite):
    assert suite.FAMILIES == ("yolov9t", "yolov10n", "yolo11n")
    assert suite.CASES == ("B1_P23_NOFTSC", "FTSC_H2")
    assert suite.SEEDS == (42, 43, 44)
    for family in suite.FAMILIES:
        b1, h2 = suite.resolved_payload(family, suite.CASES[0]), suite.resolved_payload(family, suite.CASES[1])
        assert suite.normalized_graph(b1) == suite.normalized_graph(h2)
        for case in suite.CASES:
            report = suite.static_preflight(family, case)
            assert Path(report["config"]).is_file()
            assert report["detect_inputs"] == [suite.SPECS[family]["p2"], suite.SPECS[family]["p3"]]
            assert report["detect_class"] == suite.SPECS[family]["detect"]


@pytest.mark.parametrize("suite", SUITES)
def test_ftsc_f5_fixed_alpha_and_dfl_contract(suite):
    assert "ftsc" not in suite.resolved_payload("yolov9t", "B1_P23_NOFTSC")
    for family in suite.FAMILIES:
        ftsc = suite.resolved_payload(family, "FTSC_H2")["ftsc"]
        assert ftsc["enabled"] is True and ftsc["policy"] == "f5"
        assert ftsc["position_alpha"] == 6.0 and ftsc.get("position_alpha_mode", "fixed") == "fixed"
        assert ftsc["dfl_detach"] is True and ftsc["dfl_apply_cls"] is True
        assert ftsc["dfl_apply_box"] is False and ftsc["dfl_apply_dfl"] is False


def test_dataset_specific_training_protocols_are_not_levir():
    targs = tiny.parse_args([]); targs.families = ["yolov9t"]; targs.cases = ["FTSC_H2"]
    tiny_kwargs = tiny.train_kwargs(targs, "yolov9t", "FTSC_H2", 42, Path("tiny.yaml"))
    assert (tiny_kwargs["epochs"], tiny_kwargs["imgsz"], tiny_kwargs["batch"], tiny_kwargs["patience"]) == (100, 640, 8, 20)
    assert tiny_kwargs["optimizer"] == "AdamW" and tiny_kwargs["mosaic"] == 1.0 and tiny_kwargs["close_mosaic"] == 10
    vargs = varroa.parse_args([])
    varroa_kwargs = varroa.train_kwargs(vargs, "yolov10n", "FTSC_H2", 43, Path("varroa.yaml"))
    assert (varroa_kwargs["epochs"], varroa_kwargs["imgsz"], varroa_kwargs["batch"], varroa_kwargs["patience"]) == (100, 640, 4, 20)
    assert varroa_kwargs["optimizer"] == "auto" and varroa.optimizer_contract()["iterations"] == 4400
    assert varroa_kwargs["mosaic"] == 1.0 and varroa_kwargs["close_mosaic"] == 10


def test_tinyperson_merged_protocol_and_public_results_adapter():
    assert tiny.EVALUATOR == {
        "contract": "tinybenchmark_merged_original_iou50_95_v2", "coordinate_space": "merged_original",
        "corner_geometry": "sw640_sh512", "window_conf": 0.001, "window_nms_iou": 0.5,
        "cross_window_nms_iou": 0.5, "max_det": 300, "latency_unit": "ms/window",
    }
    class Tensor:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self.value
    boxes = SimpleNamespace(xyxy=Tensor([[1, 2, 3, 4]]), conf=Tensor([0.9]), cls=Tensor([0.0]), __len__=lambda self: 1)
    # Python special methods live on the class, not an instance.
    boxes = type("Boxes", (), {"__len__": lambda self: 1, "xyxy": boxes.xyxy, "conf": boxes.conf, "cls": boxes.cls})()
    assert tiny_native.decode_ultralytics_result(SimpleNamespace(boxes=boxes)) == ([[1, 2, 3, 4]], [0.9], [0.0])
    with pytest.raises(TypeError):
        tiny_native.decode_ultralytics_result(SimpleNamespace(boxes=type("Bad", (), {"__len__": lambda self: 1})()))


def test_varroa_evaluator_keeps_two_nms_contracts():
    assert varroa.EVALUATOR["standard_nms_iou"] == 0.7
    assert varroa.EVALUATOR["size_bucket_nms_iou"] == 0.5
    source = (varroa.ROOT / "evaluate_test/size_bucket_evaluator.py").read_text(encoding="utf-8")
    assert "iou=0.5" in source


def test_initialization_parity_comparator_rejects_difference():
    class Torch:
        @staticmethod
        def equal(left, right): return left == right
    class Value:
        shape = (1,)
        def __init__(self, value): self.value = value
        def __eq__(self, other): return self.value == other.value
    common.compare_shared_state("family", {"x": Value(1)}, {"x": Value(1)}, Torch)
    with pytest.raises(RuntimeError, match="initialization mismatch"):
        common.compare_shared_state("family", {"x": Value(1)}, {"x": Value(2)}, Torch)


def _manifest(): return {"scientific_contract": {"fixed": True}}


def test_state_machine_manifest_resume_and_backfill(tmp_path):
    run = tmp_path / "run"
    assert common.run_status(run, _manifest(), prerequisites_ready=False, required_metrics=tiny.REQUIRED_METRICS)[0] == "BLOCKED"
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "NEW_RUN_REQUIRED"
    common.ensure_manifest(run, _manifest()); (run / "weights").mkdir(); (run / "weights/last.pt").write_bytes(b"last")
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "RESUME_REQUIRED"
    (run / "weights/best.pt").write_bytes(b"best")
    # A best.pt checkpoint exists midway through training; it is NOT completion proof.
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "RESUME_REQUIRED"
    (run / "results.csv").write_text("epoch\n0\n")
    (run / "args.yaml").write_text("epochs: 100\n")
    (run / "config.yaml").write_text("nc: 1\n")
    common.mark_training_complete(run, _manifest())
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "EVAL_BACKFILL_REQUIRED"
    metrics = {key: (0.5 if index < 3 else 1.0) for index, key in enumerate(tiny.REQUIRED_METRICS)}
    metrics["evaluation/checkpoint_sha256"] = common.sha256(run / "weights/best.pt")
    common.write_json(run / "evaluation_metrics_extended.json", metrics)
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "EVAL_BACKFILL_REQUIRED"
    common.mark_evaluation_complete(run, metrics)
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "COMPLETE"
    # Dataset-native validation must also be honored by the common state machine.
    def reject_native_contract(_metrics):
        raise RuntimeError("dataset evaluator contract mismatch")
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS, metric_validator=reject_native_contract)[0] == "BLOCKED"
    assert common.run_status(run, {"scientific_contract": {"fixed": False}}, prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "BLOCKED"
    # A changed best.pt or metrics file must never be silently reused.
    (run / "weights/best.pt").write_bytes(b"different-best")
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "BLOCKED"
    (run / "weights/best.pt").write_bytes(b"best")
    metrics["test_merged/AP50"] = 0.8
    common.write_json(run / "evaluation_metrics_extended.json", metrics)
    assert common.run_status(run, _manifest(), prerequisites_ready=True, required_metrics=tiny.REQUIRED_METRICS)[0] == "BLOCKED"


def test_unverified_native_cache_is_quarantined_before_backfill(tmp_path):
    run = tmp_path / "case" / "seed_42"
    common.ensure_manifest(run, _manifest())
    (run / "weights").mkdir()
    for name in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "config.yaml"):
        (run / name).write_bytes(name.encode())
    common.mark_training_complete(run, _manifest())
    (run / "evaluation_metrics.json").write_text('{"old":true}', encoding="utf-8")
    (run / "evaluation_metrics_extended.json").write_text('{"old":true}', encoding="utf-8")

    status, _ = common.run_status(run, _manifest(), prerequisites_ready=True,
                                  required_metrics=tiny.REQUIRED_METRICS)
    assert status == "EVAL_BACKFILL_REQUIRED"
    moved = common.quarantine_unverified_evaluation_cache(run)
    assert len(moved) == 2
    assert all(Path(path).is_file() for path in moved)
    assert all(not (run / name).exists() for name in ("evaluation_metrics.json", "evaluation_metrics_extended.json"))
    assert common.quarantine_unverified_evaluation_cache(run) == []

    metrics = {key: (0.5 if i < 3 else 1.0) for i, key in enumerate(tiny.REQUIRED_METRICS)}
    metrics["evaluation/checkpoint_sha256"] = common.sha256(run / "weights/best.pt")
    common.write_json(run / "evaluation_metrics_extended.json", metrics)
    common.mark_evaluation_complete(run, metrics)
    assert common.run_status(run, _manifest(), prerequisites_ready=True,
                             required_metrics=tiny.REQUIRED_METRICS)[0] == "COMPLETE"
    with pytest.raises(RuntimeError, match="existing evaluation proof"):
        common.quarantine_unverified_evaluation_cache(run)


@pytest.mark.parametrize("suite", SUITES)
def test_invalid_six_core_metrics_are_rejected(suite):
    metrics = {key: (0.5 if index < 3 else 1.0) for index, key in enumerate(suite.REQUIRED_METRICS)}
    if suite is tiny:
        metrics.update({"test_merged/evaluator_contract": tiny.EVALUATOR["contract"], "test_merged/available": 1.0, "benchmark/latency_unit": "ms/window"})
        metrics.update({key: 0.5 for key in tiny.SUPPLEMENTARY_MERGED_METRICS})
    else:
        metrics.update({"evaluation/nms_iou": 0.7, "test_size/protocol": "TinyBenchmark native", "benchmark/latency_unit": "ms/image"})
    suite.validate_metrics(metrics)
    metrics[suite.REQUIRED_METRICS[0]] = float("nan")
    with pytest.raises(RuntimeError, match="METRIC_INCOMPLETE"):
        suite.validate_metrics(metrics)


def test_incomplete_seed_aggregate_and_benchmark_mismatch(tmp_path):
    base = {"dataset": "TinyPerson", "family": "yolov9t", "case": "FTSC_H2", "status": "COMPLETE"}
    metrics = {key: (0.5 if index < 3 else 1.0) for index, key in enumerate(tiny.REQUIRED_METRICS)}
    fingerprint = {key: "same" for key in common.BENCHMARK_KEYS}
    rows = [{**base, "seed": 42, **metrics, **fingerprint}]
    _, aggregate = common.write_summaries(tmp_path, rows, tiny.REQUIRED_METRICS)
    record = next(csv.DictReader(aggregate.open(newline="")))
    assert record["canonical_complete"] == "False" and f"{tiny.REQUIRED_METRICS[0]}/mean" not in record
    rows = [{**base, "seed": seed, **metrics, **fingerprint} for seed in common.SEEDS]
    rows[-1]["benchmark/device"] = "other"
    _, aggregate = common.write_summaries(tmp_path, rows, tiny.REQUIRED_METRICS)
    record = next(csv.DictReader(aggregate.open(newline="")))
    assert record["canonical_complete"] == "True" and record["speed_comparable"] == "False"
    assert f"{tiny.REQUIRED_METRICS[-1]}/mean" not in record


@pytest.mark.parametrize("suite", SUITES)
def test_default_preflight_never_prepares_or_trains(suite, monkeypatch):
    args = suite.parse_args(["--families", "yolov9t", "--cases", "B1_P23_NOFTSC", "--seeds", "42"])
    assert args.action == "preflight"
    monkeypatch.setattr(suite, "prepare_dataset", lambda *_: pytest.fail("preflight prepared data"))
    rows = suite.build_preflight(args)
    assert len(rows) == 1 and rows[0]["status"] == "BLOCKED"


@pytest.mark.parametrize("suite", SUITES)
def test_mutating_actions_require_one_identity(suite):
    args = suite.parse_args(["--action", "train"])
    with pytest.raises(RuntimeError, match="exactly one"):
        suite.validate_args(args)


def test_archive_namespace_and_non_overwrite_contract():
    assert tiny.NAMESPACE != varroa.NAMESPACE
    source = Path(common.__file__).read_text(encoding="utf-8")
    assert "HFResultArchive" in source and "publish_run" in source
    assert "required_artifacts=required_artifacts" in source


def test_archive_refuses_existing_remote_identity(tmp_path):
    class API:
        def list_repo_files(self, **_kwargs):
            return ["tiny/run", "tinyperson_ftsc_cross_yolo_portability_v1/yolov9t__FTSC_H2/seed_42_corner_sw640_sh512/weights/best.pt"]
    class Archive:
        token = "redacted"
        def remote_path(self, dataset, variant, run): return f"{dataset}/{variant}/{run}"
        def _assert_repository_access(self): pass
        def _retry(self, operation): return operation()
        def _get_api(self): return API()
        def publish_run(self, **_kwargs): pytest.fail("overwrite reached publish")
    run = tmp_path / "seed_42_corner_sw640_sh512"; run.mkdir()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        common.archive_run(namespace=tiny.NAMESPACE, family="yolov9t", case="FTSC_H2", seed=42, run_dir=run, repo_id="TLMHoang/FTSC_results", archive=Archive())


def test_dataset_fingerprints_fail_closed_on_missing_or_wrong_protocol(tmp_path):
    assert tiny.dataset_fingerprint(tmp_path, 42) is None
    target = tmp_path / "varroa_yolo_seed42"; target.mkdir()
    (target / "varroa.yaml").write_text("gt_source: wrong\nonly_positives: true\nclass_policy: map-3-to-1\nseed: 42\n")
    assert varroa.dataset_fingerprint(tmp_path, 42) is None


def test_yolov10_runtime_contract_is_explicit_in_shared_smoke():
    source = Path(common.__file__).read_text(encoding="utf-8")
    assert "E2ELoss" in source and "criterion.one2many.ftsc_calibrator" in source
    assert "criterion.one2one.ftsc_calibrator is None" in source
    assert "total.sum().backward()" in source and "optimizer-visible FTSC parameters" in source
