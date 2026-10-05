"""Static contract tests for the cross-YOLO FTSC portability registry."""
import csv
import json
from train_levir_scripts import train_levir_ftsc_cross_yolo_portability_v1 as suite

def test_registry_and_graph_isolation_are_staticly_valid():
    assert suite.FAMILIES == ("yolov9t", "yolov10n", "yolo11n")
    assert suite.CASES == ("B0_STOCK", "B1_P23_NOFTSC", "FTSC_H2", "FTSC_H2_CP2", "FTSC_ES1_HISTORICAL", "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS", "FTSC_ES2_HISTORICAL")
    assert suite.SEEDS == (42, 43, 44)
    assert "E_H2" not in suite.CASES
    for family in suite.FAMILIES:
        assert suite.normalized_graph(suite.resolved_payload(family, "B1_P23_NOFTSC")) == suite.normalized_graph(suite.resolved_payload(family, "FTSC_H2"))
        for case in suite.CASES:
            report = suite.static_preflight(family, case)
            assert report["detect_levels"] == (3 if case == "B0_STOCK" else 2)

def test_historical_edge_graphs_preserve_pre_edge_detect_tap():
    for family in suite.FAMILIES:
        for case, schedule in (("FTSC_ES1_HISTORICAL", "constant"), ("FTSC_ES2_HISTORICAL", "ramp")):
            payload = suite.resolved_payload(family, case)
            layers = [*payload["backbone"], *payload["head"]]
            edge = next(i for i, layer in enumerate(layers) if layer[2] == "P2EdgeCueFusion")
            assert layers[edge - 1][2] == "nn.Identity"
            assert layers[-1][0][0] == suite.SPECS[family]["p2"]
            assert payload["edge_stability"]["residual_schedule"] == schedule

def test_b0_identity_and_frozen_protocol_are_levir_specific():
    assert suite.SPECS["yolov9t"]["stock"].name.endswith("levir_b0_stock.yaml")
    for family in suite.FAMILIES:
        payload = suite.resolved_payload(family, "B0_STOCK")
        assert payload["nc"] == 1
    for family in ("yolov10n", "yolo11n"):
        assert suite.resolved_payload(family, "B0_STOCK")["scale"] == "n"
    expected = {"optimizer": "AdamW", "lr0": .002, "momentum": .9, "weight_decay": .0005, "warmup_bias_lr": 0., "split_seed": 42, "workers": 4, "fitness_metric": "map50_95", "nms_iou": .5}
    assert {key: suite.TRAINING[key] for key in expected} == expected

def test_single_config_resolver_preserves_augmentation_only_cases():
    for family in suite.FAMILIES:
        assert suite.model_config_for(family, "FTSC_H2_CP2") == suite.model_config_for(family, "FTSC_H2")
        assert suite.model_config_for(family, "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS") == suite.model_config_for(family, "FTSC_ES1_HISTORICAL")
        assert suite.SPECS[family]["pretrained"] in {"yolov9t.pt", "yolov10n.pt", "yolo11n.pt"}

def test_cross_yolo_archive_contract_requires_canonical_evaluation_output():
    assert suite.REQUIRED_ARTIFACTS == ("weights/best.pt", "weights/last.pt", "results.csv", "experiment_manifest.json", "evaluation_metrics_extended.json")

def test_manifest_pretrained_identity_uses_artifact_not_machine_path(tmp_path):
    config = suite.model_config_for("yolov9t", "FTSC_H2")
    checkpoint = tmp_path / "copied.pt"; checkpoint.write_bytes(b"checkpoint")
    first = suite.pretrained_identity("yolov9t", str(checkpoint))
    second = dict(first, requested="/different/machine/path/yolov9t.pt", canonical_path="/different/machine/path/yolov9t.pt")
    a = suite.scientific_manifest("yolov9t", "FTSC_H2", 42, config, first, 4, "cuda")
    b = suite.scientific_manifest("yolov9t", "FTSC_H2", 42, config, second, 8, "cpu", requested_seeds=(42,))
    assert a["scientific_contract"] == b["scientific_contract"]

def test_validate_args_rejects_marimo_namespace_drift():
    args = suite.parse_args([])
    args.families = ["invalid"]
    try: suite.validate_args(args)
    except RuntimeError: pass
    else: raise AssertionError("invalid family accepted")
    args = suite.parse_args([]); args.workers = -1
    try: suite.validate_args(args)
    except RuntimeError: pass
    else: raise AssertionError("negative workers accepted")
    args = suite.parse_args([]); args.device = ""
    try: suite.validate_args(args)
    except RuntimeError: pass
    else: raise AssertionError("empty device accepted")

def test_validate_args_rejects_duplicate_or_invalid_case_seed():
    for field, value in (("families", ["yolov9t", "yolov9t"]), ("cases", ["FTSC_H2", "FTSC_H2"]), ("cases", ["bad"]), ("seeds", [42, 42]), ("seeds", [99])):
        args = suite.parse_args([]); setattr(args, field, value)
        try: suite.validate_args(args)
        except RuntimeError: continue
        raise AssertionError(f"{field}={value} accepted")

def test_canonical_seed_rows_retain_unrequested_seeds(tmp_path):
    args = suite.parse_args([]); args.families = ["yolov9t"]; args.cases = ["B0_STOCK"]; args.seeds = [42]; args.dataset_root = tmp_path
    rows = suite.build_preflight(args)
    assert [(row["seed"], row["requested"]) for row in rows] == [(42, True), (43, False), (44, False)]

def test_manifest_provenance_records_transfer_without_changing_science(tmp_path):
    checkpoint = tmp_path / "native.pt"; checkpoint.write_bytes(b"checkpoint")
    manifest = suite.scientific_manifest("yolov9t", "FTSC_H2", 42, suite.model_config_for("yolov9t", "FTSC_H2"), suite.pretrained_identity("yolov9t", str(checkpoint)), 4, "cuda")
    provenance = manifest["artifact_provenance"]
    assert provenance["smart_transfer"] is True
    assert provenance["smart_transfer_statistics"] == "unavailable"
    assert provenance["pretrained_family"] == "yolov9t"

def test_live_fingerprint_is_fail_closed():
    try: suite.require_live_dataset_fingerprint({"scientific_contract": {"dataset_fingerprint": None}})
    except RuntimeError: pass
    else: raise AssertionError("null dataset fingerprint accepted")

def test_status_machine_requires_extended_evaluation_and_both_checkpoints(tmp_path):
    data = tmp_path / "levir.yaml"; data.write_text("path: dataset\n")
    pretrained, manifest = {"status": "READY"}, {"scientific_contract": {"contract": "fixed"}}
    run_dir = tmp_path / "run"
    assert suite.run_status(run_dir, manifest, pretrained, data) == "NEW_RUN_REQUIRED"
    run_dir.mkdir(); (run_dir / "orphan.txt").write_text("x")
    try: suite.run_status(run_dir, manifest, pretrained, data)
    except RuntimeError: pass
    else: raise AssertionError("non-empty run without manifest accepted")
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "orphan.txt").unlink()
    assert suite.run_status(run_dir, manifest, pretrained, data) == "NEW_RUN_REQUIRED"
    weights = run_dir / "weights"; weights.mkdir(); (weights / "last.pt").write_bytes(b"last")
    assert suite.run_status(run_dir, manifest, pretrained, data) == "RESUME_REQUIRED"
    (weights / "best.pt").write_bytes(b"best")
    assert suite.run_status(run_dir, manifest, pretrained, data) == "EVAL_BACKFILL_REQUIRED"
    (run_dir / "results.csv").write_text("epoch\n0\n")
    metrics = {key: 1.0 for key in suite.REQUIRED_METRICS}
    (run_dir / "evaluation_metrics_extended.json").write_text(json.dumps(metrics))
    assert suite.run_status(run_dir, manifest, pretrained, data) == "COMPLETE"
    (run_dir / "evaluation_metrics_extended.json").unlink()
    assert suite.run_status(run_dir, manifest, pretrained, data) == "EVAL_BACKFILL_REQUIRED"
    try: suite.run_status(run_dir, {"scientific_contract": {"contract": "other"}}, pretrained, data)
    except RuntimeError: pass
    else: raise AssertionError("scientific manifest mismatch accepted")

def test_status_machine_pending_pretrained_and_data(tmp_path):
    manifest = {"scientific_contract": {"contract": "fixed"}}
    assert suite.run_status(tmp_path / "run", manifest, {"status": "PENDING_LOCAL_CHECKPOINT"}, None) == "PENDING_PRETRAINED"
    assert suite.run_status(tmp_path / "run", manifest, {"status": "READY"}, None) == "PENDING_DATA_PREPARATION"

def test_summary_completion_uses_seed_identities_and_benchmark_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "RUNS_ROOT", tmp_path)
    args = suite.parse_args([]); args.families = ["yolov9t"]; args.cases = ["FTSC_H2"]; args.seeds = [42]
    fingerprint = {key: "same" for key in suite.BENCHMARK_FINGERPRINT_KEYS}
    metric_values = {key: 1.0 for key in suite.REQUIRED_METRICS}
    rows = [{"family": "yolov9t", "case": "FTSC_H2", "seed": 42, "requested": True, "status": "COMPLETE", "metrics": {**fingerprint, **metric_values}}, {"family": "yolov9t", "case": "FTSC_H2", "seed": 43, "requested": False, "status": "NOT_REQUESTED"}, {"family": "yolov9t", "case": "FTSC_H2", "seed": 44, "requested": False, "status": "NOT_REQUESTED"}]
    suite.write_summaries(args, rows)
    with (tmp_path / "summary_aggregate.csv").open(newline="") as handle: summary = next(csv.DictReader(handle))
    assert summary["requested_complete"] == "True" and summary["canonical_complete"] == "False"
    assert summary["test_speed/inference_ms_per_image/std"] == ""
    missing_fingerprint = dict(rows[0]); missing_fingerprint["metrics"] = dict(rows[0]["metrics"]); missing_fingerprint["metrics"].pop("benchmark/gpu_name")
    suite.write_summaries(args, [missing_fingerprint, *rows[1:]])
    with (tmp_path / "summary_aggregate.csv").open(newline="") as handle: summary = next(csv.DictReader(handle))
    assert summary["speed_comparable"] == "False" and summary["test_speed/inference_ms_per_image/mean"] == ""
    args.seeds = [42, 43, 44]
    complete_rows = [{"family": "yolov9t", "case": "FTSC_H2", "seed": seed, "requested": True, "status": "COMPLETE", "metrics": {**fingerprint, **metric_values}} for seed in suite.SEEDS]
    suite.write_summaries(args, complete_rows)
    with (tmp_path / "summary_aggregate.csv").open(newline="") as handle: summary = next(csv.DictReader(handle))
    assert summary["requested_complete"] == "True" and summary["canonical_complete"] == "True"

def test_speed_comparability_requires_every_fingerprint_field():
    row = {key: "same" for key in suite.BENCHMARK_FINGERPRINT_KEYS}
    assert all(row.get(key) is not None for key in suite.BENCHMARK_FINGERPRINT_KEYS)
    row.pop("benchmark/gpu_name")
    assert not all(row.get(key) is not None for key in suite.BENCHMARK_FINGERPRINT_KEYS)

def test_runner_has_no_huggingface_network_dependency():
    source = suite.__file__
    text = open(source, encoding="utf-8").read().lower()
    assert "huggingface" not in text and "hf_result_archive" not in text

def test_reused_dispatch_branch_loads_canonical_metrics_before_marking_reused():
    text = open(suite.__file__, encoding="utf-8").read()
    branch = text.split('elif row["status"] == "COMPLETE":', 1)[1].split("return rows", 1)[0]
    assert 'load_merged_metrics(run_dir_for(family, case, seed))' in branch
    assert 'row["metrics"]' in branch and 'row["status"] = "REUSED"' in branch
