"""No-training contract checks for the frozen LEVIR Top-5 backfill."""
from __future__ import annotations

import json
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_levir_scripts import evaluate_levir_top5_fullmetric_v1 as runner


def _manifest(root: Path) -> Path:
    cases = {}
    for case in runner.CASES:
        cases[case] = {str(seed): {"run_dir": str(root / case / f"seed_{seed}")} for seed in runner.SEEDS}
    path = root / "manifest.json"; path.write_text(json.dumps({"cases": cases}))
    return path


def test_frozen_registry_and_metric_contract_are_exact(tmp_path):
    assert runner.CASES == ("A3_CP2", "ES1", "ES1_R4", "ES2", "E_H2")
    assert runner.SEEDS == (42, 43, 44)
    assert "test_size/AP50-Small" in runner.REQUIRED_METRICS
    assert "test_coco/AP_small" not in runner.REQUIRED_METRICS
    entries = runner.load_manifest(_manifest(tmp_path))
    assert len(entries) == 15


def test_manifest_rejects_missing_seed_unexpected_case_and_duplicate_checkpoint(tmp_path):
    path = _manifest(tmp_path); payload = json.loads(path.read_text())
    del payload["cases"]["ES1"]["44"]; path.write_text(json.dumps(payload))
    try: runner.load_manifest(path)
    except ValueError as error: assert "seeds" in str(error)
    else: raise AssertionError("missing seed must fail")
    path = _manifest(tmp_path); payload = json.loads(path.read_text()); payload["cases"]["OTHER"] = {}
    path.write_text(json.dumps(payload))
    try: runner.load_manifest(path)
    except ValueError as error: assert "cases" in str(error)
    else: raise AssertionError("unexpected case must fail")
    path = _manifest(tmp_path); payload = json.loads(path.read_text())
    payload["cases"]["ES1"]["42"] = payload["cases"]["A3_CP2"]["42"]
    path.write_text(json.dumps(payload))
    try: runner.load_manifest(path)
    except ValueError as error: assert "duplicate checkpoint" in str(error)
    else: raise AssertionError("duplicate checkpoint must fail")


def test_manifest_requires_exact_weights_best_pt_and_consistent_run_dir(tmp_path):
    path = _manifest(tmp_path); payload = json.loads(path.read_text())
    payload["cases"]["A3_CP2"]["42"] = {"checkpoint": str(tmp_path / "foo.pt")}; path.write_text(json.dumps(payload))
    try: runner.load_manifest(path)
    except ValueError as error: assert "weights/best.pt" in str(error)
    else: raise AssertionError("arbitrary checkpoint must fail")
    path = _manifest(tmp_path); payload = json.loads(path.read_text())
    payload["cases"]["A3_CP2"]["42"] = {"checkpoint": str(tmp_path / "x/weights/last.pt")}; path.write_text(json.dumps(payload))
    try: runner.load_manifest(path)
    except ValueError as error: assert "weights/best.pt" in str(error)
    else: raise AssertionError("last.pt must fail")
    path = _manifest(tmp_path); payload = json.loads(path.read_text())
    payload["cases"]["A3_CP2"]["42"] = {"run_dir": str(tmp_path / "one"), "checkpoint": str(tmp_path / "two/weights/best.pt")}; path.write_text(json.dumps(payload))
    try: runner.load_manifest(path)
    except ValueError as error: assert "disagree" in str(error)
    else: raise AssertionError("inconsistent run_dir/checkpoint must fail")
    path = _manifest(tmp_path); payload = json.loads(path.read_text())
    expected = tmp_path / "correct/weights/best.pt"; payload["cases"]["A3_CP2"]["42"] = {"checkpoint": str(expected)}; path.write_text(json.dumps(payload))
    assert runner.load_manifest(path)[("A3_CP2", 42)]["checkpoint"] == expected.resolve()


def test_summary_uses_tinybenchmark_small_and_sample_std(tmp_path):
    manifest = _manifest(tmp_path); entries = runner.load_manifest(manifest)
    for index, ((case, seed), entry) in enumerate(entries.items(), 1):
        entry["run_dir"].mkdir(parents=True)
        spec = runner.CASE_REGISTRY[case]
        values = {key: 1.0 + index / 100 for key in runner.REQUIRED_METRICS if key.startswith(("val/", "test/", "test_size/", "test_speed/"))}
        values.update({"model/parameters": spec["expected_parameters"], "model/parameters_M": spec["expected_parameters"] / 1e6,
                       "model/GFLOPs": spec["expected_gflops"], "model/GFLOPs_imgsz": 512})
        values.update({"model/GFLOPs_method": "thop", "benchmark/gpu_name": "GPU", "benchmark/backend": "PyTorch",
                       "benchmark/device": "cuda:0", "benchmark/imgsz": 512, "benchmark/batch": 8,
                       "benchmark/precision": "FP16", "benchmark/cuda_version": "12", "benchmark/pytorch_version": "x"})
        (entry["run_dir"] / "evaluation_metrics_extended.json").write_text(json.dumps(values))
    aggregate = runner.write_summaries(tmp_path / "results", entries)
    headers = next(csv.DictReader((tmp_path / "results" / f"{runner.NAMESPACE}_summary_runs.csv").open())).keys()
    assert "test_size/AP50-Small" in headers and "test_coco/AP_small" not in headers
    assert aggregate[0]["benchmark/speed_aggregate"] == "comparable"
    assert "test_speed/inference_ms_per_image/std" in aggregate[0]


def test_top5_reads_fresh_extended_only_and_writes_drift_ap_points(tmp_path):
    manifest = _manifest(tmp_path); entries = runner.load_manifest(manifest)
    entry = entries[("A3_CP2", 42)]; entry["run_dir"].mkdir(parents=True)
    spec = runner.CASE_REGISTRY["A3_CP2"]
    fresh = {key: 0.5 for key in runner.REQUIRED_METRICS}
    fresh.update({"model/parameters": spec["expected_parameters"], "model/parameters_M": spec["expected_parameters"] / 1e6,
                  "model/GFLOPs": spec["expected_gflops"], "model/GFLOPs_imgsz": 512, "model/GFLOPs_method": "thop",
                  "benchmark/gpu_name": "GPU", "benchmark/backend": "PyTorch", "benchmark/device": "cuda:0",
                  "benchmark/imgsz": 512, "benchmark/batch": 8, "benchmark/precision": "FP16", "benchmark/cuda_version": "12", "benchmark/pytorch_version": "x"})
    (entry["run_dir"] / "evaluation_metrics_extended.json").write_text(json.dumps(fresh))
    (entry["run_dir"] / "evaluation_metrics.json").write_text('{"test/metrics/mAP50(B)": 0.99}')
    assert runner.load_fresh_extended_metrics(entry["run_dir"])["test/metrics/mAP50(B)"] == .5
    report = runner.drift_report([{"case": "A3_CP2", **{f"{key}/mean": value for key, value in spec["reference"].items()}}], .002)
    assert report[0]["delta_ap_points"] == 0 and report[0]["tolerance"] == .002


def test_complexity_guard_and_provenance_contracts():
    spec = runner.CASE_REGISTRY["A3_CP2"]
    good = {"model/parameters": spec["expected_parameters"], "model/GFLOPs": spec["expected_gflops"], "model/GFLOPs_method": "thop"}
    runner.validate_complexity("A3_CP2", 42, good)
    for changed in ({**good, "model/parameters": 1}, {**good, "model/GFLOPs": 0.0}, {**good, "model/GFLOPs": spec["expected_gflops"] * 1.1}):
        try: runner.validate_complexity("A3_CP2", 42, changed)
        except RuntimeError: pass
        else: raise AssertionError("material complexity mismatch must fail")
    assert [runner.CASE_REGISTRY[case]["source_suite"] for case in runner.CASES] == [
        "levir_ftsc_h2_augmentation_suite:A3_CP2", "ftsc_edge_stability:ES1",
        "levir_ftsc_R_variants_augmentation_suite:A8_ES1_NEGCANVAS_R4", "ftsc_edge_stability:ES2", "ftsc_h1_h2_nextgen:E_H2",
    ]
    assert runner.parse_args(["--complexity-smoke"]).drift_tolerance == .002
    assert ".train(" not in Path(runner.__file__).read_text(encoding="utf-8")


def test_frozen_protocol_metadata_and_hashes_are_specific_and_stable():
    a3, es1, r4, es2, edge = (runner.CASE_REGISTRY[case]["protocol"] for case in runner.CASES)
    assert a3["augmentation_name"] == "CP2" and a3["augmentation"]["copies"] == 2
    assert a3["augmentation"]["p"] == .5 and a3["augmentation"]["placement"] == "random"
    assert es1["edge"]["alpha"] == .25 and es1["edge"]["schedule"] == "constant"
    assert "edge_stab_es1_scale025" in runner.CASE_REGISTRY["ES1"]["model_yaml"].name
    assert r4["augmentation"]["p"] == .30 and r4["augmentation"]["target_policy"] == "deficit"
    assert r4["augmentation"]["donor_policy"] == "larger" and r4["augmentation"]["blur_sigma"] == .5
    assert r4["augmentation"]["external_negative_bank"] is False
    assert es2["edge"] == {"name": "bounded Edge", "alpha": .25, "schedule": "ramp"}
    assert edge["edge"]["alpha"] == 1.0
    for protocol in (a3, es1, r4, es2, edge):
        assert protocol["split_seed"] == 42 and protocol["train_seeds"] == [42, 43, 44]
        assert (protocol["epochs"], protocol["imgsz"], protocol["batch"], protocol["patience"]) == (100, 512, 8, 20)
        assert protocol["mosaic"] == 0 and protocol["optimizer_generation"] == "historical AdamW via optimizer=auto"
        assert protocol["fitness_metric"] == "map50_95" and protocol["evaluation_nms_iou"] == .5
    assert runner.protocol_sha256(a3) == runner.protocol_sha256(dict(a3))
    changed = dict(a3); changed["epochs"] = 101
    assert runner.protocol_sha256(changed) != runner.protocol_sha256(a3)


def test_data_yaml_is_required_outside_smoke_and_never_prepares_a_split(tmp_path):
    manifest = _manifest(tmp_path)
    try: runner.main(["--manifest", str(manifest)])
    except ValueError as error: assert "--data-yaml is required" in str(error)
    else: raise AssertionError("normal preflight must require --data-yaml")
    try: runner.validate_fixed_data_yaml(tmp_path / "missing/levir_ship.yaml")
    except FileNotFoundError: pass
    else: raise AssertionError("missing dataset YAML must fail")
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "prepare_fixed_split(" not in source
