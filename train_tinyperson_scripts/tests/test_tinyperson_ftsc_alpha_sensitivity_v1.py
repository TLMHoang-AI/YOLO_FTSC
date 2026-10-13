from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import yaml

from train_tinyperson_scripts import train_all_tinyperson as workflow
from train_tinyperson_scripts import train_tinyperson_ftsc_alpha_sensitivity_v1 as alpha


def test_alpha_yamls_are_h2_plus_position_alpha_only():
    baseline = yaml.safe_load(alpha.H2.read_text(encoding="utf-8"))
    for case, expected in alpha.ALPHA.items():
        candidate = yaml.safe_load(alpha.YAMLS[case].read_text(encoding="utf-8"))
        assert candidate["ftsc"].get("position_alpha_mode", "fixed") == "fixed"
        assert candidate["ftsc"]["position_alpha"] == expected
        normalized = copy.deepcopy(candidate)
        normalized["ftsc"]["position_alpha"] = 6.0
        normalized["ftsc"].pop("position_alpha_mode", None)
        assert normalized == baseline


def test_only_two_fixed_alpha_cases_are_trainable():
    assert alpha.CASES == ("ALPHA4_rerun", "ALPHA8_rerun")
    assert set(alpha.ALPHA) == set(alpha.CASES)
    assert "ADAPTIVE" not in " ".join(alpha.CASES)


def test_tinyperson_native_geometry_and_merge_contract_are_preserved():
    prediction = inspect.getsource(workflow.predict_merged_test)
    assert "conf=0.001" in prediction
    assert "iou=0.5" in prediction
    assert "max_det=300" in prediction
    assert "nms_detections(items, iou_threshold=0.5)" in prediction
    args = alpha.parse_args(["--data-root", "/data"])
    assert args.imgsz == 640
    assert "sw640_sh512" in str(workflow.TRAIN_CORNER_JSON)


def test_merged_evaluator_has_separate_legacy_and_extended_sweeps():
    source = inspect.getsource(workflow.evaluate_tiny_benchmark)
    assert "np.linspace(0.50, 0.75, 6)" in source
    assert "np.linspace(0.50, 0.95, 10)" in source
    assert source.count("accumulated(") >= 3  # definition plus two independent calls
    assert '"test_merged/mAP50-75"' in source
    assert '"test_merged/mAP50-95"' in source


def test_window_metric_cannot_satisfy_merged_core_contract():
    metrics = {
        "test/metrics/mAP50(B)": 0.5,
        "test/metrics/mAP50-95(B)": 0.3,
        "test_merged/available": 1.0,
        "test_merged/evaluator_contract": "tinybenchmark_merged_original_iou50_95_v2",
        "model/parameters": 1,
        "model/GFLOPs": 1.0,
        "test_speed/inference_ms_per_image": 1.0,
    }
    with pytest.raises(RuntimeError, match="METRIC_INCOMPLETE"):
        alpha.validate_metrics(metrics)


def test_missing_merged_map5095_invalidates_completion():
    metrics = {key: 0.2 for key in alpha.CORE_METRICS}
    metrics.pop("test_merged/mAP50-95")
    metrics.update({"test_merged/available": 1.0, "test_merged/evaluator_contract": "tinybenchmark_merged_original_iou50_95_v2"})
    with pytest.raises(RuntimeError, match="METRIC_INCOMPLETE"):
        alpha.validate_metrics(metrics)


def test_json_roundtrip_is_canonical():
    assert alpha.canonical({"mapping": {0: "person"}}) == {"mapping": {"0": "person"}}


def test_commit_only_change_is_compatible_but_source_hash_change_is_not():
    recorded = {"case": "ALPHA4_rerun", "source": {"commit": "old", "clean": True, "sha256": {"runner": "same"}}}
    expected = {"case": "ALPHA4_rerun", "source": {"commit": "new", "clean": False, "sha256": {"runner": "same"}}}
    assert alpha.contracts_compatible(recorded, expected)
    expected["source"]["sha256"]["runner"] = "changed"
    assert not alpha.contracts_compatible(recorded, expected)


def test_historical_h2_is_never_a_trainable_case():
    assert "H2_MOSAIC" not in alpha.CASES
    assert "ALPHA6" not in alpha.CASES
