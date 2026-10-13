from __future__ import annotations

import copy

import pytest
import yaml

from train_levir_scripts import train_varroa_ftsc_alpha_sensitivity_v1 as alpha


def test_alpha_yamls_are_v1_ftsc_plus_position_alpha_only():
    baseline = yaml.safe_load(alpha.H2.read_text(encoding="utf-8"))
    for case, expected in alpha.ALPHA.items():
        candidate = yaml.safe_load(alpha.YAMLS[case].read_text(encoding="utf-8"))
        assert candidate["ftsc"].get("position_alpha_mode", "fixed") == "fixed"
        assert candidate["ftsc"]["position_alpha"] == expected
        normalized = copy.deepcopy(candidate)
        normalized["ftsc"]["position_alpha"] = 6.0
        normalized["ftsc"].pop("position_alpha_mode", None)
        assert normalized == baseline


def test_matrix_and_protocol_defaults_are_frozen():
    args = alpha.parse_args(["--data-root", "/data"])
    assert alpha.CASES == ("ALPHA4_rerun", "ALPHA8_rerun")
    assert args.seeds == [42, 43, 44]
    assert (args.imgsz, args.batch_size, args.epochs, args.patience) == (640, 4, 100, 20)


def test_optimizer_auto_resolves_historical_branch():
    args = alpha.parse_args(["--data-root", "/data"])
    report = alpha.optimizer_contract(args)
    assert report["requested"] == "auto"
    assert report["resolved_expected"] == "AdamW"
    assert report["iterations"] == 4400


def test_metric_contract_rejects_missing_native_small_ap():
    metrics = {key: 0.2 for key in alpha.CORE_METRICS}
    metrics.pop("test_size/AP50-Small")
    metrics.update({"evaluation/nms_iou": 0.7, "test_size/protocol": "native-test/iou=0.5"})
    with pytest.raises(RuntimeError):
        alpha.validate_metrics(metrics)


def test_standard_and_size_nms_contracts_remain_distinct():
    args = alpha.parse_args(["--data-root", "/data"])
    report = alpha.source_preflight(args)
    assert report["standard_nms_iou"] == 0.7
    assert report["size_bucket_nms_iou"] == 0.5


def test_no_amp_fallback_or_adaptive_case():
    source = alpha.Path(alpha.__file__).read_text(encoding="utf-8")
    assert "amp=False" not in source
    assert "archive_failed_amp" not in source
    assert "ALPHA_ADAPTIVE" not in source


def test_json_roundtrip_is_canonical():
    assert alpha.canonical({"mapping": {0: "mite"}}) == {"mapping": {"0": "mite"}}


def test_commit_only_change_is_compatible_but_scientific_change_is_not():
    recorded = {"alpha": 4.0, "source": {"commit": "old", "clean": True, "sha256": {"runner": "same"}}}
    expected = {"alpha": 4.0, "source": {"commit": "new", "clean": False, "sha256": {"runner": "same"}}}
    assert alpha.contracts_compatible(recorded, expected)
    expected["alpha"] = 8.0
    assert not alpha.contracts_compatible(recorded, expected)


def test_historical_alpha6_is_read_only():
    assert "ALPHA6" not in alpha.CASES
