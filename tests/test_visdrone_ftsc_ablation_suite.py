"""Focused no-training tests for the VisDrone FTSC V0--V6 suite."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from train_visdrone_scripts import train_visdrone_ftsc_ablation_suite as runner


def test_case_matrix_model_mapping_and_common_protocol():
    args = runner.parse_args([])
    assert runner.CASES == (
        "V0_H2_MOSAIC",
        "V1_H2_NOMOSAIC",
        "V2_ES1_MOSAIC",
        "V3_H2_CP2_MOSAIC",
        "V4_H2_R4_MOSAIC",
        "V5_ES1_R4_MOSAIC",
        "V6_H2_OACP_R2_MOSAIC",
    )
    assert args.seeds == [42, 43, 44]
    assert runner.model_config_for("V2_ES1_MOSAIC") == runner.ES1_MODEL_CONFIG
    assert runner.model_config_for("V5_ES1_R4_MOSAIC") == runner.ES1_MODEL_CONFIG
    assert all(
        runner.model_config_for(case) == runner.H2_MODEL_CONFIG
        for case in runner.CASES
        if case not in runner.ES1_CASES
    )
    assert runner.COMMON_TRAINING == {
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 8,
        "patience": 0,
        "optimizer": "auto",
        "deterministic": True,
        "amp": True,
        "cos_lr": False,
        "lrf": 0.01,
        "fitness_metric": "map50_95",
        "iou": 0.5,
    }


@pytest.mark.parametrize(
    ("source", "target"),
    (
        (runner.H2_SOURCE_CONFIG, runner.H2_MODEL_CONFIG),
        (runner.ES1_SOURCE_CONFIG, runner.ES1_MODEL_CONFIG),
    ),
)
def test_visdrone_yaml_differs_from_levir_only_by_nc(source: Path, target: Path):
    source_payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    target_payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert source_payload["nc"] == 1
    assert target_payload["nc"] == 10
    source_payload["nc"] = 10
    assert source_payload == target_payload


def test_ftsc_and_historical_es1_source_contract():
    report = runner.source_preflight()
    assert report["status"] == "PASS"
    assert report["ftsc"] == {
        "policy": "f5",
        "evidence": ["position_gaussian", "dfl_distribution"],
        "warmup_epochs": 5,
        "ramp_epochs": 10,
        "fixed_strengths": {"dfl_distribution": 1.0},
    }
    assert report["detect_inputs"] == [19, 22]
    assert report["es1"] == {
        "residual_scale": 0.25,
        "residual_schedule": "constant",
        "ramp_start_epoch": 5,
        "ramp_end_epoch": 15,
        "orientation_gate_mode": "learned",
        "hidden": 32,
    }


def test_mosaic_and_cp2_factor_isolation():
    v0 = runner.augmentation_for("V0_H2_MOSAIC")
    v1 = runner.augmentation_for("V1_H2_NOMOSAIC")
    v3 = runner.augmentation_for("V3_H2_CP2_MOSAIC")
    assert runner._dict_differences(v0, v1) == {"mosaic", "close_mosaic"}
    assert runner._dict_differences(v0, v3) == {"copy_paste_enabled"}
    assert (v3["copy_paste_mode"], v3["copy_paste_unit"], v3["copy_paste_copies"]) == (
        "single", "single", 2,
    )
    assert (v3["copy_paste_p"], v3["copy_paste_scale"], v3["copy_paste_padding"]) == (0.5, 1.0, 0.0)
    assert (v3["copy_paste_max_overlap"], v3["copy_paste_max_trials"], v3["copy_paste_blend"]) == (
        0.0, 30, "hard",
    )
    assert v3["copy_paste"] == 0.0


def test_r4_contract_is_shared_and_has_no_bank():
    h2 = runner.augmentation_for("V4_H2_R4_MOSAIC")
    es1 = runner.augmentation_for("V5_ES1_R4_MOSAIC")
    assert h2 == es1
    expected = {
        "negative_cp_p": 0.30,
        "negative_cp_target_policy": "deficit",
        "negative_cp_deficit_gamma": 0.5,
        "negative_cp_max_weight_ratio": 3.0,
        "negative_cp_donor_policy": "larger",
        "negative_cp_large_ratio_min": 1.5,
        "negative_cp_large_ratio_max": 2.5,
        "negative_cp_target_max_size": 20.0,
        "negative_cp_degradation": "weak_blur",
        "negative_cp_blur_sigma": 0.5,
        "copy_paste_copies": 1,
        "copy_paste_blend": "hard",
        "copy_paste_placement": "collision_aware",
        "copy_paste_max_overlap": 0.0,
    }
    assert {key: h2[key] for key in expected} == expected
    assert h2["mosaic"] == 1.0 and h2["close_mosaic"] == 10
    assert h2["mosaic_policy"] == "standard"
    assert h2["hard_negative_tile"] is False and h2["hard_negative_bank"] == ""
    assert "negative_bank" not in h2 and "negcp_bank_path" not in h2


def test_oacp_r2_is_corrected_single_pass():
    env = runner.environment_for("V6_H2_OACP_R2_MOSAIC")
    assert env == {
        "YOLO_CONTEXT_AUG": "oacp",
        "OACP_PROFILE": "r2",
        "OACP_VARIANT": "current",
        "OACP_PLACEMENT": "pre_transform",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
    }
    report = runner.transform_pipeline_preflight("V6_H2_OACP_R2_MOSAIC")
    assert report["oacp_instances"] == 1
    assert report["reported_order"] == [
        "OACP_R2", "StandardMosaic", "RandomPerspective", "photometric_transforms/flips",
    ]


def test_effective_transform_order_uses_real_pipeline():
    reports = {case: runner.transform_pipeline_preflight(case) for case in runner.CASES}
    assert reports["V0_H2_MOSAIC"]["reported_order"] == [
        "StandardMosaic", "RandomPerspective", "photometric_transforms/flips",
    ]
    assert reports["V1_H2_NOMOSAIC"]["reported_order"][0] == "Mosaic_disabled"
    assert reports["V3_H2_CP2_MOSAIC"]["top_level_types"][1] == "SmallObjectCopyPaste"
    assert reports["V4_H2_R4_MOSAIC"]["top_level_types"][1] == "NegativeCanvasCopyPaste"
    assert reports["V5_ES1_R4_MOSAIC"]["top_level_types"][1] == "NegativeCanvasCopyPaste"
    assert all(report["spatial_types"] == ["Mosaic", "RandomPerspective"] for report in reports.values())


@pytest.mark.parametrize("config", (runner.H2_MODEL_CONFIG, runner.ES1_MODEL_CONFIG))
def test_runtime_model_contract(config: Path):
    report = runner.model_preflight(config)
    assert report["nc"] == 10
    assert report["strides"] == [4.0, 8.0]
    assert report["detect_inputs"] == [19, 22]
    assert report["ftsc_policy"] == "f5"
    assert report["ftsc_evidence"] == ["position_gaussian", "dfl_distribution"]
    if config == runner.ES1_MODEL_CONFIG:
        assert report["es1_runtime_graph"]["detect_inputs"] == [19, 22]
        assert report["es1_runtime_graph"]["layer_20"] == "P2EdgeCueFusion"


def test_visdrone_evaluator_contract_and_nms():
    assert runner.EVALUATION_PROTOCOL == {
        "standard": "native ten-class VisDrone val and test-dev evaluation",
        "standard_metrics": ["val/AP50", "val/mAP50-95", "test/AP50", "test/mAP50-95"],
        "size_buckets": "evaluate_test.size_bucket_evaluator.evaluate_native_test_size_buckets",
        "nms_iou": 0.5,
        "class_agnostic_standard_eval": False,
    }
    source = (runner.ROOT / "train_visdrone_scripts/evaluate_visdrone.py").read_text(encoding="utf-8")
    assert "evaluate_native_test_size_buckets" in source
    assert 'for split in ("val", "test")' in source
    assert "iou=0.5" in source


def test_training_kwargs_freeze_all_normal_augmentations(tmp_path):
    kwargs = runner.train_kwargs("V0_H2_MOSAIC", tmp_path / "visdrone.yaml", 42, "cpu")
    for key, expected in runner.COMMON_TRAINING.items():
        assert kwargs[key] == expected
    assert {key: kwargs[key] for key in (
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
        "flipud", "fliplr", "mixup", "cutmix", "copy_paste", "rect",
    )} == {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0,
        "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5, "mixup": 0.0, "cutmix": 0.0,
        "copy_paste": 0.0, "rect": False,
    }


def test_manifest_records_reconstructable_provenance(tmp_path, monkeypatch):
    args = runner.parse_args([
        "--data-yaml", str(tmp_path / "visdrone.yaml"),
        "--pretrained", str(tmp_path / "yolov8n.pt"),
        "--device", "cpu",
    ])
    monkeypatch.setattr(runner, "git_sha", lambda: "deadbeef")
    validation = {"train": {"images": 6471}, "val": {"images": 548}, "test": {"images": 1610}}
    payload = runner._manifest(args, "V4_H2_R4_MOSAIC", 42, validation, "TEST")
    assert payload["case"] == "V4_H2_R4_MOSAIC" and payload["seed"] == 42
    assert payload["git_commit"] == "deadbeef"
    assert len(payload["model_config_sha256"]) == 64
    assert payload["visdrone_conversion_validation"] == validation
    assert payload["training_kwargs"]["fitness_metric"] == "map50_95"
    assert payload["transform_order"][2] == "NegativeCanvasCopyPaste"
    assert payload["ftsc_config"]["policy"] == "f5"
    assert payload["evaluation_protocol"]["nms_iou"] == 0.5
    json.dumps(payload)


def test_default_execution_is_read_only_and_confirm_run_is_guarded(monkeypatch, tmp_path):
    calls = []
    fake_report = {"dataset": {"status": "PENDING_DATA_PREPARATION"}}
    monkeypatch.setattr(runner, "preflight", lambda args: fake_report)
    monkeypatch.setattr(runner, "train_one", lambda *args, **kwargs: calls.append((args, kwargs)))
    runner.main(["--data-yaml", str(tmp_path / "missing.yaml")])
    assert calls == []
    with pytest.raises(RuntimeError, match="Canonical prepared VisDrone dataset is required"):
        runner.main(["--confirm-run", "--data-yaml", str(tmp_path / "missing.yaml")])
    assert calls == []


def test_expected_artifact_contract():
    assert runner.EXPECTED_ARTIFACTS == (
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "evaluation_metrics.json",
        "experiment_manifest.json",
    )
