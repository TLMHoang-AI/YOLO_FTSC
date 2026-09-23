"""No-training tests for the FTSC H2/ES1 A0--A8 augmentation suite."""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import train_levir_ftsc_augmentation_suite as runner

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO.parent / "duy_yolo_code/yolo_code"
ULTRA = REPO / "models_related/ultralytics"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(ULTRA) not in sys.path:
    sys.path.insert(0, str(ULTRA))


def _load_source_module(package: str, name: str, path: Path):
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(path.parent)]
        sys.modules[package] = module
    qualified = f"{package}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _empty_instances():
    from ultralytics.utils.instance import Instances

    return Instances(
        np.empty((0, 4), dtype=np.float32),
        segments=np.empty((0, 0, 2), dtype=np.float32),
        bbox_format="xyxy",
        normalized=False,
    )


def test_exactly_nine_cases_and_shared_training_protocol(tmp_path):
    args = runner.parse_args([])
    assert runner.CASES == (
        "A0_FTSC",
        "A1_OACP_R2",
        "A2_M5",
        "A3_CP2",
        "A4_OACP_R2_M5",
        "A5_NEGCANVAS_R1",
        "A6_NEGCANVAS_R4",
        "A7_ES1_NEGCANVAS_R1",
        "A8_ES1_NEGCANVAS_R4",
    )
    assert args.seeds == [42, 43, 44]
    assert (args.epochs, args.patience, args.imgsz, args.batch_size) == (100, 20, 512, 8)
    args.hard_negative_bank = tmp_path / "bank.json"
    invariant = None
    for case in runner.CASES:
        kwargs = runner.train_kwargs(args, case, Path("split.yaml"), 42)
        current = {key: kwargs[key] for key in (
            "data", "epochs", "imgsz", "batch", "device", "workers", "patience", "seed",
            "deterministic", "amp", "optimizer", "cos_lr", "lrf", "fitness_metric",
        )}
        invariant = invariant or current
        assert current == invariant
        assert kwargs["copy_paste"] == 0.0  # stock segmentation path always off


def test_case_augmentation_contracts(tmp_path):
    bank = tmp_path / "hard_negative_bank.json"
    cases = {case: runner.augmentation_for(case, bank) for case in runner.CASES}
    a0 = cases["A0_FTSC"]
    assert a0["mosaic"] == 0.0 and a0["close_mosaic"] == 0
    assert not a0["hard_negative_tile"] and not a0["copy_paste_enabled"]
    assert runner.environment_for("A0_FTSC") == {}

    a1 = cases["A1_OACP_R2"]
    env1 = runner.environment_for("A1_OACP_R2")
    assert a1["mosaic"] == 0.0 and not a1["copy_paste_enabled"]
    assert env1 == {
        "YOLO_CONTEXT_AUG": "oacp",
        "OACP_PROFILE": "r2",
        "OACP_VARIANT": "current",
        "YOLO_LEGACY_DOUBLE_OACP": "0",
        "OACP_PLACEMENT": "pre_transform",
    }

    a2 = cases["A2_M5"]
    assert a2["mosaic"] == 1.0 and a2["close_mosaic"] == 10
    assert a2["mosaic_policy"] == "hard_negative" and a2["hard_negative_tile"] is True
    assert a2["hardneg_mosaic_prob"] == 0.30 and not a2["copy_paste_enabled"]
    assert runner.environment_for("A2_M5") == {}

    a3 = cases["A3_CP2"]
    assert a3["copy_paste_enabled"] is True
    assert (a3["copy_paste_unit"], a3["copy_paste_copies"]) == ("single", 2)
    assert (a3["copy_paste_p"], a3["copy_paste_scale"], a3["copy_paste_padding"]) == (0.5, 1.0, 0.0)
    assert (a3["copy_paste_blend"], a3["copy_paste_placement"]) == ("hard", "random")
    assert a3["mosaic"] == 0.0 and runner.environment_for("A3_CP2") == {}

    a4 = cases["A4_OACP_R2_M5"]
    env4 = runner.environment_for("A4_OACP_R2_M5")
    assert a4["mosaic_policy"] == "hard_negative" and a4["hard_negative_tile"] is True
    assert a4["hardneg_mosaic_prob"] == 0.30 and not a4["copy_paste_enabled"]
    assert env4["OACP_PROFILE"] == "r2" and env4["YOLO_LEGACY_DOUBLE_OACP"] == "0"
    assert "OACP_PLACEMENT" not in env4  # exact source-combination precedent

    a5 = cases["A5_NEGCANVAS_R1"]
    assert a5["copy_paste_enabled"] is True and a5["copy_paste_mode"] == "negative_canvas"
    assert a5["negative_cp_p"] == 0.30
    assert a5["negative_cp_target_policy"] == "empirical"
    assert a5["negative_cp_donor_policy"] == "matched"
    assert a5["negative_cp_target_max_size"] == 20.0
    assert a5["negative_cp_degradation"] == "none"
    assert a5["negative_cp_matched_ratio_max"] == 1.5
    assert (a5["copy_paste_copies"], a5["copy_paste_placement"]) == (1, "collision_aware")
    assert (a5["copy_paste_scale"], a5["copy_paste_padding"]) == (1.0, 0.0)
    assert (a5["copy_paste_blend"], a5["copy_paste_max_overlap"]) == ("hard", 0.0)
    assert (a5["mosaic"], a5["hard_negative_tile"], a5["hard_negative_bank"]) == (0.0, False, "")
    assert runner.environment_for("A5_NEGCANVAS_R1") == {}

    a6 = cases["A6_NEGCANVAS_R4"]
    assert a6["copy_paste_enabled"] is True and a6["copy_paste_mode"] == "negative_canvas"
    assert a6["negative_cp_p"] == 0.30
    assert a6["negative_cp_target_policy"] == "deficit"
    assert (a6["negative_cp_deficit_gamma"], a6["negative_cp_max_weight_ratio"]) == (0.5, 3.0)
    assert a6["negative_cp_donor_policy"] == "larger"
    assert (a6["negative_cp_large_ratio_min"], a6["negative_cp_large_ratio_max"]) == (1.5, 2.5)
    assert a6["negative_cp_target_max_size"] == 20.0
    assert (a6["negative_cp_degradation"], a6["negative_cp_blur_sigma"]) == ("weak_blur", 0.5)
    assert (a6["copy_paste_copies"], a6["copy_paste_placement"]) == (1, "collision_aware")
    assert (a6["copy_paste_scale"], a6["copy_paste_padding"]) == (1.0, 0.0)
    assert (a6["copy_paste_blend"], a6["copy_paste_max_overlap"]) == ("hard", 0.0)
    assert (a6["mosaic"], a6["hard_negative_tile"], a6["hard_negative_bank"]) == (0.0, False, "")
    assert runner.environment_for("A6_NEGCANVAS_R4") == {}

    a7 = cases["A7_ES1_NEGCANVAS_R1"]
    a8 = cases["A8_ES1_NEGCANVAS_R4"]
    assert a7 == a5
    assert a8 == a6
    assert runner.environment_for("A7_ES1_NEGCANVAS_R1") == {}
    assert runner.environment_for("A8_ES1_NEGCANVAS_R4") == {}

    intended_copy_paste_switches = {
        "copy_paste_enabled",
        "copy_paste_mode",
        "copy_paste_copies",
        "copy_paste_placement",
    }
    for case in (a5, a6, a7, a8):
        for key, baseline_value in a0.items():
            if key not in intended_copy_paste_switches:
                assert case[key] == baseline_value, key


def test_hard_negative_bank_isolated_to_a2_and_a4():
    assert runner.hard_negative_bank_required(["A2_M5"])
    assert runner.hard_negative_bank_required(["A4_OACP_R2_M5"])
    assert runner.hard_negative_bank_required(["A0_FTSC", "A2_M5"])
    assert not runner.hard_negative_bank_required(["A5_NEGCANVAS_R1"])
    assert not runner.hard_negative_bank_required(["A6_NEGCANVAS_R4"])
    assert not runner.hard_negative_bank_required(["A5_NEGCANVAS_R1", "A6_NEGCANVAS_R4"])
    assert not runner.hard_negative_bank_required(["A7_ES1_NEGCANVAS_R1"])
    assert not runner.hard_negative_bank_required(["A8_ES1_NEGCANVAS_R4"])
    assert not runner.hard_negative_bank_required(["A7_ES1_NEGCANVAS_R1", "A8_ES1_NEGCANVAS_R4"])


def test_per_case_model_config_mapping_preserves_a0_a6():
    for case in runner.CASES[:7]:
        assert runner.model_config_for(case) == runner.H2_MODEL_CONFIG
    assert runner.model_config_for("A7_ES1_NEGCANVAS_R1") == runner.ES1_MODEL_CONFIG
    assert runner.model_config_for("A8_ES1_NEGCANVAS_R4") == runner.ES1_MODEL_CONFIG


def test_source_preflight_pins_h2_and_historical_es1_graph(tmp_path):
    args = runner.parse_args(
        [
            "--cases",
            "A5_NEGCANVAS_R1",
            "A6_NEGCANVAS_R4",
            "A7_ES1_NEGCANVAS_R1",
            "A8_ES1_NEGCANVAS_R4",
            "--dataset-root",
            str(tmp_path / "absent_data"),
        ]
    )
    report = runner.source_preflight(args)
    mapping = report["model_config_by_case"]
    assert mapping["A5_NEGCANVAS_R1"] == str(runner.H2_MODEL_CONFIG)
    assert mapping["A6_NEGCANVAS_R4"] == str(runner.H2_MODEL_CONFIG)
    assert mapping["A7_ES1_NEGCANVAS_R1"] == str(runner.ES1_MODEL_CONFIG)
    assert mapping["A8_ES1_NEGCANVAS_R4"] == str(runner.ES1_MODEL_CONFIG)
    assert report["es1_graph"] == {
        "yaml_sha256": runner.ES1_MODEL_CONFIG_SHA256,
        "identity_layer": 19,
        "edge_fusion_layer": 20,
        "edge_fusion_args": [32, 0.25, "constant", 5, 15, "learned"],
        "detect_inputs": [19, 22],
        "edge_stability": {
            "case": "ES1",
            "base_reference": "E_H2",
            "residual_scale": 0.25,
            "residual_schedule": "constant",
            "ramp_start_epoch": 5,
            "ramp_end_epoch": 15,
            "orientation_gate_mode": "learned",
            "hidden": 32,
            "description": "Fixed residual alpha=0.25; learned orientation gate.",
        },
    }
    assert report["hard_negative_bank"]["required"] is False


def test_r2_config_and_source_parity(monkeypatch):
    from ultralytics.utils.instance import Instances
    from project_ultralytics.context_augment import OACP, augmentation_config, protected_mask

    source_state = SOURCE / "project_ultralytics/oacp_state.py"
    source_context = SOURCE / "project_ultralytics/context_augment.py"
    _load_source_module("duy_project_ultralytics", "oacp_state", source_state)
    source = _load_source_module("duy_project_ultralytics", "context_augment", source_context)
    for key, value in runner.environment_for("A1_OACP_R2").items():
        monkeypatch.setenv(key, value)
    cfg = augmentation_config()
    assert cfg["oacp_probability"] == 0.40
    assert cfg["oacp_strength"] == [0.10, 0.25]
    assert cfg["oacp_resolution_scale"] == [0.80, 0.95]
    assert cfg["protected_expand"] == 3.0
    assert cfg["oacp_placement"] == "pre_transform" and cfg["legacy_double_oacp"] is False

    image = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)
    boxes = np.array([[0.50, 0.50, 0.10, 0.08]], dtype=np.float32)

    def labels():
        return {
            "img": image.copy(),
            "cls": np.array([[0]], dtype=np.float32),
            "instances": Instances(boxes.copy(), bbox_format="xywh", normalized=True),
        }

    random.seed(9182)
    expected = source.OACP(p=1.0)(labels())["img"]
    random.seed(9182)
    actual_labels = OACP(p=1.0)(labels())
    np.testing.assert_array_equal(actual_labels["img"], expected)
    absolute = np.array([[43.2, 44.16, 52.8, 51.84]], dtype=np.float32)
    protected, _ = protected_mask(absolute, 96, 96, 3.0)
    np.testing.assert_array_equal(actual_labels["img"][protected.astype(bool)], image[protected.astype(bool)])


class _CopyPasteDataset:
    def __init__(self, image_path: Path):
        self.im_files = [str(image_path)]
        self.labels = [{
            "bboxes": np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
            "cls": np.array([[0]], dtype=np.float32),
            "bbox_format": "xywh",
            "normalized": True,
            "shape": (32, 32),
        }]


def test_cp2_source_parity_and_detection_label_updates(tmp_path):
    from project_ultralytics.copy_paste import SmallObjectCopyPaste

    source = _load_source_module(
        "duy_copy_paste_package", "copy_paste", SOURCE / "project_ultralytics/copy_paste.py"
    )
    source_image = np.zeros((32, 32, 3), dtype=np.uint8)
    source_image[12:20, 12:20] = (17, 101, 233)
    image_path = tmp_path / "source.png"
    assert cv2.imwrite(str(image_path), source_image)
    dataset = _CopyPasteDataset(image_path)

    def target():
        return {
            "img": np.zeros((40, 40, 3), dtype=np.uint8),
            "cls": np.empty((0, 1), dtype=np.float32),
            "instances": _empty_instances(),
            "im_file": str(tmp_path / "target.png"),
        }

    kwargs = dict(dataset=dataset, p=1.0, unit="single", copies=2, placement="random", max_overlap=0.0,
                  padding=0.0, scale=1.0, blend="hard", max_trials=30,
                  allow_empty_target=True, allow_same_source=True)
    random.seed(401)
    expected = source.SmallObjectCopyPaste(**kwargs)(target())
    random.seed(401)
    actual = SmallObjectCopyPaste(**kwargs)(target())
    np.testing.assert_array_equal(actual["img"], expected["img"])
    np.testing.assert_array_equal(actual["instances"].bboxes, expected["instances"].bboxes)
    np.testing.assert_array_equal(actual["cls"], expected["cls"])
    assert len(actual["instances"]) == 2 and actual["cls"].shape == (2, 1)


def test_m5_helper_source_parity():
    from project_ultralytics.mosaic_policy import simulate_visible_boxes

    source = _load_source_module(
        "duy_mosaic_package", "mosaic_policy", SOURCE / "project_ultralytics/mosaic_policy.py"
    )
    metadata = [
        {"shape": (64 + i * 3, 80 - i * 2), "bboxes": np.array([[0.5, 0.5, 0.2, 0.1]], dtype=np.float32)}
        for i in range(4)
    ]
    expected = source.simulate_visible_boxes(metadata, [0, 1, 2, 3], 64, 51, 72)
    actual = simulate_visible_boxes(metadata, [0, 1, 2, 3], 64, 51, 72)
    np.testing.assert_array_equal(actual, expected)


class _MosaicDataset:
    cache = None
    buffer = []
    data = {}
    use_keypoints = False

    def __init__(self, image_paths: list[Path]):
        self.im_files = [str(path) for path in image_paths]
        self.labels = [
            {"shape": (32, 32), "bboxes": np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)}
            for _ in image_paths
        ]

    def __len__(self):
        return len(self.im_files)

    def get_image_and_label(self, index: int):
        from ultralytics.utils.instance import Instances

        image = cv2.imread(self.im_files[index])
        return {
            "img": image,
            "cls": np.array([[0]], dtype=np.float32),
            "instances": Instances(
                np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
                segments=np.empty((1, 0, 2), dtype=np.float32),
                bbox_format="xywh",
                normalized=True,
            ),
            "im_file": self.im_files[index],
            "ori_shape": image.shape[:2],
            "resized_shape": image.shape[:2],
        }


def test_m5_hard_negative_crop_has_no_gt_labels(tmp_path):
    from ultralytics.data.augment import HardNegativeMosaic
    from ultralytics.utils.instance import Instances

    image_path = tmp_path / "negative.png"
    assert cv2.imwrite(str(image_path), np.full((40, 50, 3), 99, dtype=np.uint8))
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps([{"image": str(image_path), "crop_xyxy": [5, 6, 35, 30], "fp_conf": 0.8}]))
    dataset = _MosaicDataset([image_path] * 4)
    mosaic = HardNegativeMosaic(
        dataset, imgsz=32, hard_negative_tile=True, hardneg_mosaic_prob=1.0, hard_negative_bank=bank_path
    )
    template = {
        "img": np.zeros((32, 32, 3), dtype=np.uint8),
        "cls": np.array([[0]], dtype=np.float32),
        "instances": Instances(
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            segments=np.empty((1, 0, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        ),
        "im_file": "positive.png",
    }
    random.seed(7)
    negative = mosaic._prepare_hard_negative(template)
    assert negative is not None
    assert len(negative["instances"]) == 0 and negative["cls"].shape[0] == 0
    assert negative["im_file"] == str(image_path)

    # Exercise donor replacement plus geometry/concatenation: four positive
    # source tiles can produce at most three labels after one empty M5 donor.
    random.seed(19)
    output = mosaic(dataset.get_image_and_label(0))
    assert len(output["instances"]) == len(output["cls"]) <= 3
    assert np.isfinite(output["instances"].bboxes).all()
    assert (output["instances"].bboxes >= 0).all() and (output["instances"].bboxes <= 64).all()


def test_parser_defaults_accept_ported_keys_and_pipeline_order(tmp_path):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.augment import Compose, HardNegativeMosaic, v8_transforms
    from project_ultralytics.context_augment import OACP
    from project_ultralytics.copy_paste import SmallObjectCopyPaste
    from project_ultralytics.negative_canvas_copy_paste import NegativeCanvasCopyPaste

    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    (tmp_path / "bank.json").write_text("[]\n", encoding="utf-8")
    dataset = _MosaicDataset([image_path] * 4)
    for case in runner.CASES:
        settings = runner.augmentation_for(case, tmp_path / "bank.json")
        hyp = get_cfg(overrides=settings)
        with runner.configured_environment(case):
            transforms = v8_transforms(dataset, 32, hyp)
        top = transforms.transforms
        if case == "A1_OACP_R2":
            assert isinstance(top[0], OACP) and isinstance(top[1], Compose)
        elif case == "A4_OACP_R2_M5":
            assert isinstance(top[0], Compose) and isinstance(top[0].transforms[0], HardNegativeMosaic)
            assert isinstance(top[1], OACP)
        elif case == "A2_M5":
            assert isinstance(top[0].transforms[0], HardNegativeMosaic)
        elif case == "A3_CP2":
            assert isinstance(top[0], Compose) and isinstance(top[1], SmallObjectCopyPaste)
            assert all(type(transform).__name__ != "CopyPaste" for transform in top[0].transforms)
        elif case in runner.NEGATIVE_CANVAS_CASES:
            assert isinstance(top[0], Compose)
            assert type(top[0].transforms[-1]).__name__ == "RandomPerspective"
            assert type(top[1]) is NegativeCanvasCopyPaste
            assert type(top[2]).__name__ == "MixUp"
            assert type(top[3]).__name__ == "CutMix"
            assert all(type(transform).__name__ != "CopyPaste" for transform in top[0].transforms)


def test_ftsc_h2_model_instantiates_with_p2_p3_unchanged():
    report = runner.model_preflight()
    assert report["head"] == "Detect"
    assert report["strides"] == [4.0, 8.0]
    assert report["ftsc_policy"] == "f5"
    assert report["ftsc_evidence"] == ["position_gaussian", "dfl_distribution"]


def test_es1_model_instantiates_with_historical_runtime_graph_unchanged():
    report = runner.model_preflight(runner.ES1_MODEL_CONFIG)
    assert report["head"] == "Detect"
    assert report["strides"] == [4.0, 8.0]
    assert report["ftsc_policy"] == "f5"
    assert report["ftsc_evidence"] == ["position_gaussian", "dfl_distribution"]
    assert report["es1_runtime_graph"] == {
        "identity_layer_19": "Identity",
        "edge_layer_20": "P2EdgeCueFusion",
        "edge_from": -1,
        "residual_scale": 0.25,
        "residual_schedule": "constant",
        "ramp_start_epoch": 5,
        "ramp_end_epoch": 15,
        "orientation_gate_mode": "learned",
        "hidden": 32,
        "detect_inputs": [19, 22],
    }


def test_default_path_fails_closed_without_training(monkeypatch):
    monkeypatch.setattr(runner, "model_preflight", lambda *_args: {"ok": True})
    called = []
    monkeypatch.setattr(runner, "train_one", lambda *args, **kwargs: called.append((args, kwargs)))
    runner.main([])
    assert called == []


def test_a5_a6_preflight_needs_neither_bank_nor_implicit_data_preparation(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runner, "model_preflight", lambda *_args: {"ok": True})
    prepared = []
    monkeypatch.setattr(runner.workflow, "prepare_fixed_split", lambda _args: prepared.append(True))
    runner.main(
        [
            "--cases",
            "A5_NEGCANVAS_R1",
            "A6_NEGCANVAS_R4",
            "--dataset-root",
            str(tmp_path / "absent_data"),
        ]
    )
    output = capsys.readouterr().out
    assert '"required": false' in output
    assert '"status": "PENDING_DATA_PREPARATION"' in output
    assert prepared == []


def test_a7_a8_preflight_uses_both_models_without_bank_or_data_preparation(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_model_preflight(model_config=runner.MODEL_CONFIG):
        calls.append(model_config)
        return {"model_config": str(model_config)}

    monkeypatch.setattr(runner, "model_preflight", fake_model_preflight)
    prepared = []
    monkeypatch.setattr(runner.workflow, "prepare_fixed_split", lambda _args: prepared.append(True))
    runner.main(
        [
            "--cases",
            "A7_ES1_NEGCANVAS_R1",
            "A8_ES1_NEGCANVAS_R4",
            "--dataset-root",
            str(tmp_path / "absent_data"),
        ]
    )
    output = capsys.readouterr().out
    assert calls == [runner.H2_MODEL_CONFIG, runner.ES1_MODEL_CONFIG]
    assert '"required": false' in output
    assert '"status": "PENDING_DATA_PREPARATION"' in output
    assert prepared == []


def test_manifest_records_the_model_selected_for_each_case(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "git_sha", lambda: "test-commit")
    args = runner.parse_args([])
    for case, expected in (
        ("A6_NEGCANVAS_R4", runner.H2_MODEL_CONFIG),
        ("A8_ES1_NEGCANVAS_R4", runner.ES1_MODEL_CONFIG),
    ):
        run_dir = tmp_path / case
        runner.write_manifest(run_dir, args, case, 42, Path("split.yaml"))
        manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
        assert manifest["model_config"] == str(expected)
        assert manifest["model_config_sha256"] == runner.sha256(expected)


def test_train_one_constructs_the_case_selected_model_without_training(monkeypatch, tmp_path):
    import ultralytics

    constructed = []

    class FakeYOLO:
        def __init__(self, model_config, task):
            constructed.append((Path(model_config), task))

        def load(self, _pretrained, smart_transfer):
            assert smart_transfer is True

        def train(self, **kwargs):
            run_dir = Path(kwargs["project"]) / kwargs["name"]
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            (run_dir / "weights" / "best.pt").touch()
            (run_dir / "weights" / "last.pt").touch()
            (run_dir / "results.csv").touch()

    monkeypatch.setattr(runner.workflow, "local_ultralytics", lambda: None)
    monkeypatch.setattr(runner.workflow, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(runner, "git_sha", lambda: "test-commit")
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    args = runner.parse_args(["--project", str(tmp_path / "runs")])
    args.project = args.project.resolve()
    args.pretrained = "local-test-checkpoint.pt"

    runner.train_one(args, "A6_NEGCANVAS_R4", 42, Path("split.yaml"))
    runner.train_one(args, "A8_ES1_NEGCANVAS_R4", 42, Path("split.yaml"))
    assert constructed == [
        (runner.H2_MODEL_CONFIG, "detect"),
        (runner.ES1_MODEL_CONFIG, "detect"),
    ]


def test_a5_a6_dataset_eligibility_uses_canonical_counts(tmp_path):
    dataset = tmp_path / "levir_ship_yolo_seed42"
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    assert cv2.imwrite(str(image_dir / "positive.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    assert cv2.imwrite(str(image_dir / "negative.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    (label_dir / "positive.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (label_dir / "negative.txt").write_text("", encoding="utf-8")
    (dataset / "levir_ship.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/train\ntest: images/train\nnames: [ship]\n",
        encoding="utf-8",
    )
    args = runner.parse_args(
        [
            "--cases",
            "A5_NEGCANVAS_R1",
            "A6_NEGCANVAS_R4",
            "--dataset-root",
            str(tmp_path),
        ]
    )
    report = runner.dataset_preflight(args)
    assert report["status"] == "READY"
    assert report["training_images"] == 2
    assert report["original_positive_images"] == 1
    assert report["original_negative_images"] == 1
    assert report["donor_objects"] == 1
    assert report["eligible_small_targets_le_20px"] == 1

    (label_dir / "negative.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero original-negative"):
        runner.dataset_preflight(args)

    (label_dir / "negative.txt").write_text("", encoding="utf-8")
    (label_dir / "positive.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero eligible <=20 px"):
        runner.dataset_preflight(args)
