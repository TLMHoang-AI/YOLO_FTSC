"""Deterministic contract tests for the isolated historical Y4 compatibility runner."""
from pathlib import Path
import hashlib
import pytest
import yaml

import train_all_levir_yolov8n_p2_ftsc_legacy_compat as runner

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "yolo_code_legacy_y4_repro"


def test_defaults_and_explicit_historical_training_contract():
    args = runner.parse_args([])
    assert args.variants == [runner.VARIANT]
    assert args.seeds == [42]
    assert args.epochs == 100
    assert args.patience == 20
    assert args.fitness_metric == "map50_95"
    kwargs = runner.train_kwargs(args, Path("split.yaml"), 42, True)
    assert kwargs["optimizer"] == "auto"
    assert kwargs["cos_lr"] is False
    assert kwargs["lrf"] == 0.01
    assert kwargs["mosaic"] == 1.0 and kwargs["close_mosaic"] == 10
    assert kwargs["hsv_h"] == 0.015 and kwargs["hsv_s"] == 0.7 and kwargs["hsv_v"] == 0.4
    assert kwargs["fitness_metric"] == "map50_95"


def test_compat_yaml_changes_only_explicit_fixed_dfl_strength():
    canonical = yaml.safe_load((runner.CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml").read_text())
    compat = yaml.safe_load(runner.CONFIG.read_text())
    assert compat["ftsc"].pop("fixed_strengths") == {"dfl_distribution": 1.0}
    assert compat == canonical


def test_scheduler_and_auto_optimizer_resolve_historical_values():
    expected = {0: 1.0, 1: 0.9901, 5: 0.9505, 10: 0.901, 50: 0.505, 90: 0.109, 99: 0.0199, 100: 0.01}
    for epoch, value in expected.items():
        assert runner.legacy_scheduler_factor(epoch) == pytest.approx(value)
    assert runner.legacy_auto_optimizer(10000) == {"name": "AdamW", "lr": 0.002, "beta1_or_momentum": 0.9, "beta2": 0.999}
    assert runner.legacy_auto_optimizer(10001)["name"] == "MuSGD"


def test_source_preflight_rejects_broad_freeze_and_locks_main_contract():
    report = runner.source_preflight()
    assert report["fitness_metric"] == "map50_95"
    assert report["historical_commit"] == runner.HISTORICAL_COMMIT
    trainer = (ROOT / "models_related/ultralytics/ultralytics/engine/trainer.py").read_text()
    assert "isinstance(module, DFL)" in trainer
    assert 'always_freeze_names = [".dfl"]' not in trainer


def test_shared_assignment_and_metrics_sources_are_byte_identical_to_oracle():
    if not LEGACY.is_dir():
        pytest.skip("read-only legacy oracle is not mounted")
    for relative in (
        "models_related/ultralytics/ultralytics/utils/tal.py",
        "models_related/ultralytics/ultralytics/utils/metrics.py",
    ):
        current = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        legacy = hashlib.sha256((LEGACY / relative).read_bytes()).hexdigest()
        assert current == legacy, relative


def test_historical_reference_metrics_are_recorded_without_being_training_targets():
    values = list(runner.HISTORICAL_TEST_AP50.values())
    assert values == pytest.approx([0.7878924760784733, 0.7784960372913239, 0.8037524648720230])
    assert sum(values) / len(values) == pytest.approx(0.7900469927472734)


def test_model_preflight_locks_fixed_dfl_and_live_position_when_ultralytics_deps_exist():
    pytest.importorskip("cv2")
    import sys
    sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel(runner.CONFIG, verbose=False)
    report = runner.preflight_model(model)
    assert report["ftsc_policy"] == "f5"
    assert report["fixed_strengths"] == {"dfl_distribution": 1.0}
    assert report["dfl_strength_trainable"] is False
    assert report["position_strength_trainable"] is True
    assert report["dfl_strength_effective"] == pytest.approx(1.0)
