"""Causal-audit tests for learned versus intentionally fixed DFL evidence strength."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import train_all_levir_yolov8n_p2_ftsc_dfl_strength_audit as runner
import train_all_levir_yolov8n_p2_routing as workflow
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.modules import AnchorFreeFTSCCalibrator
from ultralytics.nn.tasks import DetectionModel


def _calibrator_forward(calibrator):
    anchor_points = torch.tensor([[5.0, 5.0], [6.0, 5.0]])
    target_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]])
    target_gt_idx = torch.tensor([[0, 0]])
    fg_mask = torch.tensor([[True, True]])
    pred_distri = torch.zeros(1, 2, 64)
    pred_distri[0, 0].reshape(4, 16)[:, 0] = 0.5
    return calibrator(
        anchor_points,
        target_bboxes,
        target_gt_idx,
        fg_mask,
        pred_distri,
        epoch=calibrator.warmup_epochs + calibrator.ramp_epochs,
    )


def _optimizer(model):
    trainer = BaseTrainer.__new__(BaseTrainer)
    return trainer.build_optimizer(model, name="SGD", lr=0.1, momentum=0.9, decay=0.01)


def _in_optimizer(parameter, optimizer):
    return any(candidate is parameter for group in optimizer.param_groups for candidate in group["params"])


def test_fixed1_yaml_is_canonical_y4_plus_one_explicit_fixed_strength():
    learned = yaml.safe_load(runner.VARIANTS["F_learned"].read_text(encoding="utf-8"))
    fixed = yaml.safe_load(runner.VARIANTS["F_fixed1"].read_text(encoding="utf-8"))

    assert fixed["ftsc"].pop("fixed_strengths") == {"dfl_distribution": 1.0}
    assert fixed == learned


def test_runner_protocol_and_model_preflight_lock_learned_vs_fixed_modes():
    args = runner.parse_args([])
    assert list(runner.VARIANTS) == ["F_learned", "F_fixed1"]
    assert args.seeds == [42, 43, 44]
    assert args.fitness_metric == "map50"
    assert runner.PROTOCOL_VERSION == "ftsc_corrected_ap50_v1"
    assert runner.SUITE_NAME == "dfl_strength_fixed1_audit"
    assert workflow.train_kwargs(args, Path("data.yaml"), seed=42, amp=False)["fitness_metric"] == "map50"

    learned = DetectionModel(runner.VARIANTS["F_learned"], verbose=False)
    fixed = DetectionModel(runner.VARIANTS["F_fixed1"], verbose=False)
    learned_report = runner.preflight_model(learned, "F_learned")
    fixed_report = runner.preflight_model(fixed, "F_fixed1")

    assert learned_report["dfl_strength_mode"] == "learned"
    assert learned_report["dfl_strength_trainable"] is True
    assert learned_report["dfl_strength_effective"] == pytest.approx(1.0)
    assert fixed_report["dfl_strength_mode"] == "fixed"
    assert fixed_report["dfl_strength_trainable"] is False
    assert fixed_report["dfl_strength_effective"] == 1.0
    assert learned_report["position_strength_trainable"] is True
    assert fixed_report["position_strength_trainable"] is True
    assert learned_report["fixed_dfl_parameters"] == fixed_report["fixed_dfl_parameters"]

    learned_metadata = runner._ftsc_metadata(learned.model[-1], learned_report)
    assert learned_metadata["ftsc/dfl_strength_mode"] == "learned"
    assert learned_metadata["ftsc/dfl_strength_trainable"] == 1.0
    assert learned_metadata["ftsc/final_strength_dfl_distribution"] == pytest.approx(1.0)
    assert "ftsc/dfl_strength_fixed_value" not in learned_metadata


def test_forward_is_equal_at_initialization_and_routing_is_identical():
    learned_model = DetectionModel(runner.VARIANTS["F_learned"], verbose=False)
    fixed_model = DetectionModel(runner.VARIANTS["F_fixed1"], verbose=False)
    learned = learned_model.model[-1].ftsc_calibrator
    fixed = fixed_model.model[-1].ftsc_calibrator

    routing = (
        "dfl_apply_cls",
        "dfl_apply_box",
        "dfl_apply_dfl",
        "apply_cls",
        "apply_box",
        "apply_dfl",
    )
    assert tuple(getattr(learned, name) for name in routing) == tuple(getattr(fixed, name) for name in routing)
    assert learned.providers["dfl_distribution"].detach is fixed.providers["dfl_distribution"].detach is True

    learned_output = _calibrator_forward(learned)
    fixed_output = _calibrator_forward(fixed)
    for key in ("cls", "box", "dfl", "regularization"):
        assert torch.allclose(learned_output[key], fixed_output[key], atol=0.0, rtol=0.0)


def test_optimizer_step_updates_learned_dfl_but_fixed1_stays_one_and_position_updates():
    learned_model = DetectionModel(runner.VARIANTS["F_learned"], verbose=False)
    fixed_model = DetectionModel(runner.VARIANTS["F_fixed1"], verbose=False)
    learned = learned_model.model[-1].ftsc_calibrator
    fixed = fixed_model.model[-1].ftsc_calibrator
    learned_optimizer = _optimizer(learned_model)
    fixed_optimizer = _optimizer(fixed_model)

    learned_dfl = learned.strength_logits["dfl_distribution"]
    learned_position = learned.strength_logits["position_gaussian"]
    fixed_position = fixed.strength_logits["position_gaussian"]
    assert "dfl_distribution" not in fixed.strength_logits
    assert _in_optimizer(learned_dfl, learned_optimizer)
    assert _in_optimizer(learned_position, learned_optimizer)
    assert _in_optimizer(fixed_position, fixed_optimizer)
    assert all("strength_logits.dfl_distribution" not in name for name, _ in fixed_model.named_parameters())

    learned_before = learned_dfl.detach().clone()
    learned_position_before = learned_position.detach().clone()
    fixed_position_before = fixed_position.detach().clone()
    learned_output = _calibrator_forward(learned)
    fixed_output = _calibrator_forward(fixed)
    learned_objective = (learned_output["cls"] * torch.tensor([1.0, 2.0])).sum() + learned_output["regularization"]
    fixed_objective = (fixed_output["cls"] * torch.tensor([1.0, 2.0])).sum() + fixed_output["regularization"]
    learned_objective.backward()
    fixed_objective.backward()

    assert learned_dfl.grad is not None and torch.isfinite(learned_dfl.grad) and learned_dfl.grad.abs() > 0
    assert learned_position.grad is not None and learned_position.grad.abs() > 0
    assert fixed_position.grad is not None and fixed_position.grad.abs() > 0
    learned_optimizer.step()
    fixed_optimizer.step()

    assert not torch.allclose(learned_dfl.detach(), learned_before)
    assert not torch.allclose(learned_position.detach(), learned_position_before)
    assert not torch.allclose(fixed_position.detach(), fixed_position_before)
    assert fixed.strength("dfl_distribution", fixed_position).item() == 1.0


def test_fixed_strength_validation_rejects_ambiguous_configs():
    with pytest.raises(ValueError, match="active evidence"):
        AnchorFreeFTSCCalibrator(
            {"policy": "f5", "evidence": ["position_gaussian"], "fixed_strengths": {"dfl_distribution": 1.0}},
            reg_max=16,
        )
    with pytest.raises(ValueError, match="0 < value"):
        AnchorFreeFTSCCalibrator(
            {"policy": "f5", "evidence": ["dfl_distribution"], "fixed_strengths": {"dfl_distribution": 0.0}},
            reg_max=16,
        )


def test_metadata_distinguishes_learned_and_fixed1(monkeypatch, tmp_path):
    fixed_network = DetectionModel(runner.VARIANTS["F_fixed1"], verbose=False)

    class FakeResult:
        results_dict = {"metrics/mAP50(B)": 0.8, "metrics/mAP50-95(B)": 0.3}
        box = SimpleNamespace(map75=0.1)

    class FakeYOLO:
        def __init__(self, _):
            self.model = fixed_network

        def val(self, **_):
            return FakeResult()

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    run_dir = tmp_path / "F_fixed1" / "seed_42"
    run_dir.mkdir(parents=True)
    args = runner.parse_args([])
    args._active_variant = "F_fixed1"
    args._active_seed = 42
    args.imgsz = 512
    args.batch_size = 1
    args.device = "cpu"
    args.workers = 0

    metrics = runner.evaluate(run_dir, Path("data.yaml"), args)

    assert metrics["protocol/version"] == "ftsc_corrected_ap50_v1"
    assert metrics["protocol/fitness_metric"] == "map50"
    assert metrics["protocol/best_checkpoint_metric"] == "map50"
    assert metrics["protocol/freeze_fix_active"] == 1.0
    assert metrics["suite/dfl_strength_fixed1_audit"] == 1.0
    assert metrics["ftsc/dfl_strength_mode"] == "fixed"
    assert metrics["ftsc/dfl_strength_trainable"] == 0.0
    assert metrics["ftsc/final_strength_dfl_distribution"] == 1.0
    assert metrics["ftsc/dfl_strength_fixed_value"] == 1.0
    assert metrics["ftsc/position_strength_trainable"] == 1.0
