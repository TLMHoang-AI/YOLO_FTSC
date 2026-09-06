"""Integration tests for FTSC freezing and checkpoint-fitness protocol."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.cfg import DEFAULT_CFG_DICT, check_cfg
from ultralytics.engine.trainer import BaseTrainer, fitness_from_metrics
from ultralytics.nn.modules import DFL
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import EarlyStopping


CANONICAL_Y4 = (
    Path(__file__).resolve().parents[2]
    / "models_config/yolov8/levir/yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml"
)


def _validation_metrics(ap50: float, map50_95: float) -> dict[str, float]:
    return {
        "metrics/precision(B)": 0.8,
        "metrics/recall(B)": 0.7,
        "metrics/mAP50(B)": ap50,
        "metrics/mAP50-95(B)": map50_95,
        "fitness": map50_95,
    }


def test_fitness_metric_default_and_validation():
    assert DEFAULT_CFG_DICT["fitness_metric"] == "map50_95"
    assert fitness_from_metrics(_validation_metrics(0.80, 0.35), "map50_95") == pytest.approx(0.35)
    assert fitness_from_metrics(_validation_metrics(0.80, 0.35), "map50") == pytest.approx(0.80)
    with pytest.raises(ValueError, match="fitness_metric=banana"):
        check_cfg({"fitness_metric": "banana"})


@pytest.mark.parametrize(
    ("fitness_metric", "expected_best_epoch"),
    [("map50", 0), ("map50_95", 1)],
)
def test_fitness_controls_best_checkpoint_ordering_and_early_stopping(fitness_metric, expected_best_epoch):
    epochs = [_validation_metrics(0.82, 0.31), _validation_metrics(0.80, 0.33)]
    fitness_values = [fitness_from_metrics(metrics, fitness_metric) for metrics in epochs]
    assert max(range(len(epochs)), key=fitness_values.__getitem__) == expected_best_epoch

    stopper = EarlyStopping(patience=1)
    stopper(1, fitness_values[0])
    stopped = stopper(2, fitness_values[1])
    assert stopper.best_epoch == expected_best_epoch + 1
    assert stopped is (expected_best_epoch == 0)


def test_trainer_validate_uses_configured_ap50_fitness():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.ema = None
    trainer.world_size = 1
    trainer.args = SimpleNamespace(fitness_metric="map50")
    trainer.best_fitness = None
    trainer.validator = lambda _: _validation_metrics(0.82, 0.31)

    metrics, fitness = trainer.validate()

    assert fitness == pytest.approx(0.82)
    assert trainer.best_fitness == pytest.approx(0.82)
    assert "fitness" not in metrics


def test_default_fitness_preserves_loss_fallback_for_validator_without_fitness():
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.ema = None
    trainer.world_size = 1
    trainer.args = SimpleNamespace(fitness_metric="map50_95")
    trainer.best_fitness = None
    trainer.loss = torch.tensor(0.25)
    trainer.validator = lambda _: {"metrics/accuracy_top1": 0.9}

    _, fitness = trainer.validate()

    assert fitness == pytest.approx(-0.25)


def test_trainer_freezes_only_fixed_dfl_projection_and_updates_ftsc_dfl_strength():
    model = DetectionModel(CANONICAL_Y4, verbose=False)
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(freeze=None)

    fixed_dfl_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, DFL)
        for parameter in module.parameters()
    }
    fixed_dfl_parameters = {
        name: parameter for name, parameter in model.named_parameters() if id(parameter) in fixed_dfl_parameter_ids
    }
    assert set(fixed_dfl_parameters) == {"model.29.dfl.conv.weight"}
    for parameter in fixed_dfl_parameters.values():
        parameter.requires_grad_(True)

    trainer._freeze_layers([])
    parameters = dict(model.named_parameters())
    dfl_strength_name = "model.29.ftsc_calibrator.strength_logits.dfl_distribution"
    position_strength_name = "model.29.ftsc_calibrator.strength_logits.position_gaussian"
    dfl_strength = parameters[dfl_strength_name]

    assert fixed_dfl_parameters["model.29.dfl.conv.weight"].requires_grad is False
    assert dfl_strength.requires_grad is True
    assert parameters[position_strength_name].requires_grad is True

    optimizer = trainer.build_optimizer(model, name="SGD", lr=0.1, momentum=0.9, decay=0.01)
    containing_groups = [
        group
        for group in optimizer.param_groups
        if any(parameter is dfl_strength for parameter in group["params"])
    ]
    assert len(containing_groups) == 1
    assert containing_groups[0]["weight_decay"] == 0.0
    assert containing_groups[0]["param_group"] == "bn"

    head = model.model[-1]
    calibrator = head.ftsc_calibrator
    anchor_points = torch.tensor([[5.0, 5.0], [6.0, 5.0]])
    target_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]])
    target_gt_idx = torch.tensor([[0, 0]])
    fg_mask = torch.tensor([[True, True]])
    pred_distri = torch.zeros(1, 2, 4 * head.reg_max)
    pred_distri[0, 0].reshape(4, head.reg_max)[:, 0] = 0.5

    weights = calibrator(
        anchor_points,
        target_bboxes,
        target_gt_idx,
        fg_mask,
        pred_distri,
        epoch=calibrator.warmup_epochs + calibrator.ramp_epochs,
    )
    objective = (weights["cls"] * torch.tensor([1.0, 2.0])).sum() + weights["regularization"]
    before = dfl_strength.detach().clone()
    objective.backward()

    assert dfl_strength.grad is not None
    assert torch.isfinite(dfl_strength.grad)
    assert dfl_strength.grad.abs() > 0
    optimizer.step()
    assert not torch.allclose(dfl_strength.detach(), before)
