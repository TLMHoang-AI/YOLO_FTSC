import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.nn.modules.head import Detect, DetectClsAttention, HVDecoupledDetect, P2NUDFLDetect, P2OffsetRegression
from ultralytics.utils.loss import v8DetectionLoss

import train_all_levir_yolov8n_p2_positive_confidence_rescue as train


def _criterion(gain=0.25, gamma=1.0):
    model = YOLO(train.workflow.VARIANTS["positive_confidence_rescue"]).model
    model.args = get_cfg(DEFAULT_CFG, {"positive_confidence_rescue_gain": gain, "positive_confidence_rescue_gamma": gamma})
    return v8DetectionLoss(model)


def test_formula_gradient_isolation_and_empty_foreground():
    criterion = _criterion(gamma=1.0)
    logits = torch.tensor([[[0.2], [-0.4], [0.7]]], requires_grad=True)
    targets = torch.tensor([[[0.2], [0.8], [0.0]]])
    fg = torch.tensor([[True, True, False]])
    actual, target, selected = criterion.positive_confidence_rescue_loss(logits, targets, fg)
    expected = ((1 - torch.tensor([0.2, 0.8])) * torch.nn.functional.softplus(-torch.tensor([0.2, -0.4]))).mean()
    assert torch.allclose(actual, expected)
    actual.backward()
    assert logits.grad[0, 0, 0] != 0 and logits.grad[0, 1, 0] != 0
    assert logits.grad[0, 2, 0] == 0
    empty, _, _ = criterion.positive_confidence_rescue_loss(logits, targets, torch.zeros_like(fg))
    assert empty.item() == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_formula_matches_reference_for_gamma_and_dtype(dtype):
    criterion = _criterion(gamma=2.0)
    logits = torch.tensor([[[0.3], [-0.7]]], dtype=dtype)
    targets = torch.tensor([[[0.1], [0.6]]], dtype=dtype)
    actual, _, _ = criterion.positive_confidence_rescue_loss(logits, targets, torch.ones(1, 2, dtype=torch.bool))
    reference = sum((1 - float(t)) ** 2 * torch.nn.functional.softplus(-z.float()) for z, t in zip(logits.flatten(), targets.flatten())) / 2
    assert torch.allclose(actual.float(), reference, atol=2e-3, rtol=2e-3)


def test_opt_in_defaults_and_incompatible_classification_losses():
    assert _criterion(gain=0).positive_confidence_rescue_gain == 0
    model = YOLO(train.workflow.VARIANTS["positive_confidence_rescue"]).model
    model.args = get_cfg(DEFAULT_CFG, {"positive_confidence_rescue_gain": 0.25, "vfl": True})
    with pytest.raises(ValueError, match="default TAL-BCE"):
        v8DetectionLoss(model)


def test_gain_zero_preserves_full_detection_loss_and_gradients():
    torch.manual_seed(7)
    model = YOLO(train.workflow.VARIANTS["positive_confidence_rescue"]).model.train()
    images = torch.randn(1, 3, 64, 64)
    batch = {
        "img": images,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    preds = model(images)
    default = _criterion(gain=0)
    explicit_zero = _criterion(gain=0, gamma=3.0)
    loss_a, items_a = default(preds, batch)
    grad_a = torch.autograd.grad(loss_a.sum(), preds["scores"], retain_graph=True)[0]
    loss_b, items_b = explicit_zero(preds, batch)
    grad_b = torch.autograd.grad(loss_b.sum(), preds["scores"], retain_graph=True)[0]
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(items_a, items_b)
    assert torch.equal(grad_a, grad_b)


def test_enabled_rescue_runs_full_detection_loss_and_reports_diagnostics():
    model = YOLO(train.workflow.VARIANTS["positive_confidence_rescue"]).model.train()
    criterion = _criterion()
    images = torch.randn(1, 3, 64, 64)
    preds = model(images)
    total, items = criterion(
        preds,
        {
            "img": images,
            "batch_idx": torch.tensor([0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        },
    )
    total.sum().backward()
    assert torch.isfinite(total).all() and torch.isfinite(items).all()
    metrics = criterion.positive_confidence_rescue_metrics
    assert metrics["rescue_positive_count"] > 0
    assert metrics["rescue_raw_loss"] > 0 and metrics["rescue_applied_loss"] > 0
    assert metrics["rescue_to_cls_loss_ratio"] > 0


def test_resolved_graph_and_runner_are_plain_p2_only():
    model = YOLO(train.workflow.VARIANTS["positive_confidence_rescue"]).model
    head = model.model[-1]
    assert type(head) is Detect and head.nl == 1 and head.stride.tolist() == [4.0]
    assert not any(isinstance(module, (DetectClsAttention, HVDecoupledDetect, P2NUDFLDetect, P2OffsetRegression)) for module in model.modules())
    names = {type(module).__name__ for module in model.modules()}
    assert not names.intersection({"CBAM", "P1DRR", "P2RegLocal"})
    args = train.parse_args(["--no-upload"])
    kwargs = train.train_kwargs(args, Path("data.yaml"), 42, True)
    assert args.seeds == [42] and args.split_seed == 42
    assert kwargs["positive_confidence_rescue_gain"] == 0.25
    assert kwargs["positive_confidence_rescue_gamma"] == 1.0
    assert kwargs["vfl"] is False and kwargs["cls_iou_target"] is False
