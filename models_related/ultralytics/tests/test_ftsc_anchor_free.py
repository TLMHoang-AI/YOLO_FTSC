"""Focused tests for assignment-preserving anchor-free FTSC."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import AnchorFreeFTSCCalibrator, PositionGaussianEvidence
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import BboxLoss


def _toy_assignment():
    anchor_points = torch.tensor([[5.0, 5.0], [6.0, 5.0], [25.0, 25.0]])
    target_bboxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]]
    )
    target_gt_idx = torch.tensor([[0, 0, 1]])
    fg_mask = torch.tensor([[True, True, True]])
    pred_distri = torch.zeros(1, 3, 64)
    return anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri


def test_position_gaussian_returns_log_evidence_for_positives_only():
    anchor_points, target_bboxes, _, fg_mask, _ = _toy_assignment()
    evidence = PositionGaussianEvidence(alpha=1.0)(anchor_points, target_bboxes, fg_mask)
    assert evidence.shape == (3,)
    assert float(evidence[0]) == pytest.approx(0.0)
    assert evidence[1] < evidence[0]
    assert float(evidence[2]) == pytest.approx(0.0)


def test_e4_centers_each_gt_and_single_positive_is_identity():
    inputs = _toy_assignment()
    calibrator = AnchorFreeFTSCCalibrator(
        {
            "policy": "e4",
            "evidence": ["position_gaussian"],
            "position_alpha": 1.0,
            "log_clip": 0.35,
            "per_gt_norm": True,
        },
        reg_max=16,
    )
    output = calibrator(*inputs, epoch=0)
    assert output["cls"].shape == (3,)
    assert float(output["cls"][2]) == pytest.approx(1.0)
    assert float(output["cls"][:2].mean()) == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(output["cls"], output["box"])
    assert torch.equal(output["box"], output["dfl"])


def test_f5_is_identity_during_warmup_then_updates_bounded_strength():
    inputs = _toy_assignment()
    calibrator = AnchorFreeFTSCCalibrator(
        {
            "policy": "f5",
            "evidence": ["position_gaussian"],
            "position_alpha": 1.0,
            "warmup_epochs": 5,
            "ramp_epochs": 10,
            "strength_init": 1.0,
            "strength_max": 2.0,
        },
        reg_max=16,
    )
    warmup = calibrator(*inputs, epoch=4)
    assert torch.equal(warmup["cls"], torch.ones_like(warmup["cls"]))
    ramp = calibrator(*inputs, epoch=5)
    assert not torch.equal(ramp["cls"], torch.ones_like(ramp["cls"]))
    objective = (ramp["cls"] * torch.tensor([1.0, 2.0, 1.0])).sum() + ramp["regularization"]
    objective.backward()
    parameter = calibrator.strength_logits["position_gaussian"]
    assert parameter.grad is not None and torch.isfinite(parameter.grad)
    assert 0.0 <= float(calibrator.strength("position_gaussian", ramp["cls"])) <= 2.0
    assert calibrator.residual_fraction(14) == pytest.approx(1.0)


def test_detached_dfl_evidence_does_not_backpropagate_into_logits():
    anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri = _toy_assignment()
    pred_distri.requires_grad_(True)
    calibrator = AnchorFreeFTSCCalibrator(
        {
            "policy": "e4",
            "evidence": ["dfl_distribution"],
            "dfl_detach": True,
            "dfl_apply_cls": True,
            "dfl_apply_box": False,
            "dfl_apply_dfl": False,
        },
        reg_max=16,
    )
    output = calibrator(anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri)
    assert not output["cls"].requires_grad
    assert torch.equal(output["box"], torch.ones_like(output["box"]))
    assert torch.equal(output["dfl"], torch.ones_like(output["dfl"]))


def test_bbox_loss_accepts_independent_box_and_dfl_weights():
    criterion = BboxLoss(reg_max=1)
    pred_dist = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
    pred_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    anchor_points = torch.tensor([[0.5, 0.5]])
    target_bboxes = torch.tensor([[[0.0, 0.0, 2.0, 2.0]]])
    target_scores = torch.ones(1, 1, 1)
    fg_mask = torch.tensor([[True]])
    imgsz = torch.tensor([8.0, 8.0])
    stride = torch.ones(1, 1)
    base_iou, base_dfl = criterion(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores.sum(), fg_mask, imgsz, stride
    )
    weighted_iou, weighted_dfl = criterion(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores.sum(),
        fg_mask,
        imgsz,
        stride,
        box_weights=torch.tensor([2.0]),
        dfl_weights=torch.tensor([3.0]),
    )
    assert float(weighted_iou) == pytest.approx(2.0 * float(base_iou))
    assert float(weighted_dfl) == pytest.approx(3.0 * float(base_dfl))


def test_f5_model_yaml_owns_strength_before_criterion_and_optimizer():
    config = (
        Path(__file__).resolve().parents[2]
        / "models_config/yolov8/levir/yolov8n_p2_levir_ftsc_af_y2_f5_position.yaml"
    )
    model = DetectionModel(config, verbose=False)
    calibrator = model.model[-1].ftsc_calibrator
    assert calibrator is not None and calibrator.policy == "f5"
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    assert id(calibrator.strength_logits["position_gaussian"]) in parameter_ids


def test_f5_warmup_loss_is_exactly_baseline_identity():
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    baseline = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_y0_baseline.yaml", verbose=False)
    f5 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_af_y2_f5_position.yaml", verbose=False)
    missing, unexpected = f5.load_state_dict(baseline.state_dict(), strict=False)
    assert missing == ["model.29.ftsc_calibrator.strength_logits.position_gaussian"]
    assert not unexpected
    for model in (baseline, f5):
        model.args = IterableSimpleNamespace(**model.args)
        model.train()
    torch.manual_seed(123)
    batch = {
        "img": torch.rand(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    baseline_loss, baseline_items = baseline(batch)
    f5_loss, f5_items = f5(batch)
    assert torch.equal(f5_loss, baseline_loss)
    assert torch.equal(f5_items, baseline_items)
    assert f5.criterion.ftsc_metrics["ftsc_rho"] == 0.0
