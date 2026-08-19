"""Focused tests for assignment-preserving anchor-free FTSC."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import (
    AnchorFreeFTSCCalibrator,
    HierarchicalBackgroundSmoothing,
    PositionGaussianEvidence,
)
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


def test_task_specific_position_strengths_use_matched_mean_regularization():
    inputs = _toy_assignment()
    calibrator = AnchorFreeFTSCCalibrator(
        {
            "policy": "f5",
            "evidence": ["position_gaussian"],
            "position_task_specific_strength": True,
            "position_alpha": 1.0,
            "warmup_epochs": 0,
            "ramp_epochs": 1,
            "strength_init": 1.0,
            "strength_max": 2.0,
            "strength_reg_weight": 1.0,
        },
        reg_max=16,
    )
    expected_keys = {
        "position_gaussian_cls",
        "position_gaussian_box",
        "position_gaussian_dfl",
    }
    assert set(calibrator.strength_logits) == expected_keys
    target_strength = 1.5
    target_logit = torch.logit(torch.tensor(target_strength / calibrator.strength_max))
    with torch.no_grad():
        for parameter in calibrator.strength_logits.values():
            parameter.copy_(target_logit)
    output = calibrator(*inputs, epoch=0)
    assert float(output["regularization"]) == pytest.approx((target_strength - 1.0) ** 2)
    for task in calibrator.TASK_NAMES:
        assert calibrator.last_metrics[f"ftsc_strength_position_gaussian_{task}"] == pytest.approx(target_strength)


def test_within_gt_dfl_shuffle_is_deterministic_and_preserves_groups():
    anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri = _toy_assignment()
    logits = pred_distri.view(1, 3, 4, 16)
    logits[0, 0, :, 0] = 10.0  # low entropy
    logits[0, 1] = 0.0  # uniform/high entropy
    logits[0, 2, :, 0] = 10.0  # singleton group remains identity
    common = {
        "policy": "e4",
        "evidence": ["dfl_distribution"],
        "dfl_detach": True,
        "dfl_apply_cls": True,
        "dfl_apply_box": False,
        "dfl_apply_dfl": False,
        "per_gt_norm": True,
    }
    baseline = AnchorFreeFTSCCalibrator(common, reg_max=16)(
        anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri, epoch=0
    )
    shuffled_calibrator = AnchorFreeFTSCCalibrator(
        {**common, "dfl_shuffle_within_gt": True, "dfl_shuffle_seed": 123}, reg_max=16
    )
    shuffled = shuffled_calibrator(
        anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri, epoch=0
    )
    assert torch.allclose(shuffled["cls"][:2], baseline["cls"][:2].flip(0))
    assert float(shuffled["cls"][2]) == pytest.approx(1.0)
    assert torch.equal(shuffled["box"], torch.ones_like(shuffled["box"]))
    assert torch.equal(shuffled["dfl"], torch.ones_like(shuffled["dfl"]))
    assert shuffled_calibrator.last_metrics["ftsc_dfl_shuffle_eligible_fraction"] == pytest.approx(2 / 3)
    assert shuffled_calibrator.last_metrics["ftsc_dfl_shuffle_moved_fraction"] == pytest.approx(2 / 3)
    assert int(shuffled_calibrator.dfl_shuffle_step.item()) == 1
    assert {"dfl_shuffle_seed", "dfl_shuffle_step"} <= set(shuffled_calibrator.state_dict())

    repeated_calibrator = AnchorFreeFTSCCalibrator(
        {**common, "dfl_shuffle_within_gt": True, "dfl_shuffle_seed": 123}, reg_max=16
    )
    repeated = repeated_calibrator(anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri, epoch=0)
    assert torch.equal(repeated["cls"], shuffled["cls"])

    saved_state = {key: value.clone() for key, value in shuffled_calibrator.state_dict().items()}
    expected_next = shuffled_calibrator(
        anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri, epoch=1
    )
    resumed_calibrator = AnchorFreeFTSCCalibrator(
        {**common, "dfl_shuffle_within_gt": True, "dfl_shuffle_seed": 999}, reg_max=16
    )
    resumed_calibrator.load_state_dict(saved_state)
    resumed_next = resumed_calibrator(anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri, epoch=1)
    assert torch.equal(resumed_next["cls"], expected_next["cls"])
    assert int(resumed_calibrator.dfl_shuffle_step.item()) == 2


def test_within_gt_dfl_shuffle_rejects_non_detached_evidence():
    with pytest.raises(ValueError, match="requires detached DFL evidence"):
        AnchorFreeFTSCCalibrator(
            {
                "policy": "f5",
                "evidence": ["dfl_distribution"],
                "dfl_detach": False,
                "dfl_shuffle_within_gt": True,
            },
            reg_max=16,
        )


def test_gt_mass_rebalance_is_cls_only_bounded_and_warmup_safe():
    inputs = _toy_assignment()  # GT support counts are two and one.
    common = {
        "policy": "f5",
        "evidence": ["position_gaussian", "dfl_distribution"],
        "position_alpha": 1.0,
        "dfl_detach": True,
        "dfl_apply_cls": True,
        "dfl_apply_box": False,
        "dfl_apply_dfl": False,
        "warmup_epochs": 5,
        "ramp_epochs": 1,
        "per_gt_norm": True,
    }
    baseline = AnchorFreeFTSCCalibrator(common, reg_max=16)
    rebalance = AnchorFreeFTSCCalibrator(
        {
            **common,
            "gt_mass_rebalance_cls": True,
            "gt_mass_power": 0.5,
            "gt_mass_min_factor": 0.75,
            "gt_mass_max_factor": 1.25,
        },
        reg_max=16,
    )
    rebalance.load_state_dict(baseline.state_dict(), strict=False)

    baseline_warmup = baseline(*inputs, epoch=4)
    rebalance_warmup = rebalance(*inputs, epoch=4)
    assert torch.equal(rebalance_warmup["cls"], baseline_warmup["cls"])
    assert torch.equal(rebalance_warmup["box"], baseline_warmup["box"])
    assert torch.equal(rebalance_warmup["dfl"], baseline_warmup["dfl"])

    baseline_full = baseline(*inputs, epoch=5)
    rebalance_full = rebalance(*inputs, epoch=5)
    assert torch.equal(rebalance_full["box"], baseline_full["box"])
    assert torch.equal(rebalance_full["dfl"], baseline_full["dfl"])
    support_factor = rebalance_full["cls"] / baseline_full["cls"]
    assert float(support_factor.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(support_factor[2]) > float(support_factor[:2].mean())
    assert support_factor.min() > 0
    assert rebalance.last_metrics["ftsc_gt_group_count"] == 2.0
    assert rebalance.last_metrics["ftsc_mean_positives_per_gt"] == pytest.approx(1.5)
    assert rebalance.last_metrics["ftsc_single_positive_gt_fraction"] == pytest.approx(0.5)


def test_gt_mass_shuffle_is_deterministic_resumable_null_control():
    inputs = _toy_assignment()
    config = {
        "policy": "f5",
        "evidence": ["position_gaussian"],
        "position_alpha": 1.0,
        "warmup_epochs": 0,
        "ramp_epochs": 1,
        "gt_mass_rebalance_cls": True,
        "gt_mass_shuffle": True,
        "gt_mass_shuffle_seed": 123,
    }
    first = AnchorFreeFTSCCalibrator(config, reg_max=16)
    first_output = first(*inputs, epoch=0)
    assert int(first.gt_mass_shuffle_step.item()) == 1
    assert first.last_metrics["ftsc_gt_mass_shuffle_group_fraction"] == 1.0
    assert {"gt_mass_shuffle_seed", "gt_mass_shuffle_step"} <= set(first.state_dict())

    repeated = AnchorFreeFTSCCalibrator(config, reg_max=16)
    repeated_output = repeated(*inputs, epoch=0)
    assert torch.equal(repeated_output["cls"], first_output["cls"])

    saved_state = {key: value.clone() for key, value in first.state_dict().items()}
    expected_next = first(*inputs, epoch=1)
    resumed = AnchorFreeFTSCCalibrator({**config, "gt_mass_shuffle_seed": 999}, reg_max=16)
    resumed.load_state_dict(saved_state)
    resumed_next = resumed(*inputs, epoch=1)
    assert torch.equal(resumed_next["cls"], expected_next["cls"])
    assert int(resumed.gt_mass_shuffle_step.item()) == 2


def test_gt_mass_shuffle_requires_rebalance_branch():
    with pytest.raises(ValueError, match="requires gt_mass_rebalance_cls=true"):
        AnchorFreeFTSCCalibrator(
            {"policy": "f5", "evidence": ["position_gaussian"], "gt_mass_shuffle": True},
            reg_max=16,
        )


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


@pytest.mark.parametrize(
    ("config_name", "expected_evidence", "expected_strength_keys", "task_specific", "shuffle"),
    [
        (
            "yolov8n_p2_levir_ftsc_af_v11_a1_dflcls_only.yaml",
            ("dfl_distribution",),
            {"dfl_distribution"},
            False,
            False,
        ),
        (
            "yolov8n_p2_levir_ftsc_af_v11_a2_position_task_strengths.yaml",
            ("position_gaussian", "dfl_distribution"),
            {"position_gaussian_cls", "position_gaussian_box", "position_gaussian_dfl", "dfl_distribution"},
            True,
            False,
        ),
        (
            "yolov8n_p2_levir_ftsc_af_v11_a3_dflcls_shuffled_within_gt.yaml",
            ("position_gaussian", "dfl_distribution"),
            {"position_gaussian", "dfl_distribution"},
            False,
            True,
        ),
    ],
)
def test_v11_yaml_builds_expected_ablation(
    config_name, expected_evidence, expected_strength_keys, task_specific, shuffle
):
    config = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir" / config_name
    model = DetectionModel(config, verbose=False)
    calibrator = model.model[-1].ftsc_calibrator
    assert calibrator is not None and calibrator.policy == "f5"
    assert calibrator.evidence_names == expected_evidence
    assert set(calibrator.strength_logits) == expected_strength_keys
    assert calibrator.position_task_specific_strength is task_specific
    assert calibrator.dfl_shuffle_within_gt is shuffle
    if config_name.endswith("a1_dflcls_only.yaml"):
        assert calibrator.position_tasks == ()


@pytest.mark.parametrize(
    ("config_name", "rebalance", "shuffle"),
    [
        ("yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml", False, False),
        ("yolov8n_p2_levir_ftsc_v2_exp_s1_gt_mass_rebalance_cls.yaml", True, False),
        ("yolov8n_p2_levir_ftsc_v2_exp_s2_gt_mass_rebalance_cls_shuffled.yaml", True, True),
    ],
)
def test_v2_gt_mass_yaml_has_one_requested_factor_and_no_extra_parameters(config_name, rebalance, shuffle):
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    control = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml", verbose=False)
    model = DetectionModel(config_root / config_name, verbose=False)
    calibrator = model.model[-1].ftsc_calibrator
    assert calibrator is not None and calibrator.policy == "f5"
    assert calibrator.evidence_names == ("position_gaussian", "dfl_distribution")
    assert calibrator.gt_mass_rebalance_cls is rebalance
    assert calibrator.gt_mass_shuffle is shuffle
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in control.parameters()
    )


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


def test_hbs_mask_rasterizes_gt_union_and_handles_empty_images():
    feature = torch.randn(2, 8, 8, 8)
    batch_idx = torch.tensor([0.0, 0.0])
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.125, 0.125, 0.25, 0.25]])
    mask = HierarchicalBackgroundSmoothing.foreground_mask(feature, batch_idx, boxes)
    assert mask.shape == (2, 1, 8, 8)
    assert torch.equal(mask[0, 0, 2:6, 2:6], torch.ones(4, 4))
    assert torch.equal(mask[0, 0, :2, :2], torch.ones(2, 2))
    assert mask[1].sum() == 0
    empty = HierarchicalBackgroundSmoothing.foreground_mask(
        feature, torch.empty(0), torch.empty(0, 4)
    )
    assert torch.equal(empty, torch.zeros_like(empty))
    smoother = HierarchicalBackgroundSmoothing(8, stride=4)
    empty_output, empty_forward_mask = smoother(feature, torch.empty(0), torch.empty(0, 4))
    assert torch.equal(empty_forward_mask, empty)
    assert empty_output.shape == feature.shape and torch.isfinite(empty_output).all()


def test_hbs_implements_set_kernel_schedule_and_backpropagates():
    smoothers = [HierarchicalBackgroundSmoothing(8, stride, reduction=4) for stride in (4, 8, 16, 32)]
    assert [module.kernel_size for module in smoothers] == [3, 5, 5, 7]
    feature = torch.randn(1, 8, 12, 12, requires_grad=True)
    output, mask = smoothers[0](
        feature,
        torch.tensor([0.0]),
        torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    )
    assert output.shape == feature.shape and mask.shape == (1, 1, 12, 12)
    output.square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert smoothers[0].reduce.weight.grad is not None
    assert smoothers[0].expand.weight.grad is not None


@pytest.mark.parametrize(
    ("config_name", "expected_ftsc", "expected_hbs"),
    [
        ("yolov8n_p2_levir_ftsc_y0_baseline.yaml", False, False),
        ("yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml", True, False),
        ("yolov8n_p2_levir_ftsc_v2s_a2_y0_hbs.yaml", False, True),
        ("yolov8n_p2_levir_ftsc_v2s_a3_y4_hbs.yaml", True, True),
    ],
)
def test_v2s_factorial_yaml_has_only_the_requested_factors(config_name, expected_ftsc, expected_hbs):
    config = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir" / config_name
    model = DetectionModel(config, verbose=False)
    head = model.model[-1]
    assert (head.ftsc_calibrator is not None) is expected_ftsc
    assert head.hbs_enabled is expected_hbs
    if expected_hbs:
        assert [module.kernel_size for module in head.hbs_smoothers] == [3, 5, 5, 7]
        assert all(module.reduction == 4 for module in head.hbs_smoothers)
        parameter_ids = {id(parameter) for parameter in model.parameters()}
        assert id(head.hbs_smoothers[0].reduce.weight) in parameter_ids


def test_hbs_auxiliary_loss_uses_plain_detection_criterion_and_receives_gradients():
    config = (
        Path(__file__).resolve().parents[2]
        / "models_config/yolov8/levir/yolov8n_p2_levir_ftsc_v2s_a3_y4_hbs.yaml"
    )
    model = DetectionModel(config, verbose=False)
    model.args = IterableSimpleNamespace(**model.args)
    model.train()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, items = model(batch)
    loss.sum().backward()
    head = model.model[-1]
    assert items.numel() >= 3 and torch.isfinite(items).all()
    assert head.hbs_auxiliary_calls == 1
    assert model._hbs_auxiliary_criterion.ftsc_calibrator is None
    assert all(
        torch.equal(clean, auxiliary)
        for clean, auxiliary in zip(
            model.criterion.last_assignment,
            model._hbs_auxiliary_criterion.last_assignment,
            strict=True,
        )
    )
    assert head.hbs_smoothers[0].reduce.weight.grad is not None
    assert head.hbs_smoothers[0].expand.weight.grad is not None
    assert {"hbs_aux_box_loss", "hbs_aux_cls_loss", "hbs_aux_dfl_loss"} <= set(model.mechanism_metrics)


def test_hbs_is_not_executed_by_inference_and_can_be_stripped_exactly():
    config = (
        Path(__file__).resolve().parents[2]
        / "models_config/yolov8/levir/yolov8n_p2_levir_ftsc_v2s_a2_y0_hbs.yaml"
    )
    model = DetectionModel(config, verbose=False).eval()
    head = model.model[-1]
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        before = model(image)[0]
    assert head.hbs_auxiliary_calls == 0
    hbs_parameters = sum(parameter.numel() for module in head.hbs_smoothers for parameter in module.parameters())
    assert hbs_parameters > 0
    head.strip_hbs()
    with torch.no_grad():
        after = model(image)[0]
    assert torch.equal(after, before)
    assert not head.hbs_enabled and len(head.hbs_smoothers) == 0
