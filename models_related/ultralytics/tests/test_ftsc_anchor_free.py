"""Focused tests for assignment-preserving anchor-free FTSC."""

from pathlib import Path

import pytest
import torch

from ultralytics.cfg import DEFAULT_CFG_DICT
from ultralytics.nn.modules import (
    AnchorFreeFTSCCalibrator,
    DCFLDGMMQualityEvidence,
    DEIMPositiveMALLoss,
    DFLUncertaintyMinimization,
    FCOSCenternessEvidence,
    HierarchicalBackgroundSmoothing,
    PositionGaussianEvidence,
)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import BboxLoss, GHMCClassificationLoss, v8DetectionLoss


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


def test_fcos_centerness_is_center_high_edge_low_and_per_gt_safe():
    anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri = _toy_assignment()
    evidence = FCOSCenternessEvidence()(anchor_points, target_bboxes, fg_mask)
    assert evidence.shape == (3,)
    assert float(evidence[0]) == pytest.approx(0.0)
    assert evidence[1] < evidence[0]
    assert float(evidence[2]) == pytest.approx(0.0)
    calibrator = AnchorFreeFTSCCalibrator(
        {"policy": "e4", "evidence": ["fcos_centerness"], "per_gt_norm": True}, reg_max=16
    )
    weights = calibrator(anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri)
    assert float(weights["cls"][:2].mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(weights["cls"][2]) == pytest.approx(1.0)


def test_ghmc_is_finite_and_rebalances_dense_gradient_bins():
    criterion = GHMCClassificationLoss(bins=4, momentum=0.0)
    logits = torch.tensor([[[8.0], [0.0], [-8.0], [-1.0]]], requires_grad=True)
    targets = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    # Match the normalization used by YOLO's standard dense BCE branch.
    loss = criterion(logits, targets, normalizer=targets.sum().clamp_min(1.0))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert criterion.acc_sum.sum() == pytest.approx(float(logits.numel()))


def test_r3_ghmc_yaml_routes_config_to_detect_not_model_args():
    config = (
        Path(__file__).resolve().parents[2]
        / "models_config/yolov8/levir/yolov8n_p2_levir_ftsc_r3_ghmc.yaml"
    )
    model = DetectionModel(config, verbose=False)
    head = model.model[-1]
    assert head.ftsc_calibrator is None
    assert head.ghm_enabled is True
    assert head.ghm_config == {"enabled": True, "bins": 10, "momentum": 0.75}


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


def test_r4_positive_mal_uses_detached_iou_and_preserves_nonpositive_bce():
    mal = DEIMPositiveMALLoss(gamma=1.0)
    high_confidence_logit = torch.logit(torch.tensor([0.9, 0.9]))
    quality = torch.tensor([0.1, 0.9], requires_grad=True)
    positive_loss = mal(high_confidence_logit, quality)
    assert positive_loss[1] < positive_loss[0]

    logits = torch.tensor([[[2.0, -2.0], [-1.0, 1.0]]], requires_grad=True)
    targets = torch.tensor([[[0.7, 0.0], [0.0, 0.0]]])
    positive_mask = targets > 0
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    replaced, selected = mal.replace_dense(base, logits, positive_mask, quality[:1])
    assert torch.equal(replaced[~positive_mask], base[~positive_mask])
    selected.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert quality.grad is None


def test_r4_mal_is_finite_at_probability_and_quality_extremes():
    mal = DEIMPositiveMALLoss(gamma=1.0)
    logits = torch.tensor([-50.0, -50.0, 50.0, 50.0], requires_grad=True)
    quality = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = mal(logits, quality)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_r5_dcfl_dgmm_is_per_gt_ordered_singleton_safe_and_detached():
    provider = DCFLDGMMQualityEvidence(min_group_size=2, min_variance=1e-6, evidence_clip=3.0)
    cls_quality = torch.tensor([0.1, 0.9, 0.8, 0.9, 0.5], requires_grad=True)
    loc_quality = torch.tensor([0.1, 0.9, 0.8, 0.9, 0.5], requires_grad=True)
    group_ids = torch.tensor([0, 0, 1, 1, 2])
    evidence = provider(cls_quality, loc_quality, group_ids, group_count=3)
    assert evidence[1] > evidence[0]
    assert evidence[3] > evidence[2]
    assert evidence[1] > 0 and evidence[2] < 0  # same absolute 0.8, different object-relative role
    assert float(evidence[4]) == pytest.approx(0.0)
    assert not evidence.requires_grad
    assert cls_quality.grad is None and loc_quality.grad is None


def test_r5_dcfl_dgmm_degenerate_group_is_finite_zero_evidence():
    provider = DCFLDGMMQualityEvidence(min_group_size=4)
    quality = torch.full((4,), 0.5)
    evidence = provider(quality, quality, torch.zeros(4, dtype=torch.long), group_count=1)
    assert torch.isfinite(evidence).all()
    assert torch.equal(evidence, torch.zeros_like(evidence))
    assert provider.last_metrics["ftsc_dcfl_fallback_fraction"] == pytest.approx(1.0)


def test_r5_calibrator_keeps_assignment_tensors_and_dfl_cls_provider():
    anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri = _toy_assignment()
    original_mask, original_gt_idx = fg_mask.clone(), target_gt_idx.clone()
    calibrator = AnchorFreeFTSCCalibrator(
        {
            "policy": "f5",
            "evidence": ["dcfl_dgmm_quality", "dfl_distribution"],
            "dcfl_min_group_size": 2,
            "dfl_detach": True,
            "dfl_apply_cls": True,
            "dfl_apply_box": False,
            "dfl_apply_dfl": False,
            "warmup_epochs": 0,
            "ramp_epochs": 1,
        },
        reg_max=16,
    )
    cls_quality = torch.tensor([0.1, 0.9, 0.5], requires_grad=True)
    loc_quality = torch.tensor([0.2, 0.8, 0.5], requires_grad=True)
    output = calibrator(
        anchor_points,
        target_bboxes,
        target_gt_idx,
        fg_mask,
        pred_distri,
        classification_quality=cls_quality,
        localization_quality=loc_quality,
    )
    assert torch.equal(fg_mask, original_mask) and torch.equal(target_gt_idx, original_gt_idx)
    assert calibrator.providers["dfl_distribution"].detach
    assert output["cls"].shape == (3,) and torch.isfinite(output["cls"]).all()
    assert cls_quality.grad is None and loc_quality.grad is None


def test_r5_and_canonical_y4_produce_identical_tal_assignment_at_same_model_state():
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    y4 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml", verbose=False)
    r5 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_r5_dcfl_dgmm.yaml", verbose=False)
    missing, unexpected = r5.load_state_dict(y4.state_dict(), strict=False)
    assert len(missing) == 1 and missing[0].endswith("strength_logits.dcfl_dgmm_quality")
    assert len(unexpected) == 1 and unexpected[0].endswith("strength_logits.position_gaussian")
    for model in (y4, r5):
        model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        model.train()
        model.criterion = model.init_criterion()
        model.criterion.capture_assignment = True
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    y4(batch)
    r5(batch)
    assert all(
        torch.equal(canonical, replacement)
        for canonical, replacement in zip(y4.criterion.last_assignment, r5.criterion.last_assignment, strict=True)
    )


def test_r6_um_entropy_is_ordered_and_backpropagates_to_dfl_logits():
    regularizer = DFLUncertaintyMinimization(reg_max=16, normalized=True)
    fg_mask = torch.tensor([[True]])
    uniform = torch.zeros(1, 1, 64)
    sharp = torch.zeros(1, 1, 64)
    sharp.view(1, 1, 4, 16)[..., 0] = 12.0
    assert regularizer(uniform, fg_mask) > regularizer(sharp, fg_mask)

    logits = torch.randn(1, 1, 64, requires_grad=True)
    loss = regularizer(logits, fg_mask)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_r8_quality_guard_is_detached_and_scales_positive_dfl_gradients():
    regularizer = DFLUncertaintyMinimization(
        reg_max=16,
        normalized=True,
        quality_guard=True,
        quality_gamma=1.0,
    )
    base_logits = torch.randn(1, 1, 64)
    logits = base_logits.repeat(1, 2, 1).clone().requires_grad_(True)
    fg_mask = torch.tensor([[True, True]])
    realized_iou = torch.tensor([0.2, 0.8], requires_grad=True)
    loss = regularizer(logits, fg_mask, localization_quality=realized_iou)
    loss.backward()

    gradient_norms = logits.grad.reshape(2, -1).norm(dim=1)
    assert gradient_norms[1] > gradient_norms[0]
    assert float(gradient_norms[1] / gradient_norms[0]) == pytest.approx(4.0, rel=1e-4)
    assert realized_iou.grad is None
    assert torch.equal(regularizer.last_quality, realized_iou.detach())
    assert torch.equal(regularizer.last_guard, realized_iou.detach())


def test_r8_quality_guard_disabled_reproduces_r6_um_exactly():
    logits = torch.randn(1, 2, 64)
    fg_mask = torch.tensor([[True, True]])
    quality = torch.tensor([0.1, 0.9])
    default_r6 = DFLUncertaintyMinimization(reg_max=16, normalized=True)
    explicit_disabled = DFLUncertaintyMinimization(
        reg_max=16,
        normalized=True,
        quality_guard=False,
        quality_gamma=1.0,
    )
    expected = default_r6(logits, fg_mask)
    actual = explicit_disabled(logits, fg_mask, localization_quality=quality)
    assert torch.equal(actual, expected)
    assert explicit_disabled.last_quality is None and explicit_disabled.last_guard is None


def test_r8_quality_guard_handles_empty_singleton_and_near_zero_mass():
    regularizer = DFLUncertaintyMinimization(reg_max=16, quality_guard=True, quality_gamma=1.0)

    empty_logits = torch.randn(1, 1, 64, requires_grad=True)
    empty_loss = regularizer(empty_logits, torch.tensor([[False]]), localization_quality=torch.empty(0))
    empty_loss.backward()
    assert float(empty_loss) == pytest.approx(0.0)
    assert torch.equal(empty_logits.grad, torch.zeros_like(empty_logits))

    singleton_logits = torch.randn(1, 1, 64)
    fg_mask = torch.tensor([[True]])
    unguarded = DFLUncertaintyMinimization(reg_max=16)(singleton_logits, fg_mask)
    guarded_one = regularizer(singleton_logits, fg_mask, localization_quality=torch.tensor([1.0]))
    assert torch.allclose(guarded_one, unguarded)

    zero_logits = singleton_logits.clone().requires_grad_(True)
    guarded_zero = regularizer(zero_logits, fg_mask, localization_quality=torch.tensor([0.0]))
    guarded_zero.backward()
    assert float(guarded_zero) == pytest.approx(0.0)
    assert torch.equal(zero_logits.grad, torch.zeros_like(zero_logits))

    near_zero_logits = singleton_logits.repeat(1, 2, 1).clone().requires_grad_(True)
    near_zero_loss = regularizer(
        near_zero_logits,
        torch.tensor([[True, True]]),
        localization_quality=torch.tensor([1e-12, 2e-12]),
    )
    near_zero_loss.backward()
    assert torch.isfinite(near_zero_loss) and torch.isfinite(near_zero_logits.grad).all()


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


def test_r7_applies_s1_after_r5_and_preserves_assignment_and_localization_weights():
    anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri = _toy_assignment()
    original_assignment = tuple(value.clone() for value in (target_bboxes, target_gt_idx, fg_mask))
    r5_config = {
        "policy": "f5",
        "evidence": ["dcfl_dgmm_quality", "dfl_distribution"],
        "dcfl_min_group_size": 2,
        "dfl_detach": True,
        "dfl_apply_cls": True,
        "dfl_apply_box": False,
        "dfl_apply_dfl": False,
        "warmup_epochs": 0,
        "ramp_epochs": 1,
        "per_gt_norm": True,
    }
    r5 = AnchorFreeFTSCCalibrator(r5_config, reg_max=16)
    r7 = AnchorFreeFTSCCalibrator(
        {
            **r5_config,
            "gt_mass_rebalance_cls": True,
            "gt_mass_power": 0.5,
            "gt_mass_min_factor": 0.75,
            "gt_mass_max_factor": 1.25,
            "gt_mass_shuffle": False,
        },
        reg_max=16,
    )
    r7.load_state_dict(r5.state_dict())
    classification_quality = torch.tensor([0.1, 0.9, 0.5])
    localization_quality = torch.tensor([0.2, 0.8, 0.5])
    kwargs = {
        "epoch": 0,
        "classification_quality": classification_quality,
        "localization_quality": localization_quality,
    }
    r5_output = r5(anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri, **kwargs)
    r7_output = r7(anchor_points, target_bboxes, target_gt_idx, fg_mask, pred_distri, **kwargs)

    assert all(
        torch.equal(current, original)
        for current, original in zip((target_bboxes, target_gt_idx, fg_mask), original_assignment, strict=True)
    )
    assert torch.equal(r7_output["box"], r5_output["box"])
    assert torch.equal(r7_output["dfl"], r5_output["dfl"])
    support_factor = r7_output["cls"] / r5_output["cls"]
    assert float(support_factor.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(support_factor[2]) > float(support_factor[:2].mean())
    assert r7.last_metrics["ftsc_gt_support_count_mean"] == pytest.approx(1.5)
    assert r7.last_metrics["ftsc_corr_support_count_gt_mass_factor"] < 0
    assert r7.last_metrics["ftsc_effective_positive_cls_multiplier_mean"] == pytest.approx(
        float(r7_output["cls"].mean())
    )
    dgmm_strength = r7.strength("dcfl_dgmm_quality", r7_output["cls"])
    assert r7.strength_logits["dcfl_dgmm_quality"].requires_grad
    assert 0 < float(dgmm_strength) < r7.strength_max


def test_r7_effective_weight_changes_target_class_only():
    cls_weights = torch.ones(1, 2, 3)
    target_scores = torch.tensor([[[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    fg_mask = torch.tensor([[True, False]])
    effective_r7_weight = torch.tensor([1.75])
    weighted = v8DetectionLoss._apply_ftsc_positive_cls_weights(
        cls_weights,
        target_scores,
        fg_mask,
        effective_r7_weight,
    )
    assert float(weighted[0, 0, 0]) == pytest.approx(1.75)
    assert torch.equal(weighted[0, 0, 1:], torch.ones(2))
    assert torch.equal(weighted[0, 1], torch.ones(3))


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


@pytest.mark.parametrize(
    ("config_name", "evidence"),
    [
        ("yolov8n_p2_levir_ftsc_r4_deim_mal.yaml", ("position_gaussian",)),
        ("yolov8n_p2_levir_ftsc_r5_dcfl_dgmm.yaml", ("dcfl_dgmm_quality", "dfl_distribution")),
        ("yolov8n_p2_levir_ftsc_r6_ugs_um.yaml", ("position_gaussian",)),
    ],
)
def test_r4_r5_r6_yaml_builds_one_paper_replacement_only(config_name, evidence):
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    model = DetectionModel(config_root / config_name, verbose=False)
    head = model.model[-1]
    calibrator = head.ftsc_calibrator
    assert calibrator is not None and calibrator.policy == "f5"
    assert calibrator.evidence_names == evidence
    assert not getattr(head, "quality_head", False)
    if "r4_" in config_name:
        assert calibrator.classification_replacement == "deim_mal"
        assert "dfl_distribution" not in calibrator.providers
        assert calibrator.um_regularizer is None
    elif "r5_" in config_name:
        assert "position_gaussian" not in calibrator.providers
        assert calibrator.providers["dfl_distribution"].detach
        assert calibrator.classification_replacement == "bce" and calibrator.um_regularizer is None
    else:
        assert calibrator.um_enabled and calibrator.um_regularizer is not None
        assert "dfl_distribution" not in calibrator.providers
        assert calibrator.classification_replacement == "bce"


def test_r7_yaml_is_exact_r5_plus_ordered_s1_without_new_parameters():
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    r5 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_r5_dcfl_dgmm.yaml", verbose=False)
    r7 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_r7_dcfl_dgmm_gt_mass.yaml", verbose=False)
    head = r7.model[-1]
    calibrator = head.ftsc_calibrator
    assert calibrator.evidence_names == ("dcfl_dgmm_quality", "dfl_distribution")
    assert "position_gaussian" not in calibrator.providers
    assert calibrator.providers["dfl_distribution"].detach
    assert calibrator.dfl_apply_cls and not calibrator.dfl_apply_box and not calibrator.dfl_apply_dfl
    assert calibrator.classification_replacement == "bce" and not calibrator.um_enabled
    assert calibrator.gt_mass_rebalance_cls and not calibrator.gt_mass_shuffle
    assert calibrator.gt_mass_power == pytest.approx(0.5)
    assert calibrator.gt_mass_min_factor == pytest.approx(0.75)
    assert calibrator.gt_mass_max_factor == pytest.approx(1.25)
    assert not getattr(head, "quality_head", False)
    assert set(r7.state_dict()) == set(r5.state_dict())
    assert sum(parameter.numel() for parameter in r7.parameters()) == sum(
        parameter.numel() for parameter in r5.parameters()
    )


def test_r8_yaml_is_exact_r6_plus_parameter_free_quality_guard():
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    r6 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_r6_ugs_um.yaml", verbose=False)
    r8 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_r8_ugs_um_quality_guard.yaml", verbose=False)
    r6_calibrator = r6.model[-1].ftsc_calibrator
    head = r8.model[-1]
    calibrator = head.ftsc_calibrator
    assert r6_calibrator.um_quality_guard is False
    assert calibrator.evidence_names == ("position_gaussian",)
    assert "dfl_distribution" not in calibrator.providers
    assert "dcfl_dgmm_quality" not in calibrator.providers
    assert calibrator.um_enabled and calibrator.um_quality_guard
    assert calibrator.um_quality_gamma == pytest.approx(1.0)
    assert calibrator.um_lambda == pytest.approx(r6_calibrator.um_lambda) == pytest.approx(0.05)
    assert calibrator.classification_replacement == "bce"
    assert not calibrator.gt_mass_rebalance_cls and not getattr(head, "quality_head", False)
    assert set(r8.state_dict()) == set(r6.state_dict())
    assert sum(parameter.numel() for parameter in r8.parameters()) == sum(
        parameter.numel() for parameter in r6.parameters()
    )


@pytest.mark.parametrize(
    ("control_name", "followup_name"),
    [
        ("yolov8n_p2_levir_ftsc_r5_dcfl_dgmm.yaml", "yolov8n_p2_levir_ftsc_r7_dcfl_dgmm_gt_mass.yaml"),
        ("yolov8n_p2_levir_ftsc_r6_ugs_um.yaml", "yolov8n_p2_levir_ftsc_r8_ugs_um_quality_guard.yaml"),
    ],
)
def test_r7_r8_preserve_inference_structure_and_tal_assignment(control_name, followup_name):
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    control = DetectionModel(config_root / control_name, verbose=False)
    followup = DetectionModel(config_root / followup_name, verbose=False)
    followup.load_state_dict(control.state_dict(), strict=True)

    image = torch.rand(1, 3, 64, 64)
    control.eval()
    followup.eval()
    with torch.no_grad():
        control_prediction = control(image)[0]
        followup_prediction = followup(image)[0]
    assert torch.equal(followup_prediction, control_prediction)

    for model in (control, followup):
        model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        model.train()
        model.criterion = model.init_criterion()
        model.criterion.capture_assignment = True
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    control(batch)
    followup(batch)
    assert all(
        torch.equal(control_value, followup_value)
        for control_value, followup_value in zip(
            control.criterion.last_assignment,
            followup.criterion.last_assignment,
            strict=True,
        )
    )


def test_r6_model_total_loss_is_finite_and_um_reaches_dfl_branch():
    config = (
        Path(__file__).resolve().parents[2]
        / "models_config/yolov8/levir/yolov8n_p2_levir_ftsc_r6_ugs_um.yaml"
    )
    model = DetectionModel(config, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    model.train()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, items = model(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    head = model.model[-1]
    assert head.cv2[0][-1].weight.grad is not None
    assert model.criterion.ftsc_metrics["ftsc_um_positive_count"] > 0
    assert model.criterion.ftsc_metrics["ftsc_um_weighted_loss"] > 0


def test_r8_model_loss_logs_guard_diagnostics_and_reaches_dfl_branch():
    config = (
        Path(__file__).resolve().parents[2]
        / "models_config/yolov8/levir/yolov8n_p2_levir_ftsc_r8_ugs_um_quality_guard.yaml"
    )
    model = DetectionModel(config, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    model.train()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, items = model(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    metrics = model.criterion.ftsc_metrics
    assert model.model[-1].cv2[0][-1].weight.grad is not None
    assert metrics["ftsc_um_positive_dfl_count"] > 0
    assert metrics["ftsc_um_quality_guard"] == 1.0
    assert metrics["ftsc_um_quality_gamma"] == pytest.approx(1.0)
    assert metrics["ftsc_um_guarded_entropy_loss"] >= 0
    assert metrics["ftsc_um_weighted_loss"] >= 0
    assert 0 <= metrics["ftsc_um_quality_iou_min"] <= metrics["ftsc_um_quality_iou_max"] <= 1
    assert 0 <= metrics["ftsc_um_guard_min"] <= metrics["ftsc_um_guard_max"] <= 1
    assert metrics["ftsc_um_guard_mass"] >= 0


def test_f5_warmup_loss_is_exactly_baseline_identity():
    config_root = Path(__file__).resolve().parents[2] / "models_config/yolov8/levir"
    baseline = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_y0_baseline.yaml", verbose=False)
    f5 = DetectionModel(config_root / "yolov8n_p2_levir_ftsc_af_y2_f5_position.yaml", verbose=False)
    missing, unexpected = f5.load_state_dict(baseline.state_dict(), strict=False)
    assert missing == ["model.29.ftsc_calibrator.strength_logits.position_gaussian"]
    assert not unexpected
    for model in (baseline, f5):
        model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
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
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
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
