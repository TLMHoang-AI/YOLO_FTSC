from pathlib import Path

import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics.utils.loss import v8DetectionLoss


def test_consensus_uses_only_grouped_p2_tal_positives():
    criterion = object.__new__(v8DetectionLoss)
    pred = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 5.0, 4.0],
                          [30.0, 30.0, 40.0, 40.0], [99.0, 99.0, 100.0, 100.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0],
                            [30.0, 30.0, 40.0, 40.0], [0.0, 0.0, 4.0, 4.0]]])
    gt_idx = torch.tensor([[0, 0, 1, 0]])
    fg = torch.tensor([[True, True, True, True]])
    mask_gt = torch.tensor([[[True], [True]]])

    loss, metrics = criterion.box_consensus_loss(pred, target, gt_idx, fg, n_p2=3, mask_gt=mask_gt)
    loss.backward()

    assert loss > 0
    assert pred.grad[0, :2].abs().sum() > 0
    assert pred.grad[0, 2:].abs().sum() == 0  # singleton GT and non-P2 anchor do not contribute
    assert metrics["consensus_gt_coverage"] == 0.5
    assert metrics["consensus_p2_pos_median"] == 1.0


def test_consensus_config_keeps_assignment_off():
    config = yaml.safe_load(
        (ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_consensus.yaml").read_text()
    )
    assert config["loc_assign"] is False
    assert config["box_consensus_gain"] == 0.1
