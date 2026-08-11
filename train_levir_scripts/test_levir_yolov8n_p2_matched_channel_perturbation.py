#!/usr/bin/env python3
"""Smoke tests for MatchedChannelPerturbation."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules.conv import MatchedChannelPerturbation


def test_module_behavior():
    x = torch.ones(4, 32, 8, 8, requires_grad=True)
    m = MatchedChannelPerturbation(32, mu=0.45, sigma_delta=0.07, q01=0.2, q99=0.8)
    assert sum(p.numel() for p in m.parameters()) == 0

    m.train()
    y1, y2 = m(x), m(x)
    assert y1.shape == x.shape
    assert not torch.equal(y1, y2)
    stats = m.last_gate_stats
    assert abs(stats["gate_mean"] - 0.45) < 0.02
    assert abs(stats["gate_channel_std"] - 0.07) < 0.02
    y1.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    m.eval()
    with torch.no_grad():
        y1, y2 = m(x), m(x)
    assert torch.equal(y1, y2)
    assert torch.allclose(y1, x * 0.45)
    assert m.last_gate_stats["gate_channel_std"] == 0.0


def test_model_topology():
    model = YOLO(ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_fpn_only_matched_channel_perturbation.yaml")
    assert isinstance(model.model.model[19], MatchedChannelPerturbation)
    assert model.model.model[20].f == [19]


if __name__ == "__main__":
    test_module_behavior()
    test_model_topology()
    print("MatchedChannelPerturbation tests passed")
