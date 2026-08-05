import sys

import torch


sys.path.insert(0, "models_related/ultralytics")

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import Detect, P2OffsetRegression, v10Detect


def make_old_head():
    return torch.nn.Sequential(Conv(16, 8, 3), Conv(8, 8, 3), torch.nn.Conv2d(8, 16, 1))


def test_p2_offset_is_zero_initialized_and_differentiable():
    head = P2OffsetRegression(make_old_head(), 4)
    x = torch.randn(2, 16, 12, 15, requires_grad=True)
    output = head(x)
    assert output.shape == (2, 16, 12, 15)
    assert head.offset[-1].weight.abs().sum() == 0
    assert head.offset[-1].bias.abs().sum() == 0
    output.square().mean().backward()
    assert head.offset[-1].weight.grad is not None


def test_zero_offset_preserves_old_regression_logits():
    old = make_old_head()
    new = P2OffsetRegression(old, 4)
    x = torch.randn(1, 16, 8, 9)
    with torch.no_grad():
        assert torch.allclose(new(x), old(x), atol=1e-5, rtol=1e-5)


def test_nondefault_offset_regression_is_disabled_by_default():
    head = Detect(nc=3, ch=(16, 32, 64, 128))
    assert not isinstance(head.cv2[0], P2OffsetRegression)


def test_v10_keeps_standard_regression_on_both_assignment_heads():
    head = v10Detect(nc=3, ch=(16, 32, 64, 128))
    assert not isinstance(head.cv2[0], P2OffsetRegression)
    assert not isinstance(head.one2one_cv2[0], P2OffsetRegression)
    features = [torch.randn(1, 16, 8, 9), torch.randn(1, 32, 4, 5), torch.randn(1, 64, 2, 3), torch.randn(1, 128, 1, 2)]
    head.train()
    output = head(features)
    assert set(output) == {"one2many", "one2one"}
    assert output["one2many"]["boxes"].shape == output["one2one"]["boxes"].shape
