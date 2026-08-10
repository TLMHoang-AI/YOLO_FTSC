from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ultralytics.nn.modules import (
    CBAM,
    ChannelAttention,
    GlobalChannelContextCalibration,
    KVCompressedAttention,
    SpatialAttention,
)
from ultralytics.nn.tasks import DetectionModel


CONFIG_DIR = Path(__file__).parents[2] / "models_config/yolov8/levir"
CONFIGS = {
    "channel": ("yolov8n_p2_fpn_only_cbam_channel_only.yaml", ChannelAttention),
    "channel_gmp": ("yolov8n_p2_fpn_only_cbam_channel_gmp.yaml", ChannelAttention),
    "channel_avgmax": ("yolov8n_p2_fpn_only_cbam_channel_avgmax.yaml", ChannelAttention),
    "spatial": ("yolov8n_p2_fpn_only_cbam_spatial_only.yaml", SpatialAttention),
    "full": ("yolov8n_p2_fpn_only_cbam_matched.yaml", CBAM),
    "gccc": ("yolov8n_p2_fpn_only_gccc.yaml", GlobalChannelContextCalibration),
}


def test_channel_attention_default_is_legacy_gap_with_unchanged_state_keys():
    module = ChannelAttention(8)
    x = torch.randn(2, 8, 9, 11)
    expected = x * module.act(module.fc(module.pool(x)))
    torch.testing.assert_close(module(x), expected)
    assert module.descriptor == "avg"
    assert set(module.state_dict()) == {"fc.weight", "fc.bias"}


@pytest.mark.parametrize("descriptor", ["avg", "max", "avg_max"])
def test_channel_attention_descriptor_gate_is_spatially_uniform(descriptor):
    module = ChannelAttention(8, descriptor)
    x = torch.randn(2, 8, 9, 11)
    output = module(x)
    gate = output / x
    torch.testing.assert_close(gate, gate[..., :1, :1].expand_as(gate), rtol=1e-5, atol=1e-6)
    if descriptor == "max":
        expected = x * module.act(module.fc(module.max_pool(x)))
    elif descriptor == "avg_max":
        expected = x * module.act(module.fc(module.pool(x)) + module.max_fc(module.max_pool(x)))
        assert set(module.state_dict()) == {"fc.weight", "fc.bias", "max_fc.weight", "max_fc.bias"}
    else:
        expected = x * module.act(module.fc(module.pool(x)))
    torch.testing.assert_close(output, expected)


def test_channel_attention_rejects_unknown_descriptor():
    with pytest.raises(ValueError, match="Unsupported channel descriptor"):
        ChannelAttention(8, "median")


@pytest.mark.parametrize("sr_ratio", [0, -1, 1.5, True])
def test_gccc_rejects_invalid_sr_ratio(sr_ratio):
    with pytest.raises(ValueError, match="positive integer"):
        GlobalChannelContextCalibration(8, 8, sr_ratio=sr_ratio)


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_gccc_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="positive"):
        GlobalChannelContextCalibration(8, 8, temperature=temperature)


def test_group_weight_helper_preserves_kvca_behavior_and_handles_padding():
    module = KVCompressedAttention(8, 8, sr_ratio=4, mode="group_weight")
    x = torch.randn(2, 8, 9, 11)
    actual = module._compress_group_weight(x)

    padded = F.pad(x, (0, 1, 0, 3))
    tokens = padded.view(2, 8, 3, 4, 3, 4).permute(0, 2, 4, 3, 5, 1).contiguous()
    tokens = tokens.view(2, 3, 3, 16, 8)
    expected = (tokens * module.group_score(tokens).softmax(dim=3)).sum(dim=3).permute(0, 3, 1, 2)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 8, 3, 3)


def test_gccc_identity_alpha_gradient_and_gate_shape():
    module = GlobalChannelContextCalibration(8, 8, sr_ratio=4, alpha_init=0.0)
    x = torch.ones(2, 8, 9, 11, requires_grad=True)
    output = module(x)
    torch.testing.assert_close(output, x)
    output.sum().backward()
    assert module.last_gate_shape == (2, 8, 1, 1)
    assert module.alpha.grad is not None and torch.isfinite(module.alpha.grad) and module.alpha.grad.abs() > 0
    assert module.q.weight.grad is not None and module.q.weight.grad.abs().sum() == 0


def test_gccc_nonzero_alpha_trains_attention_branch():
    module = GlobalChannelContextCalibration(8, 8, sr_ratio=4, alpha_init=0.1)
    x = torch.randn(2, 8, 9, 11, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    for parameter in (module.q.weight, module.k.weight, module.v.weight, module.gate_proj.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", CONFIGS)
def test_matched_yaml_resolves_shared_p2_before_detect(variant):
    filename, expected_type = CONFIGS[variant]
    model = DetectionModel(CONFIG_DIR / filename, ch=3, nc=1, verbose=False).eval()
    assert isinstance(model.model[19], expected_type)
    assert model.model[20].f == [19] and model.model[20].stride.tolist() == [4.0]
    with torch.no_grad():
        assert model(torch.randn(1, 3, 128, 128)) is not None


@pytest.mark.parametrize(
    ("filename", "descriptor"),
    [
        ("yolov8n_p2_fpn_only_cbam_channel_gmp.yaml", "max"),
        ("yolov8n_p2_fpn_only_cbam_channel_avgmax.yaml", "avg_max"),
    ],
)
def test_channel_descriptor_yaml_resolves_shared_p2_before_detect(filename, descriptor):
    model = DetectionModel(CONFIG_DIR / filename, ch=3, nc=1, verbose=False).eval()
    assert isinstance(model.model[19], ChannelAttention)
    assert model.model[19].descriptor == descriptor
    assert model.model[20].f == [19] and model.model[20].stride.tolist() == [4.0]
    with torch.no_grad():
        assert model(torch.randn(1, 3, 128, 128)) is not None


def test_gccc_analytical_attention_cost_is_lower_than_global_kvca():
    height = width = 128
    channels = 32
    groups = (height // 8) * (width // 8)
    gccc = GlobalChannelContextCalibration(channels, channels, sr_ratio=8)
    kvca_macs = (
        height * width * channels * channels
        + height * width * channels
        + 2 * groups * channels * channels
        + 2 * height * width * groups * channels
        + height * width * channels * channels
    )
    assert gccc.analytical_macs(height, width) < kvca_macs
