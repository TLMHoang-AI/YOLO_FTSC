from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules import FullSelfAttention, KVCompressedAttention, PatchKVCompressedAttention
from ultralytics.nn.tasks import DetectionModel


CONFIG_DIR = Path(__file__).parents[2] / "models_config/yolov8/levir"


@pytest.mark.parametrize("radius", [-1, 0.5])
def test_patch_kvca_rejects_invalid_radius(radius):
    with pytest.raises(ValueError, match="non-negative integer"):
        PatchKVCompressedAttention(8, 8, patch_radius=radius)


@pytest.mark.parametrize("radius", [0, 1])
def test_patch_kvca_forward_backward_and_parameter_parity(radius):
    module = PatchKVCompressedAttention(32, 32, num_heads=4, sr_ratio=8, patch_radius=radius).train()
    control = KVCompressedAttention(32, 32, num_heads=4, sr_ratio=8, mode="group_weight")
    assert {k: v.shape for k, v in module.state_dict().items()} == {k: v.shape for k, v in control.state_dict().items()}

    x = torch.randn(1, 32, 128, 128, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_patch_kvca_padding_crop_and_border_mask():
    module = PatchKVCompressedAttention(8, 8, num_heads=2, sr_ratio=4, patch_radius=1).eval()
    x = torch.randn(1, 8, 9, 11)
    assert module(x).shape == x.shape

    counts = module._validity_mask(3, 3, radius=1, device=x.device).sum(-1).reshape(3, 3)
    assert counts.tolist() == [[4, 6, 4], [6, 9, 6], [4, 6, 4]]
    assert PatchKVCompressedAttention._validity_mask(3, 3, 0, x.device).sum().item() == 9


def test_full_self_attention_forward_backward_and_parameter_parity():
    module = FullSelfAttention(32, 32, num_heads=4).train()
    control = KVCompressedAttention(32, 32, num_heads=4, sr_ratio=8, mode="group_weight")
    assert {k: v.shape for k, v in module.state_dict().items()} == {k: v.shape for k, v in control.state_dict().items()}
    x = torch.randn(1, 32, 32, 32, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("radius", [0, 1])
def test_patch_kvca_yaml_parse_and_model_smoke(radius):
    config = CONFIG_DIR / f"yolov8n_p2_fpn_only_patch_kvca_r{radius}.yaml"
    model = DetectionModel(config, ch=3, nc=1, verbose=False).eval()
    attention = model.model[19]
    assert isinstance(attention, PatchKVCompressedAttention)
    assert attention.c2 == 32 and attention.num_heads == 4 and attention.patch_radius == radius
    assert model.model[20].f == [19]

    with torch.no_grad():
        output = model(torch.randn(1, 3, 128, 128))
    assert output is not None
