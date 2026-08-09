import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules.head import (
    DetectClsAttention,
    HVDecoupledDetect,
    HVDecoupledRegression,
    P2NUDFLDetect,
    P2OffsetRegression,
)

import train_all_levir_yolov8n_p2_hv_decoupled as train


def _model():
    return YOLO(train.workflow.VARIANTS["hv_decoupled"]).model


def _raw(head, x):
    return head.forward_head([x], **head.one2many)


def _grad_sum(module):
    return sum((parameter.grad.abs().sum() for parameter in module.parameters() if parameter.grad is not None), torch.tensor(0.0))


def test_resolved_graph_is_p2_only_and_opt_in():
    model = _model()
    head = model.model[-1]
    assert type(head) is HVDecoupledDetect
    assert head.stride.tolist() == [4.0]
    assert head.nl == 1 and head.reg_max == 16
    assert head.cv2[0].shared.conv.in_channels == 32
    assert head.cv2[0].shared.conv.out_channels == 64
    assert head.cv2[0].horizontal[0].conv.in_channels == 64
    assert head.cv2[0].horizontal[0].conv.out_channels == 32
    assert head.cv2[0].horizontal[-1].out_channels == 2 * head.reg_max
    assert head.cv2[0].vertical[-1].out_channels == 2 * head.reg_max
    assert not any(isinstance(module, (DetectClsAttention, P2NUDFLDetect, P2OffsetRegression)) for module in model.modules())
    assert "P2RegLocal" not in {type(module).__name__ for module in model.modules()}


def test_hv_logits_are_reordered_ltrb_and_uniform_dfl_is_unchanged():
    regression = HVDecoupledRegression(4, reg_max=3).eval()
    with torch.no_grad():
        regression.horizontal[-1].weight.zero_()
        regression.horizontal[-1].bias.copy_(torch.arange(6))
        regression.vertical[-1].weight.zero_()
        regression.vertical[-1].bias.copy_(torch.arange(10, 16))
        logits = regression(torch.randn(2, 4, 5, 7))

    assert logits.shape == (2, 12, 5, 7)
    expected = torch.tensor([0, 1, 2, 10, 11, 12, 3, 4, 5, 13, 14, 15], dtype=logits.dtype)
    assert torch.equal(logits[0, :, 0, 0], expected)

    probabilities = logits.flatten(2).view(2, 4, 3, -1).softmax(2)
    reference = (probabilities * torch.arange(3).view(1, 1, 3, 1)).sum(2)
    from ultralytics.nn.modules.block import DFL

    assert torch.allclose(DFL(3)(logits.flatten(2)), reference)


def test_direction_towers_and_task_branches_have_isolated_gradients():
    head = _model().model[-1].train()
    channels = head.cv2[0].shared.conv.in_channels

    boxes = _raw(head, torch.randn(2, channels, 8, 8))["boxes"].view(2, 4, head.reg_max, -1)
    (boxes[:, (0, 2)].sum()).backward()
    assert _grad_sum(head.cv2[0].shared) > 0
    assert _grad_sum(head.cv2[0].horizontal) > 0
    assert _grad_sum(head.cv2[0].vertical) == 0
    assert _grad_sum(head.cv3) == 0

    head.zero_grad(set_to_none=True)
    boxes = _raw(head, torch.randn(2, channels, 8, 8))["boxes"].view(2, 4, head.reg_max, -1)
    (boxes[:, (1, 3)].sum()).backward()
    assert _grad_sum(head.cv2[0].shared) > 0
    assert _grad_sum(head.cv2[0].horizontal) == 0
    assert _grad_sum(head.cv2[0].vertical) > 0

    head.zero_grad(set_to_none=True)
    _raw(head, torch.randn(2, channels, 8, 8))["scores"].sum().backward()
    assert _grad_sum(head.cv2) == 0
    assert _grad_sum(head.cv3) > 0


def test_bias_forward_backward_and_state_dict_round_trip(tmp_path):
    model = _model()
    head = model.model[-1]
    head.bias_init()
    assert torch.all(head.cv2[0].horizontal[-1].bias == 2)
    assert torch.all(head.cv2[0].vertical[-1].bias == 2)

    channels = head.cv2[0].shared.conv.in_channels
    x = torch.randn(1, channels, 8, 8)
    head.eval()
    with torch.no_grad():
        before = _raw(head, x)["boxes"]
    checkpoint = tmp_path / "hv_state.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _model()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    restored.model[-1].eval()
    with torch.no_grad():
        after = _raw(restored.model[-1], x)["boxes"]
    assert torch.equal(before, after)

    restored.train()
    loss = restored(torch.randn(1, 3, 64, 64))["boxes"].mean()
    loss.backward()
    assert _grad_sum(restored.model[-1].cv2) > 0


def test_experiment_defaults():
    args = train.parse_args(["--no-upload"])
    assert args.variants == ["hv_decoupled"]
    assert args.seeds == [42]
    assert args.split_seed == 42
    assert args.epochs == 100 and args.imgsz == 512
    assert args.no_upload
