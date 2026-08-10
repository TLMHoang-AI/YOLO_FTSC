import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect, DetectClsAttention, P2ClsContext, P2RegLocal

import train_all_levir_yolov8n_p2_asymmetric_screen as train


def _models():
    return {name: YOLO(config).model for name, config in train.workflow.VARIANTS.items()}


def _copy_shared(source, target):
    source_state, target_state = source.state_dict(), target.state_dict()
    shared = {key: value for key, value in source_state.items() if key in target_state and value.shape == target_state[key].shape}
    target.load_state_dict(shared, strict=False)
    return shared


def _raw(head, x):
    return head.forward_head([x], **head.one2many)


def test_graphs_are_p2_only_and_refiners_are_isolated():
    models = _models()
    plain, a1, a2 = (models[name].model[-1] for name in train.workflow.VARIANTS)
    assert type(plain) is Detect
    assert isinstance(a1, DetectClsAttention) and a1.attn_type == "context_mid_cbam"
    assert isinstance(a2, DetectClsAttention) and a2.attn_type == "reg_local"
    assert all(head.stride.tolist() == [4.0] for head in (plain, a1, a2))
    assert not any(isinstance(module, (P2ClsContext, P2RegLocal)) for module in plain.modules())
    assert sum(isinstance(module, P2ClsContext) for module in a1.modules()) == 1
    assert sum(isinstance(module, P2RegLocal) for module in a2.modules()) == 1
    assert a1.cls_mid.reduce.conv.in_channels // 2 == a1.cls_mid.reduce.conv.out_channels
    assert a2.box_detail[0].reduce.conv.in_channels // 2 == a2.box_detail[0].reduce.conv.out_channels


def test_identity_initialization_and_shared_tensor_transfer():
    models = _models()
    plain = models["plain_p2_only"]
    for name in ("cls_context_mid_cbam", "reg_local"):
        shared = _copy_shared(plain, models[name])
        target_state = models[name].state_dict()
        assert all(torch.equal(value, target_state[key]) for key, value in shared.items())

    plain_head = plain.model[-1].eval()
    a1 = models["cls_context_mid_cbam"].model[-1].eval()
    a2 = models["reg_local"].model[-1].eval()
    channels = plain_head.cv2[0][0].conv.in_channels
    x = torch.randn(2, channels, 32, 32)
    with torch.no_grad():
        control = _raw(plain_head, x)
        a1_raw, a2_raw = _raw(a1, x), _raw(a2, x)
    assert torch.count_nonzero(a1.cls_mid.zero.weight) == 0
    assert torch.count_nonzero(a2.box_detail[0].zero.weight) == 0
    assert torch.equal(control["boxes"], a1_raw["boxes"])
    assert torch.equal(control["scores"], a2_raw["scores"])


def test_each_refiner_receives_gradient_only_from_its_task_branch():
    models = _models()
    plain = models["plain_p2_only"]
    for name in ("cls_context_mid_cbam", "reg_local"):
        _copy_shared(plain, models[name])
    channels = plain.model[-1].cv2[0][0].conv.in_channels

    a1 = models["cls_context_mid_cbam"].model[-1].train()
    _raw(a1, torch.randn(2, channels, 16, 16))["scores"].mean().backward()
    assert a1.cls_mid.zero.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in a1.cv2.parameters())

    a2 = models["reg_local"].model[-1].train()
    _raw(a2, torch.randn(2, channels, 16, 16))["boxes"].mean().backward()
    assert a2.box_detail[0].zero.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in a2.cv3.parameters())


def test_experiment_defaults():
    args = train.parse_args([])
    assert args.variants == ["plain_p2_only", "cls_context_mid_cbam", "reg_local"]
    assert args.seeds == [42]
    assert args.split_seed == 42
    assert args.epochs == 100 and args.imgsz == 512
    assert args.no_upload
