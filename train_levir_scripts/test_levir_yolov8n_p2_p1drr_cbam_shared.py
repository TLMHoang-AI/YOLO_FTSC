import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import CBAM, Detect, P1DRR

import train_all_levir_yolov8n_p2_p1drr_cbam_shared as train


def _model():
    return YOLO(train.workflow.VARIANTS["p1drr_cbam_shared"]).model


def test_graph_is_p1drr_then_shared_cbam_then_p2_only_detect():
    model = _model()
    rescue, attention, head = model.model[-3:]
    assert type(rescue) is P1DRR
    assert type(attention) is CBAM
    assert type(head) is Detect and head.stride.tolist() == [4.0]
    assert head.cv2[0][0].conv.in_channels == head.cv3[0][0].conv.in_channels == 32
    assert not any(type(module).__name__ in {"HVDecoupledDetect", "DetectClsAttention", "P1GER"} for module in model.modules())


def test_identity_start_and_shared_cbam_receives_both_task_gradients():
    model = _model().train()
    rescue, attention, head = model.model[-3:]
    p2 = torch.randn(2, 32, 16, 16, requires_grad=True)
    p1 = torch.randn(2, 16, 32, 32, requires_grad=True)
    rescued = rescue([p2, p1])
    assert torch.equal(rescued, p2)
    shared = attention(rescued)
    raw = head.forward_head([shared], **head.one2many)
    raw["boxes"].mean().backward(retain_graph=True)
    assert attention.channel_attention.fc.weight.grad.abs().sum() > 0
    attention.zero_grad(set_to_none=True)
    raw["scores"].mean().backward()
    assert attention.channel_attention.fc.weight.grad.abs().sum() > 0


def test_defaults():
    args = train.parse_args(["--no-upload"])
    assert args.variants == ["p1drr_cbam_shared"]
    assert args.seeds == [42]
    assert args.epochs == 100 and args.imgsz == 512
