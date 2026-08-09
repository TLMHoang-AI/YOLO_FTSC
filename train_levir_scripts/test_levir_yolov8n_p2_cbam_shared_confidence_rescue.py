import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules.block import C2fCBAM
from ultralytics.nn.modules.head import Detect

import train_all_levir_yolov8n_p2_cbam_shared_confidence_rescue as train


def test_graph_is_shared_cbam_before_plain_p2_detect():
    model = YOLO(train.workflow.VARIANTS["cbam_shared_confidence_rescue"]).model
    head = model.model[-1]
    assert type(head) is Detect and head.nl == 1 and head.stride.tolist() == [4.0]
    assert isinstance(model.model[-2], C2fCBAM)
    assert {type(module).__name__ for module in model.modules()}.isdisjoint(
        {"P1DRR", "DetectClsAttention", "HVDecoupledDetect", "P2RegLocal", "P2OffsetRegression"}
    )


def test_runner_locks_rescue_and_seed42():
    args = train.parse_args(["--no-upload"])
    kwargs = train.train_kwargs(args, Path("data.yaml"), 42, True)
    assert args.seeds == [42] and args.split_seed == 42
    assert kwargs["positive_confidence_rescue_gain"] == 0.25
    assert kwargs["positive_confidence_rescue_gamma"] == 1.0
    assert kwargs["vfl"] is False and kwargs["cls_iou_target"] is False
