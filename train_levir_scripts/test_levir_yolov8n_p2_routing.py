import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import DBSS, GCTS
from ultralytics.nn.modules.head import Detect, P2OffsetRegression

import train_all_levir_yolov8n_p2_routing as train


@pytest.mark.parametrize(
    ("variant", "module_type", "module_index"),
    (("dbss_pre_p2", DBSS, 1), ("gcts_backbone_p2_p3", GCTS, 3)),
)
def test_p2_routing_architecture(variant, module_type, module_index):
    model = YOLO(train.VARIANTS[variant]).model
    assert isinstance(model.model[module_index], module_type)
    detect = model.model[-1]
    assert isinstance(detect, Detect)
    assert detect.nl == 4
    assert detect.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert not any(isinstance(module, P2OffsetRegression) for module in detect.modules())


def test_experiment_defaults_are_fixed_and_complete():
    args = train.parse_args([])
    assert args.variants == ["dbss_pre_p2", "gcts_backbone_p2_p3"]
    assert args.seeds == [42, 43, 44]
    assert args.split_seed == 42
    assert args.hf_repo_id == "duyle2408/levir-yolov8n-p2-routing-3seed"
    assert train.PUBLISHED_COUNTS == {"train": 2320, "val": 788, "test": 788}


def test_dbss_model_restores_shifted_pretrained_backbone():
    source = YOLO("yolov8n.pt").model
    target = train.model_for("dbss_pre_p2", "yolov8n.pt").model
    assert (target.model[2].conv.weight == source.model[1].conv.weight).all()
    assert (target.model[10].cv1.conv.weight == source.model[9].cv1.conv.weight).all()


def test_gcts_model_restores_pretrained_downsample_path():
    source = YOLO("yolov8n.pt").model
    target = train.model_for("gcts_backbone_p2_p3", "yolov8n.pt").model
    assert (target.model[3].conv.weight == source.model[3].conv.weight).all()
