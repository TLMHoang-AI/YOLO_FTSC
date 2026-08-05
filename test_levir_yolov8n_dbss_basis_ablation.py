import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules import DBSS

from misc import train_levir_dbss_hit as shared
import train_all_levir_yolov8n_dbss_basis_ablation as train


@pytest.mark.parametrize("variant,expected", zip(train.BASIS_VARIANTS, (4, 12, 16, 20)))
def test_basis_variant_configures_exact_basis_count(variant, expected):
    model = YOLO(shared.MODELS["yolov8n"][variant]).model
    modules = [module for module in model.modules() if isinstance(module, DBSS)]
    assert len(modules) == 1
    assert modules[0].num_bases == expected


def test_ablation_defaults_exclude_existing_k8(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner"])
    args = shared.parse_args(
        train.BASIS_VARIANTS,
        train.EXPERIMENT_SLUG,
        "levir-yolov8n-dbss-basis-ablation",
        (42,),
    )
    assert args.mechanisms == ["k4", "k12", "k16", "k20"]
    assert args.seeds == [42]
    assert args.project == ROOT / "runs/levir_yolov8n_dbss_basis_ablation"
    assert args.hf_repo_name == "levir-yolov8n-dbss-basis-ablation"
