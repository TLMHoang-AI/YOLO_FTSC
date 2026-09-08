"""Focused no-training tests for the N0--N3 generation boundary."""
from pathlib import Path
import random
import sys

import pytest
import torch
import yaml

import train_all_levir_yolov8n_p2_ftsc_nomosaic_nextgen as runner

ROOT = Path(__file__).resolve().parents[1]


def test_variants_and_no_mosaic_contract():
    args = runner.parse_args([])
    assert list(runner.VARIANTS) == [
        "N0_y4_nomosaic",
        "N1_y4_nomosaic_adaptive_zoom",
        "N2_y4_nomosaic_support_assignment",
        "N3_y4_nomosaic_localization_distill",
    ]
    assert args.seeds == [42] and args.epochs == 100 and args.patience == 20
    for variant in runner.VARIANTS:
        args._active_variant = variant
        kwargs = runner.train_kwargs(args, Path("split.yaml"), 42, True)
        assert kwargs["mosaic"] == 0.0 and kwargs["close_mosaic"] == 0
        assert kwargs["fitness_metric"] == "map50_95"
        assert kwargs["optimizer"] == "auto" and kwargs["cos_lr"] is False
    args._active_variant = "N0_y4_nomosaic"
    assert runner.train_kwargs(args, Path("split.yaml"), 42, True)["adaptive_zoom"] is False
    args._active_variant = "N1_y4_nomosaic_adaptive_zoom"
    assert runner.train_kwargs(args, Path("split.yaml"), 42, True)["adaptive_zoom"] is True


def test_configs_are_y4_fixed_dfl_copies_without_canonical_mutation():
    canonical = yaml.safe_load((runner.CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y4_legacy_compat.yaml").read_text())
    for config in runner.VARIANTS.values():
        payload = yaml.safe_load(config.read_text())
        assert payload == canonical


def test_source_preflight_reports_zoom_assignment_and_ld_boundaries():
    report = runner.source_preflight(runner.parse_args([]))
    assert report["mosaic"] == 0.0 and report["close_mosaic"] == 0
    assert report["adaptive_zoom"] == {"target_min_side": 16.0, "p": 0.5, "z_max": 2.0, "visible_ratio": 0.5}
    assert report["assignment"]["support_topk"] == 5
    assert report["localization_distillation"]["enabled_only_in"] == "N3"


def test_localization_distillation_lambda_zero_and_detached_target():
    sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
    from ultralytics.utils.loss import localization_distillation_loss
    student = torch.randn(2, 6, 16, requires_grad=True)
    teacher = torch.randn(2, 6, 16, requires_grad=True)
    mask = torch.tensor([[True, False, True, False, False, True], [False, True, False, False, True, False]])
    zero = localization_distillation_loss(student, teacher, mask, 0.0)
    assert zero.item() == 0.0
    weighted = localization_distillation_loss(student, teacher, mask, 0.1)
    weighted.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_adaptive_zoom_is_deterministic_and_preserves_valid_boxes_when_dependencies_exist():
    pytest.importorskip("cv2")
    sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
    import numpy as np
    from ultralytics.data.augment import AdaptiveZoom
    from ultralytics.utils.instance import Instances

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    boxes = np.array([[30.0, 30.0, 34.0, 38.0], [4.0, 4.0, 20.0, 20.0]], dtype=np.float32)
    labels = {"img": image.copy(), "cls": np.array([[0], [0]], dtype=np.float32),
              "instances": Instances(boxes.copy(), bbox_format="xyxy", normalized=False)}
    random.seed(7); np.random.seed(7)
    first = AdaptiveZoom(target_min_side=16, p=1.0, z_max=2.0)(labels)
    assert first["img"].shape == image.shape
    assert np.isfinite(first["instances"].bboxes).all()
    assert (first["instances"].bboxes[:, 0::2] >= 0).all() and (first["instances"].bboxes[:, 0::2] <= 64).all()
    assert (first["instances"].bboxes[:, 1::2] >= 0).all() and (first["instances"].bboxes[:, 1::2] <= 64).all()


def test_support_assigner_disabled_is_exact_standard_tal_class():
    assert runner.VARIANT_FLAGS["N0_y4_nomosaic"]["support_assignment"] is False
    assert runner.VARIANT_FLAGS["N2_y4_nomosaic_support_assignment"]["support_assignment"] is True
