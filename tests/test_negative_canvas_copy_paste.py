from __future__ import annotations

from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import numpy as np

from project_ultralytics.copy_paste import build_small_object_copy_paste, copy_paste_config
from project_ultralytics.negative_canvas_copy_paste import NegativeCanvasCopyPaste
from ultralytics.utils.instance import Instances


def _labels(image, boxes=(), *, im_file=None, image_index=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    labels = {
        "img": image,
        "instances": Instances(
            boxes.copy(), np.zeros((len(boxes), 0, 2), dtype=np.float32),
            bbox_format="xyxy", normalized=False,
        ),
        "cls": np.zeros((len(boxes), 1), dtype=np.float32),
    }
    if im_file is not None:
        labels["im_file"] = str(im_file)
    if image_index is not None:
        labels["image_index"] = image_index
    return labels


def _dataset(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.indices((16, 16))
    image[4:20, 4:20] = np.stack((xx * 10, yy * 10, (xx + yy) * 5), axis=-1)
    image[28:36, 28:36] = (40, 80, 120)
    image[40:56, 40:56] = (150, 100, 50)
    source = Path(tmp_path) / "source.png"
    negative = Path(tmp_path) / "negative.png"
    assert cv2.imwrite(str(source), image)
    assert cv2.imwrite(str(negative), np.zeros_like(image))
    return type("Dataset", (), {
        "labels": [
            {
                "bboxes": np.array([[4, 4, 20, 20], [28, 28, 36, 36], [40, 40, 56, 56]], np.float32),
                "cls": np.array([[0], [0], [0]], np.float32),
                "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2],
            },
            {"bboxes": [], "cls": [], "bbox_format": "xyxy", "normalized": False, "shape": image.shape[:2]},
        ],
        "im_files": [str(source), str(negative)],
    })()


def test_positive_original_is_untouched_and_probability_is_conditional(tmp_path):
    dataset = _dataset(tmp_path)
    transform = NegativeCanvasCopyPaste(dataset, p=1.0, rng=random.Random(1))
    labels = _labels(np.full((64, 64, 3), 7, np.uint8), [[1, 1, 5, 5]], im_file=dataset.im_files[0])
    before_image, before_boxes = labels["img"].copy(), labels["instances"].bboxes.copy()
    transform(labels)
    np.testing.assert_array_equal(labels["img"], before_image)
    np.testing.assert_array_equal(labels["instances"].bboxes, before_boxes)
    assert transform.stats["negative_seen"] == 0

    gated = NegativeCanvasCopyPaste(dataset, p=0.0, rng=random.Random(1))
    gated(_labels(np.zeros((64, 64, 3), np.uint8), im_file=dataset.im_files[1]))
    assert gated.stats["negative_seen"] == 1
    assert gated.stats["negative_selected"] == 0


def test_original_positive_with_current_boxes_dropped_is_not_a_canvas(tmp_path):
    dataset = _dataset(tmp_path)
    transform = NegativeCanvasCopyPaste(dataset, p=1.0, rng=random.Random(7))
    labels = _labels(np.zeros((64, 64, 3), np.uint8), im_file=dataset.im_files[0])
    transform(labels)
    assert len(labels["instances"]) == 0
    assert transform.stats["negative_seen"] == 0


def test_r1_ratio_and_exactly_one_target_sized_instance(tmp_path):
    dataset = _dataset(tmp_path)
    source_labels = [(label["bboxes"].copy() if hasattr(label["bboxes"], "copy") else list(label["bboxes"])) for label in dataset.labels]
    transform = NegativeCanvasCopyPaste(
        dataset, p=1.0, target_policy="empirical", donor_policy="matched",
        target_max_size=8.0, rng=random.Random(4), max_trials=100,
    )
    out = transform(_labels(np.zeros((64, 64, 3), np.uint8), image_index=1))
    assert len(out["instances"]) == 1
    assert 1.0 <= transform.stats["source_size_sum"] / transform.stats["target_size_sum"] < 1.5
    result_size = np.sqrt(np.prod(out["instances"].bboxes[0, 2:] - out["instances"].bboxes[0, :2]))
    assert result_size == 8.0
    for before, label in zip(source_labels, dataset.labels):
        np.testing.assert_array_equal(before, label["bboxes"])


def test_r1_and_r4_donor_ranges_are_exact_and_disjoint(tmp_path):
    transform = NegativeCanvasCopyPaste(_dataset(tmp_path), target_max_size=8.0)
    assert transform._donor_valid(8.0, 8.0)
    assert transform._donor_valid(11.99, 8.0)
    assert not transform._donor_valid(12.0, 8.0)
    transform.donor_policy = "larger"
    assert not transform._donor_valid(11.99, 8.0)
    assert transform._donor_valid(12.0, 8.0)
    assert transform._donor_valid(20.0, 8.0)
    assert not transform._donor_valid(20.01, 8.0)


def test_r4_and_no_blur_share_selection_box_location_but_pixels_differ(tmp_path):
    dataset = _dataset(tmp_path)
    kwargs = dict(
        dataset=dataset, p=1.0, target_policy="deficit", donor_policy="larger",
        target_max_size=8.0, max_trials=100,
    )
    no_blur = NegativeCanvasCopyPaste(**kwargs, degradation="none", rng=random.Random(42))
    r4 = NegativeCanvasCopyPaste(
        **kwargs, degradation="weak_blur", blur_sigma=0.5, rng=random.Random(42),
    )
    out_plain = no_blur(_labels(np.zeros((64, 64, 3), np.uint8), image_index=1))
    out_blur = r4(_labels(np.zeros((64, 64, 3), np.uint8), image_index=1))
    np.testing.assert_array_equal(out_plain["instances"].bboxes, out_blur["instances"].bboxes)
    np.testing.assert_array_equal(out_plain["cls"], out_blur["cls"])
    assert not np.array_equal(out_plain["img"], out_blur["img"])
    assert r4.stats["degradation_applied"] == 1
    assert 1.5 <= r4.stats["source_size_sum"] / r4.stats["target_size_sum"] <= 2.5


def test_failed_placement_keeps_labels_valid(tmp_path):
    dataset = _dataset(tmp_path)
    transform = NegativeCanvasCopyPaste(
        dataset, p=1.0, target_max_size=8.0, rng=random.Random(1), max_trials=1,
    )
    labels = _labels(np.zeros((4, 4, 3), np.uint8), image_index=1)
    transform(labels)
    assert len(labels["instances"]) == 0
    assert labels["cls"].shape == (0, 1)
    assert transform.stats["placement_failed"] == 1


def test_builder_manifest_has_all_fields_and_never_requests_negative_bank(tmp_path):
    hyp = SimpleNamespace(
        copy_paste_enabled=True, copy_paste_mode="negative_canvas",
        negative_cp_target_policy="deficit", negative_cp_donor_policy="larger",
        negative_cp_degradation="weak_blur",
    )
    transform = build_small_object_copy_paste(_dataset(tmp_path), hyp)
    assert isinstance(transform, NegativeCanvasCopyPaste) and transform.p == 0.30
    config = copy_paste_config(hyp)
    assert config["mode"] == "negative_canvas"
    assert config["negative_cp_p"] == 0.30
    assert config["negative_cp_target_policy"] == "deficit"
    assert config["negative_cp_donor_policy"] == "larger"
    assert config["negative_cp_degradation"] == "weak_blur"
    assert config["negative_cp_large_ratio_max"] == 2.5
    assert config["negative_bank_required"] is False
    assert "negcp_bank_path" not in config


def test_disabled_builder_has_no_transform_leakage(tmp_path):
    assert build_small_object_copy_paste(
        _dataset(tmp_path), SimpleNamespace(copy_paste_enabled=False, copy_paste_mode="negative_canvas")
    ) is None


def test_effective_rounded_crop_geometry_and_diagnostics_reset(tmp_path):
    transform = NegativeCanvasCopyPaste(_dataset(tmp_path), donor_policy="matched")
    record = type("Record", (), {"bbox_xyxy": (0.49, 0.49, 15.60, 15.60)})()
    assert transform._effective_source_size(record) == 16.0
    assert not transform._donor_valid(16.0, 10.0)
    transform.stats["negative_seen"] = 2
    transform.reset_stats()
    assert transform.stats["negative_seen"] == 0
    assert {"rejected_boundary", "rejected_collision", "failed_trials"}.issubset(transform.stats)
