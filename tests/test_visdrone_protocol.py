from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml

from evaluate_test import size_bucket_evaluator as size_eval
from train_visdrone_scripts.evaluate_visdrone import STANDARD_METRIC_KEYS, evaluate_standard
from train_visdrone_scripts.visdrone_protocol import (
    EXPECTED_CLASS_COUNTS,
    EXPECTED_OBJECT_COUNTS,
    EXPECTED_SPLIT_IMAGES,
    OFFICIAL_SPLITS,
    VISDRONE_CLASSES,
    compare_converted_datasets,
    convert_annotation_lines,
    prepare_dataset,
    validate_raw_dataset,
)


def _raw_fixture(root: Path) -> Path:
    specs = {
        "VisDrone2019-DET-train": ("train.jpg", (100, 50), [
            "10,10,20,10,1,1,0,0", "1,2,3,4,0,2,0,0",
            "3,4,5,6,1,10,0,0", "3,4,5,6,1,11,0,0",
        ]),
        "VisDrone2019-DET-val": ("val.png", (80, 40), ["0,0,8,4,1,2,0,0"]),
        "VisDrone2019-DET-test-dev": ("test.jpg", (60, 30), ["2,3,6,3,1,3,0,0"]),
    }
    for folder, (name, size, rows) in specs.items():
        image_dir, annotation_dir = root / folder / "images", root / folder / "annotations"
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)
        Image.new("RGB", size, (10, 20, 30)).save(image_dir / name)
        (annotation_dir / f"{Path(name).stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_official_mapping_counts_and_documented_histograms():
    assert OFFICIAL_SPLITS == {
        "VisDrone2019-DET-train": "train",
        "VisDrone2019-DET-val": "val",
        "VisDrone2019-DET-test-dev": "test",
    }
    assert EXPECTED_SPLIT_IMAGES == {"train": 6471, "val": 548, "test": 1610}
    assert {split: sum(values) for split, values in EXPECTED_CLASS_COUNTS.items()} == EXPECTED_OBJECT_COUNTS


def test_conversion_filter_mapping_and_exact_normalization():
    lines = convert_annotation_lines([
        "10,10,20,10,1,1,0,0",
        "1,2,3,4,0,2,0,0",
        "1,2,3,4,1,10,0,0",
        "1,2,3,4,1,11,0,0",
        "1,2,3,4,1,0,0,0",
    ], 100, 50)
    assert lines == [
        "0 0.200000 0.300000 0.200000 0.200000\n",
        "9 0.025000 0.080000 0.030000 0.080000\n",
    ]


def test_prepare_fixture_jpg_png_yaml_and_basename_parity(tmp_path):
    raw = _raw_fixture(tmp_path / "raw")
    output = tmp_path / "converted"
    counts = {"train": 1, "val": 1, "test": 1}
    data_yaml = prepare_dataset(
        raw, output, expected_image_counts=counts, validate_reference_statistics=False,
    )
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["test"] == "images/test"
    assert payload["names"] == VISDRONE_CLASSES
    assert (output / "images/train/train.jpg").is_file()
    assert (output / "images/val/val.png").is_file()
    for split in counts:
        images = {path.stem for path in (output / "images" / split).iterdir()}
        labels = {path.stem for path in (output / "labels" / split).iterdir()}
        assert images == labels
    assert (output / "labels/train/train.txt").read_text(encoding="utf-8").splitlines() == [
        "0 0.200000 0.300000 0.200000 0.200000",
        "9 0.055000 0.140000 0.050000 0.120000",
    ]


def test_missing_image_or_annotation_pair_fails_loudly(tmp_path):
    raw = _raw_fixture(tmp_path / "raw")
    (raw / "VisDrone2019-DET-val/images/val.png").unlink()
    with pytest.raises(RuntimeError, match="expected 1 images and annotations"):
        validate_raw_dataset(raw, {"train": 1, "val": 1, "test": 1})


def test_cross_repo_output_comparer_is_byte_exact(tmp_path):
    raw = _raw_fixture(tmp_path / "raw")
    counts = {"train": 1, "val": 1, "test": 1}
    left, right = tmp_path / "left", tmp_path / "right"
    for output in (left, right):
        prepare_dataset(raw, output, expected_image_counts=counts, validate_reference_statistics=False)
    assert compare_converted_datasets(left, right)["equal"] is True
    label = right / "labels/train/train.txt"
    label.write_text(label.read_text(encoding="utf-8") + "0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    report = compare_converted_datasets(left, right)
    assert report["equal"] is False
    assert report["mismatched_label_files"] == ["train/train.txt"]


def test_size_definitions_iou_grid_namespace_and_evaluator_path():
    assert size_eval.AREA_RANGES == (
        (1, 1e10), (1, 400), (1, 64), (64, 144),
        (144, 400), (400, 1024), (1024, 9216), (1024, 1e10),
    )
    assert size_eval.IOU_THRESHOLDS == (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
    assert size_eval.SIZE_METRIC_KEYS == (
        "test_size/available", "test_size/AP50", "test_size/AP75", "test_size/mAP50-75",
        "test_size/AP50-Tiny1", "test_size/AP50-Tiny2", "test_size/AP50-Tiny3",
        "test_size/AP50-Small", "test_size/AP50-Medium",
        "test_size/AP-Tiny1", "test_size/AP-Tiny2", "test_size/AP-Tiny3",
        "test_size/AP-Small", "test_size/AP-Medium", "test_size/protocol", "test_size/dataset",
    )
    assert size_eval.resolve_evaluator_path() == Path(__file__).resolve().parents[1] / "vendor/tinyperson_cocoeval.py"


def test_missing_tinybenchmark_evaluator_fails_clearly(monkeypatch, tmp_path):
    missing = tmp_path / "vendor/tinyperson_cocoeval.py"
    monkeypatch.setattr(size_eval, "EVALUATOR_PATH", missing)
    with pytest.raises(FileNotFoundError, match="TinyBenchmark evaluator is missing"):
        size_eval.resolve_evaluator_path()


def test_secondary_evaluator_collapses_gt_and_predictions_to_one_class(tmp_path):
    image_dir, label_dir = tmp_path / "images/test", tmp_path / "labels/test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "scene.jpg"
    Image.new("RGB", (100, 50)).save(image_path)
    (label_dir / "scene.txt").write_text("7 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    ground_truth = size_eval._coco_ground_truth([image_path])
    assert {item["category_id"] for item in ground_truth["annotations"]} == {1}
    assert ground_truth["categories"] == [{"id": 1, "name": "object"}]

    class Tensor:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self.value

    boxes = SimpleNamespace(xyxy=Tensor([[1, 2, 11, 12]]), conf=Tensor([0.7]), cls=Tensor([9]))
    model = SimpleNamespace(predict=lambda **kwargs: iter([SimpleNamespace(boxes=boxes)]))
    predictions = size_eval._predictions(
        model, [image_path], imgsz=640, batch=1, device="cpu", workers=0,
    )
    assert predictions == [{"image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10], "score": 0.7}]


def test_standard_evaluation_remains_native_multiclass_and_split_qualified(tmp_path):
    calls = []
    result = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.4, "metrics/mAP50-95(B)": 0.2})
    model = SimpleNamespace(val=lambda **kwargs: calls.append(kwargs) or result)
    metrics = evaluate_standard(model, tmp_path / "visdrone.yaml", tmp_path, device="cpu", workers=0)
    assert tuple(key for key in metrics if key in STANDARD_METRIC_KEYS) == STANDARD_METRIC_KEYS
    assert metrics["standard_eval/class_agnostic"] is False
    assert [call["split"] for call in calls] == ["val", "test"]
    assert all(call["iou"] == 0.5 and "classes" not in call for call in calls)
