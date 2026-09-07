from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from train_tinyperson_scripts import train_all_tinyperson as tiny


def _write_corner_json(path: Path, image_records: list[dict], annotations: list[dict], old_images: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "type": "instance",
                "images": image_records,
                "old_images": old_images or [],
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )


def test_yolo_labels_use_declared_crop_dimensions() -> None:
    lines = tiny.yolo_label_lines([{"bbox": [1, 2, 10, 20]}], width=20, height=40)
    assert lines == ["0 0.300000 0.300000 0.500000 0.500000\n"]


def test_write_corner_crop_uses_corner_and_dynamic_shape(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(source)
    info = {"id": 4, "width": 4, "height": 3, "corner": [2, 1, 6, 4]}
    image_out = tmp_path / "crop.jpg"
    label_out = tmp_path / "crop.txt"

    tiny.write_corner_crop(source, info, [{"bbox": [1, 1, 2, 1]}], image_out, label_out)

    assert Image.open(image_out).size == (4, 3)
    assert label_out.read_text(encoding="utf-8") == "0 0.500000 0.500000 0.500000 0.333333\n"


def test_mean_ap50_75_uses_first_six_thresholds() -> None:
    import numpy as np

    result = type("Result", (), {"box": type("Box", (), {"all_ap": np.arange(10, dtype=float)[None, :]})()})()
    assert tiny.mean_ap50_75(result) == 2.5


def test_tiny_precision_ap_ignores_missing_slices() -> None:
    import numpy as np

    precision = np.full((1, 3, 1, 1, 1), -1.0)
    precision[0, :, 0, 0, 0] = [0.25, 0.5, -1.0]
    assert tiny.precision_ap(precision, 0, 0) == 0.375


def test_nms_matches_corner_merge_behavior() -> None:
    detections = [
        {"xyxy": [0, 0, 10, 10], "score": 0.9},
        {"xyxy": [1, 1, 11, 11], "score": 0.8},
        {"xyxy": [30, 30, 40, 40], "score": 0.7},
    ]
    kept = tiny.nms_detections(detections, iou_threshold=0.5)
    assert [item["score"] for item in kept] == [0.9, 0.7]


def test_prepare_test_set_materializes_corner_windows(tmp_path: Path) -> None:
    data_root = tmp_path / "TinyPerson"
    image_root = data_root / "test" / "labeled_images"
    image_root.mkdir(parents=True)
    source_name = "labeled_images/original.jpg"
    Image.new("RGB", (8, 6), (1, 2, 3)).save(image_root / "original.jpg")
    corner_json = data_root / tiny.TEST_CORNER_JSON
    _write_corner_json(
        corner_json,
        [
            {"id": 0, "file_name": source_name, "width": 4, "height": 3, "corner": [0, 0, 4, 3]},
            {"id": 1, "file_name": source_name, "width": 4, "height": 3, "corner": [4, 3, 8, 6]},
        ],
        [{"id": 0, "image_id": 0, "bbox": [0, 0, 2, 2], "ignore": False}],
        old_images=[{"id": 10, "file_name": source_name, "width": 8, "height": 6}],
    )

    output = tiny.prepare_test_set(data_root, tmp_path / "datasets")

    assert sorted(p.name for p in (output / "images").glob("*.jpg")) == ["test_0.jpg", "test_1.jpg"]
    assert Image.open(output / "images/test_1.jpg").size == (4, 3)
    assert (output / "labels/test_0.txt").read_text(encoding="utf-8") == "0 0.250000 0.333333 0.500000 0.666667\n"
    manifest = json.loads((output / "corner_manifest.json").read_text(encoding="utf-8"))
    assert [record["corner"] for record in manifest["images"]] == [[0, 0, 4, 3], [4, 3, 8, 6]]


def test_prepare_seed_dataset_splits_by_original_file(tmp_path: Path) -> None:
    data_root = tmp_path / "TinyPerson"
    image_root = data_root / "erase_with_uncertain_dataset" / "train" / "labeled_images"
    image_root.mkdir(parents=True)
    records = []
    annotations = []
    for source_index, name in enumerate(("a.jpg", "b.jpg")):
        Image.new("RGB", (8, 6), (source_index, 2, 3)).save(image_root / name)
        for window_index, corner in enumerate(([0, 0, 4, 3], [4, 3, 8, 6])):
            image_id = source_index * 2 + window_index
            records.append({"id": image_id, "file_name": f"labeled_images/{name}", "width": 4, "height": 3, "corner": corner})
            annotations.append({"id": image_id, "image_id": image_id, "bbox": [0, 0, 1, 1], "ignore": False})
    train_json = data_root / tiny.TRAIN_CORNER_JSON
    _write_corner_json(train_json, records, annotations)

    test_out = tmp_path / "test"
    (test_out / "images").mkdir(parents=True)
    (test_out / "labels").mkdir()
    seed_dir = tiny.prepare_seed_dataset(data_root, tmp_path / "datasets", test_out, seed=1)

    manifest = json.loads((seed_dir / "corner_manifest.json").read_text(encoding="utf-8"))
    train_names = {item["file_name"] for item in manifest["train"]}
    val_names = {item["file_name"] for item in manifest["val"]}
    assert train_names.isdisjoint(val_names)
    assert train_names | val_names == {"labeled_images/a.jpg", "labeled_images/b.jpg"}
