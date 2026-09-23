#!/usr/bin/env python3
"""Prepare and validate the partner-compatible VisDrone2019-DET dataset.

The converter uses the official train/val/test-dev folders without random
reassignment. It never downloads data and has no dependency on the partner
repository at runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Iterable, Mapping

from PIL import Image
import yaml


OFFICIAL_SPLITS = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}
EXPECTED_SPLIT_IMAGES = {"train": 6471, "val": 548, "test": 1610}
VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}
EXPECTED_OBJECT_COUNTS = {"train": 343205, "val": 38759, "test": 75102}
EXPECTED_CLASS_COUNTS = {
    "train": (79337, 27059, 10480, 144867, 24956, 12875, 4812, 3246, 5926, 29647),
    "val": (8844, 5125, 1287, 14064, 1975, 750, 1045, 532, 251, 4886),
    "test": (21006, 6376, 1302, 28074, 5771, 2659, 530, 599, 2940, 5845),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _image_map(directory: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images:
                raise RuntimeError(f"Duplicate image basename {path.stem!r} under {directory}")
            images[path.stem] = path
    return images


def validate_raw_dataset(
    data_root: Path,
    expected_image_counts: Mapping[str, int] = EXPECTED_SPLIT_IMAGES,
) -> dict[str, dict[str, int]]:
    """Validate official folders, counts, and image/annotation basename parity."""
    report: dict[str, dict[str, int]] = {}
    for source_name, split in OFFICIAL_SPLITS.items():
        split_root = data_root / source_name
        image_dir, annotation_dir = split_root / "images", split_root / "annotations"
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(f"Missing images/annotations under official split {split_root}")
        images = _image_map(image_dir)
        annotations = {path.stem: path for path in annotation_dir.glob("*.txt") if path.is_file()}
        expected = expected_image_counts[split]
        if len(images) != expected or len(annotations) != expected:
            raise RuntimeError(
                f"Noncanonical VisDrone {split}: expected {expected} images and annotations, "
                f"got {len(images)} images and {len(annotations)} annotations"
            )
        if images.keys() != annotations.keys():
            missing_images = sorted(annotations.keys() - images.keys())
            missing_annotations = sorted(images.keys() - annotations.keys())
            raise RuntimeError(
                f"VisDrone {split} basename mismatch: missing images={missing_images[:10]}, "
                f"missing annotations={missing_annotations[:10]}"
            )
        report[split] = {"images": len(images), "annotations": len(annotations)}
    return report


def convert_annotation_lines(lines: Iterable[str], width: int, height: int) -> list[str]:
    """Apply the partner's filtering, class mapping, and six-decimal xywh conversion."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    dw, dh = 1.0 / width, 1.0 / height
    converted: list[str] = []
    for line in lines:
        row = [value.strip() for value in line.split(",") if value.strip()]
        if len(row) < 6 or row[4] == "0":
            continue
        x, y, box_width, box_height = map(int, row[:4])
        cls = int(row[5]) - 1
        if cls < 0 or cls > 9:
            continue
        x_center = (x + box_width / 2.0) * dw
        y_center = (y + box_height / 2.0) * dh
        converted.append(
            f"{cls} {x_center:.6f} {y_center:.6f} "
            f"{box_width * dw:.6f} {box_height * dh:.6f}\n"
        )
    return converted


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace existing output image: {destination}")
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def convert_split(source_dir: Path, images_out: Path, labels_out: Path) -> dict[str, int]:
    """Convert one already-validated official split."""
    images = _image_map(source_dir / "images")
    annotations = {path.stem: path for path in (source_dir / "annotations").glob("*.txt") if path.is_file()}
    if images.keys() != annotations.keys():
        raise RuntimeError(f"Image/annotation basename mismatch under {source_dir}")
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    object_count = 0
    class_counts: Counter[int] = Counter()
    for stem in sorted(annotations):
        source_image = images[stem]
        destination_image = images_out / source_image.name
        _link_or_copy(source_image, destination_image)
        with Image.open(source_image) as image:
            width, height = image.size
        converted = convert_annotation_lines(
            annotations[stem].read_text(encoding="utf-8").splitlines(), width, height
        )
        (labels_out / f"{stem}.txt").write_text("".join(converted), encoding="utf-8")
        object_count += len(converted)
        class_counts.update(int(line.split()[0]) for line in converted)
    return {
        "images": len(images),
        "labels": len(annotations),
        "objects": object_count,
        **{f"class_{class_id}": class_counts[class_id] for class_id in VISDRONE_CLASSES},
    }


def _yaml_payload(output_dir: Path) -> dict:
    return {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": VISDRONE_CLASSES,
    }


def prepare_dataset(
    data_root: Path,
    output_dir: Path,
    *,
    expected_image_counts: Mapping[str, int] = EXPECTED_SPLIT_IMAGES,
    validate_reference_statistics: bool = True,
) -> Path:
    """Validate and convert the official splits, returning ``visdrone.yaml``."""
    data_root, output_dir = data_root.resolve(), output_dir.resolve()
    validate_raw_dataset(data_root, expected_image_counts)
    for source_name, split in OFFICIAL_SPLITS.items():
        convert_split(
            data_root / source_name,
            output_dir / "images" / split,
            output_dir / "labels" / split,
        )
    yaml_path = output_dir / "visdrone.yaml"
    yaml_path.write_text(yaml.safe_dump(_yaml_payload(output_dir), sort_keys=False), encoding="utf-8")
    validate_converted_dataset(
        output_dir,
        expected_image_counts=expected_image_counts,
        validate_reference_statistics=validate_reference_statistics,
    )
    return yaml_path


def _converted_stats(output_dir: Path, split: str) -> dict:
    images = _image_map(output_dir / "images" / split)
    labels = {path.stem: path for path in (output_dir / "labels" / split).glob("*.txt") if path.is_file()}
    if images.keys() != labels.keys():
        raise RuntimeError(f"Generated {split} image/label basename sets differ")
    counts: Counter[int] = Counter()
    objects = 0
    for label_path in labels.values():
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            values = line.split()
            if len(values) != 5:
                raise RuntimeError(f"Malformed YOLO row {label_path}:{line_number}")
            class_id = int(values[0])
            if class_id not in VISDRONE_CLASSES:
                raise RuntimeError(f"Out-of-range class {class_id} at {label_path}:{line_number}")
            tuple(map(float, values[1:]))
            counts[class_id] += 1
            objects += 1
    return {
        "images": len(images),
        "labels": len(labels),
        "objects": objects,
        "class_counts": [counts[index] for index in VISDRONE_CLASSES],
    }


def validate_converted_dataset(
    output_dir: Path,
    *,
    expected_image_counts: Mapping[str, int] = EXPECTED_SPLIT_IMAGES,
    validate_reference_statistics: bool = True,
) -> dict[str, dict]:
    """Validate generated counts, basename parity, bbox totals, and histograms."""
    output_dir = output_dir.resolve()
    yaml_path = output_dir / "visdrone.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing generated YAML: {yaml_path}")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    expected_yaml = _yaml_payload(output_dir)
    if payload != expected_yaml:
        raise RuntimeError(f"Generated YAML semantics differ: {payload!r} != {expected_yaml!r}")
    report = {}
    for split, expected_images in expected_image_counts.items():
        stats = _converted_stats(output_dir, split)
        if stats["images"] != expected_images or stats["labels"] != expected_images:
            raise RuntimeError(f"Generated {split} count mismatch: {stats}")
        if validate_reference_statistics:
            if stats["objects"] != EXPECTED_OBJECT_COUNTS[split]:
                raise RuntimeError(
                    f"Generated {split} object count {stats['objects']} != {EXPECTED_OBJECT_COUNTS[split]}"
                )
            if tuple(stats["class_counts"]) != EXPECTED_CLASS_COUNTS[split]:
                raise RuntimeError(f"Generated {split} class histogram mismatch: {stats['class_counts']}")
        report[split] = stats
    return report


def compare_converted_datasets(left: Path, right: Path) -> dict:
    """Compare two converted datasets without invoking either training stack."""
    left, right = left.resolve(), right.resolve()
    left_yaml = yaml.safe_load((left / "visdrone.yaml").read_text(encoding="utf-8"))
    right_yaml = yaml.safe_load((right / "visdrone.yaml").read_text(encoding="utf-8"))
    for payload in (left_yaml, right_yaml):
        payload["path"] = "<dataset-root>"
    mismatched_labels: list[str] = []
    splits = {}
    for split in EXPECTED_SPLIT_IMAGES:
        left_images = set(_image_map(left / "images" / split))
        right_images = set(_image_map(right / "images" / split))
        left_labels = {path.stem: path for path in (left / "labels" / split).glob("*.txt")}
        right_labels = {path.stem: path for path in (right / "labels" / split).glob("*.txt")}
        common = left_labels.keys() & right_labels.keys()
        mismatched_labels.extend(
            f"{split}/{stem}.txt"
            for stem in sorted(common)
            if left_labels[stem].read_bytes() != right_labels[stem].read_bytes()
        )
        splits[split] = {
            "image_basenames_equal": left_images == right_images,
            "label_basenames_equal": left_labels.keys() == right_labels.keys(),
            "left": _converted_stats(left, split),
            "right": _converted_stats(right, split),
        }
    report = {
        "yaml_semantics_equal": left_yaml == right_yaml,
        "mismatched_label_files": mismatched_labels,
        "splits": splits,
    }
    report["equal"] = (
        report["yaml_semantics_equal"]
        and not mismatched_labels
        and all(
            item["image_basenames_equal"]
            and item["label_basenames_equal"]
            and item["left"] == item["right"]
            for item in splits.values()
        )
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output-dir", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        yaml_path = prepare_dataset(args.data_root, args.output_dir)
        print(json.dumps({"yaml": str(yaml_path), "validation": validate_converted_dataset(args.output_dir)}, indent=2))
    elif args.command == "validate":
        print(json.dumps(validate_converted_dataset(args.output_dir), indent=2))
    else:
        report = compare_converted_datasets(args.left, args.right)
        print(json.dumps(report, indent=2))
        if not report["equal"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
