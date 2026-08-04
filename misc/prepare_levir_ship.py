#!/usr/bin/env python3
"""Prepare reproducible random Ultralytics LEVIR-Ship splits."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path


PUBLISHED_COUNTS = {"train": 2320, "val": 788, "test": 788}
SPLITS = {name: count / sum(PUBLISHED_COUNTS.values()) for name, count in PUBLISHED_COUNTS.items()}
SCENE_RE = re.compile(r"^(.*)_(-?\d+)_(-?\d+)$")


def scene_name(stem: str) -> str:
    match = SCENE_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot extract scene from {stem!r}")
    return match.group(1)


def validate_label(path: Path) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected class + normalized xywh")
        class_id, *box = map(float, values)
        if class_id != 0 or any(value < 0 or value > 1 for value in box) or box[2] <= 0 or box[3] <= 0:
            raise ValueError(f"{path}:{line_number}: invalid ship box {values}")


def split_samples(samples: list[tuple[Path, Path, str]], seed: int) -> dict[str, list[tuple[Path, Path, str]]]:
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    raw_counts = {name: len(samples) * ratio for name, ratio in SPLITS.items()}
    counts = {name: int(value) for name, value in raw_counts.items()}
    for name in sorted(SPLITS, key=lambda key: raw_counts[key] - counts[key], reverse=True)[:len(samples) - sum(counts.values())]:
        counts[name] += 1
    output, start = {}, 0
    for name in SPLITS:
        output[name] = shuffled[start:start + counts[name]]
        start += counts[name]
    return output


def prepare(data_root: Path, out_dir: Path, seed: int = 42, limit: int = 0) -> Path:
    data_root, out_dir = data_root.resolve(), out_dir.resolve()
    image_dir, label_dir = data_root / "All Images", data_root / "All Annotations"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Expected 'All Images' and 'All Annotations' under data root")
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    if image_stems != label_stems:
        raise ValueError(f"Image/label mismatch: {len(image_stems - label_stems)} missing labels, {len(label_stems - image_stems)} orphan labels")
    samples = []
    for stem in sorted(image_stems):
        label = label_dir / f"{stem}.txt"
        validate_label(label)
        samples.append((image_dir / f"{stem}.png", label, scene_name(stem)))
    assigned = split_samples(samples, seed)
    manifest = {"seed": seed, "ratios": SPLITS, "splits": {}}
    for generated in (out_dir / "images", out_dir / "labels"):
        if generated.exists():
            shutil.rmtree(generated)
    for split, records in assigned.items():
        records = records[:limit] if limit else records
        images_out, labels_out = out_dir / "images" / split, out_dir / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True); labels_out.mkdir(parents=True, exist_ok=True)
        for image, label, _ in records:
            for source, destination in ((image, images_out / image.name), (label, labels_out / label.name)):
                if destination.is_symlink() and destination.resolve() != source.resolve():
                    destination.unlink()
                if not destination.exists():
                    destination.symlink_to(source)
        manifest["splits"][split] = {"images": len(records), "scenes": sorted({record[2] for record in records})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    yaml_path = out_dir / "levir_ship.yaml"
    yaml_path.write_text(f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ship\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "LevirShipData")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "datasets/levir_ship_yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(prepare(args.data_root, args.out_dir, args.seed, args.limit))


if __name__ == "__main__":
    main()
