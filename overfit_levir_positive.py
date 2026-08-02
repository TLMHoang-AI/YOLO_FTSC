#!/usr/bin/env python3
"""Overfit YOLOv8-P2 DBSS and HIT on the same 30 positive LEVIR images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG_DIR = ROOT / "models_related/models_config/yolov8/levir"
CONFIGS = {
    "dbss": CONFIG_DIR / "yolov8n_p2_levir_dbss_full.yaml",
    "hit": CONFIG_DIR / "yolov8n_p2_levir_hit_full.yaml",
}


def positive_samples(data_root: Path, count: int) -> list[tuple[Path, Path]]:
    images, labels = data_root / "All Images", data_root / "All Annotations"
    samples = []
    for label in sorted(labels.glob("*.txt")):
        if label.read_text(encoding="utf-8").strip():
            image = images / f"{label.stem}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            samples.append((image.resolve(), label.resolve()))
            if len(samples) == count:
                break
    if len(samples) != count:
        raise ValueError(f"Expected {count} positive samples, found {len(samples)}")
    return samples


def link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def prepare_dataset(data_root: Path, output: Path, count: int) -> Path:
    samples = positive_samples(data_root, count)
    image_out, label_out = output / "images", output / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    expected = {image.name for image, _ in samples}
    for directory in (image_out, label_out):
        for path in directory.iterdir():
            expected_name = path.name if directory == image_out else f"{path.stem}.png"
            if expected_name not in expected and path.is_symlink():
                path.unlink()
    for image, label in samples:
        link(image, image_out / image.name)
        link(label, label_out / label.name)
    manifest = {"purpose": "positive-only memorization test", "count": count, "samples": sorted(expected)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    yaml = output / "levir_positive.yaml"
    yaml.write_text(f"path: {output.resolve()}\ntrain: images\nval: images\ntest: images\nnames:\n  0: ship\n", encoding="utf-8")
    return yaml


def loss_declined(results_csv: Path) -> tuple[bool, float, float]:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    loss_keys = [key for key in rows[0] if key.startswith("train/") and key.endswith("_loss")]
    first = sum(float(rows[0][key]) for key in loss_keys)
    last = sum(float(rows[-1][key]) for key in loss_keys)
    return math.isfinite(first) and math.isfinite(last) and last < first, first, last


def run_variant(name: str, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))
    from ultralytics import YOLO

    run_dir = args.project / name
    metrics_path = run_dir / "gate_metrics.json"
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    model = YOLO(CONFIGS[name])
    model.load("yolov8n.pt", smart_transfer=True)
    model.train(
        data=str(data_yaml), epochs=10, imgsz=512, batch=8, device=args.device, workers=args.workers,
        seed=42, deterministic=True, patience=0, warmup_epochs=0.0, weight_decay=0.0,
        mosaic=0.0, mixup=0.0, cutmix=0.0, fliplr=0.0, flipud=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        project=str(args.project), name=name, exist_ok=True,
    )
    result = YOLO(run_dir / "weights/best.pt").val(
        data=str(data_yaml), split="val", imgsz=512, batch=8, device=args.device, workers=args.workers,
        project=str(run_dir / "evaluation"), name="memorization", exist_ok=True, plots=True,
    )
    declined, first_loss, last_loss = loss_declined(run_dir / "results.csv")
    finite = all(math.isfinite(float(value)) for value in (result.results_dict or {}).values())
    metrics = {
        "variant": name,
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "first_train_loss": first_loss,
        "last_train_loss": last_loss,
        "loss_declined": declined,
        "finite": finite,
    }
    metrics["passed"] = finite and declined and metrics["mAP50"] >= args.map50_threshold
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    YOLO(run_dir / "weights/best.pt").predict(
        source=str(data_yaml.parent / "images"), imgsz=512, device=args.device,
        project=str(run_dir), name="predictions", exist_ok=True, save=True, max_det=300,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-out", type=Path, default=ROOT / "datasets/levir_positive_30")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_positive_overfit")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--map50-threshold", type=float, default=0.75)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.project, args.dataset_out = args.project.resolve(), args.dataset_out.resolve()
    data_yaml = prepare_dataset(args.data_root.resolve(), args.dataset_out, args.count)
    results = [run_variant(name, data_yaml, args) for name in CONFIGS]
    summary = {"passed": all(result["passed"] for result in results), "results": results}
    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / "gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("Positive-overfit gate failed; inspect metrics and predictions before marimo launch")


if __name__ == "__main__":
    main()
