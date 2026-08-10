#!/usr/bin/env python3
"""Inference-only context-radius intervention on one trained global KVCA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))

from train_levir_scripts import analyze_p2_cbam_ranking as diagnostic  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import KVCompressedAttention, PatchKVCompressedAttention  # noqa: E402


RADII = {"mask_r0": 0, "mask_r1": 1, "mask_r2": 2}
MODELS = ("global", *RADII)


def intervention_checkpoint(source: Path, output: Path, radius: int) -> None:
    wrapper = YOLO(source)
    old = wrapper.model.model[19]
    if not isinstance(old, KVCompressedAttention) or isinstance(old, PatchKVCompressedAttention):
        raise TypeError(f"Expected global KVCompressedAttention at layer 19, got {type(old).__name__}")
    if old.mode != "group_weight" or old.sr_ratio != 8 or old.num_heads != 4:
        raise ValueError("Unexpected global KVCA configuration")
    replacement = PatchKVCompressedAttention(32, 32, 4, 8, radius, residual=old.residual)
    replacement.load_state_dict(old.state_dict(), strict=True)
    for attribute in ("i", "f", "type", "np"):
        if hasattr(old, attribute):
            setattr(replacement, attribute, getattr(old, attribute))
    replacement.train(old.training)
    wrapper.model.model[19] = replacement
    if sum(p.numel() for p in replacement.parameters()) != sum(p.numel() for p in old.parameters()):
        raise RuntimeError("Radius intervention changed parameter count")
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(output)
    reloaded = YOLO(output)
    layer = reloaded.model.model[19]
    if not isinstance(layer, PatchKVCompressedAttention) or layer.patch_radius != radius:
        raise RuntimeError(f"Failed to persist radius-{radius} intervention checkpoint")


def evaluate(checkpoint: Path, output: Path, args: argparse.Namespace) -> dict:
    result = YOLO(checkpoint).val(
        data=str(args.data), split="test", imgsz=args.imgsz, batch=args.batch_size,
        device=args.device, workers=args.workers, plots=False, iou=0.5,
        project=str(output / "evaluation"), name=checkpoint.stem, exist_ok=True,
    )
    metrics = {"nms_iou": 0.5}
    metrics.update({key: float(value) for key, value in result.results_dict.items()})
    metrics["metrics/mAP75(B)"] = float(result.box.map75)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    args = parser.parse_args()
    for path in (args.global_checkpoint, args.images, args.data):
        if not path.exists():
            raise FileNotFoundError(path)
    images = sorted(args.images.glob("*.png"))
    manifest = json.loads((args.images.parent.parent / "manifest.json").read_text())
    if len(images) != 788 or manifest.get("seed") != 42:
        raise RuntimeError("Expected fixed seed-42 test split with 788 images")
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoints = {"global": args.global_checkpoint}
    for name, radius in RADII.items():
        path = args.output / "checkpoints" / f"global_kvca_{name}.pt"
        intervention_checkpoint(args.global_checkpoint, path, radius)
        checkpoints[name] = path

    ap = {name: evaluate(checkpoint, args.output, args) for name, checkpoint in checkpoints.items()}
    diagnostic.EXPECTED_LEVELS = {name: 1 for name in MODELS}
    raw_args = SimpleNamespace(device=args.device, imgsz=args.imgsz)
    rows = {name: diagnostic.inspect_model(name, checkpoints[name], images, raw_args) for name in MODELS}
    if {len(value) for value in rows.values()} != {696}:
        raise RuntimeError(f"Expected 696 GT per model, got {dict((k, len(v)) for k, v in rows.items())}")

    fields = ["model", "image", "gt_index", "area_px2", "size_group", "candidate_count", *diagnostic.METRICS]
    with (args.output / "per_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in MODELS:
            writer.writerows(rows[name])
    summary = {
        "protocol": {
            "intervention": "same trained global KVCA weights; inference-only compressed-grid radius mask",
            "seed": 42, "split": "test", "images": 788, "gt": 696, "imgsz": args.imgsz,
            "nms_iou": 0.5,
            "candidate_rule": "P2 anchor center inside GT; overlapping anchors assigned to highest-IoU GT",
            "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        },
        "test_ap": ap,
        "raw_p2": {name: diagnostic.descriptive_summary(rows[name]) for name in MODELS},
        "paired_delta_mask_minus_global": {
            name: diagnostic.paired_delta(rows, name, "global", args.bootstrap_repeats) for name in RADII
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"test_ap": ap, "raw_all": {n: summary["raw_p2"][n]["all"] for n in MODELS}}, indent=2))


if __name__ == "__main__":
    main()
