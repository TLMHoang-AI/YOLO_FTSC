#!/usr/bin/env python3
"""Run the matched raw-P2 ranking diagnostic for global KVCA and Patch-KVCA r0/r1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from train_levir_scripts import analyze_p2_cbam_ranking as diagnostic


MODELS = ("global_kvca", "patch_kvca_r0", "patch_kvca_r1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-kvca", type=Path, required=True)
    parser.add_argument("--patch-r0", type=Path, required=True)
    parser.add_argument("--patch-r1", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    args = parser.parse_args()
    checkpoints = {
        "global_kvca": args.global_kvca,
        "patch_kvca_r0": args.patch_r0,
        "patch_kvca_r1": args.patch_r1,
    }
    for path in (*checkpoints.values(), args.images):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads((args.images.parent.parent / "manifest.json").read_text())
    images = sorted(args.images.glob("*.png"))
    if manifest.get("seed") != 42 or len(images) != 788:
        raise RuntimeError("Expected the fixed seed-42 split with 788 test images")

    diagnostic.EXPECTED_LEVELS = {name: 1 for name in MODELS}
    rows = {name: diagnostic.inspect_model(name, checkpoints[name], images, args) for name in MODELS}
    if {len(value) for value in rows.values()} != {696}:
        raise RuntimeError(f"Expected 696 GT for every model, got {dict(map(lambda x: (x[0], len(x[1])), rows.items()))}")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["model", "image", "gt_index", "area_px2", "size_group", "candidate_count", *diagnostic.METRICS]
    with (args.output / "per_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in MODELS:
            writer.writerows(rows[name])
    summary = {
        "protocol": {
            "seed": 42,
            "split": "test",
            "images": 788,
            "gt": 696,
            "imgsz": args.imgsz,
            "candidate_rule": "P2 anchor center inside GT; overlapping anchors assigned to highest-IoU GT",
            "prediction_stage": "decoded P2 boxes and sigmoid class scores before threshold and NMS",
            "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        },
        "models": {name: diagnostic.descriptive_summary(rows[name]) for name in MODELS},
        "paired_delta_r0_minus_global": diagnostic.paired_delta(
            rows, "patch_kvca_r0", "global_kvca", args.bootstrap_repeats
        ),
        "paired_delta_r1_minus_global": diagnostic.paired_delta(
            rows, "patch_kvca_r1", "global_kvca", args.bootstrap_repeats
        ),
        "paired_delta_r1_minus_r0": diagnostic.paired_delta(
            rows, "patch_kvca_r1", "patch_kvca_r0", args.bootstrap_repeats
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: summary["models"][name]["all"] for name in MODELS}, indent=2))


if __name__ == "__main__":
    main()
