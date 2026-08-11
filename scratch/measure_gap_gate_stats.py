#!/usr/bin/env python3
"""Measure train-split ChannelAttention gate stats for GAP checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
sys.path.insert(0, str(ROOT / "train_levir_scripts"))

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules.conv import ChannelAttention


def data_yaml_from_args(args):
    if args.data_yaml:
        return Path(args.data_yaml)
    import train_all_levir_yolov8n_p2_routing as workflow

    return workflow.prepare_fixed_split(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-yaml", type=Path)
    p.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/levir_ship_yolo_seed42")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=ROOT / "scratch/gap_gate_train_stats.json")
    args = p.parse_args()

    data_yaml = data_yaml_from_args(args)
    dataset = check_det_dataset(str(data_yaml))
    model = YOLO(args.checkpoint).to(args.device)
    model.model.eval()

    layers = [m for m in model.model.modules() if isinstance(m, ChannelAttention)]
    if len(layers) != 1:
        raise SystemExit(f"expected 1 ChannelAttention, found {len(layers)}")
    layer = layers[0]

    sums = {"n": 0, "sum": 0.0, "sum_delta2": 0.0}
    mins, maxs = [], []

    def capture(_module, gate):
        g = gate.detach().float().cpu().flatten(1)
        m = g.mean(dim=1, keepdim=True)
        sums["n"] += g.numel()
        sums["sum"] += g.sum().item()
        sums["sum_delta2"] += ((g - m) ** 2).sum().item()
        mins.append(g.flatten())
        return gate

    layer.override_gate_fn = capture
    validator = DetectionValidator(args={"data": str(data_yaml), "device": args.device, "imgsz": 512})
    validator.data = dataset
    validator.device = torch.device(args.device)
    loader = validator.get_dataloader(dataset["train"], batch_size=args.batch)

    with torch.no_grad():
        for batch in loader:
            batch = validator.preprocess(batch)
            model.model(batch["img"])

    flat = torch.cat(mins)
    out = {
        "checkpoint": str(args.checkpoint),
        "data_yaml": str(data_yaml),
        "split": "train",
        "gate_count": sums["n"],
        "mu": sums["sum"] / sums["n"],
        "sigma_delta": (sums["sum_delta2"] / sums["n"]) ** 0.5,
        "q01": float(torch.quantile(flat, 0.01)),
        "q99": float(torch.quantile(flat, 0.99)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
