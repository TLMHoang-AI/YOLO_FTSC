from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "investigate_gap_feature_miss"
CKPT = OUT / "hf_gap/runs/gap/seed_42/weights/best.pt"
REPO = "duyle2408/levir-yolov8n-p2-channel-descriptor-seed42"
TRAIN_HARD = [28, 1, 30, 21, 12, 29]
TEST_HARD = [28, 1, 30, 21, 12, 13]


def ensure_imports() -> None:
    sys.path.insert(0, str(ROOT / "models_related/ultralytics"))


def ensure_checkpoint() -> None:
    if CKPT.exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=OUT / "hf_gap",
        local_dir_use_symlinks=False,
        allow_patterns=["runs/gap/seed_42/**"],
    )


def resolve_split_root(dataset_root: Path) -> Path:
    direct = dataset_root / "labels"
    nested = dataset_root / "levir_ship_yolo_seed42" / "labels"
    if direct.exists():
        return dataset_root
    if nested.exists():
        return dataset_root / "levir_ship_yolo_seed42"
    raise FileNotFoundError(f"no YOLO labels dir under {dataset_root}")


def labels(split_root: Path, split: str, limit: int) -> list[dict]:
    out = []
    for label_path in sorted((split_root / f"labels/{split}").glob("*.txt")):
        rows = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[1:5]])
        image = split_root / f"images/{split}" / f"{label_path.stem}.png"
        if image.exists() and rows:
            out.append({"image": image, "boxes": np.asarray(rows, dtype=np.float32).reshape(-1, 4)})
        if limit and len(out) >= limit:
            break
    return out


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).requires_grad_(True)


def batch_for(item: dict, device: torch.device) -> dict:
    boxes = torch.as_tensor(item["boxes"], dtype=torch.float32, device=device)
    return {
        "img": image_tensor(item["image"], device),
        "batch_idx": torch.zeros(len(boxes), dtype=torch.long, device=device),
        "cls": torch.zeros((len(boxes), 1), dtype=torch.float32, device=device),
        "bboxes": boxes,
    }


def set_train_raw(model: torch.nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def score_loss(preds) -> torch.Tensor:
    if isinstance(preds, dict) and "scores" in preds:
        return preds["scores"].sum()
    if isinstance(preds, (list, tuple)):
        return sum(score_loss(p) for p in preds if isinstance(p, (dict, list, tuple, torch.Tensor)))
    if torch.is_tensor(preds):
        return preds.float().sum()
    raise TypeError(f"Unsupported prediction type: {type(preds).__name__}")


def patch_detached_gap(attn, capture: dict):
    original = attn.forward

    def forward(self, x):
        x.retain_grad()
        gate = self.act(self.fc(self.pool(x.detach())))
        y = x * gate
        y.retain_grad()
        capture["x"] = x
        capture["y"] = y
        capture["gate"] = gate.detach().flatten()
        return y

    attn.forward = types.MethodType(forward, attn)
    return original


def mean_group(v: np.ndarray, channels: list[int]) -> float:
    return float(v[channels].mean())


def summarize(rows: list[dict], split: str, easy_channels: list[int] | None) -> dict:
    occupied = set(TRAIN_HARD)
    if easy_channels:
        occupied.update(easy_channels)
    groups = {
        "train_hard": TRAIN_HARD,
        "test_hard": TEST_HARD,
        "remaining": [i for i in range(32) if i not in occupied],
    }
    if easy_channels:
        groups["train_easy"] = easy_channels
    out = {"split": split, "n_images": len(rows), "groups": {}}
    for name, channels in groups.items():
        for key in ("gate", "grad_ratio"):
            vals = [mean_group(np.asarray(r[key], dtype=float), channels) for r in rows]
            out["groups"][f"{name}_{key}"] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
            }
    out["channels"] = {}
    for c in range(32):
        out["channels"][str(c)] = {
            "gate_mean": float(np.mean([r["gate"][c] for r in rows])),
            "grad_ratio_mean": float(np.mean([r["grad_ratio"][c] for r in rows])),
        }
    return out


def run_split(model, attn, split_root: Path, split: str, args: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict]:
    rows = []
    for item in labels(split_root, split, args.max_images):
        capture: dict = {}
        original = patch_detached_gap(attn, capture)
        try:
            model.zero_grad(set_to_none=True)
            batch = batch_for(item, device)
            loss = score_loss(model(batch["img"]))
            loss.backward()
            gx = capture["x"].grad.detach().abs().mean(dim=(0, 2, 3)).cpu().numpy()
            gy = capture["y"].grad.detach().abs().mean(dim=(0, 2, 3)).cpu().numpy()
            gate = capture["gate"].cpu().numpy()
            rows.append({"image": item["image"].name, "gate": gate.tolist(), "grad_ratio": (gx / np.clip(gy, 1e-12, None)).tolist()})
        finally:
            attn.forward = original
    return rows, summarize(rows, split, args.easy_channels)


def write_outputs(rows_by_split: dict[str, list[dict]], summary: dict) -> None:
    table = OUT / "tables/gap_channel_grad_scale_by_image.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "channel", "gate", "grad_ratio"])
        w.writeheader()
        for split, rows in rows_by_split.items():
            for row in rows:
                for c, (gate, ratio) in enumerate(zip(row["gate"], row["grad_ratio"])):
                    w.writerow({"split": split, "image": row["image"], "channel": c, "gate": gate, "grad_ratio": ratio})
    (OUT / "gap_channel_grad_scale_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    easy = summary[next(iter(summary))].get("easy_channels") if summary else None
    lines = ["# GAP Channel Gradient Scale", "", f"Train hard: `{TRAIN_HARD}`", f"Test hard: `{TEST_HARD}`"]
    lines.append(f"Train easy: `{easy}`" if easy else "Train easy: not supplied")
    lines.append("")
    for split, data in summary.items():
        lines += [f"## {split}", "", "| group | gate median | grad-ratio median |", "|---|---:|---:|"]
        groups = ["train_hard", "test_hard"]
        if data.get("easy_channels"):
            groups.append("train_easy")
        groups.append("remaining")
        for group in groups:
            lines.append(
                f"| {group} | {data['groups'][group + '_gate']['median']:.4f} | {data['groups'][group + '_grad_ratio']['median']:.4f} |"
            )
        lines.append("")
    (OUT / "report_gap_channel_grad_scale.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--max-images", type=int, default=int(os.environ.get("GAP_CHANNEL_GRAD_MAX_IMAGES", "0")))
    parser.add_argument("--easy-channels", type=int, nargs="+", default=None)
    args = parser.parse_args()
    ensure_imports()
    ensure_checkpoint()
    from ultralytics import YOLO
    from ultralytics.nn.modules import ChannelAttention

    device = torch.device(args.device)
    yolo = YOLO(CKPT)
    model = yolo.model.to(device)
    set_train_raw(model)
    attn = model.model[19]
    if not isinstance(attn, ChannelAttention):
        raise TypeError(f"expected layer 19 ChannelAttention, got {type(attn).__name__}")
    rows_by_split, summary = {}, {}
    split_root = resolve_split_root(args.dataset_root)
    for split in args.splits:
        rows, split_summary = run_split(model, attn, split_root, split, args, device)
        rows_by_split[split] = rows
        split_summary["easy_channels"] = args.easy_channels
        summary[split] = split_summary
    write_outputs(rows_by_split, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
