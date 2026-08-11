from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "investigate_gap_feature_miss"
HF_DIR = OUT / "hf_gap"
CKPT = HF_DIR / "runs/gap/seed_42/weights/best.pt"
REPO = "duyle2408/levir-yolov8n-p2-channel-descriptor-seed42"
TABLE = OUT / "tables/gap_gradient_coupling_per_image.csv"
SUMMARY = OUT / "gap_gradient_coupling_summary.json"
REPORT = OUT / "report_gap_gradient_coupling.md"
EPS = 1e-12


def ensure_imports() -> None:
    sys.path.insert(0, str(ROOT / "models_related/ultralytics"))


def ensure_checkpoint() -> None:
    if CKPT.exists():
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=HF_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=["runs/gap/seed_42/**"],
    )


def load_labels() -> list[dict]:
    ann = json.loads((ROOT / "datasets/levir_ship_yolo_seed42/annotations/test.json").read_text())
    image_dir = ROOT / "LevirShipData/All Images"
    by_id = {im["id"]: im for im in ann["images"]}
    boxes_by_id: dict[int, list[list[float]]] = {im["id"]: [] for im in ann["images"]}
    for obj in ann["annotations"]:
        x, y, w, h = obj["bbox"]
        boxes_by_id[obj["image_id"]].append([(x + w / 2) / 512, (y + h / 2) / 512, w / 512, h / 512])
    labels = []
    for image_id, item in by_id.items():
        image = image_dir / item["file_name"]
        if image.exists():
            labels.append(
                {
                    "image": image,
                    "name": image.name,
                    "boxes_xywhn": np.asarray(boxes_by_id[image_id], dtype=np.float32).reshape(-1, 4),
                }
            )
    return labels


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).requires_grad_(True)


def batch_for(item: dict, device: torch.device) -> dict:
    boxes = torch.as_tensor(item["boxes_xywhn"], dtype=torch.float32, device=device)
    n = int(boxes.shape[0])
    return {
        "img": image_tensor(item["image"], device),
        "batch_idx": torch.zeros(n, dtype=torch.long, device=device),
        "cls": torch.zeros((n, 1), dtype=torch.float32, device=device),
        "bboxes": boxes,
    }


def p2_object_mask(boxes_xywhn: np.ndarray, h: int, w: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, 1, h, w), dtype=torch.bool, device=device)
    for x, y, bw, bh in boxes_xywhn:
        x1 = max(0, min(w - 1, int(math.floor((x - bw / 2) * w))))
        y1 = max(0, min(h - 1, int(math.floor((y - bh / 2) * h))))
        x2 = max(x1 + 1, min(w, int(math.ceil((x + bw / 2) * w))))
        y2 = max(y1 + 1, min(h, int(math.ceil((y + bh / 2) * h))))
        mask[:, :, y1:y2, x1:x2] = True
    return mask


def flatten_preds(preds: dict) -> torch.Tensor:
    parts = []
    for key in ("boxes", "scores"):
        if key in preds:
            parts.append(preds[key].detach().float().flatten().cpu())
    return torch.cat(parts) if parts else torch.empty(0)


def set_train_raw(model: torch.nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def cls_or_total_loss(model: torch.nn.Module, preds: dict, batch: dict) -> tuple[torch.Tensor, str, float]:
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    try:
        loss = model.criterion.get_assigned_targets_and_loss(model.criterion.parse_output(preds), batch)[1]
        return loss[1] * batch["img"].shape[0], "classification", float(loss[1].detach().item())
    except Exception:
        loss, items = model.criterion(preds, batch)
        return loss.sum(), "total_detection_fallback", float(items.sum().detach().item())


def patch_attention(attn, detach_descriptor: bool, capture: dict):
    original = attn.forward

    def forward(self, x):
        capture["x"] = x
        x.retain_grad()
        pooled = self.pool(x).detach() if detach_descriptor else self.pool(x)
        average = self.fc(pooled)
        descriptor = getattr(self, "descriptor", "avg")
        if descriptor == "avg":
            gate = average
        else:
            max_pooled = self.max_pool(x).detach() if detach_descriptor else self.max_pool(x)
            maximum = self.fc(max_pooled) if descriptor == "max" else self.max_fc(max_pooled)
            gate = maximum if descriptor == "max" else average + maximum
        gate = self.act(gate)
        capture["gate"] = gate.detach()
        return x * gate

    attn.forward = types.MethodType(forward, attn)
    return original


def one_pass(model: torch.nn.Module, attn, item: dict, device: torch.device, detach_descriptor: bool) -> dict:
    capture: dict = {}
    original = patch_attention(attn, detach_descriptor, capture)
    try:
        model.zero_grad(set_to_none=True)
        batch = batch_for(item, device)
        preds = model(batch["img"])
        loss, loss_type, loss_value = cls_or_total_loss(model, preds, batch)
        loss.backward()
        x = capture["x"]
        return {
            "grad": x.grad.detach().float().clone(),
            "gate": capture["gate"].float().clone(),
            "p2": x.detach().float().clone(),
            "preds": flatten_preds(preds),
            "loss_type": loss_type,
            "loss_value": loss_value,
        }
    finally:
        attn.forward = original


def safe_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).detach().cpu())


def mean_abs(x: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float(x.abs().masked_select(mask.expand_as(x)).mean().detach().cpu())


def read_failure_context() -> dict[str, dict]:
    path = OUT / "tables/gap_seed42_test_gt_diagnostic.csv"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rec = out.setdefault(row["image"], {"low_conf_fn": 0, "best_conf": []})
            if row["reason"] == "classification_low_conf":
                rec["low_conf_fn"] += 1
            if row["detected"].lower() == "true":
                rec["best_conf"].append(float(row["best_candidate_conf"]))
    for rec in out.values():
        vals = rec["best_conf"]
        rec["median_positive_conf"] = float(np.median(vals)) if vals else float("nan")
        del rec["best_conf"]
    return out


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def corr(rows: list[dict], x_key: str, y_key: str) -> dict:
    pairs = [(float(r[x_key]), float(r[y_key])) for r in rows if np.isfinite(float(r[x_key])) and np.isfinite(float(r[y_key]))]
    if len(pairs) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": len(pairs)}
    x, y = map(np.asarray, zip(*pairs))
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(list(x)), rankdata(list(y)))[0, 1])
    return {"pearson": pearson, "spearman": spearman, "n": len(pairs)}


def q(values: list[float], p: float) -> float:
    vals = [v for v in values if np.isfinite(v)]
    return float(np.quantile(vals, p)) if vals else float("nan")


def summarize(rows: list[dict], args: argparse.Namespace) -> dict:
    coupling = [float(r["indirect_over_full"]) for r in rows]
    cosine = [float(r["cos_full_detach"]) for r in rows]
    ob = [float(r["indirect_object_bg_ratio"]) for r in rows]
    corrs = {}
    for x_key in ("gate_mean", "gate_std", "p2_rms"):
        for y_key in ("low_conf_fn", "median_positive_conf"):
            corrs[f"{x_key}_vs_{y_key}"] = corr(rows, x_key, y_key)
    med_coupling = q(coupling, 0.5)
    med_cos = q(cosine, 0.5)
    med_ob = q(ob, 0.5)
    if med_coupling >= 0.10 or (np.isfinite(med_ob) and med_ob >= 1.5 and med_cos < 0.95):
        interp = "meaningful_gradient_coupling"
    elif med_coupling < 0.02 and med_cos > 0.99:
        interp = "negligible_gradient_coupling"
    else:
        interp = "mixed_or_small_gradient_coupling"
    return {
        "checkpoint": str(CKPT.relative_to(OUT)),
        "variant": "gap",
        "seed": 42,
        "split": "test",
        "layer": "model.model[19] ChannelAttention(avg), P2 pre-Detect",
        "loss_type": rows[0]["loss_type"] if rows else "none",
        "images": len(rows),
        "max_images": args.max_images,
        "device": args.device,
        "forward_max_abs_diff_max": q([float(r["forward_max_abs_diff"]) for r in rows], 1.0),
        "indirect_over_full": {"p25": q(coupling, 0.25), "median": med_coupling, "p75": q(coupling, 0.75)},
        "cos_full_detach": {"p25": q(cosine, 0.25), "median": med_cos, "p75": q(cosine, 0.75)},
        "indirect_object_bg_ratio": {"p25": q(ob, 0.25), "median": med_ob, "p75": q(ob, 0.75)},
        "correlations": corrs,
        "interpretation": interp,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict) -> None:
    lines = [
        "# GAP Gradient Coupling Investigation",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        f"Layer: `{summary['layer']}`",
        f"Split: `{summary['split']}`; images: **{summary['images']}**; loss: **{summary['loss_type']}**; device: `{summary['device']}`",
        "",
        "## Gradient Decomposition",
        "",
        "| metric | p25 | median | p75 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("indirect_over_full", "cos_full_detach", "indirect_object_bg_ratio"):
        v = summary[key]
        lines.append(f"| `{key}` | {v['p25']:.6f} | {v['median']:.6f} | {v['p75']:.6f} |")
    lines += [
        "",
        f"Max forward absolute diff across normal vs detached passes: `{summary['forward_max_abs_diff_max']:.6g}`.",
        f"Interpretation: **{summary['interpretation']}**.",
        "",
        "## Gate/Failure Correlations",
        "",
        "| relation | Pearson | Spearman | n |",
        "|---|---:|---:|---:|",
    ]
    for name, v in summary["correlations"].items():
        lines.append(f"| `{name}` | {v['pearson']:.4f} | {v['spearman']:.4f} | {v['n']} |")
    lines += [
        "",
        "Artifacts:",
        f"- `{TABLE.relative_to(ROOT)}`",
        f"- `{SUMMARY.relative_to(ROOT)}`",
    ]
    REPORT.write_text("\n".join(lines))

    main_report = OUT / "report.md"
    if main_report.exists():
        text = main_report.read_text()
        marker = "\n\n## GAP Gradient Coupling\n"
        section = marker + "\n".join(lines[2:])
        text = text.split(marker)[0].rstrip() + section + "\n"
        main_report.write_text(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=os.environ.get("GAP_GRAD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--max-images", type=int, default=int(os.environ.get("GAP_GRAD_MAX_IMAGES", "0")))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    ensure_imports()
    ensure_checkpoint()
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
    from ultralytics import YOLO

    torch.set_num_threads(int(os.environ.get("GAP_GRAD_THREADS", "2")))
    device = torch.device(args.device)
    yolo = YOLO(CKPT)
    model = yolo.model.to(device)
    if isinstance(getattr(model, "args", None), dict):
        model.args = get_cfg(DEFAULT_CFG_DICT, model.args)
    set_train_raw(model)
    attn = model.model[19]
    assert type(attn).__name__ == "ChannelAttention"
    assert getattr(attn, "descriptor", None) == "avg"

    labels = load_labels()
    if args.max_images:
        labels = labels[: args.max_images]
    failure = read_failure_context()
    rows = []
    for idx, item in enumerate(labels, 1):
        full = one_pass(model, attn, item, device, detach_descriptor=False)
        detached = one_pass(model, attn, item, device, detach_descriptor=True)
        g_full, g_detach = full["grad"], detached["grad"]
        g_indirect = g_full - g_detach
        _, _, h, w = g_full.shape
        obj = p2_object_mask(item["boxes_xywhn"], h, w, device)
        bg = ~obj
        fctx = failure.get(item["name"], {})
        full_norm = float(g_full.norm().detach().cpu())
        row = {
            "image": item["name"],
            "gt_count": int(len(item["boxes_xywhn"])),
            "loss_type": full["loss_type"],
            "loss_full": full["loss_value"],
            "loss_detach": detached["loss_value"],
            "forward_max_abs_diff": float((full["preds"] - detached["preds"]).abs().max().item()) if full["preds"].numel() else 0.0,
            "full_grad_norm": full_norm,
            "detach_grad_norm": float(g_detach.norm().detach().cpu()),
            "indirect_grad_norm": float(g_indirect.norm().detach().cpu()),
            "indirect_over_full": float(g_indirect.norm().detach().cpu()) / max(full_norm, EPS),
            "cos_full_detach": safe_cos(g_full, g_detach),
            "indirect_object_abs_mean": mean_abs(g_indirect, obj),
            "indirect_bg_abs_mean": mean_abs(g_indirect, bg),
            "indirect_object_bg_ratio": mean_abs(g_indirect, obj) / max(mean_abs(g_indirect, bg), EPS),
            "gate_mean": float(full["gate"].mean().detach().cpu()),
            "gate_std": float(full["gate"].std().detach().cpu()),
            "p2_rms": float(full["p2"].square().mean().sqrt().detach().cpu()),
            "low_conf_fn": int(fctx.get("low_conf_fn", 0)),
            "median_positive_conf": float(fctx.get("median_positive_conf", "nan")),
        }
        rows.append(row)
        print(f"{idx}/{len(labels)} {item['name']} indirect/full={row['indirect_over_full']:.4f} cos={row['cos_full_detach']:.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No labels selected for GAP gradient coupling diagnostic.")
    write_csv(TABLE, rows)
    summary = summarize(rows, args)
    SUMMARY.write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print(json.dumps({"report": str(REPORT), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
