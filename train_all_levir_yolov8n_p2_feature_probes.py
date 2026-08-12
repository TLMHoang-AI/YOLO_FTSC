#!/usr/bin/env python3
"""Train/evaluate/upload seed-42 P2 feature probes and low-conf FN diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
VARIANTS = {
    "context": CONFIG_ROOT / "yolov8n_p2_fpn_only_probe_context.yaml",
    "residual": CONFIG_ROOT / "yolov8n_p2_fpn_only_probe_residual.yaml",
    "energy": CONFIG_ROOT / "yolov8n_p2_fpn_only_probe_energy.yaml",
}
REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
    "gap_feature_miss_diagnostic.json",
)


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int = 42) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_split(args: argparse.Namespace) -> Path:
    from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

    data_yaml = workflow.prepare_fixed_split(argparse.Namespace(data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42))
    workflow.validate_split(data_yaml)
    return data_yaml


def model_for(variant: str, pretrained: str):
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import P2FeatureProbe

    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    probe, head = model.model.model[19], model.model.model[-1]
    if not isinstance(probe, P2FeatureProbe) or probe.mode != variant:
        raise TypeError(f"{variant}: P2FeatureProbe did not resolve")
    if head.f != [19] or head.stride.tolist() != [4.0]:
        raise ValueError(f"{variant}: expected P2FeatureProbe -> Detect stride [4.0], got {head.f}, {head.stride.tolist()}")
    return model


def complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return all((run_dir / path).is_file() for path in REQUIRED) and sum(1 for _ in results.open(encoding="utf-8")) - 1 == epochs


def train(variant: str, data_yaml: Path, seed: int, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}"
    if complete(run_dir, args.epochs):
        return run_dir
    seed_everything(seed)
    model_for(variant, args.pretrained).train(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
        device=args.device, workers=args.workers, patience=0, seed=seed, deterministic=True,
        amp=True, plots=False, project=str(args.project / variant), name=f"seed_{seed}", exist_ok=True,
    )
    missing = [path for path in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml") if not (run_dir / path).is_file()]
    if missing:
        raise RuntimeError(f"{variant}: training ended without artifacts: {missing}")
    if sum(1 for _ in (run_dir / "results.csv").open(encoding="utf-8")) - 1 != args.epochs:
        raise RuntimeError(f"{variant}: results.csv does not contain {args.epochs} completed epochs")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, float | str]:
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def xywhn_to_xyxy(boxes, w: int = 512, h: int = 512):
    import numpy as np

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x, y, bw, bh = boxes.T
    return np.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], axis=1).astype(np.float32)


def iou_mat(a, b):
    import numpy as np

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)


def greedy_match(gt, pred, thr: float = 0.5) -> tuple[set[int], set[int]]:
    mat = iou_mat(gt, pred)
    used_g: set[int] = set()
    used_p: set[int] = set()
    if mat.size == 0:
        return used_g, used_p
    for idx in mat.reshape(-1).argsort()[::-1]:
        gi, pi = divmod(int(idx), mat.shape[1])
        if mat[gi, pi] < thr:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
    return used_g, used_p


def load_test_labels(args: argparse.Namespace) -> list[dict]:
    import numpy as np

    split_root = args.dataset_root / "levir_ship_yolo_seed42"
    ann_path = split_root / "annotations/test.json"
    if ann_path.exists():
        ann = json.loads(ann_path.read_text())
        image_dir = args.data_root / "All Images"
        boxes_by_id: dict[int, list[list[float]]] = {im["id"]: [] for im in ann["images"]}
        for obj in ann["annotations"]:
            x, y, w, h = obj["bbox"]
            boxes_by_id[obj["image_id"]].append([(x + w / 2) / 512, (y + h / 2) / 512, w / 512, h / 512])
        return [
            {"image": image_dir / item["file_name"], "boxes_xywhn": np.asarray(boxes_by_id[item["id"]], dtype=np.float32).reshape(-1, 4)}
            for item in ann["images"]
            if (image_dir / item["file_name"]).exists()
        ]
    labels = []
    for label_path in sorted((split_root / "labels/test").glob("*.txt")):
        rows = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[1:5]])
        image = split_root / "images/test" / f"{label_path.stem}.png"
        if image.exists():
            labels.append({"image": image, "boxes_xywhn": np.asarray(rows, dtype=np.float32).reshape(-1, 4)})
    return labels


def diagnose(run_dir: Path, args: argparse.Namespace) -> dict[str, int | float | str]:
    local_ultralytics()
    from ultralytics import YOLO

    labels = load_test_labels(args)
    model = YOLO(run_dir / "weights/best.pt")
    missed = []
    fp = 0
    for item in labels:
        gt = xywhn_to_xyxy(item["boxes_xywhn"])
        pred_low = model.predict(str(item["image"]), imgsz=args.imgsz, conf=0.001, iou=0.5, max_det=300, batch=1, device=args.device, verbose=False)[0]
        boxes_low = pred_low.boxes.xyxy.detach().cpu().numpy() if pred_low.boxes is not None and len(pred_low.boxes) else []
        conf_low = pred_low.boxes.conf.detach().cpu().numpy() if pred_low.boxes is not None and len(pred_low.boxes) else []
        pred = model.predict(str(item["image"]), imgsz=args.imgsz, conf=0.25, iou=0.5, max_det=300, batch=1, device=args.device, verbose=False)[0]
        boxes = pred.boxes.xyxy.detach().cpu().numpy() if pred.boxes is not None and len(pred.boxes) else []
        matched_g, used_p = greedy_match(gt, boxes, 0.5)
        fp += max(0, len(boxes) - len(used_p))
        cand_iou = iou_mat(gt, boxes_low)
        for gi in range(len(gt)):
            if gi in matched_g:
                continue
            best_pi = int(cand_iou[gi].argmax()) if len(boxes_low) else -1
            best_iou = float(cand_iou[gi, best_pi]) if best_pi >= 0 else 0.0
            best_conf = float(conf_low[best_pi]) if best_pi >= 0 else 0.0
            missed.append({"low_conf": best_iou >= 0.3, "best_candidate_iou": best_iou, "best_candidate_conf": best_conf})
    low = [row for row in missed if row["low_conf"]]
    out = {
        "checkpoint": "best.pt",
        "nms_iou": 0.5,
        "test_images": len(labels),
        "gt": sum(len(item["boxes_xywhn"]) for item in labels),
        "missed": len(missed),
        "low_conf_fn": len(low),
        "lt_0_10_fn": sum(row["best_candidate_conf"] < 0.10 for row in low),
        "iou_ge_0_5_low_conf_fn": sum(row["best_candidate_iou"] >= 0.5 for row in low),
        "fp_conf_0_25": fp,
    }
    (run_dir / "gap_feature_miss_diagnostic.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_metadata(variant: str, run_dir: Path, seed: int, args: argparse.Namespace) -> None:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    shutil.copy2(VARIANTS[variant], run_dir / "config.yaml")
    model = YOLO(run_dir / "weights/best.pt")
    head, probe = model.model.model[-1], model.model.model[19]
    manifest = {
        "variant": variant,
        "seed": seed,
        "split_seed": 42,
        "config": VARIANTS[variant].name,
        "topology": "P2 -> P2FeatureProbe -> shared Detect",
        "probe_mode": probe.mode,
        "probe_layer": 19,
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required before this upload-required workflow starts")
        if not repo_id.strip():
            raise ValueError("--hf-repo-id is required before this upload-required workflow starts")
        from huggingface_hub import HfApi

        self.repo_id, self.api = repo_id, HfApi(token=token)
        self.api.whoami()
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    @staticmethod
    def retry(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def upload_run(self, variant: str, seed: int, run_dir: Path) -> None:
        missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
        if missing:
            raise RuntimeError(f"{variant}: refusing incomplete upload: {missing}")
        remote = f"runs/{variant}/seed_{seed}"
        self.retry(lambda: self.api.upload_folder(folder_path=run_dir, path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset"))
        expected = {f"{remote}/{path}" for path in REQUIRED}
        uploaded = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        missing = sorted(expected - uploaded)
        if missing:
            raise RuntimeError(f"{variant}: Hugging Face verification failed: {missing}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(json.dumps({"repo_id": self.repo_id, "variant": variant, "seed": seed, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
        self.retry(lambda: self.api.upload_file(path_or_fileobj=marker, path_in_repo=f"{remote}/{marker.name}", repo_id=self.repo_id, repo_type="dataset"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_feature_probes")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-feature-probes-seed42")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    uploader = Uploader(args.hf_repo_id)
    data_yaml = prepare_split(args)
    for seed in args.seeds:
        for variant in args.variants:
            run_dir = train(variant, data_yaml, seed, args)
            evaluate(run_dir, data_yaml, args)
            diagnose(run_dir, args)
            write_metadata(variant, run_dir, seed, args)
            if not complete(run_dir, args.epochs):
                raise RuntimeError(f"{variant}: required post-evaluation artifacts are incomplete")
            uploader.upload_run(variant, seed, run_dir)


if __name__ == "__main__":
    main()
