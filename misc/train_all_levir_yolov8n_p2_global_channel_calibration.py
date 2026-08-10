#!/usr/bin/env python3
"""Train, evaluate, diagnose, and upload the seed-42 P2 channel-calibration matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from train_levir_scripts import analyze_p2_cbam_ranking as ranking
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


ROOT = Path(__file__).resolve().parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
CONFIGS = {
    "plain": CONFIG_ROOT / "yolov8n_p2_fpn_only_plain.yaml",
    "channel_only": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_channel_only.yaml",
    "spatial_only": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_spatial_only.yaml",
    "full_cbam": CONFIG_ROOT / "yolov8n_p2_fpn_only_cbam_matched.yaml",
    "gccc": CONFIG_ROOT / "yolov8n_p2_fpn_only_gccc.yaml",
}
TRAIN_VARIANTS = ("channel_only", "spatial_only", "gccc")
REQUIRED = (
    "weights/best.pt", "weights/last.pt", "results.csv", "args.yaml",
    "evaluation_metrics.json", "experiment_manifest.json",
)
TARGETS = {
    "channel_noninferiority_margin": 0.005,
    "spatial_weakness_margin": 0.005,
    "gccc_iou_best_min": 0.797,
    "gccc_iou_topscore_min": 0.665,
    "gccc_rank_gap_max": 0.130,
    "gccc_spearman_min": 0.38,
    "gccc_test_map_min_exclusive": 0.300,
}


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


def model_for(variant: str, pretrained: str):
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.nn.modules import CBAM, ChannelAttention, GlobalChannelContextCalibration, SpatialAttention

    expected = {
        "channel_only": ChannelAttention,
        "spatial_only": SpatialAttention,
        "full_cbam": CBAM,
        "gccc": GlobalChannelContextCalibration,
    }
    model = YOLO(CONFIGS[variant])
    model.load(pretrained, smart_transfer=True)
    head = model.model.model[-1]
    if head.stride.tolist() != [4.0]:
        raise ValueError(f"{variant}: expected Detect stride [4.0], got {head.stride.tolist()}")
    if variant == "plain":
        if head.f != [18]:
            raise ValueError(f"plain: expected Detect([18]), got {head.f}")
    elif not isinstance(model.model.model[19], expected[variant]) or head.f != [19]:
        raise TypeError(f"{variant}: calibration/Detect graph did not resolve as specified")
    return model


def train_kwargs(args: argparse.Namespace, data_yaml: Path, variant: str, smoke: bool = False) -> dict:
    return {
        "data": str(data_yaml), "epochs": 1 if smoke else args.epochs,
        "imgsz": 256 if smoke else args.imgsz, "batch": 1 if smoke else args.batch_size,
        "device": args.device, "workers": 0 if smoke else args.workers, "patience": 0,
        "seed": 42, "deterministic": True, "amp": True, "plots": False,
        "project": str(args.project / ("_smoke" if smoke else variant)),
        "name": variant if smoke else "seed_42", "exist_ok": True,
        **({"fraction": args.smoke_fraction, "val": False} if smoke else {}),
    }


def custom_cost(model, imgsz: int) -> dict:
    calibration = None if len(model.model.model) == 20 else model.model.model[19]
    channels, height = 32, imgsz // 4
    payload = {"calibration_class": type(calibration).__name__ if calibration else None}
    if hasattr(calibration, "analytical_macs"):
        macs = calibration.analytical_macs(height, height)
        groups = math.ceil(height / 8) ** 2
        kvca_macs = (
            height * height * channels * channels + height * height * channels
            + 2 * groups * channels * channels + 2 * height * height * groups * channels
            + height * height * channels * channels
        )
        payload.update(
            calibration_macs=macs,
            calibration_gflops=2 * macs / 1e9,
            matched_global_kvca_macs=kvca_macs,
            custom_macs_ratio_vs_global_kvca=macs / kvca_macs,
        )
    return payload


def write_manifest(model, variant: str, run_dir: Path, args: argparse.Namespace) -> None:
    from ultralytics.utils.torch_utils import get_flops

    head = model.model.model[-1]
    calibration = None if variant == "plain" else model.model.model[19]
    payload = {
        "variant": variant, "seed": 42, "split_seed": 42, "config": CONFIGS[variant].name,
        "topology": "P2 C2f -> calibration -> shared Detect" if calibration else "P2 C2f -> shared Detect",
        "calibration_layer": 19 if calibration else None, "detect_from": head.f,
        "detect_stride": head.stride.tolist(), "params": sum(p.numel() for p in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch_size": args.batch_size,
        "amp": True, "nms_iou": 0.5, "targets": TARGETS, **custom_cost(model, args.imgsz),
    }
    if calibration is not None:
        for name in ("sr_ratio", "temperature"):
            if hasattr(calibration, name):
                payload[name] = getattr(calibration, name)
        if hasattr(calibration, "alpha"):
            payload["alpha_init"] = float(calibration.alpha.detach())
    (run_dir / "experiment_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def train(variant: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / "seed_42"
    results = run_dir / "results.csv"
    complete = all((run_dir / path).is_file() for path in workflow.REQUIRED_TRAIN_ARTIFACTS)
    complete = complete and sum(1 for _ in results.open(encoding="utf-8")) - 1 == args.epochs
    if not complete:
        seed_everything()
        model_for(variant, args.pretrained).train(**train_kwargs(args, data_yaml, variant))
    if not all((run_dir / path).is_file() for path in workflow.REQUIRED_TRAIN_ARTIFACTS):
        raise RuntimeError(f"Training ended without required artifacts: {run_dir}")
    write_manifest(model_for(variant, args.pretrained), variant, run_dir, args)
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        saved = json.loads(output.read_text())
        if saved.get("checkpoint") == "best.pt" and saved.get("nms_iou") == 0.5:
            return saved
    local_ultralytics()
    from ultralytics import YOLO

    metrics = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def image_cluster_bootstrap(left: list[dict], right: list[dict], repeats: int) -> dict:
    import numpy as np

    metrics = ("iou_best", "iou_topscore", "rank_gap", "confidence_iou_spearman")
    left_by_key = {(r["image"], r["gt_index"]): r for r in left}
    right_by_key = {(r["image"], r["gt_index"]): r for r in right}
    if left_by_key.keys() != right_by_key.keys():
        raise RuntimeError("Diagnostic GT keys differ")
    images = sorted({key[0] for key in left_by_key})
    rng = np.random.default_rng(42)
    result = {}
    for metric in metrics:
        per_image = {}
        for image in images:
            values = [left_by_key[key][metric] - right_by_key[key][metric]
                      for key in left_by_key if key[0] == image
                      and math.isfinite(left_by_key[key][metric]) and math.isfinite(right_by_key[key][metric])]
            if values:
                per_image[image] = float(np.mean(values))
        observed = float(np.mean(list(per_image.values())))
        samples = [np.mean([per_image[name] for name in rng.choice(list(per_image), len(per_image), replace=True)])
                   for _ in range(repeats)]
        result[metric] = {"unit": "image_cluster", "images": len(per_image), "mean_delta": observed,
                          "bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist()}
    return result


def run_diagnostics(run_dirs: dict[str, Path], data_yaml: Path, args: argparse.Namespace) -> dict:
    images = data_yaml.parent / "images/test"
    diagnostic_args = argparse.Namespace(**vars(args))
    if str(diagnostic_args.device).isdigit():
        diagnostic_args.device = f"cuda:{diagnostic_args.device}"
    ranking.EXPECTED_LEVELS = {name: 1 for name in CONFIGS}
    rows = {name: ranking.inspect_model(name, path / "weights/best.pt", sorted(images.glob("*.png")), diagnostic_args)
            for name, path in run_dirs.items()}
    output = args.project / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["model", "image", "gt_index", "area_px2", "size_group", "candidate_count", *ranking.METRICS]
    with (output / "per_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in CONFIGS:
            writer.writerows(rows[name])
    paired = {}
    for name in CONFIGS:
        if name != "plain":
            paired[f"{name}_minus_plain"] = {
                "gt_bootstrap": ranking.paired_delta(rows, name, "plain", args.bootstrap_repeats),
                "image_cluster_bootstrap": image_cluster_bootstrap(rows[name], rows["plain"], args.bootstrap_repeats),
            }
    summary = {
        "protocol": {"seed": 42, "split": "test", "imgsz": args.imgsz, "nms": "pre-NMS raw P2",
                     "uncertainty": "paired GT and image-cluster bootstrap; not training-seed uncertainty"},
        "models": {name: ranking.descriptive_summary(model_rows) for name, model_rows in rows.items()},
        "paired": paired,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def decision_summary(metrics: dict, diagnostics: dict) -> dict:
    test_map = lambda name: metrics[name]["test/metrics/mAP50-95(B)"]
    means = lambda name: diagnostics["models"][name]["all"]
    channel_pass = test_map("channel_only") >= test_map("full_cbam") - TARGETS["channel_noninferiority_margin"]
    spatial_weak = test_map("spatial_only") <= test_map("full_cbam") - TARGETS["spatial_weakness_margin"]
    g = means("gccc")
    gccc_checks = {
        "iou_best": g["iou_best"]["mean"] >= TARGETS["gccc_iou_best_min"],
        "iou_topscore": g["iou_topscore"]["mean"] >= TARGETS["gccc_iou_topscore_min"],
        "rank_gap": g["rank_gap"]["mean"] <= TARGETS["gccc_rank_gap_max"],
        "spearman": g["confidence_iou_spearman"]["mean"] >= TARGETS["gccc_spearman_min"],
        "test_map": test_map("gccc") > TARGETS["gccc_test_map_min_exclusive"],
    }
    return {
        "channel_hypothesis": {"supported_within_ordered_cbam": channel_pass and spatial_weak,
                               "channel_noninferior": channel_pass, "spatial_materially_weaker": spatial_weak,
                               "causal_limit": "One seed and non-additive channel-to-spatial CBAM ordering."},
        "gccc": {"passes": all(gccc_checks.values()), "checks": gccc_checks},
        "uncertainty_note": "Bootstrap intervals are sample uncertainty, not seed uncertainty.",
    }


def upload(api, run_dirs: dict[str, Path], data_yaml: Path, args: argparse.Namespace) -> None:
    def retry(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    for variant, run_dir in run_dirs.items():
        missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
        if missing:
            raise RuntimeError(f"Refusing incomplete upload for {variant}: {missing}")
        retry(lambda variant=variant, run_dir=run_dir: api.upload_folder(
            folder_path=run_dir, path_in_repo=f"runs/{variant}/seed_42",
            repo_id=args.hf_repo_id, repo_type="dataset"))
    for path, remote in (
        (args.project / "summary.json", "summary.json"),
        (args.project / "diagnostics", "diagnostics"),
        (Path(__file__), f"code/{Path(__file__).name}"),
        (data_yaml.parent / "manifest.json", "dataset/fixed_split_seed_42.json"),
    ):
        if path.is_dir():
            retry(lambda path=path, remote=remote: api.upload_folder(folder_path=path, path_in_repo=remote,
                         repo_id=args.hf_repo_id, repo_type="dataset"))
        else:
            retry(lambda path=path, remote=remote: api.upload_file(path_or_fileobj=path, path_in_repo=remote,
                         repo_id=args.hf_repo_id, repo_type="dataset"))
    for config in CONFIGS.values():
        retry(lambda config=config: api.upload_file(path_or_fileobj=config, path_in_repo=f"configs/{config.name}",
                     repo_id=args.hf_repo_id, repo_type="dataset"))
    files = set(retry(lambda: api.list_repo_files(args.hf_repo_id, repo_type="dataset")))
    expected = [f"runs/{variant}/seed_42/{path}" for variant in TRAIN_VARIANTS for path in REQUIRED]
    expected += ["summary.json", "diagnostics/summary.json", "diagnostics/per_gt.csv"]
    missing = [path for path in expected if path not in files]
    if missing:
        raise RuntimeError(f"HF verification failed: {missing}")
    marker = args.project / "upload_complete.json"
    marker.write_text(json.dumps({"repo_id": args.hf_repo_id, "verified": expected}, indent=2) + "\n")
    retry(lambda: api.upload_file(path_or_fileobj=marker, path_in_repo=marker.name,
                 repo_id=args.hf_repo_id, repo_type="dataset"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=TRAIN_VARIANTS, default=list(TRAIN_VARIANTS))
    parser.add_argument("--plain-run", type=Path, help="Existing verified seed-42 plain run directory")
    parser.add_argument("--full-cbam-run", type=Path, help="Existing verified seed-42 full-CBAM run directory")
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/levir_yolov8n_p2_global_channel_calibration")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--bootstrap-repeats", type=int, default=4000)
    parser.add_argument("--hf-repo-id", default="duyle2408/levir-yolov8n-p2-global-channel-calibration-seed42")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (
        path.resolve() for path in (args.data_root, args.dataset_root, args.project))
    data_yaml = workflow.prepare_fixed_split(argparse.Namespace(
        data_root=args.data_root, dataset_root=args.dataset_root, split_seed=42))
    for variant in args.variants:
        seed_everything()
        model_for(variant, args.pretrained).train(**train_kwargs(args, data_yaml, variant, smoke=True))
    if args.smoke_only:
        return
    if set(args.variants) != set(TRAIN_VARIANTS):
        raise ValueError("Full workflow requires all three new variants")
    if args.plain_run is None or args.full_cbam_run is None:
        raise ValueError("--plain-run and --full-cbam-run are required references; they are never retrained")
    references = {"plain": args.plain_run.resolve(), "full_cbam": args.full_cbam_run.resolve()}
    for name, run_dir in references.items():
        for required in ("weights/best.pt", "evaluation_metrics.json"):
            if not (run_dir / required).is_file():
                raise FileNotFoundError(f"{name} reference missing {required}: {run_dir}")
    trained_runs = {variant: train(variant, data_yaml, args) for variant in TRAIN_VARIANTS}
    run_dirs = {"plain": references["plain"], **trained_runs, "full_cbam": references["full_cbam"]}
    metrics = {name: json.loads((run_dir / "evaluation_metrics.json").read_text()) for name, run_dir in references.items()}
    metrics.update({variant: evaluate(run_dir, data_yaml, args) for variant, run_dir in trained_runs.items()})
    diagnostics = run_diagnostics(run_dirs, data_yaml, args)
    summary = {"metrics": metrics, "decision": decision_summary(metrics, diagnostics), "targets": TARGETS}
    (args.project / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not args.no_upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required unless --no-upload is set")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.whoami()
        api.create_repo(args.hf_repo_id, repo_type="dataset", exist_ok=True)
        upload(api, trained_runs, data_yaml, args)


if __name__ == "__main__":
    main()
