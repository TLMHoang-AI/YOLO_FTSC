#!/usr/bin/env python3
"""Post-hoc FTSC ranking audit for matched H2, E_H2, and ES1 checkpoints.

This runner performs a deterministic validation-loader pass through the normal
TAL and FTSC loss path under ``torch.inference_mode``.  It never creates an
optimizer, calls ``train()``, or changes inference outputs.  Checkpoints must
be supplied explicitly or discovered unambiguously; no substitute checkpoint
is selected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ULTRALYTICS = PROJECT_ROOT / "models_related/ultralytics"
CONFIG_ROOT = PROJECT_ROOT / "models_related/models_config/yolov8/levir"
RESULTS_ROOT = PROJECT_ROOT.parent / "results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VARIANTS = {
    "H2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml",
    "E_H2": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edgecue.yaml",
    "ES1": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml",
}


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        path = args.checkpoint.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"explicit checkpoint does not exist: {path}")
        return path
    roots = [args.runs_root.expanduser().resolve()]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("**/weights/best.pt"):
            text = str(path).lower()
            if f"seed_{args.seed}" in text and args.variant.lower() in text:
                candidates.append(path.resolve())
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"checkpoint resolution for {args.variant}/seed_{args.seed} was ambiguous: "
            f"{[str(path) for path in candidates]}; pass --checkpoint explicitly"
        )
    return candidates[0]


def _data_yaml(args: argparse.Namespace) -> Path:
    if args.data is not None:
        path = args.data.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    path = (args.dataset_root / f"levir_ship_yolo_seed{args.split_seed}" / "levir_ship.yaml").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {path}; pass --data explicitly")
    return path


def _build_loader(model, args: argparse.Namespace, data_yaml: Path):
    local_ultralytics()
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    overrides = {
        "data": str(data_yaml),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "task": "detect",
        "rect": True,
        "augment": False,
        "single_cls": False,
        "cache": False,
    }
    cfg = get_cfg(DEFAULT_CFG, overrides)
    stride = max(int(model.stride.max()), 32)
    dataset = build_yolo_dataset(cfg, data[args.split], args.batch, data, mode="val", rect=True, stride=stride)
    loader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1)
    return loader, data, cfg


def _preprocess(batch: dict, device) -> dict:
    for key, value in batch.items():
        if hasattr(value, "to"):
            batch[key] = value.to(device, non_blocking=device.type == "cuda")
    batch["img"] = batch["img"].float() / 255.0
    return batch


def _ftsc_provenance(calibrator) -> dict[str, object]:
    return {
        "policy": calibrator.policy,
        "evidence": list(calibrator.evidence_names),
        "fixed_strengths": dict(calibrator.fixed_strengths),
        "apply_cls": calibrator.apply_cls,
        "apply_box": calibrator.apply_box,
        "apply_dfl": calibrator.apply_dfl,
        "dfl_apply_cls": calibrator.dfl_apply_cls,
        "dfl_apply_box": calibrator.dfl_apply_box,
        "dfl_apply_dfl": calibrator.dfl_apply_dfl,
        "per_gt_norm": calibrator.per_gt_norm,
        "log_clip": calibrator.log_clip,
        "warmup_epochs": calibrator.warmup_epochs,
        "ramp_epochs": calibrator.ramp_epochs,
    }


def audit(args: argparse.Namespace) -> dict[str, object]:
    if args.variant not in VARIANTS:
        raise ValueError(f"unknown variant {args.variant}; choose from {sorted(VARIANTS)}")
    if args.split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    checkpoint = _resolve_checkpoint(args)
    config = VARIANTS[args.variant].resolve()
    data_yaml = _data_yaml(args)

    local_ultralytics()
    import torch
    from ultralytics import YOLO
    from train_levir_scripts.ftsc_ranking_quality import FTSCRankingAccumulator

    model_wrapper = YOLO(str(config))
    model_wrapper.load(str(checkpoint), smart_transfer=True)
    model = model_wrapper.model.to(args.device)
    # The normal loss path expects Detect to return its raw feature dict, which
    # is the training-mode contract. Freeze BatchNorm statistics explicitly so
    # this no-optimizer audit cannot update model state.
    model.train()
    import torch.nn as nn
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    model.args.device = args.device
    criterion = model.init_criterion()
    model.criterion = criterion
    if criterion.ftsc_calibrator is None:
        raise ValueError(f"{args.variant} checkpoint/config has no FTSC calibrator")
    criterion.ftsc_calibrator.audit_enabled = True
    loader, data, cfg = _build_loader(model, args, data_yaml)
    accumulator = FTSCRankingAccumulator(sample_limit=args.sample_limit, sample_seed=args.sample_seed)

    with torch.inference_mode():
        for batch in loader:
            batch = _preprocess(batch, model.device if hasattr(model, "device") else next(model.parameters()).device)
            preds = model(batch["img"])
            model.loss(batch, preds)
            accumulator.update(criterion.ftsc_calibrator.last_audit)

    head = model.model[-1]
    edge_config = None
    for module in model.modules():
        if module.__class__.__name__ == "P2EdgeCueFusion":
            edge_config = {
                "hidden": module.hidden,
                "residual_scale": module.residual_scale,
                "residual_schedule": module.residual_schedule,
                "orientation_gate_mode": module.orientation_gate_mode,
            }
            break
    provenance = {
        "variant": args.variant,
        "seed": args.seed,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config),
        "config_sha256": sha256(config),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256(data_yaml),
        "dataset_root": str(data.get("path", "")),
        "model_strides": [float(value) for value in head.stride.detach().cpu().tolist()],
        "ftsc": _ftsc_provenance(criterion.ftsc_calibrator),
        "edge": edge_config,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(args.device),
        "nms_iou": args.nms_iou,
        "audit_mode": "posthoc_validation_loss_no_optimizer",
    }
    payload = accumulator.results_payload(**provenance)
    payload["checkpoint_found_locally"] = True
    return payload


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed", "split", "level", "metric"}, key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(payload: dict[str, object], output_dir: Path) -> None:
    provenance = payload["provenance"]
    run_dir = output_dir / "ftsc_ranking_quality_runs" / str(provenance["variant"]) / f"seed_{provenance['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_payloads = []
    for path in sorted((output_dir / "ftsc_ranking_quality_runs").glob("*/seed_*/audit.json")):
        try:
            run_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    summary_rows = []
    level_rows = []
    for item in run_payloads:
        summary_rows.extend(item.get("summary", []))
        level_rows.extend(item.get("by_level", []))
    _write_csv(output_dir / "ftsc_ranking_quality_summary_runs.csv", summary_rows)
    _write_csv(output_dir / "ftsc_ranking_quality_by_level.csv", level_rows)
    aggregate = []
    grouped = {}
    for row in summary_rows:
        if row.get("level") != "all":
            continue
        grouped.setdefault((row.get("variant"), row.get("metric")), []).append(row)
    for (variant, metric), rows in sorted(grouped.items(), key=str):
        values = [float(row["mean"]) for row in rows if row.get("mean") is not None]
        valid_counts = [int(row.get("valid_gt_count", 0) or 0) for row in rows if row.get("mean") is not None]
        weighted_total = sum(count * value for count, value in zip(valid_counts, values, strict=False))
        valid_total = sum(valid_counts)
        aggregate.append({
            "variant": variant,
            "metric": metric,
            "runs": len(rows),
            "mean_of_run_means": float(np.mean(values)) if values else None,
            "std_of_run_means": float(np.std(values)) if values else None,
            "valid_gt_total": valid_total,
            "weighted_mean": weighted_total / valid_total if valid_total else None,
        })
    _write_csv(output_dir / "ftsc_ranking_quality_summary_aggregate.csv", aggregate)
    (output_dir / "ftsc_ranking_quality_results.json").write_text(
        json.dumps({"runs": run_payloads, "latest": payload}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    samples = [row for item in run_payloads for row in item.get("positive_sample", [])]
    _write_csv(output_dir / "ftsc_ranking_quality_positive_sample.csv", samples)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit(args)
    write_outputs(payload, args.output_dir.expanduser().resolve())
    print(json.dumps(payload["provenance"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
