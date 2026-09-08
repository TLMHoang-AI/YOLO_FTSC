#!/usr/bin/env python3
"""Reproduce historical anchor-free Y4 from the ca48644 source snapshot.

The runner owns one historical Y4 case only.  It delegates model construction,
training, validation, assignment, loss, optimizer, scheduler, RNG and early
stopping to the pinned legacy source.  The local additions are provenance,
read-only checkpoint shadows, and detached diagnostics already exposed by the
legacy criterion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = ROOT / "models_related/ultralytics"
CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
Y4_CONFIG = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml"
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_historical_y4_reproduction"
VARIANT = "y4_legacy"
HISTORICAL_COMMIT = "ca48644cd1abe08315de08008d138f84565bf6f7"
PUBLISHED_COUNTS = {"train": 2320, "val": 788, "test": 788}
REFERENCE = {
    "historical_reference_val_AP50": 0.8373975313576892,
    "historical_reference_test_AP50": 0.7878924760784733,
    "historical_reference_seed43_test_AP50": 0.7784960372913239,
    "historical_reference_seed44_test_AP50": 0.8037524648720230,
}
AUGMENTATION_FIELDS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "close_mosaic", "mixup",
    "cutmix", "copy_paste", "multi_scale", "rect",
)
REQUIRED_TRAIN_ARTIFACTS = (
    "weights/best.pt", "weights/last.pt", "results.csv",
)
SHADOW_NAMES = ("best_ap50.pt", "best_map50_95.pt")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ULTRALYTICS_ROOT))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

workflow.EXPERIMENT = EXPERIMENT_SLUG
workflow.VARIANTS = {VARIANT: Y4_CONFIG}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_provenance() -> dict[str, object]:
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=ROOT, capture_output=True, check=False
    ).stdout
    return {
        "git/commit": _git("rev-parse", "HEAD"),
        "git/branch": _git("branch", "--show-current"),
        "git/dirty": bool(_git("status", "--porcelain")),
        "git/instrumentation_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "legacy/base_commit": HISTORICAL_COMMIT,
        "legacy/source_snapshot_match": _git("rev-parse", "HEAD") == HISTORICAL_COMMIT,
    }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _dataset_provenance(data_yaml: Path, split_seed: int) -> dict[str, object]:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    output: dict[str, object] = {"dataset/split_seed": split_seed}
    label_manifest = hashlib.sha256()
    for split, expected in PUBLISHED_COUNTS.items():
        image_dir = root / payload[split]
        stems = sorted(path.stem for path in image_dir.iterdir() if path.suffix.lower() == ".png")
        if len(stems) != expected:
            raise ValueError(f"LEVIR {split} has {len(stems)} images; expected {expected}")
        stem_payload = "".join(f"{stem}\n" for stem in stems).encode()
        output[f"dataset/{split}_count"] = len(stems)
        output[f"dataset/{split}_stem_sha256"] = hashlib.sha256(stem_payload).hexdigest()
        labels = root / "labels" / split
        for label in sorted(labels.glob("*.txt")):
            label_manifest.update(str(label.relative_to(root)).encode())
            label_manifest.update(b"\0")
            label_manifest.update(_sha256_file(label).encode())
            label_manifest.update(b"\n")
    output["dataset/label_manifest_sha256"] = label_manifest.hexdigest()
    return output


def _pretrained_provenance(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Historical reproduction requires an existing pretrained artifact: {path}"
        )
    return {
        "pretrained/name": path.name,
        "pretrained/resolved_path": str(path.resolve()),
        "pretrained/sha256": _sha256_file(path),
        "pretrained/bytes": path.stat().st_size,
    }


def _count_transferable_items(pretrained: Path, model) -> dict[str, int]:
    import torch

    checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema") or checkpoint.get("model")
    source_state = source.state_dict() if hasattr(source, "state_dict") else source
    target_state = model.model.state_dict()
    compatible = [
        key for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    ]
    return {"pretrained/transferred_items": len(compatible), "pretrained/total_items": len(source_state)}


def _resolved_train_config(args: argparse.Namespace, data_yaml: Path, amp: bool) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics.cfg import get_cfg

    overrides = {
        "model": str(Y4_CONFIG), "data": str(data_yaml), "epochs": args.epochs,
        "imgsz": args.imgsz, "batch": args.batch_size, "device": args.device,
        "workers": args.workers, "patience": args.patience, "seed": args.seeds[0],
        "deterministic": True, "amp": amp,
        "plots": False, "project": str(args.project), "name": f"seed_{args.seeds[0]}",
        "pretrained": str(args.pretrained),
    }
    resolved = get_cfg(overrides=overrides)
    values = _jsonable(vars(resolved))
    values["model_yaml_path"] = str(Y4_CONFIG)
    values["model_yaml_text"] = Y4_CONFIG.read_text(encoding="utf-8")
    values["augmentation/resolved"] = {
        key: values.get(key, "not_available_in_legacy_source")
        for key in AUGMENTATION_FIELDS
    }
    estimated_iterations = math.ceil(PUBLISHED_COUNTS["train"] / max(args.batch_size, 64)) * args.epochs
    initial_accumulate = max(round(64 / args.batch_size), 1)
    values.update(
        {
            "optimizer/requested": "auto",
            "optimizer/estimated_iterations": estimated_iterations,
            "optimizer/nominal_batch_size": 64,
            "optimizer/effective_batch_size": args.batch_size,
            "optimizer/initial_accumulate": initial_accumulate,
            "optimizer/effective_weight_decay": args.weight_decay * args.batch_size * initial_accumulate / 64,
            "optimizer/expected_resolution": "MuSGD" if estimated_iterations > 10000 else "AdamW",
            "fitness/criterion": "Ultralytics default validator fitness = mAP50-95(B)",
            "early_stop/criterion": "same default fitness passed to EarlyStopping",
        }
    )
    return values


def source_preflight() -> dict[str, object]:
    """Audit source/config invariants without downloading data or weights."""
    import yaml

    workflow.local_ultralytics()
    from ultralytics import YOLO

    payload = yaml.safe_load(Y4_CONFIG.read_text(encoding="utf-8"))
    model = YOLO(Y4_CONFIG)
    head = model.model.model[-1]
    calibrator = getattr(head, "ftsc_calibrator", None)
    tal_path = "models_related/ultralytics/ultralytics/utils/tal.py"
    tal_snapshot_blob = _git("rev-parse", f"{HISTORICAL_COMMIT}:{tal_path}")
    tal_current_blob = _git("rev-parse", f"HEAD:{tal_path}")
    checks = {
        "legacy_commit": _git("rev-parse", "HEAD"),
        "y4_config_exists": Y4_CONFIG.is_file(),
        "y4_stride": head.stride.tolist(),
        "y4_stride_ok": head.stride.tolist() == [4.0, 8.0, 16.0, 32.0],
        "ftsc_policy": getattr(calibrator, "policy", None),
        "ftsc_evidence": getattr(calibrator, "evidence_names", None),
        "ftsc_config": payload.get("ftsc"),
        "trainer_broad_dfl_freeze_source": 'always_freeze_names = [".dfl"]' in
        (ULTRALYTICS_ROOT / "ultralytics/engine/trainer.py").read_text(encoding="utf-8"),
        "fitness_source": "metrics.pop(\"fitness\", -self.loss.detach().cpu().numpy())" in
        (ULTRALYTICS_ROOT / "ultralytics/engine/trainer.py").read_text(encoding="utf-8"),
        "tal_source_snapshot_blob": tal_snapshot_blob,
        "tal_source_current_blob": tal_current_blob,
        "tal_source_unchanged_at_snapshot": tal_snapshot_blob == tal_current_blob,
    }
    if calibrator is None or calibrator.policy != "f5":
        raise RuntimeError(f"Historical Y4 calibrator mismatch: {checks}")
    expected = {
        "position_alpha": 6.0, "dfl_entropy_tau": 1.0, "dfl_variance_tau": 0.0,
        "dfl_detach": True, "dfl_apply_cls": True, "dfl_apply_box": False,
        "dfl_apply_dfl": False, "apply_cls": True, "apply_box": True,
        "apply_dfl": True, "per_gt_norm": True, "log_clip": 0.35,
        "warmup_epochs": 5, "ramp_epochs": 10, "strength_init": 1.0,
        "strength_max": 2.0, "strength_reg_weight": 1e-4,
    }
    actual = payload["ftsc"]
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f"Y4 config invariant drift at {key}: {actual.get(key)!r} != {value!r}")
    return checks


def full_preflight(args: argparse.Namespace, data_yaml: Path) -> dict[str, object]:
    """Audit pretrained transfer, optimizer resolution, and historical freeze semantics."""
    import torch
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer

    pretrained = Path(args.pretrained).resolve()
    pretrained_meta = _pretrained_provenance(pretrained)
    model = YOLO(Y4_CONFIG)
    transfer = _count_transferable_items(pretrained, model)
    model.load(str(pretrained), smart_transfer=True)
    head = model.model.model[-1]
    if head.stride.tolist() != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"Expected P2-P5 strides, got {head.stride.tolist()}")
    named = dict(model.model.named_parameters())
    dfl_projection = named.get("model.29.dfl.conv.weight")
    dfl_strength = named.get("model.29.ftsc_calibrator.strength_logits.dfl_distribution")
    position_strength = named.get("model.29.ftsc_calibrator.strength_logits.position_gaussian")
    if dfl_projection is None or dfl_strength is None or position_strength is None:
        raise RuntimeError("Historical Y4 parameter names do not match the expected legacy snapshot")
    # Apply the exact legacy broad-freeze predicate to this in-memory preflight model.
    for name, parameter in model.model.named_parameters():
        if ".dfl" in name:
            parameter.requires_grad = False
    if dfl_projection.requires_grad or dfl_strength.requires_grad or not position_strength.requires_grad:
        raise RuntimeError("Historical broad DFL freeze invariant failed")
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.data = {"nc": 1}
    trainer.args = SimpleNamespace(lr0=0.01, momentum=0.937, warmup_bias_lr=0.1)
    iterations = math.ceil(PUBLISHED_COUNTS["train"] / max(args.batch_size, 64)) * args.epochs
    optimizer = trainer.build_optimizer(
        model.model, name="auto", lr=0.01, momentum=0.937,
        decay=args.weight_decay, iterations=iterations,
    )
    if type(optimizer).__name__ != "MuSGD":
        raise RuntimeError(f"Historical optimizer auto resolution drifted: {type(optimizer).__name__}")
    return {
        **pretrained_meta, **transfer,
        "model/strides": head.stride.tolist(),
        "freeze/actual_detect_dfl_projection": "frozen",
        "freeze/actual_ftsc_dfl_strength": "frozen",
        "freeze/actual_ftsc_position_strength": "trainable",
        "optimizer/resolved": type(optimizer).__name__,
        "optimizer/lr": optimizer.param_groups[0]["lr"],
        "optimizer/momentum": optimizer.param_groups[0].get("momentum"),
        "optimizer/weight_decay": optimizer.param_groups[0].get("weight_decay"),
        "torch/version": torch.__version__,
    }


class ShadowCheckpoints:
    """Copy serialized last.pt after normal saves; never mutate trainer state."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.best_ap50 = float("-inf")
        self.best_map50_95 = float("-inf")
        self.best_ap50_epoch = None
        self.best_map50_95_epoch = None

    def on_model_save(self, trainer) -> None:
        last = self.run_dir / "weights/last.pt"
        if not last.is_file():
            return
        metrics = trainer.metrics or {}
        ap50 = float(metrics.get("metrics/mAP50(B)", float("nan")))
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", float("nan")))
        epoch = int(trainer.epoch)
        if math.isfinite(ap50) and ap50 >= self.best_ap50:
            self.best_ap50, self.best_ap50_epoch = ap50, epoch
            shutil.copyfile(last, self.run_dir / "weights/best_ap50.pt")
        if math.isfinite(map50_95) and map50_95 >= self.best_map50_95:
            self.best_map50_95, self.best_map50_95_epoch = map50_95, epoch
            shutil.copyfile(last, self.run_dir / "weights/best_map50_95.pt")

    def metadata(self) -> dict[str, object]:
        return {
            "shadow/best_ap50_epoch": self.best_ap50_epoch,
            "shadow/best_ap50_value": self.best_ap50 if math.isfinite(self.best_ap50) else None,
            "shadow/best_map50_95_epoch": self.best_map50_95_epoch,
            "shadow/best_map50_95_value": self.best_map50_95 if math.isfinite(self.best_map50_95) else None,
        }


def model_for(pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO

    model = workflow.model_for(VARIANT, pretrained)
    head = model.model.model[-1]
    if head.stride.tolist() != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"Historical Y4 expected P2-P5, got {head.stride.tolist()}")
    if head.ftsc_calibrator is None or head.ftsc_calibrator.policy != "f5":
        raise RuntimeError("Historical Y4 model did not construct F5 FTSC")
    return model


def train_one(args: argparse.Namespace, data_yaml: Path, amp: bool) -> tuple[Path, ShadowCheckpoints]:
    run_dir = args.project / VARIANT / f"seed_{args.seeds[0]}"
    if any((run_dir / path).is_file() for path in REQUIRED_TRAIN_ARTIFACTS):
        raise RuntimeError(f"Refusing to resume or reuse a partial historical run: {run_dir}")
    workflow.seed_everything(args.seeds[0])
    kwargs = workflow.train_kwargs(args, data_yaml, args.seeds[0], amp)
    kwargs.update(
        project=str(args.project / VARIANT), name=f"seed_{args.seeds[0]}",
        exist_ok=True, weight_decay=args.weight_decay,
    )
    manager = ShadowCheckpoints(run_dir)
    model = model_for(str(args.pretrained))
    model.add_callback("on_model_save", manager.on_model_save)
    model.train(**kwargs)
    if not all((run_dir / path).is_file() for path in REQUIRED_TRAIN_ARTIFACTS):
        raise FileNotFoundError(f"Historical training ended without required artifacts: {run_dir}")
    return run_dir, manager


def evaluate_selected(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    workflow.local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(run_dir / "weights/best.pt")
    metrics: dict[str, object] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = model.val(
            data=str(data_yaml), split=split, imgsz=args.imgsz, batch=args.batch_size,
            device=args.device, workers=args.workers, plots=False, iou=0.5,
            project=str(run_dir / "evaluation"), name=split, exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
    calibrator = getattr(model.model.model[-1], "ftsc_calibrator", None)
    metrics["ftsc/enabled"] = float(calibrator is not None)
    if calibrator is not None:
        metrics.update(
            {
                "ftsc/policy_f5": float(calibrator.policy == "f5"),
                "ftsc/evidence_position": float("position_gaussian" in calibrator.evidence_names),
                "ftsc/evidence_dfl_distribution": float("dfl_distribution" in calibrator.evidence_names),
                "ftsc/dfl_strength_trainable": 0.0,
                "ftsc/dfl_strength_effective": 1.0,
                "ftsc/final_strength_position_gaussian": float(
                    (calibrator.strength_max * calibrator.strength_logits["position_gaussian"].sigmoid()).detach().cpu().item()
                ),
                "ftsc/final_strength_dfl_distribution": 1.0,
            }
        )
    metrics.update(REFERENCE)
    metrics["repro_minus_historical_val_AP50"] = metrics["val/metrics/mAP50(B)"] - REFERENCE["historical_reference_val_AP50"]
    metrics["repro_minus_historical_test_AP50"] = metrics["test/metrics/mAP50(B)"] - REFERENCE["historical_reference_test_AP50"]
    metrics["val_to_test_gap_AP50"] = metrics["val/metrics/mAP50(B)"] - metrics["test/metrics/mAP50(B)"]
    metrics["val_to_test_gap_mAP50_95"] = metrics["val/metrics/mAP50-95(B)"] - metrics["test/metrics/mAP50-95(B)"]
    return metrics


def _checkpoint_audit(run_dir: Path) -> list[dict[str, object]]:
    import torch

    def model_state_sha256(checkpoint) -> str | None:
        model = checkpoint.get("ema")
        if model is None:
            model = checkpoint.get("model")
        if model is None or not hasattr(model, "state_dict"):
            return None
        digest = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode() + b"\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(str(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    rows = []
    for name in ("best.pt", "last.pt", *SHADOW_NAMES):
        path = run_dir / "weights" / name
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(
            {
                "checkpoint": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "model_state_sha256": model_state_sha256(checkpoint),
                "epoch": checkpoint.get("epoch"),
                "best_fitness": checkpoint.get("best_fitness"),
                "optimizer_present": checkpoint.get("optimizer") is not None,
            }
        )
    return rows


def write_outputs(args: argparse.Namespace, run_dir: Path, data_yaml: Path, preflight: dict, shadow: ShadowCheckpoints, evaluation: dict):
    with (run_dir / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_rows = []
    best_fitness = float("-inf")
    for row in rows:
        current = {key: value for key, value in row.items()}
        for key, value in list(current.items()):
            try:
                current[key] = float(value)
            except (TypeError, ValueError):
                pass
        fitness = float(current.get("val/metrics/mAP50-95(B)", float("nan")))
        if math.isfinite(fitness):
            best_fitness = max(best_fitness, fitness)
        current["historical/fitness"] = fitness
        current["historical/best_fitness"] = best_fitness
        numeric_rows.append(current)
    epoch_path = run_dir / "historical_y4_reproduction_epoch_metrics.csv"
    fields = sorted({key for row in numeric_rows for key in row}, key=lambda key: (key not in {"epoch", "time"}, key))
    with epoch_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(numeric_rows)
    diagnostic_rows = []
    support_rows = []
    for row in numeric_rows:
        diagnostics = {key: row[key] for key in row if key.startswith("ftsc_")}
        diagnostics["epoch"] = row.get("epoch")
        if diagnostics:
            diagnostic_rows.append(diagnostics)
        support = {
            key: row[key] for key in row
            if key.startswith("ftsc_gt_") or "positives_per_gt" in key or key.startswith("ftsc_assigned_")
        }
        support["epoch"] = row.get("epoch")
        if support:
            support_rows.append(support)
    for filename, values in (
        ("historical_y4_reproduction_ftsc_diagnostics.csv", diagnostic_rows),
        ("historical_y4_reproduction_support_diagnostics.csv", support_rows),
    ):
        path = run_dir / filename
        fields = sorted({key for row in values for key in row}, key=lambda key: (key != "epoch", key))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or ["epoch"])
            writer.writeheader(); writer.writerows(values)
    checkpoint_rows = _checkpoint_audit(run_dir)
    checkpoint_path = run_dir / "historical_y4_reproduction_checkpoint_audit.csv"
    checkpoint_fields = ["checkpoint", "bytes", "sha256", "model_state_sha256", "epoch", "best_fitness", "optimizer_present"]
    with checkpoint_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=checkpoint_fields)
        writer.writeheader(); writer.writerows(checkpoint_rows)
    (run_dir / "historical_y4_reproduction_evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        **_git_provenance(), "historical/source_commit": HISTORICAL_COMMIT,
        "historical/y4_config_path": str(Y4_CONFIG),
        "historical/y4_config_text": Y4_CONFIG.read_text(encoding="utf-8"),
        "dataset": _dataset_provenance(data_yaml, args.split_seed),
        "preflight": _jsonable(preflight), "shadow": shadow.metadata(),
        "checkpoint_audit": checkpoint_rows,
        "resolved_training_config": _resolved_train_config(args, data_yaml, bool(args.amp)),
        "historical/source_parser_defaults": {"epochs": 100, "patience": 20},
        "historical/reproduction_protocol_overrides": {"epochs": args.epochs, "patience": args.patience},
        "historical/fitness_metric": "metrics/mAP50-95(B)",
        "historical/early_stopping": "Ultralytics EarlyStopping on default validator fitness",
        "instrumentation": {
            "scope": "detached observational summaries only",
            "extra_forward": False, "assignment_mutation": False,
            "optimizer_mutation": False, "rng_calls": False,
            "near_identity_std_tol": 0.01, "meaningful_ranking_range_tol": 0.05,
        },
        "training/actual_stop_epoch": numeric_rows[-1].get("epoch") if numeric_rows else None,
        "training/historical_best_epoch": max(numeric_rows, key=lambda row: row.get("historical/fitness", float("-inf"))).get("epoch") if numeric_rows else None,
        "training/best_ap50_epoch": max(numeric_rows, key=lambda row: row.get("metrics/mAP50(B)", float("-inf"))).get("epoch") if numeric_rows else None,
        "training/best_map50_95_epoch": max(numeric_rows, key=lambda row: row.get("metrics/mAP50-95(B)", float("-inf"))).get("epoch") if numeric_rows else None,
    }
    (run_dir / "historical_y4_reproduction_metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"variant": VARIANT, "seed": args.seeds[0], **evaluation}
    with (run_dir / "historical_y4_reproduction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader(); writer.writerow(summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT.parent / f"runs/{EXPERIMENT_SLUG}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--source-preflight-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.seeds != [42]:
        raise ValueError("Historical Y4 reproduction is intentionally locked to seed 42")
    if args.split_seed != 42:
        raise ValueError("Historical split seed is locked to 42")
    args.project = args.project.resolve()
    if args.source_preflight_only:
        print(json.dumps(_jsonable(source_preflight()), indent=2, sort_keys=True))
        return
    data_yaml = workflow.prepare_fixed_split(args)
    preflight = {"source": source_preflight(), "full": full_preflight(args, data_yaml)}
    if args.preflight_only:
        print(json.dumps(_jsonable(preflight), indent=2, sort_keys=True))
        return
    run_dir, shadow = train_one(args, data_yaml, args.amp)
    evaluation = evaluate_selected(run_dir, data_yaml, args)
    write_outputs(args, run_dir, data_yaml, preflight, shadow, evaluation)


if __name__ == "__main__":
    main()
