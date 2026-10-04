#!/usr/bin/env python3
"""Fail-closed LEVIR-only historical FTSC portability suite."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ULTRA = ROOT / "models_related/ultralytics"
import sys
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(ULTRA) not in sys.path: sys.path.insert(0, str(ULTRA))


FAMILIES = ("yolov9t", "yolov10n", "yolo11n")
CASES = ("B0_STOCK", "B1_P23_NOFTSC", "FTSC_H2", "FTSC_H2_CP2", "FTSC_ES1_HISTORICAL", "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS", "FTSC_ES2_HISTORICAL")
SEEDS = (42, 43, 44)
CANONICAL_FTSC = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
SPECS = {
    "yolov9t": {"stock": ROOT / "models_related/models_config/yolov9/levir/yolov9t_levir_b0_stock.yaml", "b1": ROOT / "models_related/models_config/yolov9/levir/yolov9t_p2_levir_b1_noftsc.yaml", "h2": ROOT / "models_related/models_config/yolov9/levir/yolov9t_p2_levir_ftsc_h2.yaml", "es1": ROOT / "models_related/models_config/yolov9/levir/yolov9t_p2_levir_ftsc_es1_historical.yaml", "es2": ROOT / "models_related/models_config/yolov9/levir/yolov9t_p2_levir_ftsc_es2_historical.yaml", "p2": 18, "p3": 21, "detect": "Detect", "native": ("ELAN1", "RepNCSPELAN4", "AConv"), "pretrained": "yolov9t.pt"},
    "yolov10n": {"stock": ROOT / "models_related/models_config/yolov10/levir/yolov10n_levir_b0_stock.yaml", "b1": ROOT / "models_related/models_config/yolov10/levir/yolov10n_p2_levir_b1_noftsc.yaml", "h2": ROOT / "models_related/models_config/yolov10/levir/yolov10n_p2_levir_ftsc_h2.yaml", "es1": ROOT / "models_related/models_config/yolov10/levir/yolov10n_p2_levir_ftsc_es1_historical.yaml", "es2": ROOT / "models_related/models_config/yolov10/levir/yolov10n_p2_levir_ftsc_es2_historical.yaml", "p2": 19, "p3": 22, "detect": "v10Detect", "native": ("C2f", "SCDown"), "pretrained": "yolov10n.pt"},
    "yolo11n": {"stock": ROOT / "models_related/models_config/yolo11/levir/yolo11n_levir_b0_stock.yaml", "b1": ROOT / "models_related/models_config/yolo11/levir/yolo11n_p2_levir_b1_noftsc.yaml", "h2": ROOT / "models_related/models_config/yolo11/levir/yolo11n_p2_levir_ftsc_h2.yaml", "es1": ROOT / "models_related/models_config/yolo11/levir/yolo11n_p2_levir_ftsc_es1_historical.yaml", "es2": ROOT / "models_related/models_config/yolo11/levir/yolo11n_p2_levir_ftsc_es2_historical.yaml", "p2": 19, "p3": 22, "detect": "Detect", "native": ("C3k2", "Conv"), "pretrained": "yolo11n.pt"},
}
TRAINING = {"split_seed": 42, "epochs": 100, "imgsz": 512, "batch": 8, "workers": 4, "patience": 20, "deterministic": True, "amp": True, "optimizer": "AdamW", "lr0": .002, "momentum": .9, "beta2_effective": .999, "weight_decay": .0005, "cos_lr": False, "lrf": .01, "warmup_epochs": 3., "warmup_momentum": .8, "warmup_bias_lr": 0., "nbs": 64, "plots": False, "mosaic": 0., "close_mosaic": 0, "mixup": 0., "cutmix": 0., "copy_paste": 0., "rect": False, "adaptive_zoom": False, "hsv_h": .015, "hsv_s": .7, "hsv_v": .4, "degrees": 0., "translate": .1, "scale": .5, "shear": 0., "perspective": 0., "flipud": 0., "fliplr": .5, "fitness_metric": "map50_95", "nms_iou": .5, "final_checkpoint": "weights/best.pt", "historical_optimizer_requested": "auto", "historical_optimizer_resolved": "AdamW", "current_optimizer_pinned": "AdamW"}
RUNS_ROOT = ROOT / "runs/levir_ftsc_cross_yolo_portability_v1"
REQUIRED_METRICS = ("val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)", "test/metrics/precision(B)", "test/metrics/recall(B)", "test/metrics/mAP50(B)", "test/metrics/mAP75(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small", "model/parameters", "model/GFLOPs", "test_speed/inference_ms_per_image")
REQUIRED_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv", "experiment_manifest.json", "evaluation_metrics_extended.json")
BENCHMARK_FINGERPRINT_KEYS = ("benchmark/gpu_name", "benchmark/backend", "benchmark/device", "benchmark/imgsz", "benchmark/batch", "benchmark/precision")
# Live-session-only caches.  Keys include immutable artifact/config identities
# so a later one-job dispatch cannot reuse a check for a different source.
_SESSION_RUNTIME_PREFLIGHT_CACHE: set[tuple[str, str]] = set()
_SESSION_PARITY_CACHE: set[tuple[str, str, str, str]] = set()
CONFIG_KEYS = {"B0_STOCK": "stock", "B1_P23_NOFTSC": "b1", "FTSC_H2": "h2", "FTSC_H2_CP2": "h2", "FTSC_ES1_HISTORICAL": "es1", "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS": "es1", "FTSC_ES2_HISTORICAL": "es2"}

def require(ok: bool, message: str) -> None:
    if not ok: raise RuntimeError(message)

def load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    require(isinstance(value, dict), f"invalid YAML: {path}")
    return value

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def model_config_for(family: str, case: str) -> Path:
    """Single source of truth for the frozen case-to-YAML mapping."""
    require(family in SPECS and case in CONFIG_KEYS, f"Unsupported family/case: {family}/{case}")
    return SPECS[family][CONFIG_KEYS[case]]

def run_dir_for(family: str, case: str, seed: int) -> Path:
    require(seed in SEEDS, f"Seed outside canonical universe: {seed}")
    return RUNS_ROOT / family / case / f"seed_{seed}"

def pretrained_identity(family: str, override: str | None = None) -> dict[str, Any]:
    """Resolve only a supplied/local native checkpoint; never download it."""
    requested = override or SPECS[family]["pretrained"]
    path = Path(override).expanduser() if override else ROOT / requested
    if not path.is_file():
        return {"family": family, "requested": requested, "status": "PENDING_LOCAL_CHECKPOINT"}
    canonical = path.resolve()
    return {"family": family, "requested": requested, "canonical_path": str(canonical), "sha256": digest(canonical), "status": "READY"}

def scientific_manifest(family: str, case: str, seed: int, config: Path, pretrained: dict[str, Any], workers: int, device: str, data_yaml: Path | None = None, requested_seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Separate immutable scientific identity from mutable execution context."""
    payload = resolved_payload(family, case)
    manifest_path = data_yaml.parent / "manifest.json" if data_yaml else None
    dataset_fingerprint = {"split_seed": TRAINING["split_seed"], "manifest_sha256": digest(manifest_path)} if manifest_path and manifest_path.is_file() else None
    pretrained_scientific = {"family": family, "artifact": SPECS[family]["pretrained"], "sha256": pretrained.get("sha256")}
    return {"scientific_contract": {"schema": "levir_ftsc_cross_yolo_portability_v1", "dataset": "LEVIR-Ship", "family": family, "case": case, "seed": seed, "canonical_seed_universe": list(SEEDS), "split_seed": TRAINING["split_seed"], "dataset_fingerprint": dataset_fingerprint, "model_config_sha256": digest(config), "detect_class": SPECS[family]["detect"], "expected_detect_inputs": payload["head"][-1][0], "expected_strides": [8,16,32] if case == "B0_STOCK" else [4,8], "ftsc": payload.get("ftsc", {"enabled": False}), "edge_historical": payload.get("edge_stability"), "augmentation": augmentation(case), "training": TRAINING, "pretrained": pretrained_scientific, "fitness_metric": TRAINING["fitness_metric"], "evaluation_nms_iou": TRAINING["nms_iou"]}, "artifact_provenance": {"model_config": str(config), "model_config_sha256": digest(config), "dataset_yaml": str(data_yaml) if data_yaml else None, "dataset_manifest": str(manifest_path) if manifest_path else None, "pretrained_family": family, "pretrained_artifact": SPECS[family]["pretrained"], "pretrained_sha256": pretrained.get("sha256"), "pretrained_requested": pretrained.get("requested"), "pretrained_canonical_path": pretrained.get("canonical_path"), "smart_transfer": True, "smart_transfer_statistics": "unavailable"}, "execution_context": {"default_workers": TRAINING["workers"], "effective_workers": workers, "device": device}, "invocation_context": {"requested_seed_selection": list(requested_seeds)}}

def require_live_dataset_fingerprint(manifest: dict[str, Any]) -> None:
    """Refuse every live state transition without a content-addressed split."""
    fingerprint = manifest["scientific_contract"].get("dataset_fingerprint")
    require(isinstance(fingerprint, dict) and bool(fingerprint.get("manifest_sha256")), "Prepared LEVIR split lacks a manifest SHA256 fingerprint")

def ftsc_contract() -> dict[str, Any]: return copy.deepcopy(load(CANONICAL_FTSC)["ftsc"])

def augmentation(case: str) -> dict[str, Any]:
    base = {"mosaic": 0., "close_mosaic": 0, "mixup": 0., "cutmix": 0., "copy_paste": 0., "copy_paste_enabled": False, "copy_paste_mode": "flip"}
    if case == "FTSC_H2_CP2":
        from train_levir_scripts.train_levir_ftsc_augmentation_suite import augmentation_for
        base.update(augmentation_for("A3_CP2"))
    if case == "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS":
        from train_levir_scripts.train_ftsc_negative_canvas_suite import augmentation_for
        base.update(augmentation_for("r4"))
    require(base["mosaic"] == 0. and base["close_mosaic"] == 0, f"{case}: Mosaic is forbidden")
    return base

def resolved_payload(family: str, case: str) -> dict[str, Any]:
    payload = load(model_config_for(family, case))
    # Upstream stock files retain ImageNet/COCO nc=80.  Resolve only the
    # LEVIR experiment copy to nc=1; never mutate vendored stock YAML.
    if case == "B0_STOCK": payload["nc"] = 1
    return payload

def collect_tensors(value: Any) -> list[Any]:
    """Return every tensor in arbitrarily nested model output containers."""
    import torch
    if isinstance(value, torch.Tensor): return [value]
    if isinstance(value, dict): return [t for child in value.values() for t in collect_tensors(child)]
    if isinstance(value, (list, tuple)): return [t for child in value for t in collect_tensors(child)]
    return []

def normalized_graph(payload: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(payload); output.pop("ftsc", None); output.pop("edge_stability", None); output.pop("ablation", None); return output

def write_resolved(family: str, case: str, directory: Path) -> Path:
    path = directory / f"{family}_{case}.yaml"; path.write_text(yaml.safe_dump(resolved_payload(family, case), sort_keys=False)); return path

def static_preflight(family: str, case: str) -> dict[str, Any]:
    payload, spec = resolved_payload(family, case), SPECS[family]
    layers = [*payload["backbone"], *payload["head"]]
    final = layers[-1]
    expected_levels = 3 if case == "B0_STOCK" else 2
    require(final[2] == spec["detect"], f"{family}/{case}: wrong detect class")
    require(len(final[0]) == expected_levels, f"{family}/{case}: wrong detect levels")
    if case != "B0_STOCK":
        require(all(name in [x[2] for x in layers] for name in spec["native"]), f"{family}/{case}: native block missing")
        historical = case in {"FTSC_ES1_HISTORICAL", "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS", "FTSC_ES2_HISTORICAL"}
        require(final[0] == [spec["p2"], spec["p3"] + (2 if historical else 0)], f"{family}/{case}: P2/P3 Detect taps wrong")
        if historical:
            edge = [i for i, layer in enumerate(layers) if layer[2] == "P2EdgeCueFusion"]
            require(len(edge) == 1 and layers[edge[0] - 1][2] == "nn.Identity", f"{family}/{case}: not historical Edge graph")
    enabled = bool(payload.get("ftsc", {}).get("enabled", False))
    ftsc_case = case.startswith("FTSC_")
    require(enabled == ftsc_case, f"{family}/{case}: FTSC state mismatch")
    if enabled: require(payload["ftsc"] == ftsc_contract(), f"{family}/{case}: FTSC contract drift")
    if case == "FTSC_H2_CP2":
        expected = {"copy_paste_enabled": True, "copy_paste_mode": "single", "copy_paste_copies": 2, "copy_paste_p": .5, "copy_paste_placement": "random"}
        require(all(augmentation(case).get(k) == v for k, v in expected.items()), f"{family}/{case}: CP2 contract drift")
    if case == "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS":
        expected = {"copy_paste_enabled": True, "copy_paste_mode": "negative_canvas", "negative_cp_target_policy": "deficit", "negative_cp_donor_policy": "larger", "negative_cp_degradation": "weak_blur"}
        require(all(augmentation(case).get(k) == v for k, v in expected.items()), f"{family}/{case}: R4 contract drift")
    return {"detect_levels": expected_levels, "detect_inputs": final[0], "ftsc": enabled, "augmentation": augmentation(case)}

def runtime_preflight(family: str, case: str, config: Path) -> dict[str, Any]:
    import torch
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss
    model = DetectionModel(str(config), verbose=False)
    # DetectionModel construction alone intentionally does not attach trainer
    # hyperparameters.  Use framework defaults for a real criterion smoke.
    model.args = get_cfg(DEFAULT_CFG, {"epochs": TRAINING["epochs"], "fitness_metric": TRAINING["fitness_metric"]})
    head = model.model[-1]; expected = [8., 16., 32.] if case == "B0_STOCK" else [4., 8.]
    require([float(x) for x in head.stride.tolist()] == expected, f"{family}/{case}: stride mismatch")
    model.train(); x = torch.randn(2, 3, 128, 128); out = model(x)
    tensors = collect_tensors(out)
    require(tensors, f"{family}/{case}: forward exposed zero tensors")
    require(all(not torch.is_floating_point(v) or torch.isfinite(v).all() for v in tensors), f"{family}/{case}: nonfinite forward")
    criterion = model.init_criterion()
    model.criterion = criterion
    criterion.epoch = 5
    if family == "yolov10n":
        require(model.end2end and isinstance(criterion, E2ELoss), "YOLOv10 native E2ELoss criterion lost")
        require((criterion.one2many.ftsc_calibrator is not None) == case.startswith("FTSC_"), "YOLOv10 one2many FTSC mismatch")
        require(criterion.one2one.ftsc_calibrator is None, "YOLOv10 one2one FTSC must be off")
    else: require((criterion.ftsc_calibrator is not None) == case.startswith("FTSC_"), f"{family}/{case}: criterion FTSC mismatch")
    # A real loss path catches errors invisible to raw inference.  Keep the
    # batch minimal but valid: one normalized xywh target per image.
    batch = {"img": torch.randn(2, 3, 128, 128), "batch_idx": torch.tensor([0, 1]), "cls": torch.tensor([0.0, 0.0]), "bboxes": torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]])}
    total, items = model.loss(batch)
    require(torch.isfinite(total).all() and torch.isfinite(items).all(), f"{family}/{case}: nonfinite criterion loss")
    total.sum().backward()
    require(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad), f"{family}/{case}: no finite detector gradient")
    if case.startswith("FTSC_"):
        active = criterion.one2many.ftsc_metrics if family == "yolov10n" else criterion.ftsc_metrics
        require(float(active.get("ftsc_rho", 0.0)) > 0.0, f"{family}/{case}: FTSC schedule did not activate")
    return {"strides": expected, "detect_class": type(head).__name__, "end2end": bool(getattr(model, "end2end", False)), "synthetic_loss": True, "synthetic_backward": True}

def verify_b1_h2_initialization_parity(family: str, pretrained: dict[str, Any]) -> None:
    """Live-only guard: shared B1/H2 state must match after one checkpoint load."""
    require(pretrained.get("status") == "READY", f"{family}: missing pretrained artifact")
    from ultralytics import YOLO
    import torch
    torch.manual_seed(0)
    b1 = YOLO(model_config_for(family, "B1_P23_NOFTSC"), task="detect")
    torch.manual_seed(0)
    h2 = YOLO(model_config_for(family, "FTSC_H2"), task="detect")
    b1.load(pretrained["canonical_path"], smart_transfer=True); h2.load(pretrained["canonical_path"], smart_transfer=True)
    compare_b1_h2_shared_state(family, b1.model.state_dict(), h2.model.state_dict(), torch)

def compare_b1_h2_shared_state(family: str, b1_state: dict[str, Any], h2_state: dict[str, Any], torch_module: Any) -> None:
    """Pure shared-state comparison; FTSC-only state is intentionally ignored."""
    shared = set(b1_state).intersection(h2_state)
    require(shared, f"{family}: B1/H2 share no state keys")
    shape_mismatched = [key for key in shared if b1_state[key].shape != h2_state[key].shape]
    require(not shape_mismatched, f"{family}: B1/H2 shared-state shape mismatch: {shape_mismatched[:5]}")
    value_mismatched = [key for key in shared if not torch_module.equal(b1_state[key], h2_state[key])]
    require(not value_mismatched, f"{family}: B1/H2 shared initialization mismatch: {value_mismatched[:5]}")

def metrics_complete(run_dir: Path) -> bool:
    if any(not (run_dir / relative).is_file() for relative in REQUIRED_ARTIFACTS): return False
    try:
        from evaluate_test.standard_detection_metrics import load_merged_metrics
        metrics = load_merged_metrics(run_dir)
    except (ImportError, OSError, ValueError, json.JSONDecodeError): return False
    return all(metrics.get(key) is not None for key in REQUIRED_METRICS)

def run_status(run_dir: Path, manifest: dict[str, Any], pretrained: dict[str, Any], data_yaml: Path | None) -> str:
    if pretrained["status"] != "READY": return "PENDING_PRETRAINED"
    if data_yaml is None or not data_yaml.is_file(): return "PENDING_DATA_PREPARATION"
    if not run_dir.exists() or not any(run_dir.iterdir()): return "NEW_RUN_REQUIRED"
    path = run_dir / "experiment_manifest.json"
    if not path.is_file(): raise RuntimeError(f"Refusing non-empty run without manifest: {run_dir}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("scientific_contract") != manifest["scientific_contract"]: raise RuntimeError(f"Scientific manifest mismatch: {run_dir}")
    if metrics_complete(run_dir): return "COMPLETE"
    if (run_dir / "weights/best.pt").is_file(): return "EVAL_BACKFILL_REQUIRED"
    if (run_dir / "weights/last.pt").is_file(): return "RESUME_REQUIRED"
    raise RuntimeError(f"Refusing incomplete run with no resumable checkpoint: {run_dir}")

def train_kwargs(args: argparse.Namespace, family: str, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    """Frozen scientific protocol; only device/workers are execution overrides."""
    kwargs = {key: value for key, value in TRAINING.items() if key not in {"split_seed", "workers", "nms_iou", "final_checkpoint", "beta2_effective", "historical_optimizer_requested", "historical_optimizer_resolved", "current_optimizer_pinned"}}
    kwargs.update(data=str(data_yaml), seed=seed, workers=args.workers, device=args.device)
    kwargs.update(augmentation(case))
    return kwargs

def train_one(args: argparse.Namespace, family: str, case: str, seed: int, data_yaml: Path, pretrained: dict[str, Any]) -> Path:
    """Live-only single-run dispatch. Call only after full confirmed preflight."""
    config, run_dir = model_config_for(family, case), run_dir_for(family, case, seed)
    manifest = scientific_manifest(family, case, seed, config, pretrained, args.workers, args.device, data_yaml, tuple(args.seeds))
    status = run_status(run_dir, manifest, pretrained, data_yaml)
    if status == "COMPLETE": return run_dir
    if status == "EVAL_BACKFILL_REQUIRED": return run_dir
    if status == "RESUME_REQUIRED":
        from train_levir_scripts.train_all_levir_yolov8n_p2_routing import local_ultralytics
        local_ultralytics()
        from ultralytics import YOLO
        last = run_dir / "weights/last.pt"
        require(last.is_file(), f"Missing resume checkpoint: {last}")
        YOLO(last, task="detect").train(resume=str(last))
        return run_dir
    require(status == "NEW_RUN_REQUIRED", f"Unsupported run status: {status}")
    from h2_generalization_completion_common import ensure_manifest
    ensure_manifest(run_dir, manifest)
    from train_levir_scripts.train_all_levir_yolov8n_p2_routing import local_ultralytics, seed_everything
    local_ultralytics(); seed_everything(seed)
    from ultralytics import YOLO
    model = YOLO(config, task="detect")
    model.load(pretrained["canonical_path"], smart_transfer=True)
    model.train(**train_kwargs(args, family, case, seed, data_yaml), project=str(run_dir.parent), name=run_dir.name, exist_ok=False)
    required = (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv")
    require(all(path.is_file() for path in required), f"Incomplete training artifacts: {run_dir}")
    return run_dir

def validate_args(args: argparse.Namespace) -> None:
    require(args.families and len(set(args.families)) == len(args.families) and all(family in FAMILIES for family in args.families), "families must be a unique canonical subset")
    require(args.cases and len(set(args.cases)) == len(args.cases) and all(case in CASES for case in args.cases), "cases must be a unique canonical subset")
    require(args.seeds and len(set(args.seeds)) == len(args.seeds) and all(seed in SEEDS for seed in args.seeds), "seeds must be a unique canonical subset")
    require(isinstance(args.workers, int) and args.workers >= 0, "workers must be a non-negative integer")
    require(isinstance(args.device, str) and bool(args.device.strip()), "device must be a non-empty string")

def build_preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Read-only status matrix; deliberately does not prepare data or build models."""
    expected_yaml = args.dataset_root / f"levir_ship_yolo_seed{TRAINING['split_seed']}" / "levir_ship.yaml"
    rows: list[dict[str, Any]] = []
    for family in args.families:
        require(normalized_graph(resolved_payload(family, "B1_P23_NOFTSC")) == normalized_graph(resolved_payload(family, "FTSC_H2")), f"{family}: B1/H2 graph parity failed")
        pretrained = pretrained_identity(family, getattr(args, f"pretrained_{family}"))
        for case in args.cases:
            config, static = model_config_for(family, case), static_preflight(family, case)
            for seed in SEEDS:
                manifest = scientific_manifest(family, case, seed, config, pretrained, args.workers, args.device, expected_yaml, tuple(args.seeds))
                requested = seed in args.seeds
                status = run_status(run_dir_for(family, case, seed), manifest, pretrained, expected_yaml) if requested else "NOT_REQUESTED"
                rows.append({"family": family, "case": case, "seed": seed, "requested": requested, "config": str(config), "config_sha256": digest(config), "static": static, "pretrained": pretrained, "manifest": manifest, "status": status})
    return rows

def evaluate_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    """Live-only canonical LEVIR evaluation of the exact best checkpoint."""
    from evaluate_test.standard_detection_metrics import evaluate_run, load_merged_metrics
    run_dir, data_yaml = run_dir_for(row["family"], row["case"], row["seed"]), Path(row["data_yaml"])
    evaluate_run(run_dir=run_dir, data_yaml=data_yaml, dataset="levir", imgsz=TRAINING["imgsz"], batch=TRAINING["batch"], workers=args.workers, device=args.device, nms_iou=TRAINING["nms_iou"])
    row["metrics"] = load_merged_metrics(run_dir); row["status"] = "COMPLETE" if metrics_complete(run_dir) else "EVAL_BACKFILL_REQUIRED"
    return row

def dispatch(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Live-only state machine; dataset preparation occurs after read-only planning."""
    from train_levir_scripts.train_all_levir_yolov8n_p2_routing import prepare_fixed_split, validate_split
    args.split_seed = TRAINING["split_seed"]
    data_yaml = prepare_fixed_split(args); validate_split(data_yaml)
    require((data_yaml.parent / "manifest.json").is_file(), "Prepared LEVIR split has no canonical manifest")
    from train_levir_scripts.train_ftsc_negative_canvas_suite import dataset_eligibility
    for row in rows:
        if not row.get("requested", True): continue
        family, case, seed = row["family"], row["case"], row["seed"]
        pretrained = pretrained_identity(family, getattr(args, f"pretrained_{family}"))
        manifest = scientific_manifest(family, case, seed, model_config_for(family, case), pretrained, args.workers, args.device, data_yaml, tuple(args.seeds))
        require_live_dataset_fingerprint(manifest)
        row.update(pretrained=pretrained, manifest=manifest, data_yaml=str(data_yaml), status=run_status(run_dir_for(family, case, seed), manifest, pretrained, data_yaml))
        require(row["status"] not in {"PENDING_PRETRAINED", "PENDING_DATA_PREPARATION"}, f"Live dispatch blocked: {family}/{case}/seed_{seed}")
        if case == "FTSC_ES1_HISTORICAL_R4_NEGATIVE_CANVAS": dataset_eligibility(data_yaml)
        if row["status"] == "NEW_RUN_REQUIRED":
            parity_key = (family, str(pretrained.get("sha256")), digest(model_config_for(family, "B1_P23_NOFTSC")), digest(model_config_for(family, "FTSC_H2")))
            if parity_key not in _SESSION_PARITY_CACHE:
                verify_b1_h2_initialization_parity(family, pretrained); _SESSION_PARITY_CACHE.add(parity_key)
            key = (family, row["config_sha256"])
            if key not in _SESSION_RUNTIME_PREFLIGHT_CACHE:
                runtime_preflight(family, case, model_config_for(family, case)); _SESSION_RUNTIME_PREFLIGHT_CACHE.add(key)
            train_one(args, family, case, seed, data_yaml, pretrained); row["status"] = "EVAL_BACKFILL_REQUIRED"
        if row["status"] == "RESUME_REQUIRED":
            train_one(args, family, case, seed, data_yaml, pretrained); row["status"] = "EVAL_BACKFILL_REQUIRED"
        if row["status"] == "EVAL_BACKFILL_REQUIRED": evaluate_one(args, row)
        elif row["status"] == "COMPLETE":
            from evaluate_test.standard_detection_metrics import load_merged_metrics
            row["metrics"] = load_merged_metrics(run_dir_for(family, case, seed))
            row["status"] = "REUSED"
    return rows

def write_summaries(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    """Persist run rows and family/case aggregates through shared completion helpers."""
    from h2_generalization_completion_common import write_csv
    from evaluate_test.standard_detection_metrics import same_benchmark_fingerprint
    flat = [{"family": row["family"], "case": row["case"], "seed": row["seed"], "requested": row.get("requested", True), "status": row["status"], **row.get("metrics", {})} for row in rows]
    write_csv(RUNS_ROOT / "summary_runs.csv", flat)
    aggregate_rows = []
    for family, case in sorted({(row["family"], row["case"]) for row in flat}):
        completed = [row for row in flat if row["family"] == family and row["case"] == case and row["status"] in {"COMPLETE", "REUSED"}]
        fingerprint_complete = bool(completed) and all(
            row.get(key) is not None for row in completed for key in BENCHMARK_FINGERPRINT_KEYS
        )
        completed_seeds = {int(row["seed"]) for row in completed}
        requested_seeds = set(args.seeds)
        summary = {"family": family, "case": case, "n_complete": len(completed), "requested_seed_count": len(requested_seeds), "canonical_seed_count": len(SEEDS), "requested_complete": requested_seeds.issubset(completed_seeds), "canonical_complete": set(SEEDS).issubset(completed_seeds), "speed_comparable": fingerprint_complete and same_benchmark_fingerprint(completed)}
        for metric in REQUIRED_METRICS:
            values = [float(row[metric]) for row in completed if row.get(metric) is not None]
            if metric == "test_speed/inference_ms_per_image" and not summary["speed_comparable"]:
                summary[f"{metric}/mean"] = summary[f"{metric}/std"] = None
            else:
                summary[f"{metric}/mean"] = statistics.fmean(values) if values else None
                summary[f"{metric}/std"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate_rows.append(summary)
    write_csv(RUNS_ROOT / "summary_aggregate.csv", aggregate_rows)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)); parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES)); parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS)); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--device", default="cuda"); parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData"); parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets"); parser.add_argument("--pretrained-yolov9t"); parser.add_argument("--pretrained-yolov10n"); parser.add_argument("--pretrained-yolo11n"); parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    rows = build_preflight(args)
    print(json.dumps({"suite": "levir_ftsc_cross_yolo_portability_v1", "training": TRAINING, "rows": rows}, indent=2, sort_keys=True, default=str))
    if not args.confirm_run: return
    rows = dispatch(args, rows)
    write_summaries(args, rows)

if __name__ == "__main__": main()
