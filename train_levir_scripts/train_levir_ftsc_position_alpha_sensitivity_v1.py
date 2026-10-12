#!/usr/bin/env python3
"""Historical-H2-matched LEVIR FTSC Position Gaussian alpha sensitivity.

The original head-pruning H2 is the baseline; only Position Gaussian alpha
changes in new ablation arms. Earlier alpha-v1 results remain isolated.
No training without --confirm-run; --live-preflight builds models but never trains.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow
from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as historical

CONFIG_ROOT = REPO / "models_related/models_config/yolov8/levir"
H2 = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
EXPERIMENT = "LEVIR_FTSC_POSITION_ALPHA_HISTORICAL_H2_V1"
NAMESPACE = "levir_ftsc_position_alpha_historical_h2_v1"
HISTORICAL_REFERENCE = REPO / "reference/levir_h2_historical_head_pruning_seed42_43_44.csv"
HISTORICAL_REFERENCE_SOURCE_SHA256 = "03c81c0800b499552311b62d965aa692bf156cffed50f061bd2065812af77779"
HISTORICAL_REFERENCE_SHA256 = "03cbe509b3e59c282bf840cdeb82a9765b8f7ffd848004b5df6f493ed80a4f88"
CASES = ("ALPHA4", "ALPHA6", "ALPHA8", "ALPHA_ADAPTIVE")
DEFAULT_CASES = ("ALPHA4", "ALPHA8")
SEEDS = (42, 43, 44)
ALPHA_POLICY = {
    "ALPHA4": ("fixed", 4.0),
    "ALPHA6": ("fixed", 6.0),
    "ALPHA8": ("fixed", 8.0),
    "ALPHA_ADAPTIVE": ("size_adaptive", 6.0),
}
YAMLS = {case: CONFIG_ROOT / f"yolov8n_p2_levir_ftsc_h2_{case.lower()}.yaml" for case in CASES}
TRAIN_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv")
TRAINING_FINALIZATION = "training_completion.json"
EVALUATION_ARTIFACTS = ("evaluation_metrics_extended.json",)
IDENTITY_ARTIFACTS = ("run_contract.json", "experiment_manifest.json", TRAINING_FINALIZATION)
COMPLETE_ARTIFACTS = TRAIN_ARTIFACTS + EVALUATION_ARTIFACTS + IDENTITY_ARTIFACTS + ("completion.json",)
# The upload contract is verified with the same local validator before publication.
HF_REQUIRED_ARTIFACTS = COMPLETE_ARTIFACTS
METRICS_REQUIRED = (
    "val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)",
    "test/metrics/precision(B)", "test/metrics/recall(B)",
    "test/metrics/mAP50(B)", "test/metrics/mAP75(B)",
    "test/metrics/mAP50-95(B)", "test_size/AP50-Small",
    "model/parameters", "model/GFLOPs", "test_speed/inference_ms_per_image",
)
SOURCE_FILES = (
    "train_levir_scripts/train_levir_ftsc_position_alpha_sensitivity_v1.py",
    "train_levir_scripts/train_all_levir_yolov8n_p2_ftsc_head_pruning.py",
    "train_levir_scripts/train_all_levir_yolov8n_p2_routing.py",
    "models_related/ultralytics/ultralytics/cfg/default.yaml",
    "models_related/ultralytics/ultralytics/utils/tal.py",
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
    "models_related/ultralytics/ultralytics/nn/modules/head.py",
    "models_related/ultralytics/ultralytics/utils/loss.py",
    "models_related/ultralytics/ultralytics/engine/trainer.py",
    "evaluate_test/standard_detection_metrics.py",
    "evaluate_test/size_bucket_evaluator.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object in {path}")
    return value


def source_identity() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPO, text=True, stderr=subprocess.DEVNULL,
        )
        state = "VERIFIED" if commit and not dirty.strip() else "DIRTY"
    except (OSError, subprocess.CalledProcessError):
        commit, dirty, state = None, None, "UNAVAILABLE"
    files = {relative: sha(REPO / relative) for relative in SOURCE_FILES}
    return {"state": state, "commit": commit, "dirty_paths_present": bool(dirty), "source_sha256": files}


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int) -> dict[str, Any]:
    """Reuse the EXACT historical head-pruning training recipe, not a rewrite."""
    require(historical.VARIANTS["H2_p2_p3_nomosaic"] == H2,
            "HISTORICAL_H2_CONFIG_PATH_DRIFT")
    require(historical.FITNESS_METRIC == "map50_95", "HISTORICAL_FITNESS_DRIFT")
    options = historical.train_kwargs(args, data_yaml, seed, True)
    require(options.get("optimizer") == "auto"
            and options.get("mosaic") == 0.0 and options.get("close_mosaic") == 0
            and options.get("amp") is True, "HISTORICAL_TRAINING_RECIPE_DRIFT")
    return options


def scientific_train_args(args: argparse.Namespace, data_yaml: Path, seed: int) -> dict[str, Any]:
    values = train_kwargs(args, data_yaml, seed)
    for operational in ("data", "device", "workers"):
        values.pop(operational)
    return values


def validate_args(args: argparse.Namespace) -> None:
    require(bool(args.cases) and len(set(args.cases)) == len(args.cases) and set(args.cases).issubset(CASES), "Invalid/duplicate alpha cases")
    require(bool(args.seeds) and len(set(args.seeds)) == len(args.seeds) and set(args.seeds).issubset(SEEDS), "Invalid/duplicate seeds")
    require(args.split_seed == 42, "LEVIR split seed is frozen at 42")
    require((args.epochs, args.imgsz, args.batch_size, args.patience) == (100, 512, 8, 20), "LEVIR scientific settings are frozen")
    require(args.workers >= 0, "workers must be nonnegative")
    require(not (args.preflight_only and args.confirm_run), "Preflight-only and confirm-run cannot both be enabled")
    require(not (args.live_preflight and args.confirm_run), "Live-preflight and confirm-run must be separate")
    require(not (args.aggregate_only and args.confirm_run), "Aggregate-only must not train")
    require("ALPHA6" not in args.cases or args.allow_alpha6_bridge or not args.confirm_run,
            "Historical H2 is the default alpha=6 control; enable --allow-alpha6-bridge only for a deliberate control replication")


def source_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import yaml
    canonical = yaml.safe_load(H2.read_text(encoding="utf-8"))
    graph = {key: value for key, value in canonical.items() if key != "ftsc"}
    original_ftsc = dict(canonical["ftsc"])
    original_ftsc.pop("position_alpha", None)
    cases = {}
    for case, path in YAMLS.items():
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        require({key: data for key, data in value.items() if key != "ftsc"} == graph, f"{case}: H2 graph drift")
        ftsc = dict(value["ftsc"])
        alpha, mode = float(ftsc.pop("position_alpha")), ftsc.pop("position_alpha_mode", "fixed")
        require(ftsc == original_ftsc, f"{case}: non-alpha FTSC drift")
        require((mode, alpha) == ALPHA_POLICY[case], f"{case}: wrong alpha mapping")
        require(value["head"][-1][0] == [19, 22], f"{case}: invalid Detect inputs")
        cases[case] = {"alpha": alpha, "alpha_mode": mode, "yaml_sha256": sha(path)}
    # Thresholds are enforced in FTSC source and covered by focused regression tests.
    return {"status": "SOURCE_CONFIG_VERIFIED", "selected_cases": list(args.cases),
            "training_seeds": list(args.seeds), "split_seed": 42, "cases": cases,
            "source": source_identity()}


def validate_dataset(data_yaml: Path, data_root: Path) -> dict[str, Any]:
    import yaml
    workflow.validate_split(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    require(cfg.get("names") in ({0: "ship"}, ["ship"]), "LEVIR one-class label mapping drift")
    root = Path(cfg["path"])
    expected = sorted(p.stem for p in (data_root / "All Images").glob("*.png"))
    original_labels = {p.stem for p in (data_root / "All Annotations").glob("*.txt")}
    require(len(expected) == 3896 and original_labels == set(expected), "LEVIR raw source count/matching failed")
    randomized = list(expected)
    random.Random(42).shuffle(randomized)
    offset = 0
    fingerprints = {}
    for split, count in (("train", 2320), ("val", 788), ("test", 788)):
        stems = sorted(p.stem for p in (root / cfg[split]).glob("*.png"))
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        require(len(stems) == count and set(stems) == labels, f"LEVIR {split}: count/annotation mismatch")
        require(set(stems) == set(randomized[offset:offset + count]), f"LEVIR {split}: deterministic membership mismatch")
        offset += count
        member_digest = hashlib.sha256("\n".join(stems).encode()).hexdigest()
        labels_digest = hashlib.sha256()
        for stem in stems:
            prepared = (root / "labels" / split / f"{stem}.txt").read_bytes()
            original = (data_root / "All Annotations" / f"{stem}.txt").read_bytes()
            require(prepared == original, f"LEVIR {split}: label content drift for {stem}")
            labels_digest.update(stem.encode() + b"\0" + prepared + b"\0")
        fingerprints[split] = {"members_sha256": member_digest,
                               "labels_sha256": labels_digest.hexdigest()}
    # Path-only differences in generated levir_ship.yaml are operational;
    # the deterministic membership and exact label bytes are scientific.
    return {"split_seed": 42, "split_algorithm": "sorted_stems_random.Random(42).shuffle",
            "split_fingerprints": fingerprints, "class_mapping": {0: "ship"},
            "counts": {"train": 2320, "val": 788, "test": 788}}


def audit_train_labels(data_yaml: Path, imgsz: int = 512) -> dict[str, Any]:
    """Label-only letterbox estimate, distinct from effective post-augmentation FTSC positives."""
    if not data_yaml.is_file():
        return {"status": "PENDING_DATA_PREPARATION", "data_yaml": str(data_yaml)}
    import yaml
    from PIL import Image
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data["path"])
    directory = root / data["train"]
    labels = root / "labels/train"
    sizes, bins, empty, invalid = [], [0] * 5, 0, 0
    for image in sorted(directory.glob("*.png")):
        annotation = labels / f"{image.stem}.txt"
        lines = annotation.read_text(encoding="utf-8").splitlines() if annotation.is_file() else []
        if not lines:
            empty += 1
        with Image.open(image) as im:
            width, height = im.size
        scale = min(imgsz / width, imgsz / height)
        for line in lines:
            try:
                values = line.split()
                bw, bh = float(values[3]) * width * scale, float(values[4]) * height * scale
                if not (math.isfinite(bw) and math.isfinite(bh) and bw > 0 and bh > 0):
                    raise ValueError("invalid size")
            except (ValueError, IndexError):
                invalid += 1
                continue
            side = math.sqrt(bw * bh)
            sizes.append(side)
            bins[0 if side < 2 else 1 if side < 8 else 2 if side < 16 else 3 if side < 32 else 4] += 1
    values = sorted(sizes)
    return {"status": "READY", "basis": "train labels at nominal letterbox 512; NOT augmented boxes",
            "images": sum(1 for _ in directory.glob("*.png")), "gt_objects": len(values),
            "empty_images": empty, "invalid_boxes": invalid,
            "bins": dict(zip(("lt2", "2to8", "8to16", "16to32", "ge32"), bins)),
            "proportions": {name: count / len(values) if values else 0
                            for name, count in zip(("lt2", "2to8", "8to16", "16to32", "ge32"), bins)},
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "percentiles": {str(p): values[round((len(values) - 1) * p / 100)] if values else None
                            for p in (1, 5, 10, 25, 75, 90, 95, 99)}}


def model_for(case: str, pretrained: str):
    workflow.local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(YAMLS[case])
    model.load(pretrained, smart_transfer=True)
    return model


def model_preflight(pretrained: str, cases: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """LIVE-only: prove identical pretrained initialization to historical H2.

    Every variant is constructed with exactly the same RNG seed and
    smart_transfer path. Alpha is a non-parameter training-time policy:
    all model parameter/buffer initial states must match bit for bit.
    """
    import torch
    workflow.local_ultralytics()
    from ultralytics import YOLO
    workflow.seed_everything(0)
    historical_model = YOLO(H2)
    historical_model.load(pretrained, smart_transfer=True)
    reference_detector = historical_model.model.model[-1]
    require(reference_detector.nl == 2 and reference_detector.nc == 1,
            "Historical H2 Detect mismatch")
    require([int(x) for x in reference_detector.stride.tolist()] == [4, 8],
            "Historical H2 strides mismatch")
    reference_state = {key: tensor.detach().cpu().clone()
                       for key, tensor in historical_model.model.state_dict().items()}
    report = {}
    for case in cases:
        workflow.seed_everything(0)
        model = model_for(case, pretrained)
        detector = model.model.model[-1]
        require(detector.nl == 2 and detector.nc == 1, f"{case}: wrong Detect levels/classes")
        require([int(x) for x in detector.stride.tolist()] == [4, 8], f"{case}: wrong strides")
        calibrator = detector.ftsc_calibrator
        require(calibrator is not None and calibrator.policy == "f5", f"{case}: FTSC F5 missing")
        require(set(calibrator.providers) == {"position_gaussian", "dfl_distribution"},
                f"{case}: FTSC evidence drift")
        require(calibrator.providers["dfl_distribution"].detach is True, f"{case}: DFL detach drift")
        provider = calibrator.providers["position_gaussian"]
        require((provider.alpha_mode, provider.alpha) == ALPHA_POLICY[case],
                f"{case}: effective alpha policy drift")
        state = model.model.state_dict()
        require(set(state) == set(reference_state), f"{case}: H2 state keys differ")
        mismatched = [key for key, value in state.items()
                      if not torch.equal(value.detach().cpu(), reference_state[key])]
        require(not mismatched,
                f"{case}: pretrained initialization differs from historical H2: {mismatched[:8]}")
        report[case] = {"strides": [4, 8], "identical_H2_initial_state": True,
                        "state_entries": len(reference_state)}
        del model
    del historical_model
    return {"status": "MODEL_PREFLIGHT_PASSED", "historical_h2_config": str(H2),
            "alpha_only_init_parity": report}


def contract(args: argparse.Namespace, case: str, seed: int, data_yaml: Path,
             data_identity: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "levir_ftsc_alpha_historical_h2_contract_v1", "case": case,
        "baseline": {"source_suite": historical.PROTOCOL_VERSION,
                     "historical_case": "H2_p2_p3_nomosaic",
                     "historical_model_yaml_sha256": sha(H2),
                     "historical_training_runner_sha256": sha(REPO / "train_levir_scripts/train_all_levir_yolov8n_p2_ftsc_head_pruning.py")},
        "training_seed": seed, "split_seed": 42,
        "alpha_policy": {"mode": ALPHA_POLICY[case][0], "alpha": ALPHA_POLICY[case][1]},
        "model_yaml_sha256": sha(YAMLS[case]), "source": source,
        "dataset": data_identity, "pretrained_sha256": sha(Path(args.pretrained)),
        "train_args": scientific_train_args(args, data_yaml, seed),
        "checkpoint_selection": "validation/mAP50-95",
        "optimizer_requested": "auto", "optimizer_resolved_expected": "AdamW",
        "evaluation": {"dataset": "levir", "imgsz": 512, "batch": 8, "nms_iou": 0.5,
                       "evaluator_sha256": sha(REPO / "evaluate_test/standard_detection_metrics.py")},
    }


def _matches_contract(run: Path, expected: dict[str, Any]) -> bool:
    try:
        return read_json(run / "run_contract.json") == expected and read_json(run / "experiment_manifest.json") == expected
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def set_state(run: Path, state: str, **detail: Any) -> None:
    write_json(run / "run_state.json", {"state": state, **detail})


def validate_metrics(metrics: dict[str, Any]) -> None:
    require(isinstance(metrics, dict), "Evaluation metrics must be an object")
    for key in METRICS_REQUIRED:
        require(key in metrics, f"Missing LEVIR metric: {key}")
        value = metrics[key]
        require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
                f"Nonfinite/non-numeric LEVIR metric {key}")
        require(value >= 0, f"Negative LEVIR metric {key}")
    for key in ("val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)", "test/metrics/mAP50(B)",
                "test/metrics/mAP75(B)", "test/metrics/mAP50-95(B)"):
        require(metrics[key] <= 1.0, f"Out-of-range AP: {key}")
    require(metrics.get("evaluation/checkpoint") == "weights/best.pt", "Incorrect evaluation checkpoint")
    require(metrics.get("evaluation/dataset") == "levir", "Incorrect evaluation dataset")
    require(metrics.get("evaluation/nms_iou") == 0.5, "Incorrect NMS IoU")
    require(metrics.get("benchmark/imgsz") == 512 and metrics.get("benchmark/batch") == 8,
            "Incorrect evaluation benchmark resolution/batch")


def _checkpoint(last: Path, *, require_resumable: bool) -> dict[str, Any]:
    """Inspect trusted project checkpoints; note final_eval() strips epoch/optimizer."""
    # PyTorch pickle needs the same vendored Ultralytics classes during aggregate/restore.
    workflow.local_ultralytics()
    import torch
    try:
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"INVALID_CHECKPOINT: {last}") from error
    require(isinstance(ckpt, dict) and isinstance(ckpt.get("train_args"), dict),
            f"INVALID_CHECKPOINT_SCHEMA: {last}")
    if require_resumable:
        require(isinstance(ckpt.get("epoch"), int) and 0 <= ckpt["epoch"] < 100,
                f"NOT_RESUMABLE_EPOCH: {last}")
        require(ckpt.get("optimizer") is not None and ckpt.get("scaler") is not None,
                f"NOT_RESUMABLE: optimizer/scaler missing from {last}")
    else:
        # Vendored BaseTrainer.final_eval() invokes strip_optimizer(last.pt)
        # which deliberately overwrites epoch=-1 and clears the optimizer.
        require(ckpt.get("epoch") == -1 and ckpt.get("optimizer") is None and ckpt.get("model") is not None,
                f"TRAINING_CHECKPOINT_NOT_FINALIZED: {last}")
    return ckpt


def _checkpoint_protocol(ckpt: dict[str, Any], expected: dict[str, Any], *,
                         finalized: bool) -> None:
    args = ckpt["train_args"]
    for key, value in expected["train_args"].items():
        if key == "data":
            continue  # Membership is checked separately, avoiding path-only drift.
        if finalized and key not in args:
            # strip_optimizer() drops non-default extension keys.
            continue
        require(args.get(key) == value, f"CHECKPOINT_PROTOCOL_DRIFT: {key}")
    # Checkpoint identity must agree with the requested alpha treatment,
    # including before its checksum is sealed in completion.json.
    checkpoint_model = ckpt.get("model")
    if checkpoint_model is None:
        checkpoint_model = ckpt.get("ema")
    require(checkpoint_model is not None, "CHECKPOINT_MODEL_MISSING")
    payload = getattr(checkpoint_model, "yaml", None)
    require(isinstance(payload, dict) and isinstance(payload.get("ftsc"), dict),
            "CHECKPOINT_FTSC_YAML_MISSING")
    position = payload["ftsc"]
    policy = expected["alpha_policy"]
    require(float(position.get("position_alpha", 6.0)) == policy["alpha"]
            and str(position.get("position_alpha_mode", "fixed")) == policy["mode"],
            "CHECKPOINT_ALPHA_POLICY_MISMATCH")
    if not finalized:
        require(0 <= ckpt["epoch"] < expected["train_args"]["epochs"],
                "RESUME_CHECKPOINT_EPOCH_INVALID")


def _last_csv_epoch(run: Path) -> int:
    with (run / "results.csv").open(newline="", encoding="utf-8") as handle:
        values = list(csv.DictReader(handle))
    require(bool(values), f"EMPTY_TRAINING_RESULTS: {run}")
    last = {key.strip(): value for key, value in values[-1].items()}
    require("epoch" in last, f"MISSING_TRAINING_EPOCH_COLUMN: {run}")
    return int(float(last["epoch"]))


def _checkpoint_results_epoch(ckpt: dict[str, Any]) -> int:
    """Ensure the finalized last.pt really contains the last logged training epoch."""
    results = ckpt.get("train_results")
    require(isinstance(results, dict), "CHECKPOINT_TRAIN_RESULTS_MISSING")
    column = next((value for key, value in results.items() if str(key).strip() == "epoch"), None)
    require(isinstance(column, (list, tuple)) and bool(column), "CHECKPOINT_EPOCH_HISTORY_MISSING")
    return int(float(column[-1]))


def _finalize_training(run: Path, expected: dict[str, Any], trained_model) -> None:
    """Produce finalization marker only after the actual trainer returned and completed."""
    trainer = getattr(trained_model, "trainer", None)
    require(trainer is not None, "TRAINER_NOT_AVAILABLE_AFTER_RETURN")
    completed = int(trainer.epoch) + 1
    target = int(expected["train_args"]["epochs"])
    stopper = getattr(trainer, "stopper", None)
    best_epoch = getattr(stopper, "best_epoch", None)
    early_stopped = completed < target and best_epoch is not None and completed - int(best_epoch) >= 20
    require(completed >= target or early_stopped,
            f"TRAINER_RETURNED_WITHOUT_CONFIRMED_COMPLETION: {completed}/{target}")
    require(all((run / item).is_file() for item in TRAIN_ARTIFACTS), "TRAIN_ARTIFACTS_MISSING")
    ckpt = _checkpoint(run / "weights/last.pt", require_resumable=False)
    _checkpoint_protocol(ckpt, expected, finalized=True)
    require(_last_csv_epoch(run) == completed, "TRAINING_CSV_FINAL_EPOCH_MISMATCH")
    require(_checkpoint_results_epoch(ckpt) == completed, "LAST_CKPT_TRAIN_RESULTS_EPOCH_MISMATCH")
    marker = {
        "schema": "levir_ftsc_training_completion_v2", "contract_sha256": sha(run / "run_contract.json"),
        "seed": expected["training_seed"], "configured_epochs": target, "completed_epochs": completed,
        "completion_reason": "early_stopped" if early_stopped else "max_epochs",
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "artifact_sha256": {item: sha(run / item) for item in TRAIN_ARTIFACTS},
    }
    write_json(run / TRAINING_FINALIZATION, marker)


def validate_training_completion(run: Path, expected: dict[str, Any]) -> bool:
    marker_file = run / TRAINING_FINALIZATION
    if not marker_file.is_file():
        return False
    try:
        marker = read_json(marker_file)
        if marker.get("schema") != "levir_ftsc_training_completion_v2":
            return False
        if marker.get("contract_sha256") != sha(run / "run_contract.json"):
            return False
        if marker.get("seed") != expected["training_seed"]:
            return False
        if marker.get("configured_epochs") != expected["train_args"]["epochs"]:
            return False
        epochs = marker.get("completed_epochs")
        if not isinstance(epochs, int) or epochs < 1 or epochs > expected["train_args"]["epochs"]:
            return False
        if epochs < expected["train_args"]["epochs"]:
            if marker.get("completion_reason") != "early_stopped":
                return False
            best = marker.get("best_epoch")
            if not isinstance(best, int) or epochs - best < 20:
                return False
        elif marker.get("completion_reason") != "max_epochs":
            return False
        for item in TRAIN_ARTIFACTS:
            if marker.get("artifact_sha256", {}).get(item) != sha(run / item):
                return False
        ckpt = _checkpoint(run / "weights/last.pt", require_resumable=False)
        _checkpoint_protocol(ckpt, expected, finalized=True)
        return _last_csv_epoch(run) == epochs and _checkpoint_results_epoch(ckpt) == epochs
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
        return False


def validate_completed_run(run: Path, expected: dict[str, Any]) -> dict[str, Any]:
    require(_matches_contract(run, expected), f"CONTRACT_MISMATCH: {run}")
    for item in COMPLETE_ARTIFACTS:
        require((run / item).is_file(), f"MISSING_COMPLETION_ARTIFACT: {run / item}")
    completion = read_json(run / "completion.json")
    require(completion.get("state") == "COMPLETE", f"COMPLETION_NOT_FINAL: {run}")
    require(completion.get("contract_sha256") == sha(run / "run_contract.json"), f"CONTRACT_HASH_MISMATCH: {run}")
    require(validate_training_completion(run, expected), f"UNVERIFIED_TRAINING_COMPLETION: {run}")
    for item in TRAIN_ARTIFACTS + EVALUATION_ARTIFACTS + IDENTITY_ARTIFACTS:
        require(completion.get("artifact_sha256", {}).get(item) == sha(run / item),
                f"ARTIFACT_HASH_MISMATCH: {run / item}")
    metrics = read_json(run / "evaluation_metrics_extended.json")
    validate_metrics(metrics)
    require(completion.get("evaluated_best_sha256") == sha(run / "weights/best.pt"),
            f"EVALUATION_CHECKPOINT_CHANGED: {run}")
    return metrics


def _resume(run: Path, expected: dict[str, Any], args: argparse.Namespace):
    require((run / "weights/last.pt").is_file(), f"MISSING_RESUME_CKPT: {run}")
    ckpt = _checkpoint(run / "weights/last.pt", require_resumable=True)
    _checkpoint_protocol(ckpt, expected, finalized=False)
    require(ckpt["epoch"] + 1 < expected["train_args"]["epochs"], f"RESUME_NOT_NEEDED_OR_IMPOSSIBLE: {run}")
    workflow.local_ultralytics()
    from ultralytics import YOLO
    model = YOLO(run / "weights/last.pt")
    require(model.ckpt and model.ckpt.get("optimizer") is not None, "ULTRALYTICS_RESUME_WOULD_FALLBACK_TO_FRESH_TRAINING")
    model.train(resume=True, device=args.device, workers=args.workers)
    return model


def _evaluate(run: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, Any]:
    from evaluate_test.standard_detection_metrics import evaluate_run
    metrics = evaluate_run(run, data_yaml, dataset="levir", imgsz=512, batch=8,
                           device=args.device, workers=args.workers, nms_iou=0.5)
    validate_metrics(metrics)
    return metrics


def run_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path,
            data_identity: dict[str, Any], source: dict[str, Any]) -> Path:
    require(case != "ALPHA6" or args.allow_alpha6_bridge,
            "Historical H2 is the alpha6 control; explicit --allow-alpha6-bridge required")
    expected = contract(args, case, seed, data_yaml, data_identity, source)
    run = args.project / case / f"seed_{seed}"
    require(source["state"] == "VERIFIED", "SOURCE_NOT_CLEAN_OR_UNAVAILABLE")
    if run.exists():
        require(_matches_contract(run, expected), f"CONTRACT_MISMATCH: {run}")
    else:
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "run_contract.json", expected)
        write_json(run / "experiment_manifest.json", expected)
    if (run / "completion.json").exists():
        validate_completed_run(run, expected)
        return run
    if not validate_training_completion(run, expected):
        set_state(run, "TRAINING")
        if (run / "weights/last.pt").exists():
            model = _resume(run, expected, args)
        else:
            require(not any((run / item).exists() for item in TRAIN_ARTIFACTS),
                    f"ORPHAN_TRAINING_ARTIFACTS: {run}")
            workflow.seed_everything(seed)
            model = model_for(case, args.pretrained)
            model.train(**train_kwargs(args, data_yaml, seed),
                        project=str(args.project / case), name=f"seed_{seed}", exist_ok=True)
        _finalize_training(run, expected, model)
    set_state(run, "TRAINED")
    # Recover trained-but-unevaluated runs without retraining; full evaluator is authoritative.
    set_state(run, "EVALUATING")
    metrics = _evaluate(run, data_yaml, args)
    set_state(run, "EVALUATED")
    # Extended evaluation writes the metric artifact before we seal the manifest.
    fingerprints = {item: sha(run / item) for item in TRAIN_ARTIFACTS + EVALUATION_ARTIFACTS + IDENTITY_ARTIFACTS}
    write_json(run / "completion.json", {
        "schema": "levir_ftsc_alpha_completion_v2", "state": "COMPLETE",
        "contract_sha256": sha(run / "run_contract.json"),
        "artifact_sha256": fingerprints, "evaluated_best_sha256": sha(run / "weights/best.pt"),
    })
    validate_completed_run(run, expected)
    set_state(run, "COMPLETE")
    return run


def historical_h2_reference(seeds: list[int]) -> list[dict[str, Any]]:
    """Historical result table is a reference, never a fabricated new alpha6 run.

    Imported verbatim (metric values) from the original LEVIR head-pruning CSV.
    Provenance is limited to the historical result export; this does not itself
    prove that old checkpoints used the current source commit.
    """
    require(HISTORICAL_REFERENCE.is_file(), "MISSING_HISTORICAL_H2_REFERENCE")
    require(sha(HISTORICAL_REFERENCE) == HISTORICAL_REFERENCE_SHA256,
            "HISTORICAL_H2_REFERENCE_SHA256_MISMATCH")
    with HISTORICAL_REFERENCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 3, "Historical H2 reference must contain exactly 3 seeds")
    require({int(row["seed"]) for row in rows} == set(SEEDS),
            "Historical H2 reference seed identity mismatch")
    require(all(
        row["variant"] == "H2_p2_p3_nomosaic"
        and row["protocol/version"] == historical.PROTOCOL_VERSION
        and row["protocol/fitness_metric"] == historical.FITNESS_METRIC
        and float(row["protocol/mosaic"]) == 0.0
        and int(row["protocol/close_mosaic"]) == 0
        and float(row["protocol/nms_iou"]) == 0.5
        and int(row["params"]) == 1821959
        for row in rows), "Historical H2 reference protocol mismatch")
    selected = []
    for row in rows:
        if int(row["seed"]) in seeds:
            selected.append({"case": "HISTORICAL_H2_ALPHA6_REFERENCE",
                             "seed": int(row["seed"]), "cohort": historical.PROTOCOL_VERSION,
                             "reference_only": True,
                             **{key: float(value) for key, value in row.items()
                                if key.startswith(("test/metrics/", "val/metrics/"))}})
    return sorted(selected, key=lambda row: row["seed"])


def aggregate(args: argparse.Namespace, data_yaml: Path, data_identity: dict[str, Any],
              source: dict[str, Any]) -> None:
    from evaluate_test.standard_detection_metrics import same_benchmark_fingerprint
    rows, statuses = [], []
    for case in args.cases:
        for seed in args.seeds:
            run = args.project / case / f"seed_{seed}"
            if not run.exists():
                statuses.append({"case": case, "seed": seed, "status": "MISSING"})
                continue
            expected = contract(args, case, seed, data_yaml, data_identity, source)
            if not (run / "completion.json").is_file():
                statuses.append({"case": case, "seed": seed, "status": "INCOMPLETE"})
                continue
            # Corrupt or mismatched completed runs are an error, NEVER silently dropped.
            metrics = validate_completed_run(run, expected)
            rows.append({"case": case, "seed": seed, **metrics})
            statuses.append({"case": case, "seed": seed, "status": "COMPLETE"})
    def write_csv(name: str, values: list[dict[str, Any]]) -> None:
        path = args.project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted(set().union(*(v.keys() for v in values))) if values else ["case", "seed"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
    write_csv("levir_ftsc_position_alpha_historical_h2_v1_summary_runs.csv", rows)
    summary = []
    for case in args.cases:
        group = [row for row in rows if row["case"] == case]
        record: dict[str, Any] = {"case": case, "runs": len(group),
                                  "missing_or_incomplete": sum(r["status"] != "COMPLETE" and r["case"] == case for r in statuses)}
        comparable = same_benchmark_fingerprint(group) if group else False
        record["speed_comparable"] = comparable
        for metric in METRICS_REQUIRED:
            if metric == "test_speed/inference_ms_per_image" and not comparable:
                continue
            values = [float(row[metric]) for row in group]
            if values:
                record[f"{metric}/mean"] = statistics.fmean(values)
                record[f"{metric}/sample_std"] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(record)
    write_csv("levir_ftsc_position_alpha_historical_h2_v1_summary_aggregate.csv", summary)
    # Keep historical H2 visibly separate: reference metrics do NOT count as
    # completed alpha6 runs or enter the new-cohort means/std.
    references = historical_h2_reference(list(args.seeds))
    write_csv("levir_ftsc_position_alpha_historical_h2_v1_historical_reference_runs.csv", references)
    reference_summary: dict[str, Any] = {
        "case": "HISTORICAL_H2_ALPHA6_REFERENCE",
        "cohort": historical.PROTOCOL_VERSION,
        "reference_only": True, "runs": len(references),
    }
    for metric in ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)",
                   "test/metrics/mAP75(B)", "test/metrics/precision(B)",
                   "test/metrics/recall(B)", "val/metrics/mAP50(B)",
                   "val/metrics/mAP50-95(B)"):
        scores = [row[metric] for row in references]
        if scores:
            reference_summary[metric + "/mean"] = statistics.fmean(scores)
            reference_summary[metric + "/sample_std"] = (
                statistics.stdev(scores) if len(scores) > 1 else None
            )
    write_csv("levir_ftsc_position_alpha_historical_h2_v1_historical_reference_aggregate.csv",
              [reference_summary])
    # Trainer's results.csv already carries epoch-indexed mechanism metrics.
    diagnostics = []
    for row in rows:
        csv_path = args.project / row["case"] / f"seed_{row['seed']}" / "results.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for epoch_row in csv.DictReader(handle):
                diag = {key.strip(): value for key, value in epoch_row.items()
                        if key.strip().startswith("ftsc_") or key.strip() == "epoch"}
                if diag:
                    diagnostics.append({"case": row["case"], "seed": row["seed"], **diag})
    write_csv("levir_ftsc_position_alpha_historical_h2_v1_diagnostics.csv", diagnostics)
    write_json(args.project / "levir_ftsc_position_alpha_historical_h2_v1_manifest.json", {
        "schema": "levir_ftsc_alpha_historical_h2_suite_v1", "namespace": NAMESPACE,
        "historical_h2_reference": {"source_cohort": historical.PROTOCOL_VERSION,
                                     "case": "H2_p2_p3_nomosaic",
                                     "source_results_sha256": HISTORICAL_REFERENCE_SOURCE_SHA256,
                                     "selected_seed_reference_csv": str(HISTORICAL_REFERENCE.relative_to(REPO)),
                                     "historical_metrics_referenced": True,
                                     "historical_training_rerun": False,
                                     "status": "HISTORICAL_REFERENCE_NOT_RETRAINED",
                                     "causal_delta_permitted_only_after_source_parity_check": True},
        "source": source, "dataset": data_identity, "statuses": statuses,
        "metric_source": "evaluation_metrics_extended.json",
        "diagnostics_basis": "epoch CSV, FTSC-positive-weighted where applicable",
    })


def restore_missing(args: argparse.Namespace, uploader, data_yaml: Path,
                    data_identity: dict[str, Any], source: dict[str, Any]) -> None:
    for case in args.cases:
        for seed in args.seeds:
            run = args.project / case / f"seed_{seed}"
            if run.exists():  # Never overwrite partial local training on restore.
                continue
            uploader.restore_completed_runs(namespace=NAMESPACE, project=args.project,
                                            variants=[case], seeds=[seed],
                                            required_artifacts=HF_REQUIRED_ARTIFACTS)
            if run.exists():
                validate_completed_run(run, contract(args, case, seed, data_yaml, data_identity, source))


def publish_completed(args: argparse.Namespace, uploader, case: str, seed: int,
                      data_yaml: Path, data_identity: dict[str, Any],
                      source: dict[str, Any]) -> dict[str, Any]:
    run = args.project / case / f"seed_{seed}"
    expected = contract(args, case, seed, data_yaml, data_identity, source)
    metrics = validate_completed_run(run, expected)
    published = uploader.publish_run(
        namespace=NAMESPACE, case=case, seed=seed, run_dir=run,
        required_artifacts=HF_REQUIRED_ARTIFACTS,
        summary_metadata={"scientific_contract_sha256": sha(run / "run_contract.json"),
                          "metrics_keys": list(METRICS_REQUIRED)},
    )
    write_json(run / "publication.json", {"state": "PUBLISHED", "archive": published})
    return metrics


def run_suite(args: argparse.Namespace, *, uploader=None) -> dict[str, Any]:
    validate_args(args)
    args.project = Path(args.project).expanduser().resolve()
    args.pretrained = str(Path(args.pretrained).expanduser().resolve())
    source_report = source_preflight(args)
    data_yaml = Path(args.dataset_root) / "levir_ship_yolo_seed42/levir_ship.yaml"
    if args.aggregate_only:
        require(data_yaml.is_file() and Path(args.pretrained).is_file(), "Aggregation needs original dataset and pretrained weights")
        data_identity = validate_dataset(data_yaml, Path(args.data_root))
        aggregate(args, data_yaml, data_identity, source_report["source"])
        return {"status": "AGGREGATED"}
    if args.preflight_only or (not args.confirm_run and not args.live_preflight):
        return {"status": "PREPARED — NOT YET RUN",
                "source": source_report, "size_audit": audit_train_labels(data_yaml)}
    require(source_report["source"]["state"] == "VERIFIED", "SOURCE_NOT_CLEAN_OR_UNAVAILABLE")
    require(Path(args.pretrained).is_file(), "Missing canonical yolov8n.pt")
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.dataset_root = Path(args.dataset_root).expanduser().resolve()
    data_yaml = workflow.prepare_fixed_split(args)
    data_identity = validate_dataset(data_yaml, args.data_root)
    size_audit = audit_train_labels(data_yaml)
    require(size_audit["status"] == "READY" and size_audit["images"] == 2320, "GT-size audit not ready")
    model_report = model_preflight(args.pretrained, list(args.cases))
    args.project.mkdir(parents=True, exist_ok=True)
    write_json(args.project / "levir_ftsc_position_alpha_historical_h2_v1_size_distribution.json", size_audit)
    if args.live_preflight:
        return {"status": "LIVE_PREFLIGHT_PASSED_NO_TRAINING",
                "model_preflight": model_report, "size_audit": size_audit,
                "dataset": data_identity, "source": source_report}
    if uploader is not None:
        restore_missing(args, uploader, data_yaml, data_identity, source_report["source"])
    for seed in args.seeds:
        for case in args.cases:
            run_one(args, case, seed, data_yaml, data_identity, source_report["source"])
            if uploader is not None:
                publish_completed(args, uploader, case, seed, data_yaml, data_identity, source_report["source"])
            aggregate(args, data_yaml, data_identity, source_report["source"])
    return {"status": "COMPLETE", "model_preflight": model_report,
            "project": str(args.project), "cases": list(args.cases), "seeds": list(args.seeds)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--allow-alpha6-bridge", action="store_true",
                        help="Explicitly permit new alpha6 control replication; not used by default")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=REPO.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=REPO / "datasets")
    parser.add_argument("--project", type=Path, default=REPO / f"runs/{NAMESPACE}")
    parser.add_argument("--pretrained", type=Path, default=REPO / "yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--live-preflight", action="store_true", help="Prepare/verify data and instantiate models; do not train")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(run_suite(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
