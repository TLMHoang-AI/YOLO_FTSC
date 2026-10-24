"""Shared fail-closed mechanics for dataset-specific cross-YOLO FTSC suites.

Dataset preparation, training arguments, evaluation, and dataset fingerprints
remain owned by the dataset runners.  This module only centralizes the model,
state-machine, metric, aggregation, and archive invariants already exercised by
the LEVIR portability implementation.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import yaml

FAMILIES = ("yolov9t", "yolov10n", "yolo11n")
CASES = ("B1_P23_NOFTSC", "FTSC_H2")
SEEDS = (42, 43, 44)
BENCHMARK_KEYS = (
    "benchmark/gpu_name", "benchmark/backend", "benchmark/device",
    "benchmark/imgsz", "benchmark/batch", "benchmark/precision",
    "benchmark/latency_unit",
)
TRAIN_ARTIFACTS = ("weights/best.pt", "weights/last.pt", "results.csv")
# These proofs distinguish an interrupted training checkpoint from a finalized model.
TRAINING_COMPLETION_FILE = "cross_yolo_training_complete.json"
EVALUATION_COMPLETION_FILE = "cross_yolo_evaluation_complete.json"
TRAINING_PROOF_ARTIFACTS = (*TRAIN_ARTIFACTS, "args.yaml", "config.yaml", "experiment_manifest.json")
FINAL_ARTIFACTS = (*TRAINING_PROOF_ARTIFACTS, "evaluation_metrics_extended.json", TRAINING_COMPLETION_FILE, EVALUATION_COMPLETION_FILE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid YAML mapping: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, default=str))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(canonical(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def normalized_graph(payload: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    for key in ("ftsc", "ablation", "edge_stability"):
        value.pop(key, None)
    return value


def static_model_preflight(
    family: str,
    case: str,
    config: Path,
    spec: dict[str, Any],
    canonical_ftsc: dict[str, Any],
) -> dict[str, Any]:
    require(family in FAMILIES and case in CASES, f"Unsupported identity: {family}/{case}")
    require(config.is_file(), f"Missing model YAML: {config}")
    payload = load_yaml(config)
    layers = [*payload.get("backbone", []), *payload.get("head", [])]
    require(layers and layers[-1][2] == spec["detect"], f"{family}/{case}: native Detect class drift")
    require(layers[-1][0] == [spec["p2"], spec["p3"]], f"{family}/{case}: P2/P3 taps drift")
    modules = {layer[2] for layer in layers}
    require(set(spec["native"]).issubset(modules), f"{family}/{case}: native family block missing")
    ftsc = payload.get("ftsc", {})
    enabled = bool(ftsc.get("enabled", False))
    require(enabled == (case == "FTSC_H2"), f"{family}/{case}: FTSC enabled state drift")
    if enabled:
        require(ftsc == canonical_ftsc, f"{family}/{case}: dataset H2 FTSC contract drift")
        require(ftsc.get("policy") == "f5", f"{family}/{case}: F5 policy required")
        require(ftsc.get("position_alpha") == 6.0, f"{family}/{case}: fixed alpha=6 required")
        require(ftsc.get("position_alpha_mode", "fixed") == "fixed", f"{family}/{case}: adaptive alpha forbidden")
        require(ftsc.get("dfl_detach") is True and ftsc.get("dfl_apply_cls") is True, f"{family}/{case}: DFL evidence drift")
    return {
        "config": str(config), "sha256": sha256(config), "detect_class": spec["detect"],
        "detect_inputs": layers[-1][0], "expected_strides": [4, 8], "ftsc_enabled": enabled,
    }


def collect_tensors(value: Any) -> list[Any]:
    import torch
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in collect_tensors(child)]
    if isinstance(value, (tuple, list)):
        return [item for child in value for item in collect_tensors(child)]
    return []


def runtime_model_preflight(family: str, case: str, config: Path, *, epochs: int, fitness_metric: str) -> dict[str, Any]:
    """Build, forward, and backward through the native loss; never trains."""
    import torch
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss

    model = DetectionModel(str(config), verbose=False)
    model.args = get_cfg(DEFAULT_CFG, {"epochs": epochs, "fitness_metric": fitness_metric})
    head = model.model[-1]
    require([float(item) for item in head.stride.tolist()] == [4.0, 8.0], f"{family}/{case}: runtime stride drift")
    require(type(head).__name__ == ("v10Detect" if family == "yolov10n" else "Detect"), f"{family}/{case}: runtime Detect drift")
    model.train()
    output = model(torch.randn(2, 3, 128, 128))
    tensors = collect_tensors(output)
    require(tensors and all(not torch.is_floating_point(item) or torch.isfinite(item).all() for item in tensors), f"{family}/{case}: invalid forward")
    criterion = model.init_criterion()
    model.criterion = criterion
    criterion.epoch = 5
    enabled = case == "FTSC_H2"
    if family == "yolov10n":
        require(bool(model.end2end) and isinstance(criterion, E2ELoss), "YOLOv10 native E2ELoss lost")
        require((criterion.one2many.ftsc_calibrator is not None) == enabled, "YOLOv10 one-to-many FTSC mismatch")
        require(criterion.one2one.ftsc_calibrator is None, "YOLOv10 one-to-one FTSC must remain disabled")
    else:
        require((criterion.ftsc_calibrator is not None) == enabled, f"{family}/{case}: criterion FTSC mismatch")
    batch = {
        "img": torch.randn(2, 3, 128, 128), "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([0.0, 0.0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]),
    }
    total, items = model.loss(batch)
    require(torch.isfinite(total).all() and torch.isfinite(items).all(), f"{family}/{case}: non-finite loss")
    total.sum().backward()
    require(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad), f"{family}/{case}: no finite detector gradient")
    if enabled:
        branch = criterion.one2many if family == "yolov10n" else criterion
        require(float(branch.ftsc_metrics.get("ftsc_rho", 0.0)) > 0.0, f"{family}/{case}: FTSC schedule inactive")
        calibrator = branch.ftsc_calibrator
        visible = [parameter for parameter in calibrator.parameters() if parameter.requires_grad]
        require(visible, f"{family}/{case}: no optimizer-visible FTSC parameters")
    return {"detect_class": type(head).__name__, "strides": [4, 8], "synthetic_loss": True, "synthetic_backward": True}


def pretrained_identity(root: Path, family: str, artifact: str, override: str | None) -> dict[str, Any]:
    requested = override or artifact
    path = Path(override).expanduser() if override else root / artifact
    if not path.is_file():
        return {"family": family, "artifact": artifact, "requested": requested, "status": "BLOCKED", "reason": "missing pretrained checkpoint"}
    path = path.resolve()
    return {"family": family, "artifact": artifact, "requested": requested, "path": str(path), "sha256": sha256(path), "status": "READY"}


def compare_shared_state(family: str, left: dict[str, Any], right: dict[str, Any], torch_module: Any) -> None:
    shared = set(left).intersection(right)
    require(shared, f"{family}: B1/H2 share no state")
    require(all(left[key].shape == right[key].shape for key in shared), f"{family}: shared-state shape mismatch")
    mismatched = [key for key in shared if not torch_module.equal(left[key], right[key])]
    require(not mismatched, f"{family}: B1/H2 initialization mismatch: {mismatched[:5]}")


def verify_initialization_parity(family: str, b1: Path, h2: Path, pretrained: dict[str, Any], seed_fn: Callable[[int], None]) -> None:
    require(pretrained.get("status") == "READY", f"{family}: pretrained checkpoint unavailable")
    import torch
    from ultralytics import YOLO
    seed_fn(0); left = YOLO(b1, task="detect"); left.load(pretrained["path"], smart_transfer=True)
    seed_fn(0); right = YOLO(h2, task="detect"); right.load(pretrained["path"], smart_transfer=True)
    compare_shared_state(family, left.model.state_dict(), right.model.state_dict(), torch)


def validate_metrics(metrics: dict[str, Any], required: tuple[str, ...], *, accuracy_count: int = 3) -> None:
    for index, key in enumerate(required):
        value = metrics.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), f"METRIC_INCOMPLETE: {key}")
        if index < accuracy_count:
            require(0.0 <= float(value) <= 1.0, f"Invalid accuracy metric: {key}")
        else:
            require(float(value) > 0.0, f"Non-positive metric: {key}")


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "evaluation_metrics_extended.json"
    require(path.is_file(), f"Missing evaluation artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid evaluation artifact: {path}")
    return value


def manifest_matches(run_dir: Path, expected: dict[str, Any]) -> None:
    path = run_dir / "experiment_manifest.json"
    require(path.is_file(), f"Non-empty run lacks manifest: {run_dir}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    require(recorded.get("scientific_contract") == expected.get("scientific_contract"), f"Scientific manifest mismatch: {run_dir}")


def _read_proof(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict) and payload.get("schema") == schema, f"Invalid completion proof: {path}")
    return payload


def mark_training_complete(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Only call after YOLO.train() returns successfully (including early stopping)."""
    manifest_matches(run_dir, manifest)
    proof = run_dir / TRAINING_COMPLETION_FILE
    require(not proof.exists(), f"Training proof already exists; refusing to rewrite: {proof}")
    # A cache from a previous evaluation cannot be reused after further training.
    for name in ("evaluation_metrics.json", "evaluation_metrics_extended.json", EVALUATION_COMPLETION_FILE):
        require(not (run_dir / name).exists(), f"Stale evaluation cache before training completion: {run_dir / name}")
    for name in TRAINING_PROOF_ARTIFACTS:
        require((run_dir / name).is_file(), f"Missing finalized training artifact: {run_dir / name}")
    write_json(proof, {
        "schema": "cross_yolo_training_complete_v1",
        "artifact_sha256": {name: sha256(run_dir / name) for name in TRAINING_PROOF_ARTIFACTS},
    })


def verify_training_complete(run_dir: Path) -> None:
    proof = _read_proof(run_dir / TRAINING_COMPLETION_FILE, "cross_yolo_training_complete_v1")
    hashes = proof.get("artifact_sha256")
    require(isinstance(hashes, dict), f"Training artifact hashes missing: {run_dir}")
    for name in TRAINING_PROOF_ARTIFACTS:
        path = run_dir / name
        require(path.is_file() and hashes.get(name) == sha256(path), f"Finalized training artifact mismatch: {path}")


def quarantine_unverified_evaluation_cache(run_dir: Path) -> list[str]:
    """Preserve but never reuse native metrics without an evaluation completion proof.

    Both dataset-native evaluators cache evaluation_metrics.json (and Varroa also
    caches evaluation_metrics_extended.json). A best.pt may have changed after
    an interrupted evaluation, so replaying either cache can silently associate
    old accuracy/speed with a new checkpoint. Move such files outside the seed
    run directory before evaluating again, without overwriting any old copy.
    """
    verify_training_complete(run_dir)
    require(not (run_dir / EVALUATION_COMPLETION_FILE).exists(),
            f"Refusing to invalidate an existing evaluation proof: {run_dir}")
    target_root = run_dir.parent / "_unverified_evaluation_cache" / run_dir.name
    moved: list[str] = []
    for name in ("evaluation_metrics.json", "evaluation_metrics_extended.json"):
        source = run_dir / name
        if not source.exists():
            continue
        require(source.is_file(), f"Evaluation cache path is not a file: {source}")
        target_root.mkdir(parents=True, exist_ok=True)
        index = 1
        target = target_root / f"{name}.stale.{index}"
        while target.exists():
            index += 1
            target = target_root / f"{name}.stale.{index}"
        source.rename(target)
        moved.append(str(target))
    return moved


def mark_evaluation_complete(run_dir: Path, metrics: dict[str, Any]) -> None:
    """Bind the persisted, validated evaluation to the exact finalized best.pt."""
    verify_training_complete(run_dir)
    expected_best = sha256(run_dir / "weights/best.pt")
    require(metrics.get("evaluation/checkpoint_sha256") == expected_best,
            f"Evaluation/checkpoint mismatch: {run_dir}")
    saved = read_metrics(run_dir)
    require(saved == metrics, f"Evaluation artifact differs from validated metrics: {run_dir}")
    write_json(run_dir / EVALUATION_COMPLETION_FILE, {
        "schema": "cross_yolo_evaluation_complete_v1",
        "checkpoint_sha256": expected_best,
        "training_proof_sha256": sha256(run_dir / TRAINING_COMPLETION_FILE),
        "metrics_sha256": sha256(run_dir / "evaluation_metrics_extended.json"),
    })


def verify_evaluation_complete(
    run_dir: Path, required_metrics: tuple[str, ...],
    metric_validator: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    verify_training_complete(run_dir)
    proof = _read_proof(run_dir / EVALUATION_COMPLETION_FILE, "cross_yolo_evaluation_complete_v1")
    best_hash = sha256(run_dir / "weights/best.pt")
    require(proof.get("checkpoint_sha256") == best_hash, f"Evaluation checkpoint hash mismatch: {run_dir}")
    require(proof.get("training_proof_sha256") == sha256(run_dir / TRAINING_COMPLETION_FILE),
            f"Evaluation training proof mismatch: {run_dir}")
    require(proof.get("metrics_sha256") == sha256(run_dir / "evaluation_metrics_extended.json"),
            f"Evaluation metrics file hash mismatch: {run_dir}")
    metrics = read_metrics(run_dir)
    require(metrics.get("evaluation/checkpoint_sha256") == best_hash,
            f"Metrics refer to a different best.pt: {run_dir}")
    validate_metrics(metrics, required_metrics)
    if metric_validator is not None:
        metric_validator(metrics)


def run_status(
    run_dir: Path, manifest: dict[str, Any], *, prerequisites_ready: bool,
    required_metrics: tuple[str, ...],
    metric_validator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, str | None]:
    if not prerequisites_ready:
        return "BLOCKED", "pretrained checkpoint or prepared dataset unavailable"
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return "NEW_RUN_REQUIRED", None
    try:
        manifest_matches(run_dir, manifest)
        proof = run_dir / TRAINING_COMPLETION_FILE
        if proof.is_file():
            verify_training_complete(run_dir)
            evaluation_proof = run_dir / EVALUATION_COMPLETION_FILE
            if not evaluation_proof.exists():
                return "EVAL_BACKFILL_REQUIRED", "Finalized training; evaluation proof not yet written"
            verify_evaluation_complete(run_dir, required_metrics, metric_validator)
            return "COMPLETE", None
    except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        return "BLOCKED", str(error)
    # best.pt may already exist midway through training. Without an explicit
    # post-train marker, never treat it as a completed model or eval backfill.
    if (run_dir / "weights/last.pt").is_file():
        return "RESUME_REQUIRED", "Training completion not verified; resume from last.pt"
    if (run_dir / "weights/best.pt").exists():
        return "BLOCKED", "best.pt exists without last.pt or a finalized training proof"
    allowed = {"experiment_manifest.json", "args.yaml", "weights", "config.yaml"}
    if {child.name for child in run_dir.iterdir()} <= allowed:
        weights = run_dir / "weights"
        if not weights.is_dir() or not any(child.is_file() for child in weights.iterdir()):
            return "NEW_RUN_REQUIRED", None
    return "BLOCKED", "non-empty run has no safe resume checkpoint"


def ensure_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    if run_dir.exists() and any(run_dir.iterdir()):
        manifest_matches(run_dir, manifest)
        return
    write_json(run_dir / "experiment_manifest.json", manifest)


def write_summaries(root: Path, rows: list[dict[str, Any]], required_metrics: tuple[str, ...]) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    run_rows = [{key: value for key, value in row.items() if key != "manifest"} for row in rows]
    fields = sorted(set().union(*(row.keys() for row in run_rows))) if run_rows else ["dataset", "family", "case", "seed", "status"]
    runs_path = root / "summary_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(run_rows)
    aggregate: list[dict[str, Any]] = []
    identities = sorted({(row.get("dataset"), row.get("family"), row.get("case")) for row in run_rows})
    for dataset, family, case in identities:
        group = [row for row in run_rows if (row.get("dataset"), row.get("family"), row.get("case")) == (dataset, family, case) and row.get("status") == "COMPLETE"]
        seeds = {int(row["seed"]) for row in group}
        record: dict[str, Any] = {"dataset": dataset, "family": family, "case": case, "n_complete": len(group), "canonical_complete": seeds == set(SEEDS)}
        fingerprints_complete = bool(group) and all(row.get(key) is not None for row in group for key in BENCHMARK_KEYS)
        record["speed_comparable"] = fingerprints_complete and len({tuple(row[key] for key in BENCHMARK_KEYS) for row in group}) == 1
        if record["canonical_complete"]:
            for key in required_metrics:
                if key.startswith("test_speed/") and not record["speed_comparable"]:
                    continue
                values = [float(row[key]) for row in group]
                record[f"{key}/mean"] = statistics.fmean(values)
                record[f"{key}/std"] = statistics.stdev(values)
        aggregate.append(record)
    aggregate_path = root / "summary_aggregate.csv"
    fields = sorted(set().union(*(row.keys() for row in aggregate))) if aggregate else ["dataset", "family", "case"]
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(aggregate)
    return runs_path, aggregate_path


def archive_run(*, namespace: str, family: str, case: str, seed: int, run_dir: Path, repo_id: str, required_artifacts: tuple[str, ...] = FINAL_ARTIFACTS, archive: Any | None = None) -> dict[str, Any]:
    """Explicit network action; callers must validate COMPLETE before entry."""
    from utils.hf_result_archive import HFResultArchive
    archive = archive or HFResultArchive(repo_id=repo_id)
    variant, run_name = f"{family}__{case}", run_dir.name
    remote = archive.remote_path(namespace, variant, run_name)
    archive._assert_repository_access()
    remote_files = archive._retry(lambda: archive._get_api().list_repo_files(repo_id=repo_id, repo_type="dataset", token=archive.token))
    require(not any(path == remote or path.startswith(remote + "/") for path in remote_files), f"Refusing to overwrite existing archive namespace: {remote}")
    return archive.publish_run(
        dataset_slug=namespace, variant=variant, seed=seed,
        run_name=run_name, run_dir=run_dir, required_artifacts=required_artifacts,
        summary_metadata={"family": family, "case": case, "seed": seed, "checkpoint": "weights/best.pt"},
    )
