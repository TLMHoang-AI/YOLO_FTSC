"""Immutable contracts and verified checkpoint transfer for new FTSC paper runs.

No HF access, training, notebook or historical-run mutation is performed here.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from utils.ftsc_paper_preflight import require, sha256

RUN_CONTRACT = "run_contract.json"
TRANSFER_REPORT = "pretrained_transfer.json"
TRAINING_COMPLETION = "training_completion.json"
TRAINING_ATTEMPT = "training_attempt.json"
REQUIRED_TRAIN_FILES = ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "config.yaml")

# Machine-specific execution settings do not alter the scientific recipe.
EXECUTION_ONLY_KEYS = {"data", "project", "name", "exist_ok", "device", "workers"}


def is_amp_related_failure(error: Exception) -> bool:
    """Classify narrowly; never turn dataset/OOM/general failures into AMP retries."""
    cause = " ".join(str(item) for item in (error, error.__cause__, error.__context__)
                     if item is not None).lower()
    return bool(re.search(
        r"\bamp\b|autocast|gradscaler|float16|half precision|"
        r"expected scalar type half|found dtype half", cause,
    ))


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_record(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def runtime_environment() -> dict[str, Any]:
    """Return stable runtime provenance without machine-specific filesystem paths."""
    import torch

    repo = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "UNAVAILABLE"
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_commit": commit,
        "executable_name": Path(sys.executable).name,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _digest_file(path: Path) -> str:
    """Stream full file bytes: dataset identity cannot be established from names/sizes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_content_identity(data_yaml: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Hash all prepared images and labels; cache per-file hashes with stat invalidation.

    The cache is only a performance optimization. The immutable identity contains
    the content digests, never mtime, inode or machine paths. First access hashes
    all bytes; later accesses rehash any changed/relocated file.
    """
    root = Path(payload.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    cache_file = data_yaml.parent / ".ftsc_paper_content_hash_cache_v1.json"
    cache: dict[str, Any] = {}
    if cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (isinstance(cached, dict) and cached.get("schema") == "ftsc_paper_content_cache_v1"
                    and isinstance(cached.get("entries"), dict)):
                cache = cached["entries"]
        except (OSError, ValueError, TypeError, AttributeError):
            cache = {}
    entries: dict[str, Any] = {}
    scientific_rows: list[tuple[str, str, int, str]] = []
    for split in ("train", "val", "test"):
        relative = Path(str(payload[split]))
        require(not relative.is_absolute() and relative.parts and relative.parts[0] == "images",
                f"Unsupported split layout ({split}): {relative}")
        image_dir = root / relative
        label_dir = root / Path("labels", *relative.parts[1:])
        for role, folder, suffixes in (
            ("image", image_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}),
            ("label", label_dir, {".txt"}),
        ):
            require(folder.is_dir(), f"Missing prepared {split}/{role} directory: {folder}")
            for file in sorted(folder.iterdir()):
                if not file.is_file() or file.suffix.lower() not in suffixes:
                    continue
                stat = file.stat()
                key = f"{split}/{role}/{file.name}"
                metadata = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                            stat.st_ino, stat.st_dev)
                old = cache.get(key)
                if (isinstance(old, dict) and
                    [old.get(k) for k in ("size", "mtime_ns", "ctime_ns", "inode", "device")] == list(metadata)
                    and isinstance(old.get("sha256"), str) and len(old["sha256"]) == 64):
                    digest = old["sha256"]
                else:
                    digest = _digest_file(file)
                entries[key] = dict(
                    size=metadata[0], mtime_ns=metadata[1], ctime_ns=metadata[2],
                    inode=metadata[3], device=metadata[4], sha256=digest)
                scientific_rows.append((key, role, stat.st_size, digest))
    require(scientific_rows, f"Prepared dataset contains no recognized image/label files: {root}")
    if entries != cache:
        atomic_json(cache_file, {"schema": "ftsc_paper_content_cache_v1", "entries": entries})
    return {
        "content_inventory_sha256": hash_record({"files": scientific_rows}),
        "content_file_count": len(scientific_rows),
        "content_verification": "SHA256_ALL_IMAGES_AND_LABELS_WITH_STAT_INVALIDATED_CACHE",
    }


def _split_identity(dataset: str, data_yaml: Path) -> dict[str, Any]:
    """Portable scientific identity for split membership AND full prepared bytes."""
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    semantic = {key: value for key, value in payload.items() if key != "path"}
    details: dict[str, Any] = {
        "data_yaml_semantic_sha256": hash_record(semantic),
        **_dataset_content_identity(data_yaml, payload),
    }
    if dataset == "TinyPerson":
        manifest = data_yaml.parent / "corner_manifest.json"
        require(manifest.is_file(), f"Missing original-image split manifest: {manifest}")
        details["split_manifest_sha256"] = sha256(manifest)
    elif dataset == "Varroa":
        details["split_contract"] = "gt_one_positives_only_class_3_to_1"
    elif dataset == "LEVIR-Ship":
        manifest = data_yaml.parent / "manifest.json"
        require(manifest.is_file(), f"Missing LEVIR split manifest: {manifest}")
        details["split_manifest_sha256"] = sha256(manifest)
    else:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    return details



def make_contract(
    *, dataset: str, case: str, seed: int, config: Path, graph_contract: str,
    expected_strides: tuple[float, ...], kwargs: dict[str, Any],
    data_yaml: Path, pretrained_name: str, pretrained_sha256: str,
) -> dict[str, Any]:
    require(data_yaml.is_file(), f"Dataset YAML missing: {data_yaml}")
    from utils.ftsc_paper_preflight import canonical_ftsc, layers, load_yaml

    model_yaml = load_yaml(config)
    recipe = {key: value for key, value in kwargs.items() if key not in EXECUTION_ONLY_KEYS}
    return {
        "schema": "ftsc_paper_new_cases_scientific_contract_v2",
        "dataset": dataset,
        "case": case,
        "seed": int(seed),
        "graph_contract": graph_contract,
        "graph_sha256": hashlib.sha256(
            json.dumps(layers(model_yaml), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "detect_strides": list(expected_strides),
        "config_sha256": sha256(config),
        "ftsc": canonical_ftsc(model_yaml),
        "pretrained_name": pretrained_name,
        "pretrained_sha256": pretrained_sha256,
        "training": recipe,
        "training_sha256": hash_record(recipe),
        "prepared_dataset": _split_identity(dataset, data_yaml),
        # Runtime belongs to training_attempt/evaluation provenance, NOT the
        # portable, content-addressed scientific identity of the checkpoint.
    }


def verify_or_create_contract(run_dir: Path, contract: dict[str, Any]) -> bool:
    """Return True if previously registered; never adopt untracked run outputs."""
    path = run_dir / RUN_CONTRACT
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        require(recorded == contract, f"Run identity/protocol mismatch: {run_dir}")
        return True
    require(not run_dir.exists() or not any(run_dir.iterdir()),
            f"Refusing to adopt nonempty run without verified contract: {run_dir}")
    atomic_json(path, contract)
    return False


def verify_checkpoint_graph(
    checkpoint: Path, *, config: Path, expected_taps: tuple[int, ...],
    expected_strides: tuple[float, ...], expected_seed: int | None = None,
    expected_training: dict[str, Any] | None = None,
) -> None:
    """Compare the actual checkpoint graph before skip/resume/evaluation."""
    require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    wrapper = YOLO(checkpoint)
    model = wrapper.model
    head = model.model[-1]
    require(isinstance(head, Detect), f"{checkpoint}: checkpoint is not a Detect model")
    require(tuple(head.f) == expected_taps, f"{checkpoint}: Detect.from mismatch")
    require(tuple(float(x) for x in head.stride.tolist()) == expected_strides,
            f"{checkpoint}: stride mismatch")
    calibrator = head.ftsc_calibrator
    require(calibrator is not None and calibrator.policy == "f5",
            f"{checkpoint}: missing FTSC calibrator")
    # Best effort graph identity check: checkpoint yaml may have extra training
    # metadata, therefore compare the parsed architecture rather than raw YAML.
    from utils.ftsc_paper_preflight import layers, load_yaml, canonical_ftsc
    ck_yaml = model.yaml
    require(layers(ck_yaml) == layers(load_yaml(config)),
            f"{checkpoint}: model graph differs from case YAML")
    require(canonical_ftsc(ck_yaml) == canonical_ftsc(load_yaml(config)),
            f"{checkpoint}: FTSC parameters differ from case YAML")
    train_args = wrapper.ckpt.get("train_args") if isinstance(wrapper.ckpt, dict) else None
    if expected_seed is not None or expected_training is not None:
        require(isinstance(train_args, dict), f"{checkpoint}: missing checkpoint train_args")
    if expected_seed is not None:
        require("seed" in train_args, f"{checkpoint}: checkpoint is missing mandatory seed")
        require(int(train_args["seed"]) == int(expected_seed),
                f"{checkpoint}: checkpoint seed differs from run contract")
    if expected_training is not None:
        # Persisted checkpoint args independently corroborate run_contract and args.yaml.
        # The full scientific recipe is checked in _validate_saved_training_args;
        # require critical optimization/data-routing keys in the checkpoint itself.
        critical = ("epochs", "imgsz", "batch", "deterministic", "optimizer",
                    "mosaic", "close_mosaic", "fitness_metric", "amp")
        for key in critical:
            if key not in expected_training:
                continue
            require(key in train_args, f"{checkpoint}: checkpoint missing training field {key}")
            require(train_args[key] == expected_training[key],
                    f"{checkpoint}: checkpoint training field mismatch: {key}")


def verify_pretrained_transfer(model: Any, pretrained: Path) -> dict[str, Any]:
    """Verify all intersecting tensors against the *actual* source checkpoint."""
    require(pretrained.is_file(), f"Pretrained checkpoint not available locally: {pretrained}")
    from ultralytics.utils.torch_utils import intersect_dicts
    import torch

    # The caller builds/initializes the target BEFORE this function.
    initial = model.model.state_dict()
    before = {key: value.detach().cpu().clone() for key, value in initial.items()}
    model.load(str(pretrained), smart_transfer=True)
    checkpoint_model = model.ckpt["model"] if isinstance(model.ckpt, dict) else model.ckpt
    source_state = checkpoint_model.float().state_dict()
    target_state = model.model.state_dict()
    transferred = intersect_dicts(source_state, target_state, smart_transfer=True)
    require(bool(transferred), f"{pretrained}: no compatible tensors transferred")
    mismatch = [key for key, value in transferred.items()
                if key not in target_state or not torch.equal(target_state[key].detach().cpu(),
                                                              value.detach().cpu())]
    require(not mismatch, f"{pretrained}: transfer verification failed: {mismatch[:8]}")
    backbone = [key for key in transferred if key.startswith("model.") and
                key.split(".")[1].isdigit() and int(key.split(".")[1]) < 10]
    require(backbone, f"{pretrained}: no matching pretrained backbone parameters")
    changed = sum(not torch.equal(before[key], target_state[key].detach().cpu())
                  for key in transferred if key in before)
    require(changed > 0, f"{pretrained}: target model retained random initialization")
    return {
        "status": "PASS",
        "verification": "EXACT_INTERSECTED_TENSOR_EQUALITY",
        "source": str(pretrained.resolve()),
        "source_sha256": sha256(pretrained),
        "source_authenticity": "LOCAL_FILE_ONLY; official-publisher digest not independently checked",
        "verified_tensor_count": len(transferred),
        "verified_backbone_tensor_count": len(backbone),
        "changed_tensor_count": changed,
        "target_tensor_count": len(target_state),
    }


def pretrained_identity(pretrained: Path, run_dir: Path) -> str:
    """Allow offline evaluation of a registered run; require bytes for new training."""
    if pretrained.is_file():
        digest = sha256(pretrained)
        contract_file = run_dir / RUN_CONTRACT
        if contract_file.is_file():
            recorded = json.loads(contract_file.read_text(encoding="utf-8"))
            require(digest == recorded.get("pretrained_sha256"),
                    f"Pretrained checkpoint changed since registration: {pretrained}")
        return digest
    contract_file = run_dir / RUN_CONTRACT
    require(contract_file.is_file(),
            f"Cannot start a new run without a verified local pretrained file: {pretrained}")
    recorded = json.loads(contract_file.read_text(encoding="utf-8"))
    require(recorded.get("pretrained_name") == pretrained.name,
            f"Pretrained identity differs from run manifest: {run_dir}")
    digest = recorded.get("pretrained_sha256")
    require(isinstance(digest, str) and len(digest) == 64,
            f"Invalid recorded pretrained SHA: {run_dir}")
    return digest


def validate_transfer_report(run_dir: Path, contract: dict[str, Any]) -> None:
    path = run_dir / TRANSFER_REPORT
    require(path.is_file(), f"Cannot reuse checkpoint without transfer report: {run_dir}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    require(recorded.get("status") == "PASS" and
            recorded.get("verification") == "EXACT_INTERSECTED_TENSOR_EQUALITY" and
            recorded.get("source_sha256") == contract["pretrained_sha256"] and
            int(recorded.get("verified_backbone_tensor_count", 0)) > 0,
            f"Unverified or mismatched pretrained transfer: {path}")


def record_training_attempt(
    run_dir: Path, contract: dict[str, Any], *, case: str, seed: int,
    effective_amp: bool, retry_reason: str | None,
) -> None:
    """Persist the exact attempt identity before training starts."""
    require(contract.get("case") == case and int(contract.get("seed")) == int(seed),
            "Training attempt differs from run contract")
    require(effective_amp or bool(retry_reason),
            "AMP-disabled retry requires an explicit failure reason")
    atomic_json(run_dir / TRAINING_ATTEMPT, {
        "case": case,
        "seed": int(seed),
        "effective_amp": bool(effective_amp),
        "amp_retry_reason": retry_reason,
        "contract_sha256": hash_record(contract),
        "environment": runtime_environment(),
    })


def validate_training_attempt(
    run_dir: Path, contract: dict[str, Any], *, case: str, seed: int,
) -> dict[str, Any]:
    """Reject cross-case/seed/protocol resumes and ambiguous AMP fallbacks."""
    path = run_dir / TRAINING_ATTEMPT
    require(path.is_file(), f"Missing training-attempt identity: {run_dir}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    require(recorded.get("case") == case and int(recorded.get("seed", -1)) == int(seed),
            f"Invalid resume case/seed identity: {path}")
    require(recorded.get("contract_sha256") == hash_record(contract),
            f"Training attempt is not bound to current contract: {path}")
    require(type(recorded.get("effective_amp")) is bool,
            f"Invalid effective AMP state: {path}")
    require(recorded["effective_amp"] or bool(recorded.get("amp_retry_reason")),
            f"AMP-disabled resume lacks retry provenance: {path}")
    require(isinstance(recorded.get("environment"), dict) and recorded["environment"],
            f"Training attempt lacks environment provenance: {path}")
    return recorded


def verify_completed_artifacts(run_dir: Path) -> bool:
    if not all((run_dir / path).is_file() and (run_dir / path).stat().st_size > 0
               for path in REQUIRED_TRAIN_FILES):
        return False
    try:
        return sum(1 for _ in (run_dir / "results.csv").open(encoding="utf-8")) > 1
    except OSError:
        return False


def _validate_saved_training_args(run_dir: Path, contract: dict[str, Any], effective_amp: bool) -> None:
    import yaml

    saved = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
    expected = contract["training"]
    for key, value in expected.items():
        if key == "amp":
            continue
        require(key in saved and saved[key] == value,
                f"Saved training argument differs from contract: {key}")
    require(bool(saved.get("amp")) is bool(effective_amp),
            f"Saved AMP state differs from completion marker: {run_dir}")


def require_resumable_last(checkpoint: Path) -> None:
    """Do not pass an optimizer-stripped or ambiguous last.pt to YOLO.resume."""
    require(checkpoint.is_file(), f"Missing resume checkpoint: {checkpoint}")
    from ultralytics.utils.patches import torch_load
    import torch
    record = torch_load(checkpoint, map_location=torch.device("cpu"))
    require(isinstance(record, dict) and isinstance(record.get("epoch"), int) and
            record["epoch"] >= 0 and isinstance(record.get("optimizer"), dict),
            f"Checkpoint is not an unfinalized, optimizer-bearing resume state: {checkpoint}")


def reconcile_finalized_training(
    run_dir: Path, *, contract: dict[str, Any], config: Path,
    expected_taps: tuple[int, ...], expected_strides: tuple[float, ...],
) -> bool:
    """Recover a run interrupted AFTER trainer.final_eval, WITHOUT retraining.

    Completed final_eval checkpoints have epoch=-1 and no optimizer/EMA/scaler;
    an ordinary last.pt with optimizer remains resumable and returns False.
    A mixed/ambiguous state blocks instead of manufacturing completion.
    """
    if (run_dir / TRAINING_COMPLETION).exists():
        return validate_completion(run_dir)
    if not verify_completed_artifacts(run_dir):
        return False
    require((run_dir / RUN_CONTRACT).is_file(), f"Untracked training artifacts: {run_dir}")
    recorded = json.loads((run_dir / RUN_CONTRACT).read_text(encoding="utf-8"))
    require(recorded == contract, f"Cannot reconcile incompatible run: {run_dir}")
    validate_transfer_report(run_dir, contract)
    attempt = validate_training_attempt(
        run_dir, contract, case=contract["case"], seed=int(contract["seed"]))
    from ultralytics.utils.patches import torch_load
    from torch import device
    last = torch_load(run_dir / "weights/last.pt", map_location=device("cpu"))
    require(isinstance(last, dict), f"Invalid last checkpoint: {run_dir}")
    if last.get("epoch") != -1:
        require(last.get("optimizer") is not None,
                f"Ambiguous unfinalized last.pt without optimizer state: {run_dir}")
        return False
    best = torch_load(run_dir / "weights/best.pt", map_location=device("cpu"))
    require(isinstance(best, dict), f"Invalid best checkpoint: {run_dir}")
    for name, ckpt in (("last", last), ("best", best)):
        require(ckpt.get("epoch") == -1 and ckpt.get("optimizer") is None and
                ckpt.get("ema") is None and ckpt.get("scaler") is None and
                ckpt.get("model") is not None,
                f"{name}.pt lacks optimizer-stripped finalization evidence: {run_dir}")
    training = dict(contract["training"], amp=attempt["effective_amp"])
    _validate_saved_training_args(run_dir, contract, attempt["effective_amp"])
    for name in ("last", "best"):
        verify_checkpoint_graph(run_dir / "weights" / f"{name}.pt", config=config,
                                expected_taps=expected_taps, expected_strides=expected_strides,
                                expected_seed=int(contract["seed"]), expected_training=training)
    finalize_training(run_dir, effective_amp=attempt["effective_amp"],
                      retry_reason=attempt.get("amp_retry_reason"))
    atomic_json(run_dir / "training_reconciliation.json", {
        "status": "RECOVERED_NO_TRAINING",
        "evidence": "BOTH_CHECKPOINTS_OPTIMIZER_STRIPPED",
        "contract_sha256": hash_record(contract),
        "best_sha256": sha256(run_dir / "weights/best.pt"),
        "last_sha256": sha256(run_dir / "weights/last.pt"),
        "environment": runtime_environment(),
    })
    return True


def validate_completion(run_dir: Path) -> bool:
    path = run_dir / TRAINING_COMPLETION
    if not path.is_file():
        return False
    recorded = json.loads(path.read_text(encoding="utf-8"))
    require(recorded.get("status") == "TRAINED", f"Invalid completion marker: {path}")
    contract_path = run_dir / RUN_CONTRACT
    require(contract_path.is_file(), f"Completion has no run contract: {run_dir}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require(recorded.get("contract_sha256") == hash_record(contract),
            f"Completion marker is not bound to current contract: {run_dir}")
    attempt = validate_training_attempt(
        run_dir, contract, case=contract["case"], seed=int(contract["seed"]),
    )
    require(recorded.get("effective_amp") is attempt["effective_amp"] and
            recorded.get("amp_retry_reason") == attempt.get("amp_retry_reason"),
            f"Completion marker differs from training attempt: {run_dir}")
    for name in ("best", "last"):
        weight = run_dir / "weights" / f"{name}.pt"
        require(weight.is_file() and sha256(weight) == recorded.get(f"{name}_sha256"),
                f"Checkpoint changed since completion: {weight}")
    require(verify_completed_artifacts(run_dir), f"Incomplete training artifacts: {run_dir}")
    _validate_saved_training_args(run_dir, contract, bool(recorded.get("effective_amp")))
    return True


def finalize_training(run_dir: Path, *, effective_amp: bool, retry_reason: str | None = None) -> None:
    require(verify_completed_artifacts(run_dir),
            f"Refusing completion without full training artifacts: {run_dir}")
    contract_path = run_dir / RUN_CONTRACT
    require(contract_path.is_file(), f"Cannot finalize without run contract: {run_dir}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    attempt = validate_training_attempt(
        run_dir, contract, case=contract["case"], seed=int(contract["seed"]),
    )
    require(attempt["effective_amp"] is bool(effective_amp) and
            attempt.get("amp_retry_reason") == retry_reason,
            f"Completion state differs from training attempt: {run_dir}")
    _validate_saved_training_args(run_dir, contract, effective_amp)
    atomic_json(run_dir / TRAINING_COMPLETION, {
        "status": "TRAINED",
        "contract_sha256": hash_record(contract),
        "best_sha256": sha256(run_dir / "weights/best.pt"),
        "last_sha256": sha256(run_dir / "weights/last.pt"),
        "effective_amp": bool(effective_amp),
        "amp_retry_reason": retry_reason,
    })
