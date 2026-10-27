#!/usr/bin/env python3
"""FTSC paper Phase 2: single-invocation, HF-first dataset-native completion.

One invocation owns the entire lifecycle. No phase switches. The 10 NEW paper
cases may train/resume; historical cases may ONLY restore and evaluate. Remote
failure, conflicting checkpoints or unverifiable provenance NEVER authorize
training. Isolated paper namespace; originals are never overwritten.

No network / training at import or in --check mode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "models_related/ultralytics"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from utils.ftsc_paper_run_integrity import (
    RUN_CONTRACT, atomic_json, hash_record, make_contract, validate_completion,
    verify_checkpoint_graph,
)
from utils.ftsc_paper_preflight import load_yaml, sha256
from utils.ftsc_completion_planner import validate_six_core
from utils.hf_result_archive import HFResultArchive

REPO_ID = "TLMHoang/FTSC_results"
SEEDS = (42, 43, 44)
OFFICIAL_YOLOV8N_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
OFFICIAL_YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
REQUIRED_NEW = (
    "weights/best.pt", "weights/last.pt", "config.yaml", "args.yaml",
    "results.csv", "run_contract.json", "pretrained_transfer.json",
    "training_attempt.json", "training_completion.json", "evaluation_metrics.json",
    "evaluation_environment.json",
)
REQUIRED_HISTORICAL = (
    "weights/best.pt", "config.yaml", "args.yaml",
    "evaluation_metrics.json", "source_provenance.json",
    "evaluation_environment.json",
)
METRICS = {
    "levir": {
        "AP50": "test/metrics/mAP50(B)", "mAP50-95": "test/metrics/mAP50-95(B)",
        "AP50-Small": "test_size/AP50-Small", "parameters": "model/parameters",
        "GFLOPs": "model/GFLOPs", "inference_ms": "test_speed/inference_ms_per_image",
    },
    "tinyperson": {
        "AP50": "test_merged/AP50", "mAP50-95": "test_merged/mAP50-95",
        "AP50-Small": "test_merged/AP50-Small", "parameters": "model/parameters",
        "GFLOPs": "model/GFLOPs", "inference_ms": "test_speed/inference_ms_per_image",
    },
    "varroa": {
        "AP50": "test/metrics/mAP50(B)", "mAP50-95": "test/metrics/mAP50-95(B)",
        "AP50-Small": "test_size/AP50-Small", "parameters": "model/parameters",
        "GFLOPs": "model/GFLOPs", "inference_ms": "test_speed/inference_ms_per_image",
    },
}
# Names are historical source aliases, not licenses to retrain.
# Additional alpha cases are resolved via the corresponding existing alpha registry.
HISTORICAL = {
    "levir": {
        "H0": ("H0_p2_p3_p4_p5_nomosaic",),
        "H1": ("H1_p2_p3_p4_nomosaic",),
        "H2": ("H2_p2_p3_nomosaic",),
        "H3": ("H3_p2_only_nomosaic",),
    },
    "tinyperson": {
        "H2_MOSAIC": ("H2_MOSAIC",),
        "ES1_MOSAIC_LEGACY": ("ES1_MOSAIC",),
    },
    "varroa": {
        "V1_FTSC": ("V1_FTSC",),
        "V2_EDGE025": ("V2_EDGE025",),
        "V3_COLOR": ("V3_COLOR",),
    },
}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid JSON object: {path}")
    return value


def ensure_official_pretrained(pretrained: Path) -> None:
    """Download once and refuse source bytes inconsistent with frozen SHA."""
    if not pretrained.is_file():
        import urllib.request
        pretrained.parent.mkdir(parents=True, exist_ok=True)
        partial = pretrained.with_name(pretrained.name + ".download")
        require(not partial.exists(), f"Stale/incomplete pretrained download: {partial}")
        try:
            urllib.request.urlretrieve(OFFICIAL_YOLOV8N_URL, partial)
            require(sha256(partial) == OFFICIAL_YOLOV8N_SHA256,
                    "Downloaded yolov8n.pt digest does not match frozen pretrained source")
            partial.replace(pretrained)
        finally:
            if partial.exists():
                partial.unlink()
    require(sha256(pretrained) == OFFICIAL_YOLOV8N_SHA256,
            "Untrusted yolov8n.pt bytes: refusing train/reuse")


def _check_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    require(bool(token), "HF_TOKEN missing. Do not run without authenticated HF access.")
    return token


def modules(dataset: str) -> tuple[Any, Any]:
    if dataset == "levir":
        from train_levir_scripts import train_levir_ftsc_cue_corrections_v2 as paper
        from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as native
    elif dataset == "tinyperson":
        from train_tinyperson_scripts import train_tinyperson_ftsc_paper_new_cases_v1 as paper
        from train_tinyperson_scripts import train_all_tinyperson as native
    else:
        from train_levir_scripts import train_varroa_ftsc_paper_new_cases_v1 as paper
        from train_levir_scripts import train_varroa_dbss_hit as native
    return paper, native


def history_models(dataset: str) -> dict[str, tuple[Path, tuple[str, ...]]]:
    cases: dict[str, tuple[Path, tuple[str, ...]]] = {}
    if dataset == "levir":
        from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as h
        for short, aliases in HISTORICAL[dataset].items():
            actual = next((name for name in aliases if name in h.VARIANTS), None)
            if actual is not None:
                cases[short] = (Path(h.VARIANTS[actual]), aliases)
        from train_levir_scripts import train_levir_ftsc_position_alpha_sensitivity_v1 as alpha
        # α6 refers to the canonical H2. Alpha 4/8 are only verified if the
        # historical source registry exposes a matching official YAML.
        for key in ("ALPHA4_rerun", "ALPHA8_rerun"):
            if hasattr(alpha, "YAMLS") and key in alpha.YAMLS:
                cases[key] = (Path(alpha.YAMLS[key]), (key,))
    elif dataset == "tinyperson":
        from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as h
        for label, aliases in HISTORICAL[dataset].items():
            case = aliases[0]
            cases[label] = (Path(h.VARIANTS[case].config), aliases)
        from train_tinyperson_scripts import train_tinyperson_ftsc_alpha_sensitivity_v1 as a
        for label in a.CASES:
            cases[label] = (Path(a.YAMLS[label]), (label,))
    else:
        from train_levir_scripts import train_varroa_ftsc_causal_suite as h
        for label, aliases in HISTORICAL[dataset].items():
            cases[label] = (Path(h.VARIANTS[aliases[0]].config), aliases)
        from train_levir_scripts import train_varroa_ftsc_alpha_sensitivity_v1 as a
        for label in getattr(a, "CASES", ()):
            cases[label] = (Path(a.YAMLS[label]), (label,))
    return cases


def _source_files(api: Any) -> tuple[str, list[str]]:
    info = api.repo_info(repo_id=REPO_ID, repo_type="dataset")
    revision = info.sha
    require(isinstance(revision, str) and len(revision) >= 30,
            "No immutable Hugging Face repository revision")
    files = sorted(set(api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", revision=revision)))
    require(isinstance(files, list), "HF list_repo_files did not succeed")
    return revision, files


def _case_paths(files: list[str], aliases: tuple[str, ...], seed: int) -> list[str]:
    import re
    paths = set()
    for remote in files:
        parts = PurePosixPath(remote).parts
        if not any(p in aliases for p in parts):
            continue
        for i, part in enumerate(parts):
            m = re.fullmatch(r"seed[_-]?0*(\d+)(?:[_-].*)?", part, flags=re.I)
            if m and int(m.group(1)) == seed and any(p in aliases for p in parts[:i]):
                paths.add("/".join(parts[:i + 1]))
    return sorted(paths)


def hf_file(remote: str, revision: str, token: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                filename=remote, revision=revision, token=token,
                                local_dir=str(cache)))


def _remote_json(remote: str, revision: str, token: str, cache: Path) -> dict[str, Any]:
    return json_load(hf_file(remote, revision, token, cache))


def _source_id_match(candidate: Path, expected: Path) -> bool:
    return load_yaml(candidate) == load_yaml(expected)


def _validate_historical_metadata(root: str, files: set[str], revision: str, token: str,
                                  cache: Path, case: str, seed: int,
                                  config: Path, dataset: str) -> dict[str, Any]:
    config_remote, args_remote = (f"{root}/{name}" for name in ("config.yaml", "args.yaml"))
    require(config_remote in files and args_remote in files,
            f"Unverifiable historical case {case}/seed_{seed}: archived YAML/args missing")
    cfg_file = hf_file(config_remote, revision, token, cache)
    args_file = hf_file(args_remote, revision, token, cache)
    require(_source_id_match(cfg_file, config),
            f"Archived historical config differs: {case}/seed_{seed}")
    import yaml
    saved = yaml.safe_load(args_file.read_text(encoding="utf-8"))
    require(isinstance(saved, dict) and int(saved.get("seed", -1)) == seed,
            f"Historical checkpoint seed unverified: {case}/{seed}")
    fixed = {
        "levir": {"imgsz": 512, "batch": 8, "epochs": 100,
                  "mosaic": 0., "close_mosaic": 0},
        "tinyperson": {"imgsz": 640, "batch": 8, "epochs": 100,
                       "mosaic": 1., "close_mosaic": 10},
        "varroa": {"imgsz": 640, "batch": 4, "epochs": 100,
                   "mosaic": 1., "close_mosaic": 10},
    }[dataset]
    for key, expected in fixed.items():
        actual = saved.get(key)
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and actual == expected,
                f"Historical training contract mismatch {case}/{seed}: {key}={actual!r}")
    require(saved.get("deterministic") is True,
            f"Historical deterministic training provenance missing: {case}/{seed}")
    data_name = str(saved.get("data", "")).lower()
    require({"levir": "levir", "tinyperson": "tinyperson",
             "varroa": "varroa"}[dataset] in data_name,
            f"Historical dataset provenance mismatch: {case}/{seed}")
    require(str(saved.get("optimizer", "")) in ("auto", "AdamW"),
            f"Historical optimizer provenance unknown: {case}/{seed}")
    return {"source_prefix": root, "model_yaml_sha256": sha256(config),
            "args_sha256": sha256(args_file), "seed": seed,
            "checkpoint_origin": "HF_HISTORICAL_REEVALUATION",
            "historical_dataset_content_sha": "UNVERIFIED_IF_NOT_ARCHIVED"}


def record_evaluation_environment(run_dir: Path, dataset: str, args: Any) -> None:
    """Preserve inference host fingerprint separately from portable train contract."""
    path = run_dir / "evaluation_environment.json"
    if path.exists():
        saved = json_load(path)
        require(saved.get("dataset") == dataset, "Existing evaluation environment differs")
        return
    from utils.ftsc_paper_run_integrity import runtime_environment
    from ultralytics.cfg import DEFAULT_CFG_DICT
    import torch
    # The TinyPerson native val() does not override half and thus follows
    # the fork's DEFAULT_CFG. Fail closed if that default changes.
    if dataset == "tinyperson":
        require(DEFAULT_CFG_DICT.get("half") is False,
                "TinyPerson precision contract changed; do not aggregate latency")
    gpu = (torch.cuda.get_device_name(torch.cuda.current_device())
           if torch.cuda.is_available() else "CPU")
    atomic_json(path, {
        "dataset": dataset, "runtime": runtime_environment(),
        "device": str(args.device), "gpu_name": gpu,
        "imgsz": int(args.imgsz), "batch": int(args.batch_size),
        "latency_unit": "ms/window" if dataset == "tinyperson" else "ms/image",
        "precision": "FP32_DEFAULT_VALIDATOR" if dataset == "tinyperson"
        else "REFER_TO_EXTENDED_NATIVE_BENCHMARK",
        "source": "native_evaluation_session",
    })


def required_artifacts(dataset: str, *, historical: bool) -> tuple[str, ...]:
    initial = REQUIRED_HISTORICAL if historical else REQUIRED_NEW
    if dataset in ("levir", "varroa"):
        return (*initial, "evaluation_metrics_extended.json")
    return initial


def complete_tiny_model_complexity(run_dir: Path) -> None:
    """Add honest model complexity using the same 640-window native profiler."""
    output = run_dir / "evaluation_metrics.json"
    require(output.is_file(), f"TinyPerson evaluator did not finish: {run_dir}")
    metrics = json_load(output)
    if all(k in metrics for k in ("model/parameters", "model/GFLOPs")):
        return
    from ultralytics import YOLO
    from evaluate_test.standard_detection_metrics import model_complexity
    model = YOLO(run_dir / "weights/best.pt")
    complexity = model_complexity(model, 640)
    metrics.update(complexity)
    atomic_json(output, metrics)


def _metric_file(run_dir: Path, dataset: str) -> dict[str, Any]:
    file = run_dir / "evaluation_metrics.json"
    require(file.is_file(), f"Native dataset evaluator did not produce {file}")
    if dataset in ("levir", "varroa"):
        from evaluate_test.standard_detection_metrics import load_merged_metrics
        require((run_dir / "evaluation_metrics_extended.json").is_file(),
                f"Extended six-core native evaluator not run: {run_dir}")
        value = load_merged_metrics(run_dir)
    else:
        value = json_load(file)
    missing = validate_six_core(value, METRICS[dataset])
    require(not missing, f"Incomplete native six-core metrics: {missing} at {file}")
    return value


def _native_evaluate(dataset: str, native: Any, args: Any, run_dir: Path,
                     data_yaml: Path, prepared_test: Path | None) -> None:
    if dataset == "tinyperson":
        require(prepared_test is not None, "TinyPerson original-image test set not materialized")
        native.evaluate(run_dir, data_yaml, prepared_test, args.data_root, args)
        complete_tiny_model_complexity(run_dir)
    elif dataset == "levir":
        native.evaluate(run_dir, data_yaml, args)
        from evaluate_test.standard_detection_metrics import evaluate_run
        evaluate_run(run_dir, data_yaml, dataset="levir", imgsz=args.imgsz,
                     batch=args.batch_size, device=args.device, workers=args.workers,
                     nms_iou=0.5, include_size=True)
    else:
        native.evaluate(run_dir, args)


def _prepare_data(dataset: str, paper: Any, native: Any, args: Any,
                  seed: int) -> tuple[Path, Path | None]:
    if dataset == "levir":
        from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow
        return workflow.prepare_fixed_split(args), None
    if dataset == "tinyperson":
        test_dir = native.prepare_test_set(args.data_root, args.dataset_root)
        seed_dir = native.prepare_seed_dataset(args.data_root, args.dataset_root, test_dir, seed)
        return seed_dir / "tinyperson.yaml", test_dir
    args.seed_scope = "all"
    data_yaml = native.dataset_for_seed(args, seed)
    args.data_yaml = data_yaml
    return data_yaml, None


def _new_expected_contract(paper: Any, dataset: str, args: Any, case: str,
                           seed: int, data_yaml: Path) -> dict[str, Any]:
    spec = paper.CASES[case]
    if dataset == "levir":
        kwargs = paper.train_kwargs(args, data_yaml, seed)
    elif dataset == "tinyperson":
        kwargs = paper.train_kwargs(args, case, seed, data_yaml)
    else:
        kwargs = paper.train_kwargs(args, data_yaml, seed)
    return make_contract(
        dataset={"levir": "LEVIR-Ship", "tinyperson": "TinyPerson", "varroa": "Varroa"}[dataset],
        case=case, seed=seed, config=spec.yaml, graph_contract=spec.graph_contract,
        expected_strides=spec.strides, kwargs=kwargs, data_yaml=data_yaml,
        pretrained_name=args.pretrained.name, pretrained_sha256=sha256(args.pretrained),
    )


def _run_path(dataset: str, args: Any, spec: Any, seed: int) -> Path:
    suffix = "_corner_sw640_sh512" if dataset == "tinyperson" else ""
    return args.project / spec.namespace / f"seed_{seed}{suffix}"


def _archive_new(run_dir: Path, archive: HFResultArchive, dataset_slug: str,
                 case: str, seed: int, existing_files: list[str], revision: str,
                 token: str, cache: Path, dataset: str) -> str:
    prefix = f"{dataset_slug}/{case}/{run_dir.name}"
    marker_remote = prefix + "/upload_complete.json"
    required = required_artifacts(dataset, historical=False)
    for relative in required:
        require((run_dir / relative).is_file(),
                f"Cannot publish without {relative}: {run_dir}")
    if marker_remote in existing_files:
        marker = _remote_json(marker_remote, revision, token, cache)
        require(marker.get("dataset") == dataset_slug and marker.get("case") == case
                and marker.get("seed") == seed,
                f"Conflicting archived case/seed at {prefix}")
        digest = marker.get("artifact_sha256", {})
        require(all(digest.get(name) == sha256(run_dir / name) for name in required),
                f"Remote marker/checkpoint conflicts with current run: {prefix}")
        return "VERIFIED_ALREADY_ON_HF"
    require(not any(p.startswith(prefix + "/") for p in existing_files),
            f"Unmarked partial HF upload exists; refusing overwrite: {prefix}")
    archive.publish_run(
        dataset_slug=dataset_slug, variant=case, seed=seed,
        run_name=run_dir.name, run_dir=run_dir,
        required_artifacts=required,
        summary_metadata={"checkpoint_sha256": sha256(run_dir / "weights/best.pt"),
                          "run_contract_sha256": sha256(run_dir / RUN_CONTRACT)},
    )
    return "PUBLISHED"


def _archive_historical(
    run_dir: Path, archive: HFResultArchive, dataset_slug: str, case: str, seed: int,
    files: list[str], revision: str, token: str, cache: Path, dataset: str,
) -> str:
    required = required_artifacts(dataset, historical=True)
    variant = "historical__" + case
    prefix = f"{dataset_slug}/{variant}/{run_dir.name}"
    marker_remote = prefix + "/upload_complete.json"
    require(all((run_dir / relative).is_file() for relative in required),
            f"Historical evaluation artifacts missing: {run_dir}")
    if marker_remote in files:
        marker = _remote_json(marker_remote, revision, token, cache)
        require(marker.get("dataset") == dataset_slug and marker.get("case") == variant
                and marker.get("seed") == seed,
                f"Historical archive identity collision: {prefix}")
        digest = marker.get("artifact_sha256", {})
        require(all(digest.get(name) == sha256(run_dir / name) for name in required),
                f"Historical archive checksum conflict: {prefix}")
        return "HISTORICAL_VERIFIED_ON_HF"
    require(not any(p.startswith(prefix + "/") for p in files),
            f"Unmarked historical HF output, refusing overwrite: {prefix}")
    archive.publish_run(
        dataset_slug=dataset_slug, variant=variant,
        seed=seed, run_name=run_dir.name, run_dir=run_dir,
        required_artifacts=required,
        summary_metadata={"source_checkpoint_sha256": sha256(run_dir / "weights/best.pt"),
                          "provenance": "read-only historical checkpoint, native reevaluation"},
    )
    return "HISTORICAL_PUBLISHED"


def _restore_new(run_dir: Path, prefix: str, revision: str, token: str,
                 cache: Path, contract: dict[str, Any], files: set[str],
                 dataset: str) -> None:
    marker = _remote_json(prefix + "/upload_complete.json", revision, token, cache)
    require(marker.get("case") == contract["case"] and
            marker.get("seed") == contract["seed"] and
            marker.get("run_relative_path") == f"{contract['case']}/{run_dir.name}",
            f"HF marker identity mismatch: {prefix}")
    hashes = marker.get("artifact_sha256", {})
    required = required_artifacts(dataset, historical=False)
    require(all(isinstance(hashes.get(name), str) and len(hashes[name]) == 64
                and prefix + "/" + name in files for name in required),
            f"HF archive lacks required scientific artifacts: {prefix}")
    remote_contract = hf_file(prefix + "/" + RUN_CONTRACT, revision, token, cache)
    require(sha256(remote_contract) == hashes[RUN_CONTRACT] and json_load(remote_contract) == contract,
            f"HF contract incompatible with current graph/protocol/prepared data: {prefix}")
    if run_dir.exists() and any(run_dir.iterdir()):
        # Never erase local interrupted training or clobber a valid completed run.
        require(all((run_dir / name).is_file() and sha256(run_dir / name) == hashes[name]
                    for name in required),
                f"Local work conflicts with HF archive, manual investigation required: {run_dir}")
        return
    # Materialize to an isolated sibling, then promote atomically only after
    # every byte and checksum has been verified. HF interruption cannot leave
    # a misleading half-restored training directory.
    staging = run_dir.with_name(run_dir.name + ".hf_restore_staging")
    require(not staging.exists(), f"Stale HF restore staging folder: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for relative in required:
            source = hf_file(prefix + "/" + relative, revision, token, cache)
            require(sha256(source) == hashes[relative],
                    f"HF checksum mismatch: {prefix}/{relative}")
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            require(sha256(destination) == hashes[relative], "Local copy SHA mismatch")
        require(not run_dir.exists() or not any(run_dir.iterdir()),
                f"Local run appeared during HF restore: {run_dir}")
        if run_dir.exists():
            run_dir.rmdir()
        staging.replace(run_dir)
    except Exception:
        # Only our own temporary staging tree is cleaned. Never delete the
        # destination or any pre-existing training artifact.
        shutil.rmtree(staging)
        raise


def _legacy_run(dataset: str, native: Any, args: Any, case: str, seed: int,
                config: Path, remote: str, files: set[str], revision: str, token: str,
                cache: Path, data_yaml: Path, prepared_test: Path | None) -> dict[str, Any]:
    source_info = _validate_historical_metadata(
        remote, files, revision, token, cache, case, seed, config, dataset)
    best_remote = remote + "/weights/best.pt"
    require(best_remote in files, f"Historical best.pt absent: {remote}")
    run_dir = args.project / "_historical_evaluation" / case / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "weights/best.pt"
    downloaded = hf_file(best_remote, revision, token, cache)
    if best.is_file():
        require(sha256(best) == sha256(downloaded),
                f"Existing historical checkpoint differs: {best}")
    else:
        best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, best)
    for name in ("config.yaml", "args.yaml"):
        source = hf_file(remote + "/" + name, revision, token, cache)
        dest = run_dir / name
        if dest.is_file():
            require(sha256(dest) == sha256(source), f"Historical source conflict: {dest}")
        else:
            shutil.copy2(source, dest)
    from ultralytics import YOLO
    from utils.ftsc_paper_preflight import layers, canonical_ftsc
    loaded = YOLO(best)
    head = loaded.model.model[-1]
    expected_config = load_yaml(config)
    require(tuple(head.f) == tuple(layers(expected_config)[-1][0]),
            f"Historical checkpoint Detect tap mismatch: {case}/{seed}")
    require(layers(loaded.model.yaml) == layers(expected_config)
            and canonical_ftsc(loaded.model.yaml) == canonical_ftsc(expected_config),
            f"Historical checkpoint architecture/FTSC differs from source: {case}/{seed}")
    embedded = loaded.ckpt.get("train_args", {}) if isinstance(loaded.ckpt, dict) else {}
    require(isinstance(embedded, dict) and "seed" in embedded
            and int(embedded["seed"]) == seed,
            f"Historical best.pt embedded seed mismatch: {case}/{seed}")
    source_info.update(
        source_checkpoint_sha256=sha256(best), source_revision=revision,
        source_remote_best=best_remote, evaluation_graph="HISTORICAL_UNMODIFIED")
    marker = run_dir / "source_provenance.json"
    if marker.exists():
        recorded = json_load(marker)
        require(recorded.get("source_checkpoint_sha256") == source_info["source_checkpoint_sha256"],
                "Historical provenance changed")
    else:
        atomic_json(marker, source_info)
    # Native eval may overwrite ONLY the isolated paper recovery directory.
    metrics = run_dir / "evaluation_metrics.json"
    try:
        if dataset == "tinyperson" and metrics.is_file():
            complete_tiny_model_complexity(run_dir)
        _metric_file(run_dir, dataset)
    except RuntimeError:
        # Native evaluator owns the missing-metric backfill; no training.
        _native_evaluate(dataset, native, args, run_dir, data_yaml, prepared_test)
    _metric_file(run_dir, dataset)
    record_evaluation_environment(run_dir, dataset, args)
    return {"status": "HISTORICAL_EVALUATED", "checkpoint_sha256": sha256(best),
            "run_dir": str(run_dir), "source": remote}


def _aggregate(dataset: str, project: Path, table: dict[str, Any],
               cohort: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = METRICS[dataset]
    csv_rows: list[dict[str, Any]] = []
    for case, rows in sorted(table.items()):
        if set(rows) != {str(s) for s in SEEDS} or any(
                rows[str(seed)].get("status") not in
                {"PUBLISHED", "VERIFIED_ALREADY_ON_HF", "HISTORICAL_PUBLISHED", "HISTORICAL_VERIFIED_ON_HF"}
                for seed in SEEDS):
            result[case] = {"status": "INCOMPLETE_SEEDS_NOT_FOR_PAPER"}
            continue
        folders = [Path(rows[str(seed)]["run_dir"]) for seed in SEEDS]
        metrics = [_metric_file(folder, dataset) for folder in folders]
        environments = [json_load(folder / "evaluation_environment.json") for folder in folders]
        # Latency requires hardware/backend/precision/methodology parity.
        if dataset in ("levir", "varroa"):
            from evaluate_test.standard_detection_metrics import same_benchmark_fingerprint
            comparable_speed = same_benchmark_fingerprint(metrics)
            required_benchmark = ("benchmark/gpu_name", "benchmark/backend",
                                  "benchmark/device", "benchmark/imgsz",
                                  "benchmark/batch", "benchmark/precision")
            comparable_speed = comparable_speed and all(
                all(key in row and row[key] is not None for key in required_benchmark)
                for row in metrics)
        else:
            fingerprint_keys = ("device", "gpu_name", "imgsz", "batch",
                                "latency_unit", "precision")
            signatures = {
                tuple(env.get(key) for key in fingerprint_keys)
                + (env.get("runtime", {}).get("torch"),
                   env.get("runtime", {}).get("cuda_runtime"))
                for env in environments
            }
            comparable_speed = len(signatures) == 1 and all(
                env.get("precision") == "FP32_DEFAULT_VALIDATOR"
                for env in environments
            )
        recorded = {"status": "COMPLETE_3_SEEDS", "n": 3,
                    "latency_comparable": bool(comparable_speed)}
        csv_row: dict[str, Any] = {"case": case, "n": 3, "status": "COMPLETE_3_SEEDS"}
        for name, key in keys.items():
            if name == "inference_ms" and not comparable_speed:
                recorded[name] = {"status": "BENCHMARK_INCOMPARABLE_OR_UNVERIFIED"}
                csv_row["inference_ms/status"] = "BENCHMARK_INCOMPARABLE_OR_UNVERIFIED"
                continue
            values = [float(m[key]) for m in metrics]
            average, sd = statistics.mean(values), statistics.stdev(values)
            recorded[name] = {"mean": average, "sample_sd": sd}
            csv_row[f"{name}/mean"] = average
            csv_row[f"{name}/sample_sd"] = sd
        result[case] = recorded
        csv_rows.append(csv_row)
    atomic_json(project / f"phase2_paper_aggregate_{cohort}.json", result)
    if csv_rows:
        headers = sorted(set().union(*(row.keys() for row in csv_rows)))
        with (project / f"phase2_paper_aggregate_{cohort}.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
    return result


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("levir", "tinyperson", "varroa"))
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pretrained", type=Path, default=ROOT / "yolov8n.pt")
    p.add_argument("--check", action="store_true", help="No HF/training: source and registry check")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    opt = cli(argv)
    paper, native = modules(opt.dataset)
    historical = history_models(opt.dataset)
    require(tuple(paper.DEFAULT_SEEDS) == SEEDS, "Seed registry changed")
    require({"levir": 2, "tinyperson": 4, "varroa": 4}[opt.dataset] == len(paper.CASES),
            "Case matrix is not the frozen 30-run policy")
    paper.source_preflight()
    if opt.check:
        print(json.dumps({"mode": "CHECK_ONLY", "dataset": opt.dataset,
                          "new_cases": list(paper.CASES),
                          "historical_cases": list(historical)}, indent=2))
        return
    token = _check_token()
    require(opt.data_root.is_dir() and opt.dataset_root.is_dir(), "Source/prepared data missing")
    ensure_official_pretrained(opt.pretrained)
    # Full forward/loss/backward smoke runs once for EVERY new architecture,
    # before any seed training. No optimizer steps or checkpoint mutation.
    require(callable(getattr(paper, "model_preflight", None)),
            "Paper registry missing actual runtime graph/gradient preflight")
    live_preflight = paper.model_preflight(str(opt.pretrained), True)
    require(isinstance(live_preflight, dict) and
            set(paper.CASES).issubset(set(live_preflight)),
            "Runtime graph/gradient preflight did not cover all new cases")
    project = ROOT / "runs" / f"{opt.dataset}_ftsc_journal_phase2_v1"
    project.mkdir(parents=True, exist_ok=True)
    args = paper.parse_args([])
    args.data_root = opt.data_root.resolve()
    args.dataset_root = opt.dataset_root.resolve()
    args.project = project
    args.device = opt.device
    args.workers = opt.workers
    args.pretrained = opt.pretrained.resolve()
    paper.validate_args(args)
    import huggingface_hub
    api = huggingface_hub.HfApi(token=token)
    revision, files = _source_files(api)
    file_set = set(files)
    cache = Path("/marimo/hf_results_cache") / f"{opt.dataset}_ftsc_journal_phase2_v1"
    cache.mkdir(parents=True, exist_ok=True)
    dataset_slug = f"{opt.dataset}_ftsc_journal_phase2_v1"
    archive = HFResultArchive(repo_id=REPO_ID, token=token, cache_dir=cache)
    progress: dict[str, Any] = {"schema": "ftsc_journal_phase2_v1",
                                "dataset": opt.dataset, "HF_revision": revision,
                                "new_cases": {}, "historical_cases": {}}
    atomic_json(project / "phase2_progress.json", progress)
    # Freeze all dataset preparations before any possible training.
    prepared = {seed: _prepare_data(opt.dataset, paper, native, args, seed) for seed in SEEDS}
    require(all(v[0].is_file() for v in prepared.values()), "Prepared dataset YAML absent")
    from ultralytics import YOLO  # ensure the vendored model loader exists
    del YOLO

    for case, spec in paper.CASES.items():
        progress["new_cases"][case] = {}
        for seed in SEEDS:
            key = str(seed)
            run_dir = _run_path(opt.dataset, args, spec, seed)
            try:
                args.case, args.seed = case, seed
                data_yaml, _ = prepared[seed]
                args.data_yaml = data_yaml
                contract = _new_expected_contract(paper, opt.dataset, args, case, seed, data_yaml)
                found = _case_paths(files, (case,), seed)
                expected_remote = f"{dataset_slug}/{case}/{run_dir.name}"
                # Exact case/seed anywhere on HF is a candidate; an unmarked or
                # alternate namespace cannot be assumed missing.
                if found:
                    require(found == [expected_remote],
                            f"HF conflicting/new-case namespace(s): {found}")
                    require(f"{expected_remote}/upload_complete.json" in file_set,
                            f"HF has unmarked case/seed artifacts: {expected_remote}")
                    _restore_new(run_dir, expected_remote, revision, token, cache, contract, file_set, opt.dataset)
                args.confirm_run = True
                # A verified complete archive is immutable. Re-evaluation may
                # generate different benchmark timestamps/speed values and would
                # invalidate the already uploaded marker on a second Run All.
                already_done = False
                if validate_completion(run_dir):
                    try:
                        _metric_file(run_dir, opt.dataset)
                        already_done = True
                    except RuntimeError:
                        pass
                if not already_done:
                    result_dir = paper.run_one(args)  # eval/resume/train only as needed
                    require(result_dir == run_dir and validate_completion(run_dir),
                            "New-case native runner did not finish safely")
                    if opt.dataset == "levir":
                        # LEVIR head-pruning trainer does not emit the six-core
                        # extended metrics. Re-evaluate the SAME best.pt only.
                        data_yaml, _ = prepared[seed]
                        _native_evaluate(opt.dataset, native, args, run_dir, data_yaml, None)
                if opt.dataset == "tinyperson":
                    complete_tiny_model_complexity(run_dir)
                _metric_file(run_dir, opt.dataset)
                record_evaluation_environment(run_dir, opt.dataset, args)
                # Check remote status against a FRESH pinned inventory; any
                # unexpected remote publication blocks rather than overwrites.
                latest_revision, latest_files = _source_files(api)
                if latest_revision != revision:
                    # Revisions naturally change after our own upload. Identity
                    # still rechecks remote file hashes before any publish.
                    revision, files, file_set = latest_revision, latest_files, set(latest_files)
                status = _archive_new(run_dir, archive, dataset_slug, case, seed,
                                      files, revision, token, cache, opt.dataset)
                progress["new_cases"][case][key] = {
                    "status": status, "run_dir": str(run_dir),
                    "checkpoint_sha256": sha256(run_dir / "weights/best.pt")}
                print(f"[{opt.dataset}] NEW {case}/seed_{seed}: {status}", flush=True)
            except Exception as exc:
                progress["new_cases"][case][key] = {
                    "status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
                atomic_json(project / "phase2_progress.json", progress)
                # Do not continue unattended training after any HF/provenance
                # failure; a new inventory and explicit audit is necessary.
                raise
            atomic_json(project / "phase2_progress.json", progress)

    # Historical runs are evaluation-only and NEVER train; missing or ambiguous
    # historical source produces an explicit BLOCKED record, not a new run.
    revision, files = _source_files(api)
    file_set = set(files)
    for case, (config, aliases) in historical.items():
        progress["historical_cases"][case] = {}
        for seed in SEEDS:
            try:
                data_yaml, prepared_test = prepared[seed]
                args.seed = seed
                args.data_yaml = data_yaml
                candidates = _case_paths(files, aliases, seed)
                require(len(candidates) == 1, f"Historical candidate count {len(candidates)}: {candidates}")
                row = _legacy_run(opt.dataset, native, args, case, seed, config,
                                  candidates[0], file_set, revision, token, cache,
                                  data_yaml, prepared_test)
                revision, files = _source_files(api)
                file_set = set(files)
                row["status"] = _archive_historical(
                    Path(row["run_dir"]), archive, dataset_slug, case, seed,
                    files, revision, token, cache, opt.dataset)
                progress["historical_cases"][case][str(seed)] = row
            except Exception as exc:
                progress["historical_cases"][case][str(seed)] = {
                    "status": "BLOCKED_UNVERIFIED",
                    "reason": f"{type(exc).__name__}: {exc}"}
            atomic_json(project / "phase2_progress.json", progress)
    progress["new_aggregates"] = _aggregate(opt.dataset, project, progress["new_cases"], "new")
    progress["historical_aggregates"] = _aggregate(opt.dataset, project, progress["historical_cases"], "historical")
    atomic_json(project / "phase2_progress.json", progress)
    print(json.dumps({"dataset": opt.dataset, "status": "DONE_WITH_RECORDED_BLOCKERS",
                      "progress_file": str(project / "phase2_progress.json"),
                      "new_cases": list(paper.CASES),
                      "historical_cases": list(historical)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
