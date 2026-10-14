#!/usr/bin/env python3
"""Non-destructive Varroa HF recovery and six-core evaluation orchestration.

phases: plan (HF metadata only), restore (verified best.pt), evaluate (GPU),
aggregate (local only). No automatic training. TRAIN_REQUIRED is advisory.
The historical run and HF source namespace are NEVER modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.hf_checkpoint_recovery import HFCheckpointRecovery, CheckpointRecoveryError
from utils.ftsc_completion_planner import CORE, decide, validate_six_core
from train_levir_scripts import train_varroa_dbss_hit as shared
from train_levir_scripts import train_varroa_ftsc_causal_suite as causal
from train_levir_scripts import train_varroa_ftsc_edge_placement_headscale as placement
from train_levir_scripts import train_varroa_ftsc_next_ablation_suite as next_suite
from train_levir_scripts import train_varroa_ftsc_alpha_sensitivity_v1 as alpha_suite

NAMESPACE = "varroa_ftsc_fullmetric_recovery_v1"
REPO = "TLMHoang/FTSC_results"
SEEDS = (42, 43, 44)
METRIC_KEYS = {
    "AP50": "test/metrics/mAP50(B)",
    "mAP50-95": "test/metrics/mAP50-95(B)",
    "AP50-Small": "test_size/AP50-Small",
    "parameters": "model/parameters",
    "GFLOPs": "model/GFLOPs",
    "inference_ms": "test_speed/inference_ms_per_image",
}
PREFIXES = (
    "varroa", "varroa_ftsc_edge_placement_headscale",
    "varroa_ftsc_next_ablation_suite", "varroa_ftsc_position_alpha_h2_v1",
    "varroa_h2_generalization_completion_v1",
)

@dataclass(frozen=True)
class Case:
    yaml: Path
    prefixes: tuple[str, ...]
    mosaic: float
    close_mosaic: int
    cp2: bool = False
    alpha: float | None = None

CASES: dict[str, Case] = {}
for _name, _spec in causal.VARIANTS.items():
    CASES[_name] = Case(_spec.config, ("varroa",), 1.0, 10)
for _name, _spec in placement.CASES.items():
    if _spec.stage == 1:  # 4 cases with historical results
        CASES[_name] = Case(_spec.config, ("varroa_ftsc_edge_placement_headscale", "varroa"), 1.0, 10)
for _name, _spec in next_suite.CASES.items():
    CASES[_name] = Case(_spec.config,
                        ("varroa_ftsc_next_ablation_suite", "varroa"),
                        0.0 if _spec.no_mosaic else 1.0,
                        0 if _spec.no_mosaic else 10, _spec.cp2_enabled)
for _name in alpha_suite.CASES:
    CASES[_name] = Case(alpha_suite.YAMLS[_name],
                        ("varroa_ftsc_position_alpha_h2_v1",),
                        1.0, 10, False, alpha_suite.ALPHA[_name])
assert len(CASES) == 19, "Varroa cases drifted; review the completion registry"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Not an object: {path}")
    return data


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"Stale temporary artifact: {tmp}")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def yaml_file(path: Path) -> dict[str, Any]:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return data


def scientific_verification(case: str, seed: int, metadata: dict[str, Path]) -> tuple[bool, str]:
    """No checkpoint download, no model load. Fail closed on ambiguous lineage."""
    try:
        if "config.yaml" not in metadata or "args.yaml" not in metadata:
            return False, "Missing archived config.yaml/args.yaml"
        spec = CASES[case]
        if yaml_file(metadata["config.yaml"]) != yaml_file(spec.yaml):
            return False, "Saved model graph/FTSC YAML differs from registered case"
        saved = yaml_file(metadata["args.yaml"])
        summary_file = metadata.get("run_summary.json")
        if summary_file is not None:
            summary = read_json(summary_file)
            if summary.get("case") not in (None, case) or summary.get("seed") not in (None, seed):
                return False, "Archive run_summary contradicts case/seed"
        required = {"seed": seed, "imgsz": 640, "batch": 4, "epochs": 100,
                    "amp": True, "mosaic": spec.mosaic, "close_mosaic": spec.close_mosaic}
        for key, value in required.items():
            if saved.get(key) != value or type(saved.get(key)) is not type(value):
                # numeric ints/floats (e.g. YAML 1 vs 1.0) are compatible, but booleans aren't
                if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and isinstance(saved.get(key), (int, float))
                        and not isinstance(saved.get(key), bool) and saved[key] == value):
                    return False, f"Training recipe differs: {key} expected {value!r}, got {saved.get(key)!r}"
        if saved.get("optimizer") not in ("auto", "AdamW"):
            return False, "Optimizer not verified as canonical auto/AdamW"
        data_path = str(saved.get("data", ""))
        if not data_path.endswith("/varroa.yaml") or f"varroa_yolo_seed{seed}/" not in data_path:
            return False, "Historical seed-specific dataset lineage cannot be established"
        if spec.cp2:
            if saved.get("copy_paste_enabled") is not True:
                return False, "CP2 treatment missing enabled flag"
        elif saved.get("copy_paste_enabled", False) is not False:
            return False, "Unexpected CP2 treatment in FTSC control"
        if spec.alpha is not None:
            cfg = yaml_file(metadata["config.yaml"])
            if cfg.get("ftsc", {}).get("position_alpha") != spec.alpha:
                return False, "Position alpha mismatch"
        # train config + args are necessary but not a proof of the original data bytes.
        # Preserve this limitation explicitly in provenance.
        return True, "Matched archived YAML, seed-specific data path and full recorded training recipe; original split-content SHA not archived"
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return False, f"Metadata cannot be verified: {exc}"


def evaluator_identity() -> dict[str, str]:
    files = {
        "standard": ROOT / "evaluate_test/standard_detection_metrics.py",
        "size": ROOT / "evaluate_test/size_bucket_evaluator.py",
        "shared": ROOT / "train_levir_scripts/train_varroa_dbss_hit.py",
    }
    return {name: digest(file) for name, file in files.items()}


def local_metrics(run: Path) -> dict[str, Any] | None:
    metric = run / "evaluation_metrics_completion_v1.json"
    complete = run / "completion_source.json"
    if not metric.is_file() or not complete.is_file():
        return None
    record = read_json(complete)
    if record.get("evaluator_sources") != evaluator_identity():
        return None
    if not (run / "weights/best.pt").is_file() or record.get("checkpoint_sha256") != digest(run / "weights/best.pt"):
        return None
    metrics = read_json(metric)
    if record.get("metrics_sha256") != digest(metric):
        return None
    return {**metrics, "checkpoint_sha256": record["checkpoint_sha256"],
            "evaluator_verified": record.get("status") == "COMPLETE"}


def handle_plan(args: argparse.Namespace, *, recover: bool = False) -> dict[str, Any]:
    agent = HFCheckpointRecovery(args.repo_id, token=args.token)
    report: dict[str, Any] = {"dataset": "varroa", "namespace": NAMESPACE,
                              "cases": list(args.cases), "seeds": list(args.seeds),
                              "phase": "restore" if recover else "plan",
                              "rows": []}
    try:
        inventory = agent.inventory(PREFIXES)
    except Exception as exc:
        # Network outage is never TRAIN_REQUIRED.
        report["inventory_status"] = "BLOCKED_HF_UNAVAILABLE"
        report["error"] = f"{type(exc).__name__}: {exc}"
        for case in args.cases:
            for seed in args.seeds:
                report["rows"].append(decide(
                    dataset="varroa", case=case, seed=seed,
                    inventory_successful=False, candidates=[],
                    existing_metrics=None, metric_keys=METRIC_KEYS).as_dict())
        write_json(args.project / "recovery_plan.json", report)
        return report
    report["hf_revision"] = inventory["revision"]
    report["inventory_status"] = "OK"
    for case in args.cases:
        spec = CASES[case]
        for seed in args.seeds:
            run = args.project / case / f"seed_{seed}"
            named = agent.candidates(inventory, case=case, seed=seed,
                                     aliases=(case,), prefixes=spec.prefixes)
            # Old uploads can rename a folder: inspect metadata from same-seed
            # candidates when exact case names are absent. Never infer absence
            # from a single naming scheme.
            fallback = not named
            found = named if named else agent.candidates(
                inventory, case=case, seed=seed, aliases=(case,),
                prefixes=spec.prefixes, match_case=False)
            verified: list[dict[str, Any]] = []
            unmatched: list[dict[str, Any]] = []
            unknown_metadata = False
            for candidate in found:
                entry: dict[str, Any] = candidate.as_dict()
                try:
                    metadata = agent.download_metadata(candidate, args.cache_dir)
                    ok, evidence = scientific_verification(case, seed, metadata)
                    entry.update(verified=ok, evidence=evidence,
                                 sha256=agent.marker_checksum(metadata) if ok else None)
                    if fallback and not ok:
                        unmatched.append(entry)
                        if "Missing archived" in evidence or "cannot be verified" in evidence:
                            unknown_metadata = True
                        continue
                except Exception as exc:
                    entry.update(verified=False, evidence=f"{type(exc).__name__}: {exc}", sha256=None)
                    if fallback:
                        unknown_metadata = True
                        unmatched.append(entry)
                        continue
                verified.append(entry)
            decision = decide(dataset="varroa", case=case, seed=seed,
                              inventory_successful=True, candidates=verified,
                              existing_metrics=local_metrics(run), metric_keys=METRIC_KEYS)
            row = {**decision.as_dict(), "candidates": verified,
                   "unmatched_candidates": unmatched, "fallback_name_search": fallback}
            if fallback and not verified and unknown_metadata:
                row["action"] = "BLOCKED_UNMAPPED"
                row["reason"] = "An unnamed same-seed checkpoint lacks enough metadata to exclude scientific compatibility"
            if recover and decision.action == "EVAL_ONLY":
                chosen = next(c for c in found if c.checkpoint_path == decision.checkpoint_path)
                try:
                    prior = run / "recovery_source.json"
                    if prior.is_file():
                        previous = read_json(prior)
                        if (previous.get("hf_checkpoint") != chosen.checkpoint_path
                                or previous.get("revision") != chosen.revision):
                            raise RuntimeError("Recovery source already pinned to another HF path/revision")
                    source = agent.restore(chosen, destination=run / "weights/best.pt",
                                           verified=True, cache_dir=args.cache_dir,
                                           expected_sha256=decision.checkpoint_sha256)
                    if prior.is_file() and read_json(prior).get("checkpoint_sha256") != source["checkpoint_sha256"]:
                        raise RuntimeError("Conflicting recovery provenance; no overwrite")
                    source.update(case=case, seed=seed, verification=next(
                        c["evidence"] for c in verified if c["checkpoint_path"] == chosen.checkpoint_path),
                        model_yaml_sha256=digest(spec.yaml))
                    write_json(prior, source)
                    row["restore_status"] = "RESTORED"
                    row["checkpoint_sha256"] = source["checkpoint_sha256"]
                except Exception as exc:
                    row["action"] = "BLOCKED_RECOVERY"
                    row["restore_status"] = "BLOCKED_RECOVERY"
                    row["restore_error"] = f"{type(exc).__name__}: {exc}"
            report["rows"].append(row)
    write_json(args.project / "recovery_plan.json", report)
    return report


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    outcomes = []
    for case in args.cases:
        for seed in args.seeds:
            run = args.project / case / f"seed_{seed}"
            row = {"case": case, "seed": seed}
            source = run / "recovery_source.json"
            if not source.is_file():
                row["status"] = "BLOCKED_NO_VERIFIED_CHECKPOINT"
                outcomes.append(row)
                continue
            record = read_json(source)
            if record.get("case") != case or record.get("seed") != seed:
                row["status"] = "BLOCKED_RECOVERY_IDENTITY"
                outcomes.append(row)
                continue
            best = run / "weights/best.pt"
            if not best.is_file() or digest(best) != record.get("checkpoint_sha256"):
                row["status"] = "BLOCKED_CHECKPOINT_SHA"
                outcomes.append(row)
                continue
            if record.get("model_yaml_sha256") != digest(CASES[case].yaml):
                row["status"] = "BLOCKED_MODEL_YAML_CHANGED"
                outcomes.append(row)
                continue
            if local_metrics(run) is not None:
                row["status"] = "SKIP_COMPLETE"
                outcomes.append(row)
                continue
            if (run / "evaluation_metrics_completion_v1.json").exists():
                row["status"] = "BLOCKED_STALE_EVALUATOR_OR_METRICS"
                outcomes.append(row)
                continue
            data = shared.dataset_for_seed(args, seed)
            args.data_yaml = data
            metrics = shared.evaluate(run, args)
            missing = validate_six_core(metrics, METRIC_KEYS)
            if metrics.get("evaluation/nms_iou") != 0.7 or not metrics.get("test_size/protocol"):
                missing = (*missing, "NMS_OR_SIZE_PROTOCOL")
            if missing:
                row.update(status="METRIC_INCOMPLETE", missing=list(missing))
                outcomes.append(row)
                continue
            output = run / "evaluation_metrics_completion_v1.json"
            write_json(output, metrics)
            write_json(run / "completion_source.json", {
                "status": "COMPLETE", "case": case, "seed": seed,
                "source": record, "checkpoint_sha256": record["checkpoint_sha256"],
                "metrics_sha256": digest(output), "evaluator_sources": evaluator_identity(),
                "standard_nms_iou": 0.7, "size_bucket_nms_iou": 0.5,
            })
            row["status"] = "COMPLETE"
            outcomes.append(row)
    report = {"rows": outcomes, "phase": "evaluate", "namespace": NAMESPACE}
    write_json(args.project / "evaluation_status.json", report)
    return report


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if tuple(sorted(args.seeds)) != SEEDS:
        raise RuntimeError("PAPER_AGGREGATE_REQUIRES_ALL_3_SEEDS: 42,43,44")
    rows = []
    for case in args.cases:
        for seed in args.seeds:
            run = args.project / case / f"seed_{seed}"
            metrics = local_metrics(run)
            if metrics is None or validate_six_core(metrics, METRIC_KEYS):
                raise RuntimeError(f"METRIC_INCOMPLETE: {case}/seed_{seed}")
            rows.append({"case": case, "seed": seed, **metrics})
    if not rows:
        raise RuntimeError("No complete cases selected")
    def csv_write(path: Path, items: list[dict[str, Any]]) -> None:
        fields = sorted({key for row in items for key in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(items)
    args.project.mkdir(parents=True, exist_ok=True)
    csv_write(args.project / "summary_runs.csv", rows)
    summary = []
    for case in args.cases:
        group = [r for r in rows if r["case"] == case]
        assert len(group) == len(args.seeds)
        fingerprint = ("benchmark/gpu_name", "benchmark/device", "benchmark/imgsz",
                       "benchmark/batch", "benchmark/precision", "benchmark/backend")
        comparable = (all(all(row.get(k) not in (None, "") for k in fingerprint) for row in group)
                      and len({tuple(str(row.get(k)) for k in fingerprint) for row in group}) == 1)
        aggregate_row: dict[str, Any] = {"case": case, "runs": len(group),
                                          "benchmark/speed_comparable": comparable}
        for field in METRIC_KEYS.values():
            if field.startswith("test_speed/") and not comparable:
                continue
            values = [float(row[field]) for row in group]
            aggregate_row[field + "/mean"] = statistics.fmean(values)
            aggregate_row[field + "/std"] = statistics.stdev(values) if len(values) > 1 else None
        summary.append(aggregate_row)
    csv_write(args.project / "summary_aggregate.csv", summary)
    return {"status": "COMPLETE", "cases": list(args.cases), "seeds": list(args.seeds),
            "run_rows": len(rows), "aggregate_rows": len(summary)}


def publish(args: argparse.Namespace) -> dict[str, Any]:
    """Explicit upload only, never overwrite an existing remote canonical path."""
    from utils.hf_result_archive import HFResultArchive
    agent = HFCheckpointRecovery(args.repo_id, token=args.token)
    archive = HFResultArchive(args.repo_id, token=args.token, cache_dir=args.cache_dir)
    results = []
    for case in args.cases:
        for seed in args.seeds:
            run = args.project / case / f"seed_{seed}"
            if local_metrics(run) is None:
                raise RuntimeError(f"Refusing to publish incomplete run: {case}/seed_{seed}")
            folder = f"{NAMESPACE}/{case}/seed_{seed}"
            # Recheck remote before EACH upload; earlier uploads change the repo revision.
            remote = agent.inventory((NAMESPACE,))
            if any(p.startswith(folder + "/") for p in remote["files"]):
                raise RuntimeError(f"Remote destination already exists, refusing overwrite: {folder}")
            result = archive.publish_run(
                dataset_slug=NAMESPACE, variant=case, seed=seed,
                run_name=f"seed_{seed}", run_dir=run,
                required_artifacts=(
                    "weights/best.pt", "recovery_source.json",
                    "evaluation_metrics_completion_v1.json", "completion_source.json"),
                summary_metadata={"historical_source": read_json(run / "recovery_source.json"),
                                  "workflow": NAMESPACE, "training": "never_retrained"},
            )
            results.append({"case": case, "seed": seed, "status": "PUBLISHED", "archive": result})
    return {"phase": "publish", "rows": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "restore", "evaluate", "aggregate", "publish"), default="plan")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--repo-id", default=REPO)
    parser.add_argument("--token", default=None, help="Prefer HF_TOKEN environment variable")
    parser.add_argument("--project", type=Path, default=ROOT / f"runs/{NAMESPACE}")
    parser.add_argument("--cache-dir", type=Path, default=Path("/marimo/hf_results_cache/varroa_fullmetric_recovery"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not args.cases or not args.seeds or len(set(args.cases)) != len(args.cases) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Cases and seeds must be non-empty and unique")
    if not set(args.seeds).issubset(SEEDS) or (args.imgsz, args.batch_size) != (640, 4):
        parser.error("Frozen Varroa seed/input/batch contract violated")
    args.token = args.token or os.environ.get("HF_TOKEN")
    args.seed_scope = "all"
    result = (
        handle_plan(args, recover=args.phase == "restore")
        if args.phase in ("plan", "restore")
        else evaluate(args) if args.phase == "evaluate"
        else aggregate(args) if args.phase == "aggregate"
        else publish(args)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
