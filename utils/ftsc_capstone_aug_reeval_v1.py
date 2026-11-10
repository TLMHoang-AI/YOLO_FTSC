"""FTSC capstone augmentation re-evaluation: immutable, evaluation-only workflow.

Phases:
  plan       no network/GPU and no files changed
  evaluate   explicit HF checkpoint recovery and evaluator execution
  aggregate  local, sample-standard-deviation summary
  publish    explicit opt-in to HF derived_evaluations; NEVER uploads weights

All source run paths come from the November 2026 canonical Hub layout.
LEVIR's original A0 augmentation control is retained separately from the
Main Results H2 reference to make cohort mismatches visible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from utils.ftsc_hf_layout import (
    AUGMENTATION_RUNS, DATASETS, SEEDS, REPO_ID,
    augmentation_run, augmentation_checkpoint, evaluation_root, seed_dir,
)

ROOT = Path(__file__).resolve().parents[1]
SUITE = "capstone_aug_sixcore_v1"
CONTRACT_VERSION = "FTSC_CAPSTONE_AUG_HF_CANONICAL_20261110_V2"
METRICS = {
    "TinyPerson": ("test_merged/AP50", "test_merged/mAP50-95", "test_merged/AP50-Small"),
    "LEVIR-Ship": ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small"),
    "Varroa": ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small"),
}
COMPLEXITY = ("model/parameters", "model/GFLOPs")
LABELS = ("AP50", "mAP50:95", "AP50_small")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def output_run(project: Path, dataset: str, case: str, seed: int) -> Path:
    return project / dataset / case / seed_dir(dataset, seed)


def planned_rows(dataset: str, cases: tuple[str, ...], seeds: tuple[int, ...],
                 project: Path) -> list[dict[str, Any]]:
    out = []
    for case in cases:
        for seed in seeds:
            source = augmentation_checkpoint(dataset, case, seed)
            run = output_run(project, dataset, case, seed)
            out.append(dict(dataset=dataset, case=case, seed=seed,
                            hf_source_checkpoint=source,
                            hf_source_run=augmentation_run(dataset, case, seed),
                            local_checkpoint=str(run / "weights/best.pt"),
                            output_metrics=str(run / "evaluation_metrics_extended.json"),
                            publish_prefix=f"{evaluation_root(dataset, SUITE)}/{case}/{seed_dir(dataset, seed)}"))
    return out


def verify_hf_sources(archive: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Online, read-only preflight of every selected checkpoint and merged prediction.

    Pin a single immutable Hub revision; do not accept a moved repository
    revision halfway through the verification.
    """
    api = archive._get_api()
    info = archive._retry(lambda: api.repo_info(repo_id=REPO_ID, repo_type="dataset"))
    revision = getattr(info, "sha", None)
    require(isinstance(revision, str) and len(revision) >= 30,
            "Cannot pin a canonical Hugging Face revision")
    files = set(archive._retry(lambda: api.list_repo_files(
        repo_id=REPO_ID, repo_type="dataset", revision=revision)))
    result = []
    for row in rows:
        dataset, case, seed = row["dataset"], row["case"], row["seed"]
        source = row["hf_source_checkpoint"]
        root = row["hf_source_run"]
        required = (source, root + "/checkpoint_reference.json")
        if dataset == "TinyPerson":
            required += (root + "/evaluation/test_merged_predictions.json",)
        missing = sorted(set(required) - files)
        require(not missing, f"Missing canonical source artifacts for {dataset}/{case}/{seed}: {missing}")
        ref, ref_revision = archive._source_reference(dataset, case, seed)
        require(ref_revision == revision,
                "Hugging Face revision moved during preflight; retry before evaluation")
        result.append(dict(dataset=dataset, case=case, seed=seed,
                           status="SOURCE_REFERENCE_VERIFIED",
                           source_checkpoint=source,
                           source_sha256=ref["best_sha256"],
                           hf_revision=revision,
                           merged_predictions_available=dataset == "TinyPerson"))
    return result


def yaml_for_seed(template: str | None, seed: int) -> Path:
    require(bool(template), "Native LEVIR/Varroa evaluation requires --data-yaml-template")
    path = Path(template.format(seed=seed)).expanduser().resolve()
    require(path.is_file(), f"Dataset test YAML missing: {path}")
    return path


def evaluate_one(args: argparse.Namespace, case: str, seed: int, archive: Any) -> dict[str, Any]:
    """Evaluate a canonical checkpoint without training or overwriting HF artifacts."""
    run_dir = output_run(args.project, args.dataset, case, seed)
    output_path = run_dir / "evaluation_metrics_extended.json"
    if output_path.exists():
        try:
            validated_output(output_path, args.dataset, case, seed)
        except (ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
            if not args.recover_incomplete:
                raise RuntimeError(
                    f"Unverified/incomplete local evaluation: {output_path}. "
                    "Inspect it, then explicitly pass --recover-incomplete to back it up "
                    "and re-evaluate; original checkpoint remains untouched."
                ) from exc
            backup = output_path.with_name(output_path.name + ".incomplete_backup")
            require(not backup.exists(), f"Refusing to replace existing backup: {backup}")
            output_path.replace(backup)
        else:
            ref, revision = archive._source_reference(args.dataset, case, seed)
            require(previous["evaluation/source_checkpoint_sha256"] == ref["best_sha256"],
                    f"Local metric file source SHA does not match canonical HF checkpoint: {output_path}")
            local_checkpoint = run_dir / "weights/best.pt"
            require(local_checkpoint.is_file() and sha256(local_checkpoint) == ref["best_sha256"],
                    f"Local checkpoint missing or has changed since evaluation: {local_checkpoint}")
            if args.dataset == "TinyPerson":
                require(args.tiny_merged_gt is not None, "--tiny-merged-gt required to reuse merged metrics")
                gt = args.tiny_merged_gt.expanduser().resolve()
                require(gt.is_file() and sha256(gt) == previous.get("evaluation/gt_sha256"),
                        f"Merged original-image annotations differ from cached evaluation: {gt}")
                prediction = run_dir / "evaluation/test_merged_predictions.json"
                require(prediction.is_file() and sha256(prediction) == previous.get("evaluation/predictions_sha256"),
                        f"TinyPerson merged predictions differ from cached evaluation: {prediction}")
            else:
                data_yaml = yaml_for_seed(args.data_yaml_template, seed)
                require(sha256(data_yaml) == previous.get("evaluation/data_yaml_sha256"),
                        f"Dataset YAML differs from cached evaluation: {data_yaml}")
            return {"dataset": args.dataset, "case": case, "seed": seed,
                    "status": "REUSED_VERIFIED_LOCAL", "path": str(output_path),
                    "source_revision_verified": revision}
    checkpoint = run_dir / "weights/best.pt"
    provenance = archive.restore_augmentation_source(
        dataset=args.dataset, case=case, seed=seed, destination=checkpoint,
        include_predictions=args.dataset == "TinyPerson",
    )
    require(checkpoint.is_file() and sha256(checkpoint) == provenance["sha256"],
            f"Checkpoint SHA-256 verification failed: {args.dataset}/{case}/{seed}")
    if args.dataset == "TinyPerson":
        require(args.tiny_merged_gt is not None, "--tiny-merged-gt required for TinyPerson")
        gt = args.tiny_merged_gt.expanduser().resolve()
        require(gt.is_file(), f"Missing TinyPerson original-image merged GT: {gt}")
        from train_tinyperson_scripts.train_all_tinyperson import evaluate_tiny_benchmark
        from evaluate_test.standard_detection_metrics import (
            ensure_local_ultralytics, model_complexity,
        )
        prediction = Path(provenance["merged_predictions"])
        require(prediction.is_file(), f"Merged-image predictions are missing: {prediction}")
        metrics = evaluate_tiny_benchmark(prediction, gt)
        require(metrics.get("test_merged/evaluator_contract")
                == "tinybenchmark_merged_original_iou50_95_v2",
                "TinyPerson merged IoU 0.50:0.95 evaluator contract drift")
        ensure_local_ultralytics()
        from ultralytics import YOLO
        model = YOLO(checkpoint)
        metrics.update(model_complexity(model, imgsz=640))
        evaluator = metrics["test_merged/evaluator_contract"]
        metrics["evaluation/gt_sha256"] = sha256(gt)
        metrics["evaluation/predictions_sha256"] = sha256(prediction)
    else:
        from evaluate_test.standard_detection_metrics import evaluate_run
        data_yaml = yaml_for_seed(args.data_yaml_template, seed)
        is_levir = args.dataset == "LEVIR-Ship"
        metrics = evaluate_run(
            run_dir, data_yaml, dataset="levir" if is_levir else "varroa",
            imgsz=512 if is_levir else 640,
            batch=8 if is_levir else 4,
            device=args.device, workers=args.workers,
            nms_iou=0.5 if is_levir else 0.7,
            include_size=True,
        )
        evaluator = "ftsc_native_sixcore_v1"
        metrics["evaluation/data_yaml_sha256"] = sha256(data_yaml)

    for k in (*METRICS[args.dataset], *COMPLEXITY):
        require(k in metrics and isinstance(metrics[k], (int, float))
                and math.isfinite(float(metrics[k])) and float(metrics[k]) > 0,
                f"{args.dataset}/{case}/{seed}: missing or invalid metric {k}")
    metrics.update({
        "evaluation/source_checkpoint": provenance["source"],
        "evaluation/source_checkpoint_sha256": provenance["sha256"],
        "evaluation/source_revision": provenance["revision"],
        "evaluation/source_dataset": args.dataset,
        "evaluation/source_case": case,
        "evaluation/source_seed": seed,
        "evaluation/evaluator_contract": evaluator,
        "evaluation/no_retraining": True,
    })
    out = run_dir / "evaluation_metrics_extended.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dataset": args.dataset, "case": case, "seed": seed,
            "status": "EVALUATED", "path": str(out)}


def validated_output(path: Path, dataset: str, case: str, seed: int) -> dict[str, Any]:
    require(path.is_file(), f"Missing evaluation result: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data.get("evaluation/source_checkpoint") == augmentation_checkpoint(dataset, case, seed)
            and data.get("evaluation/source_case") == case
            and data.get("evaluation/source_seed") == seed
            and data.get("evaluation/source_dataset") == dataset
            and data.get("evaluation/no_retraining") is True,
            f"Evaluation identity mismatch: {path}")
    source_sha = data.get("evaluation/source_checkpoint_sha256")
    require(isinstance(source_sha, str) and len(source_sha) == 64
            and all(c in "0123456789abcdef" for c in source_sha),
            f"Missing immutable source checkpoint digest: {path}")
    expected_contract = ("tinybenchmark_merged_original_iou50_95_v2"
                         if dataset == "TinyPerson" else "ftsc_native_sixcore_v1")
    require(data.get("evaluation/evaluator_contract") == expected_contract,
            f"Evaluation contract mismatch: {path}")
    require(isinstance(data.get("evaluation/source_revision"), str)
            and len(data["evaluation/source_revision"]) >= 30,
            f"Missing HF source revision: {path}")
    for key in (*METRICS[dataset], *COMPLEXITY):
        require(key in data and isinstance(data[key], (int, float))
                and math.isfinite(float(data[key])) and float(data[key]) > 0,
                f"Missing full six-core key {key}: {path}")
    return data


def aggregate(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    profile_contracts: list[tuple[str, int, str]] = []
    prepared_data: dict[int, set[str]] = {seed: set() for seed in args.seeds}
    for case in args.cases:
        sources = [
            validated_output(output_run(args.project, args.dataset, case, seed)
                             / "evaluation_metrics_extended.json",
                             args.dataset, case, seed)
            for seed in args.seeds
        ]
        require(set(args.seeds) == set(SEEDS), "Publishing journal means requires exactly 42/43/44")
        for seed, source in zip(args.seeds, sources):
            method = source.get("model/GFLOPs_method")
            imgsz = source.get("model/GFLOPs_imgsz")
            require(isinstance(method, str) and method
                    and isinstance(imgsz, int)
                    and imgsz == (512 if args.dataset == "LEVIR-Ship" else 640),
                    f"Invalid/unmatched FLOPs measurement protocol: {args.dataset}/{case}/{seed}")
            profile_contracts.append((method, imgsz, source["evaluation/evaluator_contract"]))
            key = "evaluation/gt_sha256" if args.dataset == "TinyPerson" else "evaluation/data_yaml_sha256"
            fingerprint = source.get(key)
            require(isinstance(fingerprint, str) and len(fingerprint) == 64,
                    f"Missing evaluation data fingerprint: {args.dataset}/{case}/{seed}")
            prepared_data[seed].add(fingerprint)
        out: dict[str, Any] = {"dataset": args.dataset, "case": case,
                               "n": len(sources), "metrics_source": "reevaluated_sixcore"}
        for label, metric in zip(LABELS, METRICS[args.dataset]):
            values = [100 * float(x[metric]) for x in sources]
            out[label] = round(statistics.fmean(values), 6)
            out[label + "_sample_std"] = round(statistics.stdev(values), 6)
        for name, metric in (("Params_M", "model/parameters"), ("GFLOPs", "model/GFLOPs")):
            values = [float(x[metric]) / (1e6 if name == "Params_M" else 1)
                      for x in sources]
            require(max(values) - min(values) < 1e-4,
                    f"Model complexity differs across seeds for {args.dataset}/{case}/{name}")
            out[name] = round(statistics.fmean(values), 6)
        rows.append(out)
    require(len(set(profile_contracts)) == 1,
            f"GFLOPs profiler/resolution or evaluator contract differs between cases: {set(profile_contracts)}")
    require(all(len(hashes) == 1 for hashes in prepared_data.values()),
            f"Different cases evaluated with different prepared data for the same seed: {prepared_data}")
    if args.dataset == "TinyPerson":
        require(len(set.union(*prepared_data.values())) == 1,
                "TinyPerson original-image merged GT must be identical for all seeds")
    result = args.project / args.dataset / "summary_aggregate.csv"
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--contract-version", help="Notebook/codebase protocol version guard")
    p.add_argument("--phase", default="plan", choices=("plan", "verify", "evaluate", "aggregate", "publish"))
    p.add_argument("--cases", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--project", type=Path, default=ROOT / "runs/ftsc_capstone_aug_reeval_v1")
    p.add_argument("--data-yaml-template", help="Path with optional {seed} placeholder; LEVIR/Varroa only")
    p.add_argument("--tiny-merged-gt", type=Path, help="TinyPerson original-image TinyBenchmark test annotations JSON")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--recover-incomplete", action="store_true", help="Back up and recompute only invalid local evaluation JSON; never modify weights")
    p.add_argument("--confirm-eval", action="store_true")
    p.add_argument("--confirm-publish", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    require(args.contract_version in (None, CONTRACT_VERSION),
            "Notebook requested a different FTSC re-evaluation code contract; sync the updated yolo_code checkout")
    args.project = args.project.expanduser().resolve()
    args.cases = tuple(args.cases or AUGMENTATION_RUNS[args.dataset])
    args.seeds = tuple(args.seeds)
    require(len(set(args.cases)) == len(args.cases), "Duplicate cases")
    require(len(set(args.seeds)) == len(args.seeds) and all(x in SEEDS for x in args.seeds),
            "Unsupported or duplicate seeds")
    require(all(x in AUGMENTATION_RUNS[args.dataset] for x in args.cases), "Unknown case")
    rows = planned_rows(args.dataset, args.cases, args.seeds, args.project)
    if args.phase == "plan":
        print(json.dumps({"status": "PLAN_ONLY", "repo_id": REPO_ID, "rows": rows}, indent=2))
        return
    if args.phase == "aggregate":
        print(json.dumps(aggregate(args), indent=2))
        return
    if args.phase == "evaluate" and not args.confirm_eval:
        raise RuntimeError("Evaluation requires explicit --confirm-eval")
    if args.phase == "publish" and not args.confirm_publish:
        print(json.dumps({"status": "PUBLISH_DRY_RUN", "rows": rows}, indent=2))
        return

    from utils.ftsc_hf_canonical_archive import CanonicalFTSCArchive
    identity = os.environ.get("HF_TOKEN")
    require(bool(identity), "HF_TOKEN must be provided by the notebook/runtime")
    archive = CanonicalFTSCArchive(
        repo_id=REPO_ID, token=identity,
        cache_dir=args.project / "_hf_cache",
    )
    if args.phase == "publish":
        # Fail closed before uploading even the first seed: a partial publication
        # must never be mistaken for a complete three-seed paper comparison.
        require(set(args.seeds) == set(SEEDS), "Publishing requires all seeds 42/43/44")
        aggregate(args)
        existing = set(archive._retry(lambda: archive._get_api().list_repo_files(
            repo_id=REPO_ID, repo_type="dataset")))
        for row in rows:
            destination = row["publish_prefix"].rstrip("/") + "/"
            require(not any(name.startswith(destination) for name in existing),
                    f"HF evaluation prefix already exists: {destination}")
            identity = validated_output(Path(row["output_metrics"]), args.dataset,
                                        row["case"], row["seed"])
            reference, _revision = archive._source_reference(args.dataset,
                                                             row["case"], row["seed"])
            require(identity["evaluation/source_checkpoint_sha256"] ==
                    reference["best_sha256"],
                    f"Cannot publish metrics for a different source checkpoint: {destination}")
    if args.phase == "verify":
        print(json.dumps({"status": "HF_VERIFIED_READ_ONLY",
                          "rows": verify_hf_sources(archive, rows)}, indent=2))
        return
    outputs = []
    for row in rows:
        case, seed = row["case"], row["seed"]
        if args.phase == "evaluate":
            outputs.append(evaluate_one(args, case, seed, archive))
            continue
        result_path = Path(row["output_metrics"])
        data = validated_output(result_path, args.dataset, case, seed)
        outputs.append(archive.publish_evaluation(
            dataset=args.dataset, case=case, seed=seed,
            metric_file=result_path,
            source_sha256=data["evaluation/source_checkpoint_sha256"],
            evaluator_contract=data["evaluation/evaluator_contract"],
            suite=SUITE, dry_run=False,
        ))
    print(json.dumps({"status": args.phase.upper(), "rows": outputs}, indent=2))


if __name__ == "__main__":
    main()
