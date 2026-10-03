"""Evaluation-only completion of the four historical LEVIR H2 augmentations."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from evaluate_test.standard_detection_metrics import evaluate_run, load_merged_metrics, same_benchmark_fingerprint
from h2_generalization_completion_common import selected_cases, selected_seeds, status_fields

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
DEFAULT_SEEDS=SEEDS; DEFAULT_EPOCHS=100; DEFAULT_IMGSZ=512; DEFAULT_BATCH_SIZE=8; DEFAULT_WORKERS=4; DEFAULT_PATIENCE=20; DEFAULT_OPTIMIZER="AdamW"
CASES = {
    "LEVIR_H2": "A0_FTSC", "LEVIR_H2_CP2": "A3_CP2",
    "LEVIR_H2_OACP_R2": "A1_OACP_R2", "LEVIR_H2_R4": "A6_NEGCANVAS_R4",
}
SOURCE_REGISTRY={
 "LEVIR_H2":("levir_ftsc_h2_augmentation_suite","A0_FTSC","source_project"),
 "LEVIR_H2_CP2":("levir_ftsc_h2_augmentation_suite","A3_CP2","source_project"),
 "LEVIR_H2_OACP_R2":("levir_ftsc_h2_augmentation_suite","A1_OACP_R2","source_project"),
 "LEVIR_H2_R4":("levir_ftsc_R_variants_augmentation_suite","A6_NEGCANVAS_R4","r_variants_project"),
}
REQUIRED_METRICS = (
    "val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)", "test/metrics/mAP50(B)",
    "test/metrics/mAP75(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small",
    "model/parameters", "model/GFLOPs", "test_speed/inference_ms_per_image",
)
PROTOCOL = {"split_seed":42,"epochs":100,"imgsz":512,"batch":8,"workers":4,"patience":20,"deterministic":True,"amp":True,"optimizer_historical_requested":"auto","optimizer_historical_resolved":"AdamW","optimizer_current_pinned":"AdamW","lr0":.002,"momentum":.9,"beta2":.999,"weight_decay":.0005,"cos_lr":False,"lrf":.01}


def source_run_dir(args: argparse.Namespace, source_case: str, seed: int, root_field: str="source_project") -> Path:
    """Use the historical augmentation suite's established run layout."""
    return getattr(args,root_field,getattr(args,"source_project")) / source_case / f"seed_{seed}"


def reusable_fullmetric_dir(args: argparse.Namespace, source_case: str, seed: int) -> Path | None:
    # Top-5 explicitly backfilled A3_CP2 under the same frozen source case.
    if source_case != "A3_CP2":
        return None
    candidate = args.top5_project / "A3_CP2" / f"seed_{seed}"
    return candidate if (candidate / "weights/best.pt").is_file() else None


def metric_missing(values: dict[str, Any]) -> list[str]:
    missing = []
    for key in REQUIRED_METRICS:
        try:
            value = float(values[key])
            if not value >= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            missing.append(key)
    return missing

def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 0: raise ValueError("workers must be non-negative")
    selected_seeds(args); selected_cases(args,CASES)
    from train_levir_scripts.evaluate_levir_top5_fullmetric_v1 import validate_fixed_data_yaml
    validate_fixed_data_yaml(args.data_yaml)

def _same_checkpoint(first: Path, second: Path) -> bool:
    """Identity verification for local Top5 reuse; no Hub/archive operation."""
    if not (first.is_file() and second.is_file()) or first.stat().st_size != second.stat().st_size: return False
    import hashlib
    def digest(path: Path) -> str:
        h=hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""): h.update(block)
        return h.hexdigest()
    return digest(first)==digest(second)


def preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    requested=selected_seeds(args); cases=selected_cases(args,CASES)
    for case, source_case in CASES.items():
        for seed in SEEDS:
            if case not in cases or seed not in requested:
                rows.append({"case":case,"seed":seed,"status":"NOT_REQUESTED"}); continue
            source_suite, source_case, root_field=SOURCE_REGISTRY[case]
            source = source_run_dir(args, source_case, seed, root_field)
            best = source / "weights/best.pt"
            if not best.is_file():
                raise FileNotFoundError(f"{case}/seed_{seed}: historical checkpoint missing: {best}")
            reuse = reusable_fullmetric_dir(args, source_case, seed)
            if reuse and not _same_checkpoint(best,reuse/"weights/best.pt"): reuse=None
            run_dir = reuse or source
            values = load_merged_metrics(run_dir) if (run_dir / "evaluation_metrics_extended.json").is_file() else {}
            missing = metric_missing(values)
            rows.append({"case": case, "source_case": source_case, "seed": seed,
                         "source_suite":source_suite, "source_experiment":source_suite, "source_checkpoint":str(best),
                         "evaluation_checkpoint":str(run_dir/"weights/best.pt"), "evaluation_run_dir":str(run_dir),
                         "training_enabled": False, "protocol":json.dumps(PROTOCOL,sort_keys=True),
                         "status": "REUSED" if not missing else "EVAL_BACKFILL_REQUIRED",
                         "provenance": "LEVIR_TOP5_FULLMETRIC_V1:A3_CP2" if reuse else f"{source_suite}:{source_case}",
                         "missing_metrics": ";".join(missing), **values})
    return status_fields(rows,requested)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_summaries(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    _write(args.project / "summary_runs.csv", rows)
    aggregate = []
    for case in CASES:
        group = [row for row in rows if row["case"] == case]
        complete = [row for row in group if row["status"] in {"REUSED", "COMPLETE"} and not metric_missing(row)]
        requested_status=next((row.get("requested_status") for row in group if row.get("requested_status")),"PENDING")
        canonical_status=next((row.get("canonical_status") for row in group if row.get("canonical_status")),"PARTIAL")
        record: dict[str, Any] = {"case": case, "runs": len(complete), "status": requested_status, "requested_status":requested_status,"canonical_status":canonical_status}
        if complete:
            comparable = same_benchmark_fingerprint(complete)
            record["benchmark/speed_aggregate"] = "comparable" if comparable else "mixed hardware / non-comparable"
            for metric in REQUIRED_METRICS:
                values = [float(row[metric]) for row in complete]
                if metric.startswith("test_speed/") and not comparable:
                    continue
                record[f"{metric}/mean"] = statistics.fmean(values)
                record[f"{metric}/std"] = statistics.stdev(values) if len(values) > 1 else None
        aggregate.append(record)
    _write(args.project / "summary_aggregate.csv", aggregate)


def evaluate_backfill(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate only incomplete historical checkpoints; no training path exists."""
    for row in rows:
        if row["status"] != "EVAL_BACKFILL_REQUIRED":
            continue
        run_dir = Path(row["evaluation_run_dir"])
        values = evaluate_run(run_dir, args.data_yaml, dataset="levir", imgsz=512, batch=8,
                              device=args.device, workers=args.workers, nms_iou=.5)
        missing = metric_missing(values)
        if missing:
            raise RuntimeError(f"{row['case']}/seed_{row['seed']}: evaluator omitted {missing}")
        row.update(values); row["status"] = "COMPLETE"; row["missing_metrics"] = ""
    return status_fields(rows,selected_seeds(args))

def evaluate_one(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    """Public evaluation-only API for one selected historical checkpoint."""
    if row.get("status") != "EVAL_BACKFILL_REQUIRED": return row
    return evaluate_backfill(args,[row])[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, default=ROOT / "runs/levir_ftsc_h2_augmentation_suite")
    parser.add_argument("--r-variants-project",type=Path,default=ROOT/"runs/levir_ftsc_R_variants_augmentation_suite")
    parser.add_argument("--top5-project", type=Path, default=ROOT / "runs/levir_top5_fullmetric_v1")
    parser.add_argument("--project",type=Path,default=ROOT/"runs/levir_h2_generalization_completion_v1")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases",nargs="+",default=list(CASES));parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    parser.add_argument("--confirm-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args); rows = build_preflight(args)
    if not args.confirm_eval:
        for row in rows:
            print(f"{row['case']}/seed_{row['seed']}: {row['status']}")
        return
    write_summaries(args,evaluate_backfill(args, rows))

def build_preflight(args: argparse.Namespace) -> list[dict[str, Any]]:
    return preflight(args)


if __name__ == "__main__":
    main()
