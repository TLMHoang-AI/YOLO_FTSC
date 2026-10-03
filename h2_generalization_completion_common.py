"""Small dataset-agnostic persistence helpers for H2 completion runners."""
from __future__ import annotations
import csv, json, math, statistics
from pathlib import Path
from typing import Any

SEEDS = (42, 43, 44)
COMPLETE_STATUSES = {"REUSED", "COMPLETE"}

def selected_seeds(args: Any) -> tuple[int, ...]:
    values=tuple(getattr(args,"seeds",SEEDS))
    if not values or len(set(values)) != len(values) or any(seed not in SEEDS for seed in values): raise ValueError(f"seeds must be a non-empty unique subset of {SEEDS}: {values}")
    return values

def selected_cases(args: Any, cases: tuple[str, ...] | dict[str, Any]) -> tuple[str, ...]:
    allowed=tuple(cases); values=tuple(getattr(args,"cases",allowed))
    if not values or len(set(values)) != len(values) or any(case not in allowed for case in values): raise ValueError(f"cases must be a non-empty unique subset of {allowed}: {values}")
    return values

def status_fields(rows: list[dict[str, Any]], requested: tuple[int, ...]) -> list[dict[str, Any]]:
    for case in dict.fromkeys(row["case"] for row in rows):
        group=[row for row in rows if row["case"]==case]; by_seed={int(row["seed"]):row for row in group}
        wanted=[by_seed[seed] for seed in requested if seed in by_seed]
        requested_status="COMPLETE" if wanted and all(row["status"] in COMPLETE_STATUSES|{"NOT_APPLICABLE"} for row in wanted) else "PENDING"
        canonical_status="COMPLETE" if all(by_seed.get(seed) and by_seed[seed]["status"] in COMPLETE_STATUSES|{"NOT_APPLICABLE"} for seed in SEEDS) else "PARTIAL"
        for row in group:
            row["requested_in_this_invocation"]=int(row["seed"]) in requested; row["seed_status"]=row["status"] if row["requested_in_this_invocation"] else "NOT_REQUESTED"; row["requested_status"]=requested_status; row["canonical_status"]=canonical_status
    return rows

def number(value: Any) -> float | None:
    try: value = float(value)
    except (TypeError, ValueError): return None
    return value if math.isfinite(value) else None

def require_metrics(metrics: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    return [key for key in required if number(metrics.get(key)) is None]

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["case", "seed", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def aggregate(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    output=[]
    for case in dict.fromkeys(row["case"] for row in rows):
        group=[row for row in rows if row["case"] == case]; completed=[row for row in group if row.get("status") in COMPLETE_STATUSES and not require_metrics(row, metrics)]
        requested_status=next((row.get("requested_status") for row in group if row.get("requested_status")),"PENDING")
        canonical_status=next((row.get("canonical_status") for row in group if row.get("canonical_status")),"PARTIAL")
        item={"case":case,"runs":len(completed),"status":requested_status,"requested_status":requested_status,"canonical_status":canonical_status}
        for metric in metrics:
            values=[number(row.get(metric)) for row in completed]; values=[v for v in values if v is not None]
            if values: item[f"{metric}/mean"]=statistics.fmean(values); item[f"{metric}/std"]=statistics.stdev(values) if len(values)>1 else None
        output.append(item)
    return output

def ensure_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Create a contract for an empty run or fail closed on a mismatch."""
    path = run_dir / "experiment_manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not path.is_file(): raise RuntimeError(f"Refusing untracked non-empty run directory: {run_dir}")
        existing=json.loads(path.read_text(encoding="utf-8"))
        # Execution context (workers/device) is deliberately resumable; every
        # causal/scientific setting must still match exactly.
        if existing.get("scientific_contract",existing) != manifest.get("scientific_contract",manifest): raise RuntimeError(f"Incompatible existing experiment manifest: {run_dir}")
        return path
    run_dir.mkdir(parents=True, exist_ok=True); write_json(path, manifest); return path

def write_completion_marker(run_dir: Path, *, case: str, seed: int, required: tuple[str, ...], metrics: dict[str, Any]) -> Path:
    """Mark only an evaluated, archive-complete native run as complete."""
    missing=require_metrics(metrics,required)
    artifacts=("weights/best.pt","weights/last.pt","results.csv","args.yaml","config.yaml","evaluation_metrics.json","experiment_manifest.json")
    absent=[item for item in artifacts if not (run_dir/item).is_file()]
    if missing or absent: raise RuntimeError(f"Cannot mark incomplete {case}/seed_{seed}: metrics={missing}, artifacts={absent}")
    path=run_dir/"causal_suite_complete.json";write_json(path,{"schema":"ftsc_h2_completion_v1","status":"complete","case":case,"seed":seed,"checkpoint":"weights/best.pt"});return path
