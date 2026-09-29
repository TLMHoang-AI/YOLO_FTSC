#!/usr/bin/env python3
"""Frozen, evaluation-only backfill for the existing LEVIR Top-5 checkpoints.

The default is a read-only manifest/checkpoint preflight.  Evaluation is only
reachable with ``--confirm-eval``; this module deliberately has no training,
remote-artifact, or model-initialisation path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXPERIMENT = "LEVIR_TOP5_FULLMETRIC_V1"
NAMESPACE = "levir_top5_fullmetric_v1"
CASES = ("A3_CP2", "ES1", "ES1_R4", "ES2", "E_H2")
SEEDS = (42, 43, 44)
CONFIG_ROOT = REPO / "models_related/models_config/yolov8/levir"
COMMON_PROTOCOL = {
    "dataset": "LEVIR-Ship", "split_seed": 42, "train_seeds": list(SEEDS), "epochs": 100,
    "imgsz": 512, "batch": 8, "workers": 4, "patience": 20, "deterministic": True,
    "AMP": True, "fitness_metric": "map50_95", "evaluation_nms_iou": .5,
    "checkpoint": "weights/best.pt", "mosaic": .0, "close_mosaic": 0, "adaptive_zoom": False,
    "mixup": .0, "cutmix": .0, "stock_copy_paste": .0, "rect": False,
    "optimizer_generation": "historical AdamW via optimizer=auto", "scheduler": "linear", "lrf": .01,
}
CASE_REGISTRY: dict[str, dict[str, Any]] = {
    "A3_CP2": {
        "source_suite": "levir_ftsc_h2_augmentation_suite:A3_CP2",
        "model_yaml": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml",
        "expected_parameters": 1_821_959, "expected_gflops": 6.670778,
        "protocol": {**COMMON_PROTOCOL, "architecture_name": "H2 P2/P3", "augmentation_name": "CP2",
                     "augmentation": {"detection_copy_paste": "CP2", "copies": 2, "p": .5, "scale": 1.0,
                                      "padding": 0, "blend": "hard", "placement": "random",
                                      "allow_empty_target": True, "allow_same_source": True,
                                      "max_overlap_note": "historically present but inert for placement=random"}},
        "reference": {"val/metrics/mAP50(B)": .8308096769372955, "val/metrics/mAP50-95(B)": .3311774442185916, "test/metrics/mAP50(B)": .8248308426963091, "test/metrics/mAP75(B)": .13460563328906508, "test/metrics/mAP50-95(B)": .3166189416099076},
    },
    "ES1": {
        "source_suite": "ftsc_edge_stability:ES1",
        "model_yaml": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml",
        "expected_parameters": 1_888_923, "expected_gflops": 7.6858,
        "protocol": {**COMMON_PROTOCOL, "architecture_name": "historical ES1 graph", "augmentation_name": "none",
                     "edge": {"name": "bounded Edge", "alpha": .25, "schedule": "constant"},
                     "historical_graph_note": "historical ES1 graph confound preserved; not Clean Edge"},
        "reference": {"val/metrics/mAP50(B)": .8392594280223311, "val/metrics/mAP50-95(B)": .33195113369515716, "test/metrics/mAP50(B)": .8234946598102618, "test/metrics/mAP75(B)": .1409579467383025, "test/metrics/mAP50-95(B)": .319644989520679},
    },
    "ES1_R4": {
        "source_suite": "levir_ftsc_R_variants_augmentation_suite:A8_ES1_NEGCANVAS_R4",
        "model_yaml": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml",
        "expected_parameters": 1_888_923, "expected_gflops": 7.6858,
        "protocol": {**COMMON_PROTOCOL, "architecture_name": "historical ES1 graph", "augmentation_name": "R4 negative-canvas",
                     "edge": {"name": "bounded Edge", "alpha": .25, "schedule": "constant"},
                     "augmentation": {"enabled": True, "unit": "single", "copies": 1, "mode": "negative_canvas",
                                      "p": .30, "target_policy": "deficit", "donor_policy": "larger", "target_max_size": 20,
                                      "degradation": "weak_blur", "blur_sigma": .5, "canvases": "original-negative only",
                                      "external_negative_bank": False}},
        "reference": {"val/metrics/mAP50(B)": .8464896262246397, "val/metrics/mAP50-95(B)": .33188815316975817, "test/metrics/mAP50(B)": .8211882174356887, "test/metrics/mAP75(B)": .1226357103146291, "test/metrics/mAP50-95(B)": .3159005550509001},
    },
    "ES2": {
        "source_suite": "ftsc_edge_stability:ES2",
        "model_yaml": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es2_ramp025.yaml",
        "expected_parameters": 1_888_923, "expected_gflops": 7.6858,
        "protocol": {**COMMON_PROTOCOL, "architecture_name": "ES2 ramp Edge graph", "augmentation_name": "none",
                     "edge": {"name": "bounded Edge", "alpha": .25, "schedule": "ramp"}},
        "reference": {"val/metrics/mAP50(B)": .8370139598395211, "val/metrics/mAP50-95(B)": .33510442277031455, "test/metrics/mAP50(B)": .8164796816329073, "test/metrics/mAP75(B)": .13869942233699906, "test/metrics/mAP50-95(B)": .31695781444087495},
    },
    "E_H2": {
        "source_suite": "ftsc_h1_h2_nextgen:E_H2",
        "model_yaml": CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edgecue.yaml",
        "expected_parameters": 1_888_923, "expected_gflops": 7.6858,
        "protocol": {**COMMON_PROTOCOL, "architecture_name": "E_H2 original Edge graph", "augmentation_name": "none",
                     "edge": {"name": "original Edge", "alpha": 1.0}},
        "reference": {"val/metrics/mAP50(B)": .8401199229816924, "val/metrics/mAP50-95(B)": .33245367729421554, "test/metrics/mAP50(B)": .8156289225649541, "test/metrics/mAP75(B)": .12723072459866197, "test/metrics/mAP50-95(B)": .3133347769435626},
    },
}
REQUIRED_METRICS = (
    "val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)", "test/metrics/mAP50(B)",
    "test/metrics/mAP75(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small",
    "model/parameters", "model/parameters_M", "model/GFLOPs", "model/GFLOPs_imgsz",
    "model/GFLOPs_method", "test_speed/inference_ms_per_image", "benchmark/gpu_name",
    "benchmark/backend", "benchmark/device", "benchmark/imgsz", "benchmark/batch",
    "benchmark/precision", "benchmark/cuda_version", "benchmark/pytorch_version",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_sha256(protocol: dict[str, Any]) -> str:
    """Stable hash of frozen, JSON-native protocol metadata."""
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_fixed_data_yaml(data_yaml: Path) -> dict[str, Any]:
    """Validate the supplied existing seed-42 split without creating anything."""
    resolved = data_yaml.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"fixed LEVIR split YAML missing: {resolved}")
    if "levir_ship_yolo_seed42" not in resolved.parts:
        raise ValueError(f"data YAML must be the frozen levir_ship_yolo_seed42 split, got {resolved}")
    from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow
    workflow.validate_split(resolved)
    candidates = (resolved.parent / "split_manifest.json", resolved.parent / "manifest.json")
    manifest = next((path for path in candidates if path.is_file()), None)
    return {"data_yaml": str(resolved), "split_seed": 42,
            "dataset_manifest": str(manifest) if manifest else None,
            "dataset_manifest_sha256": sha256(manifest) if manifest else None}


def _case_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    mapping = payload.get("cases", payload)
    if not isinstance(mapping, dict) or set(mapping) != set(CASES):
        raise ValueError(f"manifest cases must be exactly {CASES}; got {sorted(mapping) if isinstance(mapping, dict) else mapping!r}")
    return mapping


def load_manifest(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read a local JSON case->seed manifest and enforce the frozen 15-run matrix."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = _case_mapping(payload)
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    checkpoints: set[Path] = set()
    for case in CASES:
        seeds = mapping[case]
        if not isinstance(seeds, dict) or {int(seed) for seed in seeds} != set(SEEDS):
            raise ValueError(f"{case}: manifest seeds must be exactly {SEEDS}")
        for raw_seed, entry in seeds.items():
            seed = int(raw_seed)
            if not isinstance(entry, dict):
                raise ValueError(f"{case}/{seed}: manifest entry must be an object")
            supplied_source = entry.get("source_suite")
            if supplied_source is not None and supplied_source != CASE_REGISTRY[case]["source_suite"]:
                raise ValueError(f"{case}/{seed}: source_suite is frozen to {CASE_REGISTRY[case]['source_suite']!r}")
            supplied_yaml = entry.get("model_yaml")
            if supplied_yaml is not None and Path(supplied_yaml).name != CASE_REGISTRY[case]["model_yaml"].name:
                raise ValueError(f"{case}/{seed}: model_yaml does not match the frozen historical graph")
            supplied_checkpoint = entry.get("checkpoint")
            supplied_run_dir = entry.get("run_dir")
            if supplied_checkpoint is None and supplied_run_dir is None:
                raise ValueError(f"{case}/{seed}: requires checkpoint or run_dir")
            run_dir = Path(supplied_run_dir).expanduser().resolve() if supplied_run_dir is not None else None
            checkpoint = Path(supplied_checkpoint).expanduser().resolve() if supplied_checkpoint is not None else None
            if checkpoint is None:
                checkpoint = run_dir / "weights" / "best.pt"
            if checkpoint.name != "best.pt" or checkpoint.parent.name != "weights":
                raise ValueError(f"{case}/{seed}: checkpoint must be exactly <run_dir>/weights/best.pt, got {checkpoint}")
            derived_run_dir = checkpoint.parent.parent
            if run_dir is not None and checkpoint != run_dir / "weights" / "best.pt":
                raise ValueError(f"{case}/{seed}: run_dir and checkpoint disagree; expected {run_dir / 'weights' / 'best.pt'}")
            run_dir = derived_run_dir
            if checkpoint in checkpoints:
                raise ValueError(f"duplicate checkpoint in manifest: {checkpoint}")
            checkpoints.add(checkpoint)
            # Keep paths normalized even when the JSON used the convenient
            # ``run_dir`` form; do not let the raw string overwrite them.
            entries[(case, seed)] = {**entry, "checkpoint": checkpoint, "run_dir": run_dir}
    if len(entries) != 15:
        raise ValueError(f"manifest must resolve exactly 15 checkpoints, got {len(entries)}")
    return entries


def preflight(args: argparse.Namespace, entries: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    from evaluate_test.standard_detection_metrics import resolve_device
    try:
        import torch
        device, _ = resolve_device(args.device, torch)
    except Exception as error:
        device = f"unresolved ({error})"
    dataset = validate_fixed_data_yaml(args.data_yaml)
    rows = []; case_protocols = []
    for case in CASES:
        spec = CASE_REGISTRY[case]; yaml_path = spec["model_yaml"]
        protocol = spec["protocol"]; protocol_hash = protocol_sha256(protocol)
        case_protocols.append({"case": case, "source_suite": spec["source_suite"], "model_yaml": str(yaml_path),
                               "model_yaml_sha256": sha256(yaml_path), "protocol": protocol,
                               "protocol_sha256": protocol_hash, "expected_parameters": spec["expected_parameters"],
                               "expected_gflops": spec["expected_gflops"]})
        for seed in SEEDS:
            item = entries[(case, seed)]; checkpoint = item["checkpoint"]
            rows.append({"case": case, "seed": seed, "source_suite": item.get("source_suite", spec["source_suite"]),
                         "checkpoint": str(checkpoint), "run_dir": str(item["run_dir"]),
                         "model_yaml": str(yaml_path), "model_yaml_sha256": sha256(yaml_path),
                         "expected_parameters": spec["expected_parameters"], "expected_gflops": spec["expected_gflops"],
                         "protocol": protocol, "protocol_sha256": protocol_hash,
                         "checkpoint_status": "READY" if checkpoint.is_file() else "MISSING"})
    return {"status": "PREFLIGHT — READ ONLY", "experiment": EXPERIMENT, "namespace": NAMESPACE,
            "cases": list(CASES), "seeds": list(SEEDS), "checkpoint_count": len(rows), **dataset,
            "imgsz": 512, "batch": 8, "workers": args.workers, "nms_iou": .5, "drift_tolerance": args.drift_tolerance,
            "benchmark_device": device, "case_protocols": case_protocols, "matrix": rows}


def load_fresh_extended_metrics(run_dir: Path) -> dict[str, Any]:
    """Load only the newly-created extended artifact; never merge history into it."""
    path = run_dir / "evaluation_metrics_extended.json"
    if not path.is_file():
        raise FileNotFoundError(f"fresh extended evaluation artifact is missing: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_METRICS if key not in values]
    if missing:
        raise RuntimeError(f"{run_dir}: fresh extended evaluation misses required metrics {missing}")
    return values


def validate_complexity(case: str, seed: int, values: dict[str, Any]) -> None:
    """Fail closed when a fresh profile no longer matches its frozen graph."""
    expected = CASE_REGISTRY[case]
    parameters = values["model/parameters"]
    measured = values["model/GFLOPs"]
    method = values["model/GFLOPs_method"]
    if parameters != expected["expected_parameters"]:
        raise RuntimeError(f"{case}/{seed}: parameters mismatch: measured={parameters}, expected={expected['expected_parameters']}; "
                           f"GFLOPs={measured}, expected_GFLOPs={expected['expected_gflops']}, method={method}")
    try:
        measured = float(measured)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{case}/{seed}: invalid GFLOPs {measured!r}, method={method}") from error
    expected_gflops = float(expected["expected_gflops"])
    tolerance = max(.05, expected_gflops * .02)
    if not math.isfinite(measured) or measured <= 0 or abs(measured - expected_gflops) > tolerance:
        raise RuntimeError(f"{case}/{seed}: GFLOPs mismatch: measured={measured}, expected={expected_gflops}, "
                           f"tolerance={tolerance}, params={parameters}, expected_params={expected['expected_parameters']}, method={method}")


def _rows_from_evaluation(entries: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        spec = CASE_REGISTRY[case]
        for seed in SEEDS:
            item = entries[(case, seed)]
            values = load_fresh_extended_metrics(item["run_dir"])
            validate_complexity(case, seed, values)
            protocol = spec["protocol"]
            rows.append({"case": case, "seed": seed, "source_suite": item.get("source_suite", spec["source_suite"]),
                         "checkpoint": str(item["checkpoint"]), "model_yaml": str(spec["model_yaml"]),
                         "model_yaml_sha256": sha256(spec["model_yaml"]), "protocol": json.dumps(protocol, sort_keys=True),
                         "protocol_sha256": protocol_sha256(protocol), "protocol/dataset": protocol["dataset"],
                         "protocol/split_seed": protocol["split_seed"], "protocol/optimizer_generation": protocol["optimizer_generation"],
                         "protocol/epochs": protocol["epochs"], "protocol/imgsz": protocol["imgsz"], "protocol/batch": protocol["batch"],
                         "protocol/fitness_metric": protocol["fitness_metric"], "protocol/nms_iou": protocol["evaluation_nms_iou"],
                         "protocol/mosaic": protocol["mosaic"], "protocol/augmentation_name": protocol["augmentation_name"],
                         "protocol/architecture_name": protocol["architecture_name"], **values})
    return rows


def write_summaries(results_dir: Path, entries: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    from evaluate_test.standard_detection_metrics import same_benchmark_fingerprint
    rows = _rows_from_evaluation(entries); results_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}, key=lambda k: (k not in {"case", "seed", "source_suite", "checkpoint", "model_yaml", "model_yaml_sha256"}, k))
    with (results_dir / f"{NAMESPACE}_summary_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for case in CASES:
        group = [row for row in rows if row["case"] == case]
        record: dict[str, Any] = {"case": case, "runs": len(group)}
        comparable = same_benchmark_fingerprint(group)
        record["benchmark/speed_aggregate"] = "comparable" if comparable else "mixed hardware / non-comparable"
        for key in set.intersection(*(set(row) for row in group)) - {"case", "seed"}:
            if key.startswith("test_speed/") and not comparable:
                continue
            try: values = [float(row[key]) for row in group]
            except (TypeError, ValueError): continue
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values)
        aggregate.append(record)
    aggregate_fields = sorted({key for row in aggregate for key in row}, key=lambda k: (k not in {"case", "runs"}, k))
    with (results_dir / f"{NAMESPACE}_summary_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields); writer.writeheader(); writer.writerows(aggregate)
    return aggregate


def drift_report(aggregate: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    report = []
    for row in aggregate:
        case = str(row["case"])
        for key, historical in CASE_REGISTRY[case]["reference"].items():
            fresh = float(row[f"{key}/mean"]); delta = fresh - historical
            report.append({"case": case, "metric": key, "fresh": fresh, "historical_reference": historical,
                           "delta": delta, "delta_ap_points": 100 * delta, "tolerance": tolerance,
                           "status": "REVIEW_REQUIRED" if abs(delta) > tolerance else "within_tolerance"})
    return report


def write_drift(results_dir: Path, report: list[dict[str, Any]]) -> Path:
    path = results_dir / f"{NAMESPACE}_drift.csv"
    fields = ("case", "metric", "fresh", "historical_reference", "delta", "delta_ap_points", "tolerance", "status")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(report)
    return path


def complexity_smoke_check() -> list[dict[str, Any]]:
    """Real, no-checkpoint profile of the two historical architecture families."""
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics, model_complexity
    ensure_local_ultralytics()
    from ultralytics.nn.tasks import DetectionModel
    representatives = ("A3_CP2", "ES1")
    report = []
    for case in representatives:
        spec = CASE_REGISTRY[case]
        model = DetectionModel(spec["model_yaml"], verbose=False).eval()
        measured = model_complexity(SimpleNamespace(model=model), imgsz=512)
        validate_complexity(case, 0, measured)
        row = {"case": case, "yaml": str(spec["model_yaml"]), "params": measured["model/parameters"],
               "GFLOPs": measured["model/GFLOPs"], "GFLOPs_method": measured["model/GFLOPs_method"],
               "expected_params": spec["expected_parameters"], "expected_GFLOPs": spec["expected_gflops"],
               "delta": float(measured["model/GFLOPs"]) - float(spec["expected_gflops"])}
        report.append(row)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="local JSON mapping all frozen case/seed checkpoints")
    parser.add_argument("--data-yaml", type=Path, help="existing fixed seed-42 LEVIR YAML; required outside architecture smoke")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drift-tolerance", type=float, default=.002)
    parser.add_argument("--complexity-smoke", action="store_true", help="profile H2 and historical ES1 before any evaluation")
    parser.add_argument("--confirm-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.drift_tolerance < 0:
        raise ValueError("--drift-tolerance must be non-negative")
    if args.complexity_smoke:
        print(json.dumps({"complexity_smoke": complexity_smoke_check()}, indent=2, sort_keys=True))
        if not args.confirm_eval:
            return
    if args.manifest is None:
        raise ValueError("--manifest is required for preflight or --confirm-eval")
    if args.data_yaml is None:
        raise ValueError("--data-yaml is required for preflight or --confirm-eval")
    entries = load_manifest(args.manifest)
    report = preflight(args, entries); print(json.dumps(report, indent=2, sort_keys=True))
    if not args.confirm_eval:
        return
    missing = [str(item["checkpoint"]) for item in entries.values() if not item["checkpoint"].is_file()]
    if missing: raise FileNotFoundError("evaluation requires all 15 existing local best.pt files; missing: " + ", ".join(missing))
    validate_fixed_data_yaml(args.data_yaml)
    if not args.complexity_smoke:
        print(json.dumps({"complexity_smoke": complexity_smoke_check()}, indent=2, sort_keys=True))
    from evaluate_test.standard_detection_metrics import evaluate_run
    for case in CASES:
        for seed in SEEDS:
            item = entries[(case, seed)]
            evaluate_run(item["run_dir"], args.data_yaml, dataset="levir", imgsz=512, batch=8,
                         device=args.device, workers=args.workers, nms_iou=.5, include_size=True)
    aggregate = write_summaries(args.results_dir, entries)
    drift = drift_report(aggregate, args.drift_tolerance)
    write_drift(args.results_dir, drift)
    print(json.dumps({"historical_drift": drift}, indent=2, sort_keys=True))
    if any(row["status"] == "REVIEW_REQUIRED" for row in drift):
        raise RuntimeError("historical metric drift exceeds tolerance; summaries were written, but backfill is not protocol-compatible")


if __name__ == "__main__":
    main()
