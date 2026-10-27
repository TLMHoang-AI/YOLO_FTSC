#!/usr/bin/env python3
"""Register and preflight new structural RepC2f Varroa FTSC paper cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for entry in (ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from train_levir_scripts import train_varroa_dbss_hit as shared
from train_levir_scripts import train_varroa_ftsc_causal_suite as historical
from utils.ftsc_paper_preflight import layers, load_yaml, require, runtime_case_report, static_case_report, synthetic_loss_backward, sha256
from utils.ftsc_paper_run_integrity import (
    TRANSFER_REPORT, atomic_json, finalize_training, is_amp_related_failure, make_contract, pretrained_identity,
    record_training_attempt, validate_completion, validate_training_attempt, validate_transfer_report,
    verify_checkpoint_graph, verify_completed_artifacts, verify_or_create_contract, verify_pretrained_transfer,
    reconcile_finalized_training, require_resumable_last,
)


CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/varroa"
CANONICAL_V1 = CONFIG_ROOT / "yolov8n_varroa_ftsc_v1_p2p3.yaml"
HISTORICAL_P234 = CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p234.yaml"
HISTORICAL_V3_COLOR = CONFIG_ROOT / "yolov8n_varroa_ftsc_v3_p2p3_color.yaml"
DEFAULT_SEEDS = (42, 43, 44)
EXPERIMENT = "varroa_ftsc_paper_new_cases_v1"


@dataclass(frozen=True)
class CaseSpec:
    yaml: Path
    detect_from: tuple[int, ...]
    strides: tuple[float, ...]
    graph_contract: str
    namespace: str
    historical_candidates: tuple[str, ...] = ()


CASES: dict[str, CaseSpec] = {
    "VR_STRUCT_H0_P2345": CaseSpec(CONFIG_ROOT / "yolov8n_varroa_ftsc_struct_h0_p2345.yaml",
        (18, 21, 24, 27), (4., 8., 16., 32.), "repc2f_structural_p2345_v1", "VR_STRUCT_H0_P2345"),
    "VR_STRUCT_H1_P234": CaseSpec(CONFIG_ROOT / "yolov8n_varroa_ftsc_struct_h1_p234.yaml",
        (18, 21, 24), (4., 8., 16.), "repc2f_structural_p234_v1", "VR_STRUCT_H1_P234"),
    "VR_STRUCT_H2_P23": CaseSpec(CONFIG_ROOT / "yolov8n_varroa_ftsc_struct_h2_p23.yaml",
        (18, 21), (4., 8.), "repc2f_structural_p23_v1", "VR_STRUCT_H2_P23"),
    "VR_STRUCT_H3_P2": CaseSpec(CONFIG_ROOT / "yolov8n_varroa_ftsc_struct_h3_p2.yaml",
        (18,), (4.,), "repc2f_structural_p2_v1", "VR_STRUCT_H3_P2"),
}

HISTORICAL_REFERENCES = {
    "V1_FTSC": {"yaml": str(CANONICAL_V1), "training_enabled": False, "graph": "full neck, Detect P2/P3"},
    "HEAD_P234": {"yaml": str(HISTORICAL_P234), "training_enabled": False, "graph": "full neck, Detect P2/P3/P4"},
    "FTSC_P34": {"yaml": str(CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p34.yaml"),
                  "training_enabled": False, "graph": "full neck, Detect P3/P4"},
    "V2_EDGE025": {"yaml": str(historical.VARIANTS["V2_EDGE025"].config),
                    "training_enabled": False, "action": "REUSE_HISTORICAL_PROPAGATED_EDGE"},
    "V3_COLOR": {"yaml": str(HISTORICAL_V3_COLOR), "training_enabled": False,
                  "action": "REUSE_AFTER_PHASE2_CHECKPOINT_PROVENANCE"},
    "EDGE_P2_DETECT_ONLY": {"yaml": str(CONFIG_ROOT / "yolov8n_varroa_ftsc_edge_p2_detect_only.yaml"),
                            "training_enabled": False, "action": "REUSE_HISTORICAL_PREDICTION_LOCAL_EDGE"},
    "EDGE_P3_DETECT_ONLY": {"yaml": str(CONFIG_ROOT / "yolov8n_varroa_ftsc_edge_p3_detect_only.yaml"),
                            "training_enabled": False, "action": "REUSE_HISTORICAL_PREDICTION_LOCAL_EDGE"},
}

# Strict paper cohorts: do not relabel old full-neck checkpoints as structural.
# This is how we preserve all existing Varroa Edge/Color training artifacts.
PAPER_COHORTS = {
    "STRUCTURAL_HEAD_PRUNING": tuple(CASES),
    "HISTORICAL_FULL_NECK_CUES": (
        "V1_FTSC", "V2_EDGE025", "V3_COLOR",
        "EDGE_P2_DETECT_ONLY", "EDGE_P3_DETECT_ONLY",
    ),
    "HISTORICAL_FULL_NECK_HEAD_SCALE": (
        "V1_FTSC", "HEAD_P234", "FTSC_P34",
    ),
}
PAPER_COMPARISON_POLICY = {
    "head_pruning_table": {
        "cohort": "STRUCTURAL_HEAD_PRUNING",
        "baseline": "VR_STRUCT_H2_P23",
        "training_action": "NEW_INDEPENDENT_TRAINING_ALL_FOUR_CASES",
        "note": "Only independently trained structurally pruned RepC2f H0/H1/H2/H3 models.",
    },
    "edge_color_table": {
        "cohort": "HISTORICAL_FULL_NECK_CUES",
        "baseline": "V1_FTSC",
        "note": "Retain all original Varroa full-neck checkpoints; no retraining.",
    },
    "historical_head_scale_appendix": {
        "cohort": "HISTORICAL_FULL_NECK_HEAD_SCALE",
        "baseline": "V1_FTSC",
        "note": "Legacy full-neck head-scale comparisons are not structural pruning.",
    },
}


def assert_paper_comparison(case_ids: list[str] | tuple[str, ...], table: str) -> None:
    """Fail closed if a paper exporter mixes old and new Varroa H2 definitions."""
    require(table in PAPER_COMPARISON_POLICY, f"Unknown Varroa paper table: {table}")
    permitted = set(PAPER_COHORTS[PAPER_COMPARISON_POLICY[table]["cohort"]])
    require(bool(case_ids), f"Empty Varroa paper table: {table}")
    require(len(case_ids) == len(set(case_ids)), "Duplicate comparison rows")
    invalid = sorted(set(case_ids) - permitted)
    require(not invalid, f"Non-comparable Varroa rows in {table}: {invalid}")


def protocol_metadata() -> dict[str, Any]:
    return {
        "parent": "train_varroa_ftsc_causal_suite.V1_FTSC",
        "epochs": 100, "imgsz": 640, "batch": 4, "patience": 20,
        "optimizer": "auto", "mosaic": 1.0, "close_mosaic": 10,
        "pretrained": "yolov8n.pt smart_transfer", "deterministic": True,
        "dataset": "gt_one, positives only, class 3->1, per-seed split",
        "evaluation": "native Varroa val/test + native size bucket; latency ms/image",
        "standard_nms_iou": .7, "size_bucket_nms_iou": .5, "seed_policy": list(DEFAULT_SEEDS),
    }


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int) -> dict[str, object]:
    values = shared.train_kwargs(args, data_yaml, seed, True)
    values.update(optimizer="auto", mosaic=1.0, close_mosaic=10)
    return values


def source_preflight() -> dict[str, Any]:
    assert_paper_comparison(list(CASES), "head_pruning_table")
    assert_paper_comparison(list(PAPER_COHORTS["HISTORICAL_FULL_NECK_CUES"]), "edge_color_table")
    canonical = load_yaml(CANONICAL_V1)
    historical_p234 = load_yaml(HISTORICAL_P234)
    reports = {}
    for case, spec in CASES.items():
        payload = load_yaml(spec.yaml)
        report = static_case_report(spec.yaml, canonical, spec.detect_from, "RepC2f", case)
        report.update({"expected_strides": list(spec.strides), "graph_contract": spec.graph_contract,
                       "namespace": spec.namespace, "historical_candidates": list(spec.historical_candidates)})
        reports[case] = report
    require(set(CASES).isdisjoint(historical.VARIANTS), "New IDs collide with historical Varroa IDs")
    require(layers(load_yaml(CASES["VR_STRUCT_H2_P23"].yaml)) != layers(canonical),
            "Structural H2 unexpectedly equals historical V1_FTSC")
    require(layers(load_yaml(CASES["VR_STRUCT_H1_P234"].yaml)) != layers(historical_p234),
            "Structural H1 unexpectedly equals historical HEAD_P234")
    obsolete = CONFIG_ROOT / "yolov8n_varroa_ftsc_head_p2_fullneck_v1.yaml"
    require(all(spec.yaml != obsolete for spec in CASES.values()),
            "Obsolete full-neck P2 case entered the active structural registry")
    v3 = load_yaml(HISTORICAL_V3_COLOR)
    v3_graph = layers(v3)
    require(tuple(v3_graph[-1][0]) == (19, 22) and v3_graph[19][2] == "P2ColorCueFusion" and
            v3_graph[20][0] == -1 and v3_graph[21][0] == [-1, 15] and v3_graph[22][2] == "RepC2f",
            "Historical V3_COLOR does not satisfy the post-fusion semantic contract")
    normalized_v3 = copy.deepcopy(v3)
    normalized_v3["head"].pop(9)
    normalized_v3["head"][-1][0] = [18, 21]
    require(normalized_v3 == canonical, "Historical V3_COLOR differs from V1 beyond Color insertion")
    from utils.ftsc_paper_preflight import ancestors
    v3_active = ancestors(v3_graph, len(v3_graph) - 1)
    require({19, 20, 21, 22}.issubset(v3_active), "Historical V3_COLOR cue/P3 path is not active")
    retained_inactive = sorted(set(range(len(v3_graph) - 1)) - v3_active)
    require(retained_inactive == [23, 24, 25, 26, 27, 28],
            f"Historical V3_COLOR full-neck structure changed: {retained_inactive}")
    protocol = protocol_metadata()
    return {
        "suite": EXPERIMENT, "status": "PREPARED_NOT_RUN", "dataset": "Varroa",
        "seeds": list(DEFAULT_SEEDS), "cases": reports, "historical_references": HISTORICAL_REFERENCES,
        "paper_comparison_policy": PAPER_COMPARISON_POLICY,
        "paper_cohorts": {key: list(value) for key, value in PAPER_COHORTS.items()},
        "historical_v3_color_reuse": {
            "status": "SOURCE_GRAPH_PASS_CHECKPOINT_PROVENANCE_PENDING_PHASE2",
            "yaml": str(HISTORICAL_V3_COLOR), "detect_from": [19, 22],
            "p2": "post_fusion", "p3": "post_RepC2f", "training_enabled": False,
            "retained_full_neck_layers_not_used_by_detect": retained_inactive,
        },
        "protocol": protocol,
        "training_protocol_fingerprint": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "source_checkpoint_identity": {"name": "yolov8n.pt", "sha256": "PHASE2_VERIFICATION_REQUIRED"},
        "phase2_states": [
            "HF_REUSE_COMPLETE", "HF_RESTORE_EVAL_REQUIRED", "LOCAL_RESUME_REQUIRED",
            "NEW_TRAIN_REQUIRED", "BLOCKED_IDENTITY_OR_REMOTE",
        ],
    }


def model_preflight(pretrained: str | None = None, loss_backward: bool = False) -> dict[str, Any]:
    shared.local_ultralytics()
    reports = {}
    for case, spec in CASES.items():
        item = runtime_case_report(spec.yaml, spec.detect_from, spec.strides, pretrained, imgsz=640)
        if loss_backward:
            item["synthetic_loss_backward"] = synthetic_loss_backward(spec.yaml)
        reports[case] = item
    ordered = [reports[name] for name in CASES]
    params, gflops = [item["params"] for item in ordered], [item["gflops"] for item in ordered]
    require(params == sorted(params, reverse=True), f"Varroa parameter trend invalid: {params}")
    require(gflops == sorted(gflops, reverse=True), f"Varroa GFLOP trend invalid: {gflops}")
    color = runtime_case_report(HISTORICAL_V3_COLOR, (19, 22), (4., 8.), pretrained, imgsz=640)
    if loss_backward:
        color["synthetic_loss_backward"] = synthetic_loss_backward(HISTORICAL_V3_COLOR)
    color["reuse_status"] = "RUNTIME_GRAPH_PASS_CHECKPOINT_PROVENANCE_PENDING_PHASE2"
    reports["HISTORICAL_V3_COLOR_REUSE_AUDIT"] = color
    return reports


def _amp_related_failure(error: Exception) -> bool:
    """Compatibility wrapper for existing tests/callers."""
    return is_amp_related_failure(error)


def run_one(args: argparse.Namespace) -> Path:
    case, seed = args.case, args.seed
    require(case in CASES and seed in DEFAULT_SEEDS, "Expected a registered case and seed")
    spec = CASES[case]
    args.seed_scope = "all"
    data_yaml = shared.dataset_for_seed(args, seed)
    args.data_yaml = data_yaml
    run_dir = args.project / spec.namespace / f"seed_{seed}"
    pretrained = Path(args.pretrained)
    pre_sha = pretrained_identity(pretrained, run_dir)
    contract = make_contract(
        dataset="Varroa", case=case, seed=seed,
        config=spec.yaml, graph_contract=spec.graph_contract,
        expected_strides=spec.strides, kwargs=train_kwargs(args, data_yaml, seed),
        data_yaml=data_yaml, pretrained_name=pretrained.name, pretrained_sha256=pre_sha,
    )
    registered = verify_or_create_contract(run_dir, contract)
    if (run_dir / "weights/last.pt").is_file() or (run_dir / "training_completion.json").is_file():
        validate_transfer_report(run_dir, contract)
    shared.local_ultralytics()
    from ultralytics import YOLO

    def _expected_checkpoint_training():
        state = validate_training_attempt(run_dir, contract, case=case, seed=seed)
        return {**contract["training"], "amp": state["effective_amp"]}

    completed = validate_completion(run_dir) or reconcile_finalized_training(
        run_dir, contract=contract, config=spec.yaml,
        expected_taps=spec.detect_from, expected_strides=spec.strides)
    if completed:
        require(registered, f"Unregistered completion: {run_dir}")
        verify_checkpoint_graph(run_dir / "weights/best.pt", config=spec.yaml,
                                expected_taps=spec.detect_from, expected_strides=spec.strides,
                                expected_seed=seed, expected_training=_expected_checkpoint_training())
    else:
        shared.seed_everything(seed)
        last = run_dir / "weights/last.pt"
        effective_amp, retry_reason = True, None
        if last.is_file():
            require(registered, f"Unregistered resume is forbidden: {run_dir}")
            attempt = validate_training_attempt(run_dir, contract, case=case, seed=seed)
            effective_amp = attempt["effective_amp"]
            retry_reason = attempt.get("amp_retry_reason")
            verify_checkpoint_graph(last, config=spec.yaml,
                                    expected_taps=spec.detect_from, expected_strides=spec.strides,
                                    expected_seed=seed, expected_training=_expected_checkpoint_training())
            require_resumable_last(last)
            YOLO(last).train(resume=True)
        else:
            require(not (run_dir / "results.csv").exists() and
                    not (run_dir / "weights/best.pt").exists(),
                    f"Orphaned training artifacts without last.pt: {run_dir}")
            require(pretrained.is_file(), "Verified local yolov8n.pt required for NEW Varroa training")
            def start_fresh(amp: bool) -> None:
                import shutil
                config_copy = run_dir / "config.yaml"
                if config_copy.is_file():
                    require(sha256(config_copy) == sha256(spec.yaml),
                            f"Run config changed: {config_copy}")
                else:
                    shutil.copy2(spec.yaml, config_copy)
                shared.seed_everything(seed)
                model = YOLO(spec.yaml)
                transfer = verify_pretrained_transfer(model, pretrained)
                require(transfer["source_sha256"] == contract["pretrained_sha256"],
                        "Pretrained source changed during construction")
                atomic_json(run_dir / TRANSFER_REPORT, transfer)
                record_training_attempt(run_dir, contract, case=case, seed=seed,
                                        effective_amp=amp, retry_reason=retry_reason)
                kwargs = train_kwargs(args, data_yaml, seed)
                kwargs.update(amp=amp, project=str(args.project / spec.namespace),
                              name=f"seed_{seed}", exist_ok=True)
                model.train(**kwargs)

            try:
                start_fresh(amp=True)
            except Exception as error:
                if not _amp_related_failure(error):
                    raise
                retry_reason = f"{type(error).__name__}: {error}"
                archive = shared.archive_failed_amp(run_dir)
                require(archive is not None, f"Failed to archive AMP attempt: {run_dir}")
                # Archive contains the first run contract and partial artifacts;
                # the retry gets a fresh directory with the SAME immutable identity.
                verify_or_create_contract(run_dir, contract)
                effective_amp = False
                start_fresh(amp=False)
        require(shared.complete(run_dir), f"Missing Varroa training artifacts: {run_dir}")
        verify_checkpoint_graph(run_dir / "weights/best.pt", config=spec.yaml,
                                expected_taps=spec.detect_from, expected_strides=spec.strides,
                                expected_seed=seed, expected_training=_expected_checkpoint_training())
        finalize_training(run_dir, effective_amp=effective_amp, retry_reason=retry_reason)
    shared.evaluate(run_dir, args)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int, choices=DEFAULT_SEEDS)
    parser.add_argument("--runtime-preflight", action="store_true")
    parser.add_argument("--loss-backward", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / EXPERIMENT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    frozen = {"epochs": (args.epochs, 100), "imgsz": (args.imgsz, 640),
              "batch": (args.batch_size, 4), "patience": (args.patience, 20)}
    require(not {k: v for k, (v, expected) in frozen.items() if v != expected}, "Frozen Varroa protocol changed")
    require(Path(args.pretrained).name == "yolov8n.pt", "Pretrained source must be yolov8n.pt")
    require(not args.loss_backward or args.runtime_preflight, "--loss-backward requires --runtime-preflight")
    require(not args.confirm_run or (args.case is not None and args.seed is not None),
            "--confirm-run requires --case and --seed")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    report: dict[str, Any] = {"source": source_preflight()}
    if args.runtime_preflight:
        report["model"] = model_preflight(args.pretrained, args.loss_backward)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.confirm_run:
        return
    args.data_root, args.dataset_root, args.project = map(Path.resolve, (args.data_root, args.dataset_root, args.project))
    run_one(args)


if __name__ == "__main__":
    main()
