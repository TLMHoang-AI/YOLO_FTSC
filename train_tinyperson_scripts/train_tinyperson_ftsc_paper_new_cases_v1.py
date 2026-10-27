#!/usr/bin/env python3
"""Register and preflight the new TinyPerson FTSC paper cases (Phase 1).

The default command is source-only and cannot train. Training one explicitly
selected case/seed requires ``--confirm-run``. No HF or notebook logic lives here.
"""

from __future__ import annotations

import argparse
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

from train_tinyperson_scripts import train_all_tinyperson as tiny
from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as historical
from utils.ftsc_paper_preflight import (
    layers, load_yaml, require, runtime_case_report, static_case_report, synthetic_loss_backward, sha256,
)
from utils.ftsc_paper_run_integrity import (
    TRANSFER_REPORT, atomic_json, finalize_training, make_contract, pretrained_identity,
    record_training_attempt, validate_completion, validate_training_attempt,
    validate_transfer_report, verify_checkpoint_graph, verify_completed_artifacts,
    verify_or_create_contract, verify_pretrained_transfer, reconcile_finalized_training,
    require_resumable_last,
)


CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/tinyperson"
CANONICAL_H2 = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
LEVIR_COLOR_H2 = ROOT / "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_colorcue.yaml"
DEFAULT_SEEDS = (42, 43, 44)
EXPERIMENT = "tinyperson_ftsc_paper_new_cases_v1"


@dataclass(frozen=True)
class CaseSpec:
    yaml: Path
    detect_from: tuple[int, ...]
    strides: tuple[float, ...]
    graph_contract: str
    namespace: str
    color: bool = False
    historical_candidates: tuple[str, ...] = ()


CASES: dict[str, CaseSpec] = {
    "TP_PRUNE_H0_P2345": CaseSpec(CONFIG_ROOT / "yolov8n_tinyperson_ftsc_prune_h0_p2345.yaml",
        (19, 22, 25, 28), (4., 8., 16., 32.), "c2f_structural_p2345_v1", "TP_PRUNE_H0_P2345"),
    "TP_PRUNE_H1_P234": CaseSpec(CONFIG_ROOT / "yolov8n_tinyperson_ftsc_prune_h1_p234.yaml",
        (19, 22, 25), (4., 8., 16.), "c2f_structural_p234_v1", "TP_PRUNE_H1_P234"),
    "TP_PRUNE_H3_P2": CaseSpec(CONFIG_ROOT / "yolov8n_tinyperson_ftsc_prune_h3_p2.yaml",
        (19,), (4.,), "c2f_structural_p2_v1", "TP_PRUNE_H3_P2"),
    "TP_CUE_COLOR_H2_MOSAIC": CaseSpec(CONFIG_ROOT / "yolov8n_tinyperson_ftsc_color_h2_mosaic.yaml",
        (20, 23), (4., 8.), "c2f_color_postfusion_p2_postc2f_p3_v1", "TP_CUE_COLOR_H2_MOSAIC", True),
}

HISTORICAL_REFERENCES = {
    "H2_MOSAIC": {
        "yaml": str(CANONICAL_H2),
        "training_enabled": False,
        "meaning": "immutable historical TinyPerson P2/P3 reference",
    },
    "ES1_MOSAIC": {
        "yaml": str(historical.VARIANTS["ES1_MOSAIC"].config),
        "training_enabled": False,
        "action": "REUSE_ORIGINAL_HISTORICAL_EDGE",
        "graph_contract": "legacy_prefusion_p2_preblock_p3_edge_alpha025",
    },
}


def train_kwargs(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    """Reuse the exact historical H2_MOSAIC recipe, changing only identity/output."""
    values = historical.train_kwargs(args, "H2_MOSAIC", seed, data_yaml)
    values.update(project=str(args.project / CASES[case].namespace),
                  name=f"seed_{seed}_corner_sw640_sh512", exist_ok=True)
    return values


def source_preflight() -> dict[str, Any]:
    canonical = load_yaml(CANONICAL_H2)
    reference_color = load_yaml(LEVIR_COLOR_H2)
    reports: dict[str, Any] = {}
    for case, spec in CASES.items():
        payload = load_yaml(spec.yaml)
        report = static_case_report(spec.yaml, canonical, spec.detect_from, "C2f", case)
        graph = layers(payload)
        modules = [str(layer[2]) for layer in graph]
        color_indices = [index for index, module in enumerate(modules) if module == "P2ColorCueFusion"]
        edge_indices = [index for index, module in enumerate(modules) if "EdgeCue" in module]
        require(not edge_indices, f"{case}: Edge Cue is forbidden")
        require(color_indices == ([20] if spec.color else []), f"{case}: Color Cue placement changed")
        if spec.color:
            require(graph[20][3] == reference_color["head"][10][3] == [32], f"{case}: Color Cue args changed")
            require(graph[21][0] == -1 and graph[22][0] == [-1, 15], f"{case}: fused P2 does not feed P3")
            require(spec.detect_from == (20, 23), f"{case}: Detect must consume fused P2 and post-C2f P3")
        report.update({
            "expected_strides": list(spec.strides), "graph_contract": spec.graph_contract,
            "namespace": spec.namespace, "historical_candidates": list(spec.historical_candidates),
            "color_cue_count": len(color_indices),
        })
        reports[case] = report
    require(set(CASES).isdisjoint(historical.VARIANTS), "New IDs collide with historical TinyPerson IDs")
    h2_graph = layers(canonical)
    h0_graph = layers(load_yaml(CASES["TP_PRUNE_H0_P2345"].yaml))
    h1_graph = layers(load_yaml(CASES["TP_PRUNE_H1_P234"].yaml))
    h3_graph = layers(load_yaml(CASES["TP_PRUNE_H3_P2"].yaml))
    require(h2_graph[:-1] == h0_graph[:len(h2_graph) - 1] == h1_graph[:len(h2_graph) - 1],
            "Historical H2 is not the shared structural prefix of TinyPerson H0/H1")
    require(h3_graph[:-1] == h2_graph[:len(h3_graph) - 1],
            "TinyPerson H3 is not a structural prefix of historical H2")
    require(tuple(h2_graph[-1][0]) == (19, 22), "Historical TinyPerson H2 Detect routing changed")
    require(historical.VARIANTS["H2_MOSAIC"].config == CANONICAL_H2 and
            historical.VARIANTS["H2_MOSAIC"].mosaic == 1.0 and
            historical.VARIANTS["H2_MOSAIC"].close_mosaic == 10,
            "Historical H2_MOSAIC registry is incompatible with the paper cohort")
    protocol = protocol_metadata()
    return {
        "suite": EXPERIMENT, "status": "PREPARED_NOT_RUN", "dataset": "TinyPerson",
        "seeds": list(DEFAULT_SEEDS), "cases": reports,
        "historical_references": HISTORICAL_REFERENCES,
        "historical_h2_compatibility": {
            "status": "SOURCE_CONTRACT_PASS_CHECKPOINT_PROVENANCE_PENDING_PHASE2",
            "structural_prefix_match": True, "detect_from": [19, 22],
            "ftsc_match": True, "mosaic": 1.0, "close_mosaic": 10,
            "pretrained_policy": "yolov8n.pt smart_transfer",
            "data_and_evaluator_owner": "train_all_tinyperson",
        },
        "historical_edge_reuse": {
            "case": "ES1_MOSAIC", "action": "REUSE_IN_ORIGINAL_EXPERIMENT",
            "semantic_warning": "Detect [19,22] is pre-fusion P2 and pre-C2f P3; not equivalent to corrected LEVIR Edge",
            "cue": "P2EdgeCueFusion hidden32 alpha0.25 constant learned orientation gate",
            "checkpoint_provenance": "PHASE2_VERIFICATION_REQUIRED",
        },
        "protocol": protocol,
        "training_protocol_fingerprint": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        "source_checkpoint_identity": {"name": "yolov8n.pt", "sha256": "PHASE2_VERIFICATION_REQUIRED"},
        "phase2_states": [
            "HF_REUSE_COMPLETE", "HF_RESTORE_EVAL_REQUIRED", "LOCAL_RESUME_REQUIRED",
            "NEW_TRAIN_REQUIRED", "BLOCKED_IDENTITY_OR_REMOTE",
        ],
    }


def protocol_metadata() -> dict[str, Any]:
    return {
        "parent": "train_tinyperson_ftsc_causal_suite.H2_MOSAIC",
        "epochs": 100, "imgsz": 640, "batch": 8, "patience": 20,
        "optimizer": "AdamW", "lr0": .002, "momentum": .9, "weight_decay": .0005,
        "scheduler": "linear", "mosaic": 1.0, "close_mosaic": 10,
        "pretrained": "yolov8n.pt smart_transfer", "corner_geometry": "sw640_sh512",
        "evaluation": "original-image merged TinyBenchmark; latency ms/window",
        "nms_iou": .5, "seed_policy": list(DEFAULT_SEEDS),
    }


def model_preflight(pretrained: str | None = None, loss_backward: bool = False) -> dict[str, Any]:
    historical.local_ultralytics()
    import torch
    from ultralytics.nn.modules import EdgeCueFusion, P2ColorCueFusion
    from ultralytics.nn.tasks import DetectionModel

    reports = {}
    for case, spec in CASES.items():
        item = runtime_case_report(spec.yaml, spec.detect_from, spec.strides, pretrained, imgsz=640)
        inspected = DetectionModel(spec.yaml, verbose=False)
        colors = [module for module in inspected.modules() if isinstance(module, P2ColorCueFusion)]
        edges = [module for module in inspected.modules() if isinstance(module, EdgeCueFusion)]
        require(len(colors) == int(spec.color), f"{case}: runtime Color Cue count wrong")
        require(not edges, f"{case}: runtime Edge Cue is forbidden")
        if colors:
            cue = colors[0]
            zero_init = bool(torch.count_nonzero(cue.projection.weight) == 0 and
                             torch.count_nonzero(cue.projection.bias) == 0)
            require(zero_init, f"{case}: Color projection is not zero initialized")
            item["color_cue"] = {"count": 1, "hidden": cue.encoder[0].conv.out_channels,
                                 "zero_initialized_projection": True}
        else:
            item["color_cue"] = {"count": 0}
        item["edge_cue_count"] = 0
        if loss_backward:
            item["synthetic_loss_backward"] = synthetic_loss_backward(spec.yaml)
        reports[case] = item
    pruning = [reports[name] for name in ("TP_PRUNE_H0_P2345", "TP_PRUNE_H1_P234", "TP_PRUNE_H3_P2")]
    params = [item["params"] for item in pruning]
    gflops = [item["gflops"] for item in pruning]
    require(params == sorted(params, reverse=True), f"TinyPerson parameter trend invalid: {params}")
    require(gflops == sorted(gflops, reverse=True), f"TinyPerson GFLOP trend invalid: {gflops}")
    historical_h2 = runtime_case_report(CANONICAL_H2, (19, 22), (4., 8.), pretrained, imgsz=640)
    require(reports["TP_PRUNE_H1_P234"]["params"] > historical_h2["params"] >
            reports["TP_PRUNE_H3_P2"]["params"], "Historical H2 complexity is outside structural cohort trend")
    reports["HISTORICAL_H2_MOSAIC_REFERENCE"] = historical_h2
    return reports


def run_one(args: argparse.Namespace) -> Path:
    """One identity-pinned NEW case/seed. Never adopt an untracked training folder."""
    case, seed = args.case, args.seed
    require(case in CASES and seed in DEFAULT_SEEDS, "Expected a registered case and seed")
    spec = CASES[case]
    test_dir = tiny.prepare_test_set(args.data_root, args.dataset_root)
    seed_dir = tiny.prepare_seed_dataset(args.data_root, args.dataset_root, test_dir, seed)
    data_yaml = seed_dir / "tinyperson.yaml"
    require(data_yaml.is_file(), f"Prepared TinyPerson YAML missing: {data_yaml}")
    run_dir = args.project / spec.namespace / f"seed_{seed}_corner_sw640_sh512"
    pretrained = Path(args.pretrained)
    pre_sha = pretrained_identity(pretrained, run_dir)
    contract = make_contract(
        dataset="TinyPerson", case=case, seed=seed,
        config=spec.yaml, graph_contract=spec.graph_contract,
        expected_strides=spec.strides, kwargs=train_kwargs(args, case, seed, data_yaml),
        data_yaml=data_yaml, pretrained_name=pretrained.name, pretrained_sha256=pre_sha,
    )
    registered = verify_or_create_contract(run_dir, contract)
    if (run_dir / "weights/last.pt").is_file() or (run_dir / "training_completion.json").is_file():
        validate_transfer_report(run_dir, contract)
    historical.local_ultralytics()
    from ultralytics import YOLO

    def _expected_checkpoint_training():
        state = validate_training_attempt(run_dir, contract, case=case, seed=seed)
        return {**contract["training"], "amp": state["effective_amp"]}

    completed = validate_completion(run_dir) or reconcile_finalized_training(
        run_dir, contract=contract, config=spec.yaml,
        expected_taps=spec.detect_from, expected_strides=spec.strides)
    if completed:
        require(registered, f"Completion without registered provenance: {run_dir}")
        verify_checkpoint_graph(run_dir / "weights/best.pt", config=spec.yaml,
                                expected_taps=spec.detect_from, expected_strides=spec.strides,
                                expected_seed=seed, expected_training=_expected_checkpoint_training())
    else:
        historical._seed_everything(seed)
        last = run_dir / "weights/last.pt"
        if last.is_file():
            require(registered, f"Unregistered resume is forbidden: {run_dir}")
            attempt = validate_training_attempt(run_dir, contract, case=case, seed=seed)
            require(attempt["effective_amp"] is True,
                    f"TinyPerson does not define AMP fallback: {run_dir}")
            verify_checkpoint_graph(last, config=spec.yaml,
                                    expected_taps=spec.detect_from, expected_strides=spec.strides,
                                    expected_seed=seed, expected_training=_expected_checkpoint_training())
            require_resumable_last(last)
            YOLO(last).train(resume=True)
        else:
            require(not (run_dir / "results.csv").exists() and
                    not (run_dir / "weights/best.pt").exists(),
                    f"Orphaned training artifacts without last.pt: {run_dir}")
            require(pretrained.is_file(), "Official local yolov8n.pt required for NEW training")
            # A failed previous startup is safe to retry, but never accept foreign artifacts.
            config_copy = run_dir / "config.yaml"
            if config_copy.exists():
                require(sha256(config_copy) == sha256(spec.yaml),
                        f"Run config mutated: {config_copy}")
            else:
                import shutil
                shutil.copy2(spec.yaml, config_copy)
            model = YOLO(spec.yaml)
            transfer = verify_pretrained_transfer(model, pretrained)
            require(transfer["source_sha256"] == contract["pretrained_sha256"],
                    "Pretrained source changed during model construction")
            atomic_json(run_dir / TRANSFER_REPORT, transfer)
            record_training_attempt(run_dir, contract, case=case, seed=seed,
                                    effective_amp=True, retry_reason=None)
            model.train(**train_kwargs(args, case, seed, data_yaml))
        require(tiny.training_complete(run_dir, args.epochs),
                f"Incomplete training artifacts: {run_dir}")
        verify_checkpoint_graph(run_dir / "weights/best.pt", config=spec.yaml,
                                expected_taps=spec.detect_from, expected_strides=spec.strides,
                                expected_seed=seed, expected_training=_expected_checkpoint_training())
        finalize_training(run_dir, effective_amp=True)

    # Reuses only the exact best.pt belonging to this case+seed.
    tiny.evaluate(run_dir, data_yaml, test_dir, args.data_root, args)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int, choices=DEFAULT_SEEDS)
    parser.add_argument("--runtime-preflight", action="store_true")
    parser.add_argument("--loss-backward", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / EXPERIMENT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    frozen = {"epochs": (args.epochs, 100), "imgsz": (args.imgsz, 640),
              "batch": (args.batch_size, 8), "patience": (args.patience, 20)}
    require(not {k: v for k, (v, expected) in frozen.items() if v != expected}, "Frozen TinyPerson protocol changed")
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
