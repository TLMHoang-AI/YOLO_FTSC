#!/usr/bin/env python3
"""Register corrected post-fusion LEVIR H2 Color/Edge cases.

Default execution is a no-network, no-training preflight. Exactly one new
case/seed may train only behind ``--confirm-run``. Historical C_H2/E_H2 are
comparison evidence only and can never be selected by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for entry in (ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as reference
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow
from utils.ftsc_paper_preflight import (
    layers, load_yaml, require, runtime_case_report, sha256, static_case_report,
    synthetic_loss_backward,
)
from utils.ftsc_paper_run_integrity import (
    TRANSFER_REPORT, atomic_json, finalize_training, is_amp_related_failure,
    make_contract, pretrained_identity, record_training_attempt, validate_completion,
    validate_training_attempt, validate_transfer_report, verify_checkpoint_graph,
    verify_completed_artifacts, verify_or_create_contract, verify_pretrained_transfer,
    reconcile_finalized_training, require_resumable_last,
)


CONFIG_ROOT = ROOT / "models_related/models_config/yolov8/levir"
CANONICAL_H2 = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml"
LEGACY_COLOR = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_colorcue.yaml"
LEGACY_EDGE = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edgecue.yaml"
LEGACY_ES1 = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml"
DEFAULT_SEEDS = (42, 43, 44)
EXPERIMENT = "levir_ftsc_cue_corrections_v2"


@dataclass(frozen=True)
class CaseSpec:
    yaml: Path
    cue_module: str
    detect_from: tuple[int, int]
    strides: tuple[float, float]
    graph_contract: str
    namespace: str
    legacy_yaml: Path


CASES: dict[str, CaseSpec] = {
    "C_H2_POSTFUSION_V2": CaseSpec(
        CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_colorcue_postfusion_v2.yaml",
        "P2ColorCueFusion", (20, 23), (4., 8.),
        "c2f_color_postfusion_p2_postc2f_p3_v2", "C_H2_POSTFUSION_V2", LEGACY_COLOR,
    ),
    "E_H2_POSTFUSION_V2": CaseSpec(
        CONFIG_ROOT / "yolov8n_p2_levir_ftsc_h2_edgecue_postfusion_v2.yaml",
        "P2EdgeCueFusion", (20, 23), (4., 8.),
        "c2f_edge_es1_scale025_postfusion_v2", "E_H2_POSTFUSION_V2", LEGACY_ES1,
    ),
}

HISTORICAL_REFERENCES = {
    "H2_p2_p3_nomosaic": {"yaml": str(CANONICAL_H2), "action": "REUSE"},
    "C_H2": {"yaml": str(LEGACY_COLOR), "action": "HISTORICAL_ONLY_NOT_CORRECTED"},
    "E_H2": {"yaml": str(LEGACY_EDGE), "action": "HISTORICAL_ONLY_NOT_CORRECTED"},
    "ES1": {"yaml": str(LEGACY_ES1), "action": "HISTORICAL_ONLY_NOT_CORRECTED"},
}


def protocol_metadata() -> dict[str, Any]:
    return {
        "parent": "train_all_levir_yolov8n_p2_ftsc_head_pruning.H2_p2_p3_nomosaic",
        "epochs": 100, "imgsz": 512, "batch": 8, "patience": 20,
        "optimizer": "auto", "scheduler": "linear", "lrf": .01,
        "fitness_metric": "map50_95", "mosaic": 0., "close_mosaic": 0,
        "pretrained": "yolov8n.pt smart_transfer", "deterministic": True,
        "fixed_split_seed": 42, "evaluation_nms_iou": .5, "latency_unit": "ms/image",
        "edge_cue": "ES1 bounded scale 0.25 constant learned gate, corrected postfusion routing",
        "seed_policy": list(DEFAULT_SEEDS),
    }


def train_kwargs(args: argparse.Namespace, data_yaml: Path, seed: int, amp: bool = True) -> dict[str, object]:
    return reference.train_kwargs(args, data_yaml, seed, amp)


def source_preflight() -> dict[str, Any]:
    canonical = load_yaml(CANONICAL_H2)
    reports = {}
    for case, spec in CASES.items():
        payload, legacy = load_yaml(spec.yaml), load_yaml(spec.legacy_yaml)
        report = static_case_report(spec.yaml, canonical, spec.detect_from, "C2f", case)
        graph, old_graph = layers(payload), layers(legacy)
        require(graph[:-1] == old_graph[:-1], f"{case}: correction changed more than Detect routing/metadata")
        require(tuple(old_graph[-1][0]) == (19, 22), f"{case}: historical routing evidence changed")
        require(graph[20][2] == spec.cue_module, f"{case}: cue placement changed")
        require(graph[21][0] == -1 and graph[22][0] == [-1, 15] and graph[23][2] == "C2f",
                f"{case}: fused P2 does not reach post-C2f P3")
        cue_count = sum(layer[2] == spec.cue_module for layer in graph)
        other = "P2EdgeCueFusion" if spec.cue_module == "P2ColorCueFusion" else "P2ColorCueFusion"
        require(cue_count == 1 and all(layer[2] != other for layer in graph), f"{case}: cue isolation failed")
        require(graph[20][3] == old_graph[20][3], f"{case}: historical cue parameters changed")
        report.update({
            "expected_strides": list(spec.strides), "graph_contract": spec.graph_contract,
            "namespace": spec.namespace, "cue_module": spec.cue_module,
            "historical_result_reusable_as_corrected": False,
        })
        reports[case] = report
    protocol = protocol_metadata()
    return {
        "suite": EXPERIMENT, "dataset": "LEVIR-Ship", "status": "PREPARED_NOT_RUN",
        "seeds": list(DEFAULT_SEEDS), "cases": reports,
        "historical_references": HISTORICAL_REFERENCES, "protocol": protocol,
        "training_protocol_fingerprint": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "source_checkpoint_identity": {"name": "yolov8n.pt", "sha256": "PHASE2_VERIFICATION_REQUIRED"},
        "phase2_states": ["HF_REUSE_COMPLETE", "HF_RESTORE_EVAL_REQUIRED", "LOCAL_RESUME_REQUIRED",
                          "NEW_TRAIN_REQUIRED", "BLOCKED_IDENTITY_OR_REMOTE"],
    }


def model_preflight(pretrained: str | None = None, loss_backward: bool = False) -> dict[str, Any]:
    workflow.local_ultralytics()
    import torch
    from ultralytics.nn.modules import EdgeCueFusion, P2ColorCueFusion
    from ultralytics.nn.tasks import DetectionModel

    reports = {}
    for case, spec in CASES.items():
        item = runtime_case_report(spec.yaml, spec.detect_from, spec.strides, pretrained, imgsz=512)
        model = DetectionModel(spec.yaml, verbose=False)
        colors = [module for module in model.modules() if isinstance(module, P2ColorCueFusion)]
        edges = [module for module in model.modules() if isinstance(module, EdgeCueFusion)]
        require((len(colors), len(edges)) == ((1, 0) if spec.cue_module == "P2ColorCueFusion" else (0, 1)),
                f"{case}: runtime cue isolation failed")
        cue = (colors or edges)[0]
        require(bool(torch.count_nonzero(cue.projection.weight) == 0 and
                     torch.count_nonzero(cue.projection.bias) == 0), f"{case}: projection is not zero initialized")
        item["cue"] = {"module": spec.cue_module, "hidden": cue.encoder[0].conv.out_channels,
                       "zero_initialized_projection": True}
        if loss_backward:
            item["synthetic_loss_backward"] = synthetic_loss_backward(spec.yaml)
        reports[case] = item
    return reports


def run_one(
    args: argparse.Namespace, *, registry: dict[str, CaseSpec] | None = None,
) -> Path:
    # An explicit isolated registry permits reproducible reconstruction of
    # checkpoint-missing LEVIR historical cases WITHOUT widening the two
    # original Color/Edge V2 paper cases or altering shared training logic.
    cases = CASES if registry is None else registry
    case, seed = args.case, args.seed
    require(case in cases and seed in DEFAULT_SEEDS, "Expected an authorized case and seed")
    spec = cases[case]
    data_yaml = workflow.prepare_fixed_split(args)
    run_dir = args.project / spec.namespace / f"seed_{seed}"
    pretrained = Path(args.pretrained)
    pre_sha = pretrained_identity(pretrained, run_dir)
    contract = make_contract(
        dataset="LEVIR-Ship", case=case, seed=seed, config=spec.yaml,
        graph_contract=spec.graph_contract, expected_strides=spec.strides,
        kwargs=train_kwargs(args, data_yaml, seed), data_yaml=data_yaml,
        pretrained_name=pretrained.name, pretrained_sha256=pre_sha,
    )
    registered = verify_or_create_contract(run_dir, contract)
    if (run_dir / "weights/last.pt").is_file() or (run_dir / "training_completion.json").is_file():
        validate_transfer_report(run_dir, contract)
    workflow.local_ultralytics()
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
        last = run_dir / "weights/last.pt"
        effective_amp, retry_reason = True, None
        if last.is_file():
            attempt = validate_training_attempt(run_dir, contract, case=case, seed=seed)
            effective_amp, retry_reason = bool(attempt["effective_amp"]), attempt.get("amp_retry_reason")
            verify_checkpoint_graph(last, config=spec.yaml, expected_taps=spec.detect_from,
                                    expected_strides=spec.strides, expected_seed=seed, expected_training=_expected_checkpoint_training())
            require_resumable_last(last)
            YOLO(last).train(resume=True)
        else:
            require(not (run_dir / "results.csv").exists() and
                    not (run_dir / "weights/best.pt").exists(),
                    f"Orphaned training artifacts without last.pt: {run_dir}")
            require(pretrained.is_file(), "Official local yolov8n.pt required for NEW LEVIR training")

            def start_fresh(amp: bool) -> None:
                config_copy = run_dir / "config.yaml"
                if config_copy.is_file():
                    require(sha256(config_copy) == sha256(spec.yaml), f"Run config changed: {config_copy}")
                else:
                    shutil.copy2(spec.yaml, config_copy)
                workflow.seed_everything(seed)
                model = YOLO(spec.yaml)
                transfer = verify_pretrained_transfer(model, pretrained)
                require(transfer["source_sha256"] == contract["pretrained_sha256"],
                        "Pretrained source changed during construction")
                atomic_json(run_dir / TRANSFER_REPORT, transfer)
                record_training_attempt(run_dir, contract, case=case, seed=seed,
                                        effective_amp=amp, retry_reason=retry_reason)
                kwargs = train_kwargs(args, data_yaml, seed, amp)
                kwargs.update(project=str(args.project / spec.namespace), name=f"seed_{seed}", exist_ok=True)
                model.train(**kwargs)

            try:
                start_fresh(True)
            except Exception as error:
                if not is_amp_related_failure(error):
                    raise
                retry_reason = f"{type(error).__name__}: {error}"
                archive = workflow.archive_failed_amp(run_dir) if hasattr(workflow, "archive_failed_amp") else None
                if archive is None:
                    import time
                    archive = run_dir.with_name(f"{run_dir.name}_amp_failed_{int(time.time())}")
                    shutil.move(run_dir, archive)
                verify_or_create_contract(run_dir, contract)
                effective_amp = False
                start_fresh(False)
        require(workflow.trained(run_dir), f"Missing LEVIR training artifacts: {run_dir}")
        verify_checkpoint_graph(run_dir / "weights/best.pt", config=spec.yaml,
                                expected_taps=spec.detect_from, expected_strides=spec.strides,
                                expected_seed=seed, expected_training=_expected_checkpoint_training())
        finalize_training(run_dir, effective_amp=effective_amp, retry_reason=retry_reason)
    reference.evaluate(run_dir, data_yaml, args)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int, choices=DEFAULT_SEEDS)
    parser.add_argument("--runtime-preflight", action="store_true")
    parser.add_argument("--loss-backward", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / EXPERIMENT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    frozen = {"epochs": (args.epochs, 100), "imgsz": (args.imgsz, 512), "batch": (args.batch_size, 8),
              "patience": (args.patience, 20), "split_seed": (args.split_seed, 42)}
    require(not {key: value for key, (value, expected) in frozen.items() if value != expected},
            "Frozen LEVIR protocol changed")
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
