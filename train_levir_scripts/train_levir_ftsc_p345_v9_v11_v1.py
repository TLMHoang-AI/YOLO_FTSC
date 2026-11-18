#!/usr/bin/env python3
"""LEVIR journal follow-up: ONLY YOLOv9t P345+FTSC and YOLO11n P345+FTSC.

Two new YAMLs, three seeds each. No stock, P2/P3, or YOLOv8n training.
Reuses frozen LEVIR cross-YOLO AdamW/no-Mosaic protocol and evaluator.
Read-only static preflight is the CLI default. Checkpoints must be archived
after EACH completed seed by the paired Marimo notebook/HF helper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

from train_levir_scripts import train_levir_ftsc_cross_yolo_portability_v1 as cross
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "runs/levir_ftsc_p345_v9_v11_v1"
NAMESPACE = "levir_ftsc_p345_v9_v11_v1"
SEEDS = (42, 43, 44)
CASES = {
    "V9T_P345_FTSC": {
        "family": "yolov9t",
        "model": ROOT / "models_related/models_config/yolov9/levir/yolov9t_levir_p345_ftsc_factorial.yaml",
        "stock": ROOT / "models_related/models_config/yolov9/levir/yolov9t_levir_b0_stock.yaml",
        "h2": ROOT / "models_related/models_config/yolov9/levir/yolov9t_p2_levir_ftsc_h2.yaml",
        "weights": "yolov9t.pt",
        "detect": (15, 18, 21),
    },
    "V11N_P345_FTSC": {
        "family": "yolo11n",
        "model": ROOT / "models_related/models_config/yolo11/levir/yolo11n_levir_p345_ftsc_factorial.yaml",
        "stock": ROOT / "models_related/models_config/yolo11/levir/yolo11n_levir_b0_stock.yaml",
        "h2": ROOT / "models_related/models_config/yolo11/levir/yolo11n_p2_levir_ftsc_h2.yaml",
        "weights": "yolo11n.pt",
        "detect": (16, 19, 22),
    },
}
METRICS = {
    "AP50": "test/metrics/mAP50(B)",
    "mAP50_95": "test/metrics/mAP50-95(B)",
    "AP50_small": "test_size/AP50-Small",
}
SOURCE_FILES = (
    "train_levir_scripts/train_levir_ftsc_p345_v9_v11_v1.py",
    "train_levir_scripts/train_levir_ftsc_cross_yolo_portability_v1.py",
    "train_levir_scripts/train_all_levir_yolov8n_p2_routing.py",
    "models_related/ultralytics/ultralytics/nn/tasks.py",
    "models_related/ultralytics/ultralytics/nn/modules/head.py",
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
    "models_related/ultralytics/ultralytics/utils/loss.py",
    "models_related/ultralytics/ultralytics/utils/tal.py",
    "models_related/ultralytics/ultralytics/engine/trainer.py",
    "models_related/ultralytics/ultralytics/cfg/default.yaml",
    "evaluate_test/standard_detection_metrics.py",
    "evaluate_test/size_bucket_evaluator.py",
)


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise RuntimeError(detail)


def sha(path: Path) -> str:
    require(path.is_file(), f"File missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = cross.load(path)
    require(isinstance(payload, dict), f"Bad model YAML: {path}")
    return payload


def graph(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "ftsc"}


def source_preflight() -> dict[str, Any]:
    require(set(CASES) == {"V9T_P345_FTSC", "V11N_P345_FTSC"},
            "Do not extend the two-case training cohort implicitly")
    cases = {}
    for name, spec in CASES.items():
        model, stock, h2 = (load(spec[key]) for key in ("model", "stock", "h2"))
        require(graph(model) == graph(stock),
                f"{name}: P345 treatment must retain the EXACT stock graph and model settings")
        require(model.get("ftsc", {}).get("enabled") is True, f"{name}: FTSC absent")
        require("ftsc" not in stock or not stock["ftsc"].get("enabled", True),
                f"{name}: stock already has FTSC")
        require(model["ftsc"] == h2["ftsc"] == cross.ftsc_contract(),
                f"{name}: FTSC differs from published cross-YOLO F5 H2 contract")
        require(model["head"][-1][2] == "Detect", f"{name}: wrong Detect class")
        require(tuple(model["head"][-1][0]) == spec["detect"],
                f"{name}: wrong stock P345 Detect input taps")
        require(tuple(h2["head"][-1][0]) != spec["detect"],
                f"{name}: P2/P3 accidentally used as P345")
        require(model["backbone"] == h2["backbone"],
                f"{name}: backbone differs from existing FTSC H2")
        require(spec["family"] in ("yolov9t", "yolo11n"), "Unexpected family")
        cases[name] = {
            "model_sha256": sha(spec["model"]),
            "stock_sha256": sha(spec["stock"]),
            "ftsc_reference_sha256": sha(spec["h2"]),
            "head_configuration": "P3/P4/P5",
            "ftsc_enabled": True,
            "family": spec["family"],
            "pretrained_file": spec["weights"],
        }
    return {
        "status": "STATIC_TWO_CASE_PREFLIGHT_PASSED",
        "cases": cases,
        "code_sha256": {path: sha(ROOT / path) for path in SOURCE_FILES},
        "training_protocol": cross.TRAINING,
    }


def pretrained_for(args: argparse.Namespace, case: str) -> Path:
    spec = CASES[case]
    path = args.pretrained_v9 if spec["family"] == "yolov9t" else args.pretrained_v11
    return path.expanduser().resolve()


def dataset(args: argparse.Namespace) -> Path:
    split = args.dataset_root / "levir_ship_yolo_seed42/levir_ship.yaml"
    require(split.is_file(), f"Missing frozen LEVIR split: {split}")
    workflow.validate_split(split, args.data_root)
    require((split.parent / "manifest.json").is_file(),
            f"Frozen LEVIR split manifest missing: {split.parent}")
    return split


def training(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str, Any]:
    spec = CASES[case]
    train = cross.train_kwargs(args, spec["family"], "FTSC_H2", seed, data_yaml)
    expected = cross.TRAINING
    require(train["optimizer"] == "AdamW" and train["mosaic"] == 0
            and train["close_mosaic"] == 0 and train["fitness_metric"] == "map50_95"
            and train["epochs"] == 100 and train["imgsz"] == 512 and train["batch"] == 8
            and train["patience"] == 20 and train["deterministic"] is True,
            "Cross-YOLO journal training protocol changed")
    require(expected["optimizer"] == "AdamW" and expected["lr0"] == .002
            and expected["nms_iou"] == .5, "Pinned cross-YOLO optimizer/evaluator drift")
    return train


def contract(args: argparse.Namespace, case: str, seed: int,
             data_yaml: Path, source: dict[str, Any]) -> dict[str, Any]:
    require(case in CASES and seed in SEEDS, "Invalid two-case seed cell")
    train = training(args, case, seed, data_yaml)
    return {
        "schema": "levir_ftsc_p345_v9_v11_v1",
        "dataset": "LEVIR-Ship",
        "case": case,
        "family": CASES[case]["family"],
        "seed": seed,
        "canonical_seeds": list(SEEDS),
        "split_seed": 42,
        "head_configuration": "P3/P4/P5",
        "ftsc_enabled": True,
        "model_yaml_sha256": source["cases"][case]["model_sha256"],
        "stock_yaml_sha256": source["cases"][case]["stock_sha256"],
        "cross_yolo_ftsc_yaml_sha256": source["cases"][case]["ftsc_reference_sha256"],
        "code_sha256": source["code_sha256"],
        "data_yaml_sha256": sha(data_yaml),
        "split_manifest_sha256": sha(data_yaml.parent / "manifest.json"),
        "pretrained_file": CASES[case]["weights"],
        "pretrained_sha256": sha(pretrained_for(args, case)),
        "training": {k: v for k, v in train.items() if k not in ("device", "workers", "data")},
        "evaluation": {"imgsz": 512, "batch": 8, "nms_iou": 0.5,
                       "dataset": "levir", "evaluator": "standard_detection_metrics"},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def state(run: Path, expected: dict[str, Any]) -> str:
    if not run.exists():
        return "NEW"
    require((run / "experiment_manifest.json").is_file()
            and (run / "p345_ftsc_contract.json").is_file(),
            f"Nonempty run missing scientific contract: {run}")
    for name in ("experiment_manifest.json", "p345_ftsc_contract.json"):
        require(json.loads((run / name).read_text(encoding="utf-8")) == expected,
                f"Scientific contract mismatch in {name}: {run}")
    complete = run / "p345_ftsc_completion.json"
    if complete.is_file():
        proof = json.loads(complete.read_text(encoding="utf-8"))
        require(proof["contract_sha256"] == sha(run / "experiment_manifest.json")
                and proof["best_sha256"] == sha(run / "weights/best.pt")
                and proof["metrics_sha256"] == sha(run / "evaluation_metrics_extended.json"),
                f"Completed run SHA mismatch: {run}")
        for name in ("config.yaml", "args.yaml", "results.csv", "weights/last.pt"):
            require((run / name).is_file(), f"Completed run missing {name}: {run}")
        return "COMPLETE"
    if all((run / rel).is_file() for rel in (
        "weights/best.pt", "weights/last.pt", "results.csv")):
        return "EVALUATE"
    if (run / "weights/last.pt").is_file():
        return "RESUME_REQUIRED"
    require(False, f"Orphan run cannot be implicitly retrained: {run}")
    raise AssertionError("Unreachable")


def model_preflight(pretrained_v9: Path | None = None,
                    pretrained_v11: Path | None = None) -> dict[str, Any]:
    """Real stock stride + FTSC-loss + backward tests for ONLY two new YAMLs."""
    workflow.local_ultralytics()
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.nn.tasks import DetectionModel

    reports: dict[str, Any] = {}
    for case, spec in CASES.items():
        torch.manual_seed(0)
        model = DetectionModel(str(spec["model"]), verbose=False)
        detect = model.model[-1]
        actual = [float(v) for v in detect.stride.detach().cpu().tolist()]
        require(actual == [8.0, 16.0, 32.0],
                f"{case}: Detect strides expected [8,16,32], got {actual}")
        require(tuple(detect.f) == spec["detect"],
                f"{case}: runtime Detect taps mismatch")
        require(getattr(detect, "ftsc_calibrator", None) is not None,
                f"{case}: runtime FTSC is disabled")
        model.args = get_cfg(DEFAULT_CFG,
                             {"epochs": 100, "fitness_metric": "map50_95"})
        model.train()
        torch.manual_seed(0)
        batch = {
            "img": torch.randn(2, 3, 128, 128),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([0.0, 0.0]),
            "bboxes": torch.tensor([[.5, .5, .1, .1], [.4, .4, .1, .1]]),
        }
        criterion = model.init_criterion()
        model.criterion = criterion
        require(getattr(criterion, "ftsc_calibrator", None) is not None,
                f"{case}: loss criterion does not receive FTSC")
        criterion.epoch = 5
        loss, items = model.loss(batch)
        require(torch.isfinite(loss).all() and torch.isfinite(items).all(),
                f"{case}: nonfinite synthetic loss")
        loss.sum().backward()
        require(any(p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad),
                f"{case}: no finite detector gradients")
        metrics = criterion.ftsc_metrics
        require(float(metrics.get("ftsc_rho", 0)) > 0,
                f"{case}: FTSC schedule did not activate")
        report = {"strides": actual, "detect_taps": list(spec["detect"]),
                  "ftsc": True, "loss_and_backward": "PASSED"}

        ckpt = pretrained_v9 if spec["family"] == "yolov9t" else pretrained_v11
        if ckpt is not None:
            require(ckpt.is_file(), f"{case}: pretrained missing: {ckpt}")
            # Matched initialization checks shared stock graph weights exactly.
            torch.manual_seed(0)
            control = YOLO(str(spec["stock"]), task="detect")
            torch.manual_seed(0)
            treatment = YOLO(str(spec["model"]), task="detect")
            control.load(str(ckpt), smart_transfer=True)
            treatment.load(str(ckpt), smart_transfer=True)
            shared = set(control.model.state_dict()) & set(treatment.model.state_dict())
            require(len(shared) > 50, f"{case}: too few shared initialization keys")
            for key in sorted(shared):
                require(torch.equal(control.model.state_dict()[key],
                                    treatment.model.state_dict()[key]),
                        f"{case}: stock and FTSC weights differ immediately after pretrained load: {key}")
            report["stock_pretrained_shared_weights"] = "IDENTICAL"
        reports[case] = report
    return reports


def run_one(args: argparse.Namespace, case: str, seed: int,
            data_yaml: Path, source: dict[str, Any]) -> str:
    """Single seed; no implicit subset expansion, no automatic HF upload."""
    require(case in CASES and seed in SEEDS, "Unexpected training cell")
    run = args.project / case / f"seed_{seed}"
    expected = contract(args, case, seed, data_yaml, source)
    status = state(run, expected)
    if status == "COMPLETE":
        print(f"[MATCHED COMPLETE] {case}/seed_{seed}", flush=True)
        return status
    if status == "NEW":
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / "experiment_manifest.json", expected)
        write_json(run / "p345_ftsc_contract.json", expected)
        workflow.local_ultralytics()
        from ultralytics import YOLO
        workflow.seed_everything(seed)
        model = YOLO(str(CASES[case]["model"]), task="detect")
        model.load(str(pretrained_for(args, case)), smart_transfer=True)
        kw = training(args, case, seed, data_yaml)
        model.train(**kw, project=str(run.parent), name=run.name, exist_ok=True)
        require(all((run / x).is_file() for x in (
            "weights/best.pt", "weights/last.pt", "results.csv", "args.yaml")),
            f"Training missing required artifacts: {run}")
    elif status == "RESUME_REQUIRED":
        require(args.resume, f"Interrupted run {run}: explicit --resume required")
        import torch
        ckpt = torch.load(run / "weights/last.pt", map_location="cpu", weights_only=False)
        require(isinstance(ckpt, dict) and ckpt.get("optimizer") is not None,
                f"Nonresumable last.pt: {run}")
        recorded = ckpt.get("train_args", {})
        kw = training(args, case, seed, data_yaml)
        require(isinstance(recorded, dict), f"Missing checkpoint train_args: {run}")
        for field in ("epochs", "imgsz", "batch", "seed", "optimizer", "mosaic", "close_mosaic"):
            require(str(recorded.get(field)) == str(kw[field]),
                    f"Resume training argument drift: {field}/{run}")
        workflow.local_ultralytics()
        from ultralytics import YOLO
        YOLO(str(run / "weights/last.pt"), task="detect").train(
            resume=True, device=args.device, workers=args.workers)
        require((run / "weights/best.pt").is_file(),
                f"Resume produced no best.pt: {run}")
    if status in {"NEW", "RESUME_REQUIRED", "EVALUATE"}:
        from evaluate_test.standard_detection_metrics import evaluate_run, load_merged_metrics
        evaluate_run(run_dir=run, data_yaml=data_yaml, dataset="levir",
                     imgsz=512, batch=8, workers=args.workers, device=args.device,
                     nms_iou=0.5, include_size=True)
        metrics = load_merged_metrics(run)
        for label, key in METRICS.items():
            require(key in metrics and math.isfinite(float(metrics[key])),
                    f"Missing/invalid {label} on {run}")
        config_copy = run / "config.yaml"
        if config_copy.is_file():
            require(sha(config_copy) == sha(CASES[case]["model"]),
                    f"config.yaml changed: {run}")
        else:
            shutil.copy2(CASES[case]["model"], config_copy)
        write_json(run / "p345_ftsc_completion.json", {
            "state": "COMPLETE",
            "contract_sha256": sha(run / "experiment_manifest.json"),
            "best_sha256": sha(run / "weights/best.pt"),
            "metrics_sha256": sha(run / "evaluation_metrics_extended.json"),
        })
        require(state(run, expected) == "COMPLETE",
                f"Completion integrity failed: {run}")
        print(f"[COMPLETE] {case}/seed_{seed}", flush=True)
    return "COMPLETE"


def aggregate(args: argparse.Namespace, data_yaml: Path, source: dict[str, Any]) -> dict[str, Any]:
    """Summarize just the six new runs; do NOT fabricate 2x2 factorial effects."""
    from evaluate_test.standard_detection_metrics import load_merged_metrics
    rows = []
    summaries = []
    for case in CASES:
        values = {metric: [] for metric in METRICS}
        for seed in SEEDS:
            run = args.project / case / f"seed_{seed}"
            require(state(run, contract(args, case, seed, data_yaml, source)) == "COMPLETE",
                    f"Cannot aggregate incomplete six-run cohort: {case}/seed_{seed}")
            metrics = load_merged_metrics(run)
            row = {"case": case, "family": CASES[case]["family"], "seed": seed}
            for metric, key in METRICS.items():
                score = float(metrics[key])
                require(math.isfinite(score), f"Non-finite metric: {run}/{metric}")
                row[metric] = score
                values[metric].append(score)
            rows.append(row)
        summary = {"case": case, "family": CASES[case]["family"], "n": 3}
        for metric, scores in values.items():
            summary[f"{metric}_mean"] = statistics.fmean(scores)
            summary[f"{metric}_sample_std"] = statistics.stdev(scores)
        summaries.append(summary)
    args.project.mkdir(parents=True, exist_ok=True)
    for filename, records in (
        ("p345_ftsc_runs.csv", rows),
        ("p345_ftsc_aggregate.csv", summaries),
    ):
        with (args.project / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return {"status": "SIX_FTSC_P345_RUNS_AGGREGATED",
            "runs": len(rows), "files": ["p345_ftsc_runs.csv", "p345_ftsc_aggregate.csv"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--model-preflight", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-root", type=Path, default=ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--pretrained-v9", type=Path, default=ROOT / "yolov9t.pt")
    parser.add_argument("--pretrained-v11", type=Path, default=ROOT / "yolo11n.pt")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    require(len(args.cases) == len(set(args.cases)) and all(c in CASES for c in args.cases),
            "Must select distinct authorized cases")
    require(len(args.seeds) == len(set(args.seeds)) and set(args.seeds).issubset(SEEDS),
            "Seeds must be distinct from (42,43,44)")
    require(sum(bool(x) for x in (args.confirm_run, args.model_preflight, args.aggregate)) <= 1,
            "Only one operation permitted at a time")
    require(not args.resume or args.confirm_run, "Resume requires --confirm-run")
    source = source_preflight()
    result: dict[str, Any] = {"status": "STATIC_PREFLIGHT_ONLY", "source": source}
    if args.model_preflight:
        result = {"status": "MODEL_PREFLIGHT_PASSED",
                  "models": model_preflight()}
    elif args.confirm_run or args.aggregate:
        args.data_root = args.data_root.expanduser().resolve()
        args.dataset_root = args.dataset_root.expanduser().resolve()
        args.project = args.project.expanduser().resolve()
        args.pretrained_v9 = args.pretrained_v9.expanduser().resolve()
        args.pretrained_v11 = args.pretrained_v11.expanduser().resolve()
        for case in args.cases:
            require(pretrained_for(args, case).is_file(),
                    f"Missing native pretrained {CASES[case]['weights']} for {case}")
        data_yaml = dataset(args)
        if args.aggregate:
            require(set(args.cases) == set(CASES) and set(args.seeds) == set(SEEDS),
                    "Aggregate requires BOTH families × three seeds")
            result = aggregate(args, data_yaml, source)
        else:
            # Real model/loss/pretrained identity preflight before any GPU training.
            model_preflight(args.pretrained_v9, args.pretrained_v11)
            for seed in args.seeds:
                for case in args.cases:
                    run_one(args, case, seed, data_yaml, source)
            result = {"status": "REQUESTED_SIX_CASE_CELLS_PROCESSED",
                      "cases": args.cases, "seeds": args.seeds}
    print(json.dumps(result, sort_keys=True, indent=2))
    return result


if __name__ == "__main__":
    main()
