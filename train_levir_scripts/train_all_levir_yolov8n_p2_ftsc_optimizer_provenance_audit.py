#!/usr/bin/env python3
"""Run the five-case FTSC optimizer and checkpoint-provenance audit on LEVIR-Ship."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_corrected_controls as corrected
from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_dfl_strength_audit as fixed1
from train_levir_scripts import train_all_levir_yolov8n_p2_routing as workflow


CONFIG_ROOT = PROJECT_ROOT / "models_related/models_config/yolov8/levir"
EXPERIMENT_SLUG = "levir_yolov8n_p2_ftsc_optimizer_provenance_audit"
HF_REPO = "duyle2408/levir-yolov8n-p2-ftsc-optimizer-provenance-audit"
PROTOCOL_VERSION = "ftsc_optimizer_provenance_v1"
SUITE_NAME = "optimizer_provenance_audit"
HISTORICAL_Y4_TEST_AP50 = 0.7878924760784733
HISTORICAL_Y4_VAL_AP50 = 0.8373975313576892
TRAIN_DATASET_SIZE = workflow.PUBLISHED_COUNTS["train"]
NOMINAL_BATCH_SIZE = 64

Y0_CONFIG = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_y0_baseline.yaml"
Y4_CONFIG = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls.yaml"
Y4_FIXED1_CONFIG = CONFIG_ROOT / "yolov8n_p2_levir_ftsc_af_y4_f5_position_dflcls_fixed1.yaml"

VARIANTS = {
    "A0_current_y0_musgd": Y0_CONFIG,
    "A1_current_y4_musgd": Y4_CONFIG,
    "B0_y0_sgd": Y0_CONFIG,
    "B1_y4_sgd": Y4_CONFIG,
    "H1_y4_fixed1_sgd_legacyfit": Y4_FIXED1_CONFIG,
}


@dataclass(frozen=True)
class FactorSpec:
    model_semantics: str
    optimizer: str
    fitness_metric: str
    lr0: float
    momentum: float


FACTORS = {
    "A0_current_y0_musgd": FactorSpec("y0", "auto", "map50", 0.01, 0.937),
    "A1_current_y4_musgd": FactorSpec("y4_learned", "auto", "map50", 0.01, 0.937),
    "B0_y0_sgd": FactorSpec("y0", "SGD", "map50", 0.01, 0.9),
    "B1_y4_sgd": FactorSpec("y4_learned", "SGD", "map50", 0.01, 0.9),
    "H1_y4_fixed1_sgd_legacyfit": FactorSpec("y4_fixed1", "SGD", "map50_95", 0.01, 0.9),
}

DUAL_CHECKPOINT_METRICS = {
    "ap50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
CHECKPOINTS = {
    "selected": "best.pt",
    "best_ap50": "best_ap50.pt",
    "best_map50_95": "best_map50_95.pt",
}
PRIMARY_METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/mAP75(B)",
)


def _require(condition: bool, context: str, message: str) -> None:
    if not condition:
        raise RuntimeError(f"{context}: {message}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_local_pretrained(pretrained: str | Path) -> Path:
    """Resolve a local checkpoint from the launch directory or repository root."""
    path = Path(pretrained).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (path.resolve(), (PROJECT_ROOT / path).resolve())
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def pretrained_provenance(pretrained: str | Path) -> dict[str, object]:
    """Describe an already-local pretrained artifact without downloading anything."""
    path = Path(pretrained).expanduser()
    resolved = resolve_local_pretrained(path)
    metadata: dict[str, object] = {
        "pretrained/name": path.name,
        "pretrained/path": str(resolved),
        "pretrained/sha256": None,
        "pretrained/bytes": None,
    }
    if resolved.is_file():
        metadata["pretrained/sha256"] = _sha256_file(resolved)
        metadata["pretrained/bytes"] = resolved.stat().st_size
    return metadata


def _hash_names(names: list[str]) -> str:
    payload = "".join(f"{name}\n" for name in sorted(names)).encode()
    return hashlib.sha256(payload).hexdigest()


def dataset_provenance(data_yaml: Path, split_seed: int) -> dict[str, object]:
    """Hash the exact image stems in the prepared split without copying data."""
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    split_stems: dict[str, list[str]] = {}
    metadata: dict[str, object] = {"dataset/split_seed": split_seed}
    for split in ("train", "val", "test"):
        stems = sorted(path.stem for path in (root / payload[split]).iterdir() if path.suffix.lower() == ".png")
        split_stems[split] = stems
        metadata[f"dataset/{split}_count"] = len(stems)
        metadata[f"dataset/{split}_stem_sha256"] = _hash_names(stems)
    all_stems = [stem for split in ("train", "val", "test") for stem in split_stems[split]]
    _require(len(all_stems) == len(set(all_stems)), "dataset", "prepared splits overlap")
    metadata["dataset/all_stem_sha256"] = _hash_names(all_stems)
    return metadata


def git_provenance() -> dict[str, object]:
    """Return local source identity without contacting the remote."""
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    workflow.local_ultralytics()
    import ultralytics

    return {
        "git/commit": git("rev-parse", "HEAD"),
        "git/branch": git("branch", "--show-current"),
        "ultralytics/version": ultralytics.__version__,
    }


def estimated_iterations(epochs: int, batch_size: int, dataset_size: int = TRAIN_DATASET_SIZE) -> int:
    """Mirror BaseTrainer's optimizer-selection iteration estimate."""
    return math.ceil(dataset_size / max(batch_size, NOMINAL_BATCH_SIZE)) * epochs


def model_preflight(model, variant: str) -> dict[str, object]:
    """Reuse the locked Y0/Y4/fixed1 checks and normalize their reports."""
    semantics = FACTORS[variant].model_semantics
    if semantics == "y0":
        report = corrected.preflight_model(model, "B_tal_baseline")
        return {**report, "ftsc_mode": "disabled"}
    if semantics == "y4_learned":
        report = corrected.preflight_model(model, "F_canonical_y4")
        return {**report, "ftsc_mode": "learned"}
    report = fixed1.preflight_model(model, "F_fixed1")
    return {**report, "ftsc_mode": "fixed1"}


def _network(model):
    return corrected._network_and_head(model)[0]


def model_for(variant: str, pretrained: str):
    """Build one case, transfer the common pretrained weights, and verify model semantics."""
    workflow.local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(VARIANTS[variant])
    model.load(pretrained, smart_transfer=True)
    model_preflight(model, variant)
    return model


def _tensor_state_sha256(model, *, omit_dfl_strength: bool) -> str:
    """Hash ordered state names, shapes, dtypes, and exact CPU tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(_network(model).state_dict().items()):
        if omit_dfl_strength and name.endswith("strength_logits.dfl_distribution"):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_hashes(models: dict[str, object]) -> dict[str, dict[str, object]]:
    """Prove paired pretrained initialization equivalence before optimizer construction."""
    reports = {
        variant: {
            "init/full_state_sha256": _tensor_state_sha256(model, omit_dfl_strength=False),
            "init/shared_state_sha256": _tensor_state_sha256(model, omit_dfl_strength=True),
        }
        for variant, model in models.items()
    }
    pairs = (
        ("A0_current_y0_musgd", "B0_y0_sgd", "full_state_sha256"),
        ("A1_current_y4_musgd", "B1_y4_sgd", "full_state_sha256"),
        ("B1_y4_sgd", "H1_y4_fixed1_sgd_legacyfit", "shared_state_sha256"),
    )
    for left, right, kind in pairs:
        key = f"init/{kind}"
        _require(reports[left][key] == reports[right][key], f"{left} vs {right}", "initial state differs")
    reports["A0_current_y0_musgd"]["init/equal_to_B0"] = True
    reports["B0_y0_sgd"]["init/equal_to_A0"] = True
    reports["A1_current_y4_musgd"]["init/equal_to_B1"] = True
    reports["B1_y4_sgd"]["init/equal_to_A1"] = True
    reports["B1_y4_sgd"]["init/shared_equal_to_H1"] = True
    reports["H1_y4_fixed1_sgd_legacyfit"]["init/shared_equal_to_B1"] = True
    return reports


def optimizer_preflight(model, variant: str, args: argparse.Namespace) -> dict[str, object]:
    """Construct the real optimizer path and fail unless the requested factor resolves exactly."""
    workflow.local_ultralytics()
    import torch
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.optim import MuSGD

    spec = FACTORS[variant]
    iterations = estimated_iterations(args.epochs, args.batch_size)
    accumulate = max(round(NOMINAL_BATCH_SIZE / args.batch_size), 1)
    effective_decay = args.weight_decay * args.batch_size * accumulate / NOMINAL_BATCH_SIZE
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(lr0=spec.lr0, momentum=spec.momentum, warmup_bias_lr=0.1)
    trainer.data = {"nc": 1}
    optimizer = trainer.build_optimizer(
        _network(model),
        name=spec.optimizer,
        lr=spec.lr0,
        momentum=spec.momentum,
        decay=effective_decay,
        iterations=iterations,
    )
    resolved = type(optimizer).__name__
    expected = "MuSGD" if spec.optimizer == "auto" else "SGD"
    _require(resolved == expected, variant, f"optimizer={spec.optimizer} resolved to {resolved}, expected {expected}")
    if expected == "MuSGD":
        _require(type(optimizer) is MuSGD, variant, f"expected exact MuSGD class, got {type(optimizer)!r}")
    else:
        _require(
            type(optimizer) is torch.optim.SGD,
            variant,
            f"expected exact torch.optim.SGD, got {type(optimizer)!r}",
        )
    lrs = {float(group["lr"]) for group in optimizer.param_groups if group["params"]}
    momenta = {float(group["momentum"]) for group in optimizer.param_groups if group["params"]}
    _require(lrs == {0.01}, variant, f"resolved learning rates are {sorted(lrs)}")
    _require(momenta == {0.9}, variant, f"resolved momentum values are {sorted(momenta)}")
    return {
        "optimizer/requested": spec.optimizer,
        "optimizer/resolved": resolved,
        "optimizer/lr0": 0.01,
        "optimizer/momentum": 0.9,
        "optimizer/weight_decay": effective_decay,
        "optimizer/estimated_iterations": iterations,
    }


def verify_runtime_optimizer(trainer, variant: str) -> None:
    """Recheck the optimizer created by the actual trainer before its first epoch."""
    workflow.local_ultralytics()
    import torch
    from ultralytics.optim import MuSGD

    expected_type = MuSGD if FACTORS[variant].optimizer == "auto" else torch.optim.SGD
    _require(type(trainer.optimizer) is expected_type, variant, f"runtime optimizer is {type(trainer.optimizer)!r}")
    spec = FACTORS[variant]
    _require(trainer.args.optimizer == spec.optimizer, variant, "runtime optimizer request drifted")
    _require(trainer.args.fitness_metric == spec.fitness_metric, variant, "runtime fitness metric drifted")
    groups = [group for group in trainer.optimizer.param_groups if group["params"]]
    _require({float(group["lr"]) for group in groups} == {0.01}, variant, "runtime learning rate drifted")
    _require({float(group["momentum"]) for group in groups} == {0.9}, variant, "runtime momentum drifted")
    expected_decay = trainer.args.weight_decay * trainer.batch_size * trainer.accumulate / NOMINAL_BATCH_SIZE
    observed_decay = {float(group.get("weight_decay", 0.0)) for group in groups}
    _require(observed_decay == {0.0, expected_decay}, variant, f"runtime weight decay drifted: {observed_decay}")


class DualCheckpointManager:
    """Observe validation metrics and copy the current epoch checkpoint under two independent criteria."""

    def __init__(self, run_dir: Path, selected_fitness_metric: str) -> None:
        self.run_dir = Path(run_dir)
        self.metadata_path = self.run_dir / "dual_checkpoint_metadata.json"
        self.state: dict[str, object] = {
            "selected_fitness_metric": selected_fitness_metric,
            "best_ap50_epoch": None,
            "best_ap50_value": None,
            "best_map50_95_epoch": None,
            "best_map50_95_value": None,
        }
        if self.metadata_path.is_file():
            loaded = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            _require(
                loaded.get("selected_fitness_metric") == selected_fitness_metric,
                str(self.run_dir),
                "resume checkpoint fitness does not match this case",
            )
            self.state.update(loaded)

    def observe(self, epoch: int, metrics: dict[str, object], checkpoint: Path) -> tuple[str, ...]:
        """Copy on improvement or tie, matching the trainer's latest-tie best.pt semantics."""
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        improved = []
        self.run_dir.joinpath("weights").mkdir(parents=True, exist_ok=True)
        for short_name, metric_key in DUAL_CHECKPOINT_METRICS.items():
            value = float(metrics[metric_key])
            value_key = f"best_{short_name}_value"
            epoch_key = f"best_{short_name}_epoch"
            previous = self.state[value_key]
            # BaseTrainer raises best_fitness only on `>`, then writes best.pt
            # whenever `best_fitness == fitness`; a tie selects the later epoch.
            if previous is None or value >= float(previous):
                destination = self.run_dir / "weights" / f"best_{short_name}.pt"
                temporary = destination.with_suffix(".pt.tmp")
                shutil.copyfile(checkpoint, temporary)
                temporary.replace(destination)
                self.state[value_key] = value
                self.state[epoch_key] = int(epoch)
                improved.append(short_name)
        metadata_temporary = self.metadata_path.with_suffix(".json.tmp")
        metadata_temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_temporary.replace(self.metadata_path)
        return tuple(improved)

    def on_model_save(self, trainer) -> None:
        self.observe(trainer.epoch + 1, trainer.metrics, Path(trainer.last))


def train_kwargs(
    args: argparse.Namespace, variant: str, data_yaml: Path, seed: int, amp: bool
) -> dict[str, object]:
    spec = FACTORS[variant]
    kwargs: dict[str, object] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": seed,
        "deterministic": True,
        "amp": amp,
        "plots": False,
        "optimizer": spec.optimizer,
        "lr0": spec.lr0,
        "momentum": spec.momentum,
        "weight_decay": args.weight_decay,
        "fitness_metric": spec.fitness_metric,
    }
    _require(args.augmentation_policy == "yolo_default", variant, "augmentation policy drifted")
    return kwargs


def _attach_audit_callbacks(model, manager: DualCheckpointManager, variant: str) -> None:
    model.add_callback("on_pretrain_routine_end", lambda trainer: verify_runtime_optimizer(trainer, variant))
    model.add_callback("on_model_save", manager.on_model_save)


def _trained(run_dir: Path) -> bool:
    required = (
        "weights/best.pt",
        "weights/last.pt",
        "weights/best_ap50.pt",
        "weights/best_map50_95.pt",
        "results.csv",
        "dual_checkpoint_metadata.json",
    )
    return all((run_dir / relative).is_file() for relative in required)


def train(variant: str, seed: int, data_yaml: Path, amp: bool, args: argparse.Namespace) -> Path:
    """Train or resume one case with observational dual-checkpoint callbacks."""
    run_dir = args.project / variant / f"seed_{seed}"
    if _trained(run_dir):
        print(f"Reusing trained audit run: {variant}/seed_{seed}", flush=True)
        return run_dir
    manager = DualCheckpointManager(run_dir, FACTORS[variant].fitness_metric)
    last = run_dir / "weights/last.pt"
    workflow.seed_everything(seed)
    if last.is_file():
        workflow.local_ultralytics()
        from ultralytics import YOLO

        model = YOLO(last)
        _attach_audit_callbacks(model, manager, variant)
        print(f"Resuming {run_dir}", flush=True)
        model.train(resume=True)
    else:
        kwargs = train_kwargs(args, variant, data_yaml, seed, amp)
        kwargs.update(project=str(args.project / variant), name=f"seed_{seed}", exist_ok=True)
        model = model_for(variant, args.pretrained)
        _attach_audit_callbacks(model, manager, variant)
        try:
            model.train(**kwargs)
        except Exception as error:
            if not amp:
                raise
            archived = run_dir.with_name(f"{run_dir.name}_amp_failed_{int(time.time())}")
            if run_dir.exists():
                shutil.move(run_dir, archived)
            print(f"AMP failed for {variant}/seed_{seed}: {error!r}; retrying without AMP", flush=True)
            workflow.seed_everything(seed)
            manager = DualCheckpointManager(run_dir, FACTORS[variant].fitness_metric)
            model = model_for(variant, args.pretrained)
            _attach_audit_callbacks(model, manager, variant)
            kwargs["amp"] = False
            model.train(**kwargs)
    if not _trained(run_dir):
        raise FileNotFoundError(f"Training ended without complete dual-checkpoint artifacts: {run_dir}")
    return run_dir


def _results_csv_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        return {}
    return {"training/actual_stop_epoch": int(float(rows[-1]["epoch"]))}


def _actual_amp(run_dir: Path, fallback: bool) -> bool:
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.is_file():
        return fallback
    import yaml

    return bool(yaml.safe_load(args_yaml.read_text(encoding="utf-8")).get("amp", fallback))


def _case_metadata(variant: str, seed: int, args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    spec = FACTORS[variant]
    metadata = {
        "protocol/version": PROTOCOL_VERSION,
        "protocol/fitness_metric": spec.fitness_metric,
        "protocol/best_checkpoint_metric": spec.fitness_metric,
        f"suite/{SUITE_NAME}": 1.0,
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "workers": args.workers,
        "device": args.device,
        "deterministic": True,
        "amp": _actual_amp(run_dir, args._active_amp),
        "augmentation_policy": args.augmentation_policy,
        "fitness_metric": spec.fitness_metric,
        "best_checkpoint_metric": spec.fitness_metric,
        "nms_iou": 0.5,
        **args._optimizer_reports[variant],
        **args._initialization_reports[variant],
        **args._pretrained_provenance,
        **args._dataset_provenance,
        **args._git_provenance,
    }
    return metadata


def _ftsc_metadata(model, variant: str) -> dict[str, object]:
    _, head = corrected._network_and_head(model)
    if FACTORS[variant].model_semantics == "y0":
        return {"ftsc/enabled": 0.0}
    if FACTORS[variant].model_semantics == "y4_fixed1":
        report = fixed1.preflight_model(model, "F_fixed1")
        return fixed1._ftsc_metadata(head, report)
    report = corrected.preflight_model(model, "F_canonical_y4")
    return corrected._ftsc_metadata(head, report)


def evaluate(run_dir: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, object]:
    """Evaluate selected and both observational checkpoints on val/test without key collisions."""
    variant = args._active_variant
    seed = int(args._active_seed)
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if (
            cached.get("protocol/version") == PROTOCOL_VERSION
            and cached.get(f"suite/{SUITE_NAME}") == 1.0
            and cached.get("variant") == variant
            and int(cached.get("seed")) == seed
        ):
            return cached

    workflow.local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, object] = {}
    selected_path = run_dir / "weights/best.pt"
    matching_shadow_name = (
        "best_ap50.pt" if FACTORS[variant].fitness_metric == "map50" else "best_map50_95.pt"
    )
    matching_shadow_path = run_dir / "weights" / matching_shadow_name
    _require(
        _sha256_file(selected_path) == _sha256_file(matching_shadow_path),
        variant,
        f"best.pt bytes differ from fitness-matched shadow {matching_shadow_name}",
    )
    metrics["checkpoint/selected_shadow_identical"] = 1.0
    selected_model = None
    for checkpoint_name, filename in CHECKPOINTS.items():
        checkpoint = run_dir / "weights" / filename
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        trained_model = YOLO(checkpoint)
        if checkpoint_name == "selected":
            selected_model = trained_model
            model_preflight(trained_model, variant)
        for split in ("val", "test"):
            result = trained_model.val(
                data=str(data_yaml),
                split=split,
                imgsz=args.imgsz,
                batch=args.batch_size,
                device=args.device,
                workers=args.workers,
                plots=False,
                iou=0.5,
                project=str(run_dir / "evaluation" / checkpoint_name),
                name=split,
                exist_ok=True,
            )
            prefix = f"{checkpoint_name}/{split}"
            metrics.update({f"{prefix}/{key}": float(value) for key, value in result.results_dict.items()})
            metrics[f"{prefix}/metrics/mAP75(B)"] = float(result.box.map75)

    dual = json.loads((run_dir / "dual_checkpoint_metadata.json").read_text(encoding="utf-8"))
    metrics.update(
        {
            **_case_metadata(variant, seed, args, run_dir),
            **_results_csv_metadata(run_dir / "results.csv"),
            **_ftsc_metadata(selected_model, variant),
            "protocol/best_ap50_epoch": dual["best_ap50_epoch"],
            "protocol/best_map50_95_epoch": dual["best_map50_95_epoch"],
            "protocol/best_ap50_value": dual["best_ap50_value"],
            "protocol/best_map50_95_value": dual["best_map50_95_value"],
            "checkpoint_delta_test_ap50": (
                metrics["best_ap50/test/metrics/mAP50(B)"]
                - metrics["best_map50_95/test/metrics/mAP50(B)"]
            ),
            "checkpoint_epoch_gap": dual["best_ap50_epoch"] - dual["best_map50_95_epoch"],
        }
    )
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _metric_deltas(left: dict[str, object], right: dict[str, object]) -> dict[str, float]:
    output = {}
    for split in ("val", "test"):
        for metric in PRIMARY_METRICS:
            key = f"selected/{split}/{metric}"
            output[f"delta/{split}/{metric}"] = float(left[key]) - float(right[key])
    return output


def write_factorial_analysis(args: argparse.Namespace) -> None:
    """Write requested matched contrasts whenever their seed-matched inputs exist."""
    records: dict[tuple[str, int], dict[str, object]] = {}
    for variant in args.variants:
        for seed in args.seeds:
            path = args.project / variant / f"seed_{seed}" / "evaluation_metrics.json"
            if path.is_file():
                records[(variant, seed)] = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    ordinary = (
        ("A1_minus_A0", "A1_current_y4_musgd", "A0_current_y0_musgd"),
        ("B0_minus_A0", "B0_y0_sgd", "A0_current_y0_musgd"),
        ("B1_minus_A1", "B1_y4_sgd", "A1_current_y4_musgd"),
        ("B1_minus_B0", "B1_y4_sgd", "B0_y0_sgd"),
        ("H1_minus_B1", "H1_y4_fixed1_sgd_legacyfit", "B1_y4_sgd"),
    )
    for seed in args.seeds:
        for contrast, left_name, right_name in ordinary:
            left, right = records.get((left_name, seed)), records.get((right_name, seed))
            if left is not None and right is not None:
                rows.append(
                    {"contrast": contrast, "seed": seed, "left": left_name, "right": right_name,
                     **_metric_deltas(left, right)}
                )
        needed = {
            name: records.get((name, seed))
            for name in ("A0_current_y0_musgd", "A1_current_y4_musgd", "B0_y0_sgd", "B1_y4_sgd")
        }
        if all(record is not None for record in needed.values()):
            interaction = {}
            for split in ("val", "test"):
                for metric in PRIMARY_METRICS:
                    key = f"selected/{split}/{metric}"
                    interaction[f"delta/{split}/{metric}"] = (
                        float(needed["B1_y4_sgd"][key])
                        - float(needed["B0_y0_sgd"][key])
                        - float(needed["A1_current_y4_musgd"][key])
                        + float(needed["A0_current_y0_musgd"][key])
                    )
            rows.append({"contrast": "optimizer_ftsc_interaction", "seed": seed, **interaction})
        historical = records.get(("H1_y4_fixed1_sgd_legacyfit", seed))
        if seed == 42 and historical is not None:
            rows.append(
                {
                    "contrast": "H1_minus_historical_y4_seed42",
                    "seed": seed,
                    "delta/test/metrics/mAP50(B)": (
                        float(historical["selected/test/metrics/mAP50(B)"]) - HISTORICAL_Y4_TEST_AP50
                    ),
                    "delta/val/metrics/mAP50(B)": (
                        float(historical["selected/val/metrics/mAP50(B)"]) - HISTORICAL_Y4_VAL_AP50
                    ),
                    "reference/test/metrics/mAP50(B)": HISTORICAL_Y4_TEST_AP50,
                    "reference/val/metrics/mAP50(B)": HISTORICAL_Y4_VAL_AP50,
                }
            )
    if rows:
        workflow.write_csv(args.project / "factorial_analysis.csv", rows)


def write_outputs(args: argparse.Namespace) -> None:
    workflow.write_summaries(args)
    write_factorial_analysis(args)
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            path = args.project / variant / f"seed_{seed}" / "evaluation_metrics.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    if rows:
        output = args.project / "ftsc_optimizer_provenance_audit_results.json"
        output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "LevirShipData")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / f"runs/{EXPERIMENT_SLUG}")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--augmentation-policy", choices=("yolo_default",), default="yolo_default")
    parser.add_argument("--smoke-fraction", type=float, default=0.01)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO)
    args = parser.parse_args(argv)
    args.runner = Path(__file__)
    return args


def preflight_suite(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """Load identical pretrained states, validate models, and resolve all optimizers before data/training."""
    _require(list(args.variants) == list(VARIANTS), "suite", "all five variants must run in canonical order")
    pretrained_path = resolve_local_pretrained(args.pretrained)
    load_pretrained = pretrained_path.is_file()
    if not load_pretrained and not args.preflight_only:
        raise FileNotFoundError(
            f"Pretrained checkpoint must already exist locally for provenance hashing: {pretrained_path}"
        )
    if load_pretrained:
        models = {variant: model_for(variant, str(pretrained_path)) for variant in args.variants}
        init_source = "pretrained"
    else:
        workflow.local_ultralytics()
        import torch
        from ultralytics import YOLO

        models = {}
        for variant in args.variants:
            torch.manual_seed(0)
            models[variant] = YOLO(VARIANTS[variant])
            model_preflight(models[variant], variant)
        init_source = "deterministic_config_init_preflight_only"
    initialization = initialization_hashes(models)
    reports = {}
    for variant, model in models.items():
        reports[variant] = {
            **model_preflight(model, variant),
            **optimizer_preflight(model, variant, args),
            **initialization[variant],
            "init/source": init_source,
            "fitness_metric": FACTORS[variant].fitness_metric,
        }
    return reports


def _configure_workflow(args: argparse.Namespace) -> None:
    workflow.EXPERIMENT = EXPERIMENT_SLUG
    workflow.HF_REPO = args.hf_repo_id
    workflow.VARIANTS = VARIANTS


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.project = args.project.resolve()
    _require(args.split_seed == 42, "protocol", "split_seed must remain 42")
    _configure_workflow(args)

    args._pretrained_provenance = pretrained_provenance(args.pretrained)
    reports = preflight_suite(args)
    args._optimizer_reports = {
        variant: {key: value for key, value in report.items() if key.startswith("optimizer/")}
        for variant, report in reports.items()
    }
    args._initialization_reports = {
        variant: {key: value for key, value in report.items() if key.startswith("init/")}
        for variant, report in reports.items()
    }
    for variant, report in reports.items():
        print(f"Preflight passed: {variant}: {report}", flush=True)

    if args.preflight_only:
        return

    data_yaml = workflow.prepare_fixed_split(args)
    args._dataset_provenance = dataset_provenance(data_yaml, args.split_seed)
    args._git_provenance = git_provenance()
    uploader = None if args.no_upload or args.smoke_only else workflow.Uploader(args)
    amp = {variant: True for variant in args.variants}
    if not args.no_smoke:
        raise RuntimeError("This audit requires --no-smoke; smoke trajectories would not create comparable artifacts")
    if args.smoke_only:
        return

    for seed in args.seeds:
        for variant in args.variants:
            args._active_variant = variant
            args._active_seed = seed
            args._active_amp = amp[variant]
            run_dir = train(variant, seed, data_yaml, amp[variant], args)
            evaluate(run_dir, data_yaml, args)
            write_outputs(args)
            if uploader:
                uploader.upload_run(run_dir, variant, seed)
                uploader.upload_metadata(args, data_yaml)


if __name__ == "__main__":
    main()
