"""Tests for the five-case FTSC optimizer and checkpoint-provenance audit."""

from pathlib import Path
import csv
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_all_levir_yolov8n_p2_ftsc_optimizer_provenance_audit as runner
from ultralytics.engine.trainer import fitness_from_metrics
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import EarlyStopping


EXPECTED_VARIANTS = [
    "A0_current_y0_musgd",
    "A1_current_y4_musgd",
    "B0_y0_sgd",
    "B1_y4_sgd",
    "H1_y4_fixed1_sgd_legacyfit",
]


def _model(variant: str):
    torch.manual_seed(0)
    return DetectionModel(runner.VARIANTS[variant], verbose=False)


def test_exact_variants_factor_mapping_and_locked_defaults():
    args = runner.parse_args([])

    assert list(runner.VARIANTS) == EXPECTED_VARIANTS
    assert list(runner.FACTORS) == EXPECTED_VARIANTS
    assert args.seeds == [42]
    assert args.split_seed == 42
    assert args.epochs == 1000
    assert args.imgsz == 512
    assert args.batch_size == 8
    assert args.workers == 4
    assert args.device == "cuda"
    assert args.patience == 40
    assert args.augmentation_policy == "yolo_default"

    assert [runner.FACTORS[name].optimizer for name in EXPECTED_VARIANTS] == [
        "auto", "auto", "SGD", "SGD", "SGD"
    ]
    assert [runner.FACTORS[name].fitness_metric for name in EXPECTED_VARIANTS] == [
        "map50", "map50", "map50", "map50", "map50_95"
    ]
    for name in EXPECTED_VARIANTS[2:]:
        assert runner.FACTORS[name].lr0 == 0.01
        assert runner.FACTORS[name].momentum == 0.9


def test_train_kwargs_apply_only_the_requested_optimizer_and_fitness_factors():
    args = runner.parse_args([])
    for variant in EXPECTED_VARIANTS:
        kwargs = runner.train_kwargs(args, variant, Path("data.yaml"), seed=42, amp=True)
        spec = runner.FACTORS[variant]
        assert kwargs["optimizer"] == spec.optimizer
        assert kwargs["lr0"] == spec.lr0
        assert kwargs["momentum"] == spec.momentum
        assert kwargs["fitness_metric"] == spec.fitness_metric
        assert kwargs["deterministic"] is True
        assert kwargs["epochs"] == 1000
        assert kwargs["patience"] == 40


def test_model_preflight_covers_y0_learned_y4_and_fixed1_semantics():
    reports = {variant: runner.model_preflight(_model(variant), variant) for variant in EXPECTED_VARIANTS}

    assert reports["A0_current_y0_musgd"]["ftsc_mode"] == "disabled"
    assert reports["B0_y0_sgd"]["ftsc_mode"] == "disabled"
    assert reports["A1_current_y4_musgd"]["ftsc_mode"] == "learned"
    assert reports["B1_y4_sgd"]["ftsc_mode"] == "learned"
    fixed = reports["H1_y4_fixed1_sgd_legacyfit"]
    assert fixed["ftsc_mode"] == "fixed1"
    assert fixed["dfl_strength_mode"] == "fixed"
    assert fixed["dfl_strength_trainable"] is False
    assert fixed["dfl_strength_effective"] == 1.0
    assert fixed["position_strength_trainable"] is True


def test_initialization_equivalence_for_optimizer_pairs_and_fixed1_shared_state():
    models = {variant: _model(variant) for variant in EXPECTED_VARIANTS}
    reports = runner.initialization_hashes(models)

    assert (
        reports["A0_current_y0_musgd"]["init/full_state_sha256"]
        == reports["B0_y0_sgd"]["init/full_state_sha256"]
    )
    assert (
        reports["A1_current_y4_musgd"]["init/full_state_sha256"]
        == reports["B1_y4_sgd"]["init/full_state_sha256"]
    )
    assert (
        reports["B1_y4_sgd"]["init/shared_state_sha256"]
        == reports["H1_y4_fixed1_sgd_legacyfit"]["init/shared_state_sha256"]
    )
    assert (
        reports["B1_y4_sgd"]["init/full_state_sha256"]
        != reports["H1_y4_fixed1_sgd_legacyfit"]["init/full_state_sha256"]
    )


def test_optimizer_preflight_constructs_exact_musgd_and_sgd_classes():
    args = runner.parse_args([])
    assert runner.estimated_iterations(args.epochs, args.batch_size) == 37000

    for variant in EXPECTED_VARIANTS:
        report = runner.optimizer_preflight(_model(variant), variant, args)
        expected = "MuSGD" if runner.FACTORS[variant].optimizer == "auto" else "SGD"
        assert report["optimizer/requested"] == runner.FACTORS[variant].optimizer
        assert report["optimizer/resolved"] == expected
        assert report["optimizer/lr0"] == 0.01
        assert report["optimizer/momentum"] == 0.9
        assert report["optimizer/weight_decay"] == pytest.approx(0.0005)
        assert report["optimizer/estimated_iterations"] == 37000


def test_fixed1_strength_is_not_an_optimizer_parameter():
    model = _model("H1_y4_fixed1_sgd_legacyfit")
    args = runner.parse_args([])
    report = runner.optimizer_preflight(model, "H1_y4_fixed1_sgd_legacyfit", args)
    calibrator = model.model[-1].ftsc_calibrator

    assert report["optimizer/resolved"] == "SGD"
    assert "dfl_distribution" not in calibrator.strength_logits
    assert all("strength_logits.dfl_distribution" not in name for name, _ in model.named_parameters())
    reference = calibrator.strength_logits["position_gaussian"]
    assert calibrator.strength("dfl_distribution", reference).item() == 1.0


def test_dual_checkpoint_manager_tracks_independent_maxima(tmp_path):
    run_dir = tmp_path / "run"
    source = run_dir / "weights" / "last.pt"
    source.parent.mkdir(parents=True)
    selected = run_dir / "weights" / "best.pt"
    selected.write_bytes(b"selected-best-semantics")
    manager = runner.DualCheckpointManager(run_dir, "map50")
    sequence = [(0.70, 0.30), (0.72, 0.29), (0.71, 0.32)]

    for epoch, (ap50, map50_95) in enumerate(sequence, 1):
        source.write_bytes(f"epoch-{epoch}".encode())
        manager.observe(
            epoch,
            {"metrics/mAP50(B)": ap50, "metrics/mAP50-95(B)": map50_95},
            source,
        )

    assert manager.state["best_ap50_epoch"] == 2
    assert manager.state["best_map50_95_epoch"] == 3
    assert manager.state["best_ap50_value"] == pytest.approx(0.72)
    assert manager.state["best_map50_95_value"] == pytest.approx(0.32)
    assert (run_dir / "weights/best_ap50.pt").read_bytes() == b"epoch-2"
    assert (run_dir / "weights/best_map50_95.pt").read_bytes() == b"epoch-3"
    assert selected.read_bytes() == b"selected-best-semantics"
    assert manager.state["selected_fitness_metric"] == "map50"


@pytest.mark.parametrize(
    ("fitness_metric", "matching_shadow"),
    [("map50", "best_ap50.pt"), ("map50_95", "best_map50_95.pt")],
)
def test_dual_audit_preserves_normal_control_flow_and_matches_latest_tie(
    tmp_path, fitness_metric, matching_shadow
):
    sequence = [(0.70, 0.30), (0.72, 0.29), (0.72, 0.32), (0.71, 0.31), (0.70, 0.30)]

    def simulate(run_dir, audit_enabled):
        weights = run_dir / "weights"
        weights.mkdir(parents=True)
        last = weights / "last.pt"
        best = weights / "best.pt"
        manager = runner.DualCheckpointManager(run_dir, fitness_metric) if audit_enabled else None
        best_fitness = None
        best_trajectory = []
        fitness_trajectory = []
        selected_epoch = None
        stop_epoch = None
        stopper = EarlyStopping(patience=2)
        for epoch, (ap50, map50_95) in enumerate(sequence, 1):
            metrics = {
                "metrics/mAP50(B)": ap50,
                "metrics/mAP50-95(B)": map50_95,
                "fitness": map50_95,
            }
            fitness = fitness_from_metrics(dict(metrics), fitness_metric)
            fitness_trajectory.append(fitness)
            if not best_fitness or best_fitness < fitness:
                best_fitness = fitness
            last.write_bytes(f"epoch-{epoch}".encode())
            if best_fitness == fitness:
                best.write_bytes(last.read_bytes())
                selected_epoch = epoch
            if manager is not None:
                manager.observe(epoch, metrics, last)
            best_trajectory.append(best_fitness)
            if stopper(epoch, fitness):
                stop_epoch = epoch
                break
        return {
            "best_fitness": best_fitness,
            "best_trajectory": best_trajectory,
            "fitness_trajectory": fitness_trajectory,
            "selected_epoch": selected_epoch,
            "stop_epoch": stop_epoch,
            "best_bytes": best.read_bytes(),
        }

    without_audit = simulate(tmp_path / "off", False)
    with_audit = simulate(tmp_path / "on", True)

    assert with_audit == without_audit
    assert (tmp_path / "on/weights" / matching_shadow).read_bytes() == with_audit["best_bytes"]
    assert {path.name for path in (tmp_path / "off/weights").iterdir()} == {"best.pt", "last.pt"}
    assert {path.name for path in (tmp_path / "on/weights").iterdir()} == {
        "best.pt", "last.pt", "best_ap50.pt", "best_map50_95.pt"
    }
    assert {path.name for path in (tmp_path / "on").iterdir()} - {"weights"} == {
        "dual_checkpoint_metadata.json"
    }
    metadata = json.loads((tmp_path / "on/dual_checkpoint_metadata.json").read_text(encoding="utf-8"))
    matching_epoch = "best_ap50_epoch" if fitness_metric == "map50" else "best_map50_95_epoch"
    assert metadata[matching_epoch] == with_audit["selected_epoch"]


def test_shadow_save_does_not_advance_rng_or_invoke_validation_or_model(tmp_path):
    run_dir = tmp_path / "run"
    last = run_dir / "weights/last.pt"
    best = run_dir / "weights/best.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"serialized-epoch")
    best.write_bytes(b"normal-selected")
    manager = runner.DualCheckpointManager(run_dir, "map50")
    calls = {"validation": 0, "forward": 0}

    def validator():
        calls["validation"] += 1

    def model():
        calls["forward"] += 1

    trainer = SimpleNamespace(
        epoch=0,
        metrics={"metrics/mAP50(B)": 0.7, "metrics/mAP50-95(B)": 0.3},
        last=last,
        validator=validator,
        model=model,
        best_fitness=0.7,
        fitness=0.7,
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    manager.on_model_save(trainer)

    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert calls == {"validation": 0, "forward": 0}
    assert trainer.best_fitness == 0.7
    assert trainer.fitness == 0.7
    assert last.read_bytes() == b"serialized-epoch"
    assert best.read_bytes() == b"normal-selected"


def test_resume_restores_shadow_history_and_rejects_worse_later_epoch(tmp_path):
    run_dir = tmp_path / "run"
    last = run_dir / "weights/last.pt"
    last.parent.mkdir(parents=True)
    manager = runner.DualCheckpointManager(run_dir, "map50")
    last.write_bytes(b"epoch-1")
    manager.observe(1, {"metrics/mAP50(B)": 0.72, "metrics/mAP50-95(B)": 0.32}, last)

    resumed = runner.DualCheckpointManager(run_dir, "map50")
    last.write_bytes(b"epoch-2-worse")
    assert resumed.observe(
        2, {"metrics/mAP50(B)": 0.71, "metrics/mAP50-95(B)": 0.31}, last
    ) == ()

    assert resumed.state["best_ap50_epoch"] == 1
    assert resumed.state["best_map50_95_epoch"] == 1
    assert (run_dir / "weights/best_ap50.pt").read_bytes() == b"epoch-1"
    assert (run_dir / "weights/best_map50_95.pt").read_bytes() == b"epoch-1"


def test_pretrained_and_dataset_provenance_are_deterministic(tmp_path):
    checkpoint = tmp_path / "yolov8n.pt"
    checkpoint.write_bytes(b"fixed checkpoint bytes")
    first = runner.pretrained_provenance(checkpoint)
    second = runner.pretrained_provenance(checkpoint)
    assert first == second
    assert first["pretrained/bytes"] == len(b"fixed checkpoint bytes")
    assert len(first["pretrained/sha256"]) == 64

    dataset = tmp_path / "dataset"
    for split, stems in {"train": ["b", "a"], "val": ["c"], "test": ["d"]}.items():
        folder = dataset / "images" / split
        folder.mkdir(parents=True)
        for stem in stems:
            (folder / f"{stem}.png").touch()
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset}\ntrain: images/train\nval: images/val\ntest: images/test\n",
        encoding="utf-8",
    )
    provenance = runner.dataset_provenance(data_yaml, 42)
    assert provenance["dataset/train_count"] == 2
    assert provenance["dataset/val_count"] == 1
    assert provenance["dataset/test_count"] == 1
    assert provenance["dataset/split_seed"] == 42
    assert provenance == runner.dataset_provenance(data_yaml, 42)


def test_factorial_analysis_writes_all_requested_seed42_contrasts(tmp_path):
    args = runner.parse_args([])
    args.project = tmp_path
    values = {
        "A0_current_y0_musgd": 0.70,
        "A1_current_y4_musgd": 0.71,
        "B0_y0_sgd": 0.72,
        "B1_y4_sgd": 0.76,
        "H1_y4_fixed1_sgd_legacyfit": 0.78,
    }
    for variant, base in values.items():
        run_dir = tmp_path / variant / "seed_42"
        run_dir.mkdir(parents=True)
        metrics = {}
        for split in ("val", "test"):
            for offset, metric in enumerate(runner.PRIMARY_METRICS):
                metrics[f"selected/{split}/{metric}"] = base + offset / 100
        (run_dir / "evaluation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    runner.write_factorial_analysis(args)

    with (tmp_path / "factorial_analysis.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["contrast"] for row in rows} == {
        "A1_minus_A0",
        "B0_minus_A0",
        "B1_minus_A1",
        "B1_minus_B0",
        "optimizer_ftsc_interaction",
        "H1_minus_B1",
        "H1_minus_historical_y4_seed42",
    }
    interaction = next(row for row in rows if row["contrast"] == "optimizer_ftsc_interaction")
    assert float(interaction["delta/test/metrics/mAP50(B)"]) == pytest.approx((0.76 - 0.72) - (0.71 - 0.70))


def test_evaluation_keeps_three_checkpoint_namespaces_and_exports_metadata(monkeypatch, tmp_path):
    network = _model("A0_current_y0_musgd")

    class FakeResult:
        def __init__(self, ap50):
            self.results_dict = {
                "metrics/precision(B)": 0.8,
                "metrics/recall(B)": 0.7,
                "metrics/mAP50(B)": ap50,
                "metrics/mAP50-95(B)": 0.3,
            }
            self.box = SimpleNamespace(map75=0.2)

    class FakeYOLO:
        def __init__(self, checkpoint):
            self.model = network
            self.checkpoint = Path(checkpoint).name

        def val(self, **_):
            ap50 = {"best.pt": 0.70, "best_ap50.pt": 0.72, "best_map50_95.pt": 0.69}[self.checkpoint]
            return FakeResult(ap50)

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    run_dir = tmp_path / "A0_current_y0_musgd" / "seed_42"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    for filename in runner.CHECKPOINTS.values():
        (weights / filename).touch()
    (run_dir / "dual_checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "selected_fitness_metric": "map50",
                "best_ap50_epoch": 2,
                "best_ap50_value": 0.72,
                "best_map50_95_epoch": 3,
                "best_map50_95_value": 0.32,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.7\n3,0.69\n", encoding="utf-8")

    args = runner.parse_args([])
    args.project = tmp_path
    args._active_variant = "A0_current_y0_musgd"
    args._active_seed = 42
    args._active_amp = True
    args._optimizer_reports = {
        "A0_current_y0_musgd": {
            "optimizer/requested": "auto",
            "optimizer/resolved": "MuSGD",
            "optimizer/lr0": 0.01,
            "optimizer/momentum": 0.9,
            "optimizer/weight_decay": 0.0005,
            "optimizer/estimated_iterations": 37000,
        }
    }
    args._initialization_reports = {"A0_current_y0_musgd": {"init/shared_state_sha256": "abc"}}
    args._pretrained_provenance = {"pretrained/sha256": "def"}
    args._dataset_provenance = {"dataset/all_stem_sha256": "ghi"}
    args._git_provenance = {"git/commit": "123", "git/branch": "main", "ultralytics/version": "test"}

    metrics = runner.evaluate(run_dir, Path("data.yaml"), args)

    assert metrics["selected/test/metrics/mAP50(B)"] == pytest.approx(0.70)
    assert metrics["best_ap50/test/metrics/mAP50(B)"] == pytest.approx(0.72)
    assert metrics["best_map50_95/test/metrics/mAP50(B)"] == pytest.approx(0.69)
    assert metrics["checkpoint_delta_test_ap50"] == pytest.approx(0.03)
    assert metrics["checkpoint_epoch_gap"] == -1
    assert metrics["training/actual_stop_epoch"] == 3
    assert metrics["protocol/best_checkpoint_metric"] == "map50"
    assert metrics["optimizer/resolved"] == "MuSGD"
