"""Focused regression tests for LEVIR FTSC Position Gaussian alpha sensitivity.

Prepared for live/test CI execution; intentionally NOT executed during local static review.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from train_levir_scripts import train_levir_ftsc_position_alpha_sensitivity_v1 as alpha
from ultralytics.nn.modules.ftsc import PositionGaussianEvidence


def fake_args(tmp_path: Path):
    return SimpleNamespace(
        cases=["ALPHA4_rerun"], seeds=[42], split_seed=42, epochs=100, imgsz=512,
        batch_size=8, patience=20, workers=4, device="cuda",
        data_root=tmp_path, dataset_root=tmp_path, project=tmp_path / "runs",
        pretrained=tmp_path / "yolov8n.pt", confirm_run=True,
        aggregate_only=False, preflight_only=False, live_preflight=False,
        allow_alpha6_bridge=False,
    )


def test_fixed_alpha6_preserves_historical_formula():
    points = torch.tensor([[2., 3.], [7., 7.]])
    boxes = torch.tensor([[[0., 0., 10., 10.], [0., 0., 10., 10.]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    observed = PositionGaussianEvidence(alpha=6.)(points, boxes, mask)
    expected = -0.5 * (((points - 5.) / (10. / 6.)) ** 2).sum(-1)
    torch.testing.assert_close(observed, expected)


@pytest.mark.parametrize("alpha", [4.0, 6.0, 8.0])
def test_fixed_alpha_numeric(alpha):
    points = torch.tensor([[7., 5.]])
    boxes = torch.tensor([[[0., 0., 10., 10.]]])
    value = PositionGaussianEvidence(alpha=alpha)(points, boxes, torch.ones(1, 1, dtype=torch.bool))
    assert value.item() == pytest.approx(-0.5 * (2 * alpha / 10) ** 2)


@pytest.mark.parametrize("side,expected", [(1.0, 4.), (7.999, 4.), (8., 6.),
                                            (15.999, 6.), (16., 8.), (31.999, 8.),
                                            (32., 10.), (64., 10.)])
def test_adaptive_boundaries(side, expected):
    provider = PositionGaussianEvidence(alpha_mode="size_adaptive")
    boxes = torch.tensor([[0., 0., side, side]])
    assert provider._adaptive_alpha(boxes).item() == pytest.approx(expected)


def test_adaptive_multiple_gt_and_batch_indices():
    provider = PositionGaussianEvidence(alpha_mode="size_adaptive")
    # First two positives share GT; a second batch image has a different GT.
    boxes = torch.tensor([[
        [0., 0., 7., 7.], [0., 0., 7., 7.], [0., 0., 15., 15.],
    ], [
        [0., 0., 33., 33.], [0., 0., 33., 33.], [0., 0., 33., 33.],
    ]])
    mask = torch.tensor([[True, True, True], [True, False, False]])
    points = torch.tensor([[3., 3.], [4., 4.], [7., 7.]])
    result = provider(points, boxes, mask)
    assert result.shape == (4,)
    assert provider.last_alpha.tolist() == [4., 4., 6., 10.]


def test_invalid_boxes_fail_closed_and_valid_box_unchanged():
    provider = PositionGaussianEvidence(alpha_mode="size_adaptive")
    point = torch.tensor([[5., 5.]])
    for bad in ([0., 0., 0., 5.], [float("nan"), 0., 8., 8.],
                [0., 0., float("inf"), 8.]):
        with pytest.raises(ValueError, match="invalid matched GT"):
            provider(point, torch.tensor([[bad]]), torch.ones(1, 1, dtype=torch.bool))
    assert torch.isfinite(provider(point, torch.tensor([[[0., 0., 8., 8.]]]),
                                   torch.ones(1, 1, dtype=torch.bool))).all()


def test_zero_positives_return_empty():
    provider = PositionGaussianEvidence(alpha_mode="size_adaptive")
    out = provider(torch.tensor([[5., 5.]]), torch.tensor([[[0., 0., 8., 8.]]]),
                   torch.zeros(1, 1, dtype=torch.bool))
    assert out.numel() == 0


def test_train_kwargs_match_exact_historical_head_pruning(tmp_path):
    args = fake_args(tmp_path)
    data_yaml = tmp_path / "dataset.yaml"
    kwargs = alpha.train_kwargs(args, data_yaml, 42)
    assert kwargs == alpha.historical.train_kwargs(args, data_yaml, 42, True)
    assert kwargs["optimizer"] == "auto"
    assert "lr0" not in kwargs  # historical auto decides effective AdamW
    assert kwargs["amp"] is True and kwargs["mosaic"] == 0.0
    assert kwargs["fitness_metric"] == "map50_95"


def test_preflight_does_not_launch_training(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    args.confirm_run = False
    monkeypatch.setattr(alpha, "source_preflight",
                        lambda unused: {"source": {"state": "UNAVAILABLE"}})
    monkeypatch.setattr(alpha, "audit_train_labels", lambda unused: {"status": "PENDING_DATA_PREPARATION"})
    monkeypatch.setattr(alpha.workflow, "prepare_fixed_split",
                        lambda unused: pytest.fail("Preflight must not prepare data"))
    result = alpha.run_suite(args)
    assert result["status"] == "PREPARED — NOT YET RUN"


def test_training_marker_uses_finalized_checkpoint_contract(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    for name in alpha.TRAIN_ARTIFACTS:
        file = run / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("trained\n", encoding="utf-8")
    (run / "results.csv").write_text("epoch,train/box_loss\n1,0.1\n20,0.01\n", encoding="utf-8")
    alpha.write_json(run / "run_contract.json", {"case": "ALPHA4_rerun", "seed": 42})
    expected = {"training_seed": 42, "train_args": {"epochs": 100}}
    monkeypatch.setattr(alpha, "_checkpoint",
                        lambda path, require_resumable: {"epoch": -1, "optimizer": None, "train_results": {"epoch": [1, 20]}})
    monkeypatch.setattr(alpha, "_checkpoint_protocol", lambda *args, **kwargs: None)
    model = SimpleNamespace(trainer=SimpleNamespace(
        epoch=19, stopper=SimpleNamespace(best_epoch=0)))
    alpha._finalize_training(run, expected, model)
    assert alpha.validate_training_completion(run, expected)
    assert alpha.read_json(run / alpha.TRAINING_FINALIZATION)["completion_reason"] == "early_stopped"


def test_incomplete_training_with_all_three_artifacts_must_resume(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    run = args.project / "ALPHA4_rerun" / "seed_42"
    (run / "weights").mkdir(parents=True)
    expected = {"case": "ALPHA4_rerun", "train_args": {"epochs": 100}}
    for artifact in alpha.TRAIN_ARTIFACTS:
        path = run / artifact
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("partial", encoding="utf-8")
    alpha.write_json(run / "run_contract.json", expected)
    alpha.write_json(run / "experiment_manifest.json", expected)
    monkeypatch.setattr(alpha, "contract", lambda *unused: expected)
    monkeypatch.setattr(alpha, "validate_training_completion", lambda *unused: False)
    monkeypatch.setattr(alpha, "_resume", lambda *unused: pytest.fail("Resume branch reached safely"))
    with pytest.raises(pytest.fail.Exception, match="Resume branch reached"):
        alpha.run_one(args, "ALPHA4_rerun", 42, tmp_path / "dataset.yaml", {}, {"state": "VERIFIED"})


def test_completed_run_detects_tampered_checkpoint(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for artifact in alpha.TRAIN_ARTIFACTS + alpha.EVALUATION_ARTIFACTS + alpha.IDENTITY_ARTIFACTS:
        target = run / artifact
        target.parent.mkdir(parents=True, exist_ok=True)
        if artifact in ("run_contract.json", "experiment_manifest.json"):
            alpha.write_json(target, {"case": "ALPHA4_rerun"})
        elif artifact == "evaluation_metrics_extended.json":
            metrics = {key: 0.5 for key in alpha.METRICS_REQUIRED}
            metrics.update({"evaluation/checkpoint": "weights/best.pt",
                            "evaluation/dataset": "levir", "evaluation/nms_iou": .5,
                            "benchmark/imgsz": 512, "benchmark/batch": 8})
            alpha.write_json(target, metrics)
        else:
            target.write_text("test", encoding="utf-8")
    fingerprints = {item: alpha.sha(run / item) for item in
                    alpha.TRAIN_ARTIFACTS + alpha.EVALUATION_ARTIFACTS + alpha.IDENTITY_ARTIFACTS}
    alpha.write_json(run / "completion.json",
                     {"state": "COMPLETE", "contract_sha256": alpha.sha(run / "run_contract.json"),
                      "artifact_sha256": fingerprints, "evaluated_best_sha256": alpha.sha(run / "weights/best.pt")})
    monkeypatch.setattr(alpha, "validate_training_completion", lambda *unused: True)
    assert alpha.validate_completed_run(run, {"case": "ALPHA4_rerun"})
    (run / "weights/best.pt").write_text("TAMPERED", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ARTIFACT_HASH_MISMATCH"):
        alpha.validate_completed_run(run, {"case": "ALPHA4_rerun"})


def test_aggregate_invokes_authoritative_validator(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    run = args.project / "ALPHA4_rerun" / "seed_42"
    run.mkdir(parents=True)
    (run / "completion.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(alpha, "contract", lambda *unused: {"case": "ALPHA4_rerun"})
    def reject(*args):
        raise RuntimeError("ARTIFACT_HASH_MISMATCH")
    monkeypatch.setattr(alpha, "validate_completed_run", reject)
    with pytest.raises(RuntimeError, match="ARTIFACT_HASH_MISMATCH"):
        alpha.aggregate(args, tmp_path / "dataset.yaml", {}, {"state": "VERIFIED"})


def test_required_full_levir_metric_is_mandatory():
    incomplete = {"test/metrics/mAP50(B)": 0.5}
    with pytest.raises(RuntimeError, match="Missing LEVIR metric"):
        alpha.validate_metrics(incomplete)


def test_alpha_case_mapping_locked():
    assert alpha.ALPHA_POLICY == {
        "ALPHA4_rerun": ("fixed", 4.0),
        "ALPHA6_rerun": ("fixed", 6.0),
        "ALPHA8": ("fixed", 8.0),
        "ALPHA_ADAPTIVE": ("size_adaptive", 6.0),
    }


def test_adaptive_fresh_dataset_is_prepared_before_audit(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    args.cases = ["ALPHA_ADAPTIVE"]
    args.pretrained.write_bytes(b"dummy official checkpoint fixture")
    calls = []
    monkeypatch.setattr(alpha, "source_preflight",
                        lambda unused: {"source": {"state": "VERIFIED"}})
    def prepare(unused):
        calls.append("prepare")
        return tmp_path / "levir_ship.yaml"
    def audit(*unused):
        calls.append("audit")
        return {"status": "READY", "images": 2320}
    monkeypatch.setattr(alpha.workflow, "prepare_fixed_split", prepare)
    monkeypatch.setattr(alpha, "validate_dataset", lambda *unused: {"split_seed": 42})
    monkeypatch.setattr(alpha, "audit_train_labels", audit)
    monkeypatch.setattr(alpha, "model_preflight", lambda *unused: {"status": "MODEL_PREFLIGHT_PASSED"})
    monkeypatch.setattr(alpha, "run_one", lambda *unused: calls.append("train"))
    monkeypatch.setattr(alpha, "aggregate", lambda *unused: calls.append("aggregate"))
    alpha.run_suite(args)
    assert calls.index("prepare") < calls.index("audit") < calls.index("train")


def test_live_preflight_builds_models_but_never_trains(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    args.confirm_run = False
    args.live_preflight = True
    args.pretrained.write_bytes(b"synthetic")
    monkeypatch.setattr(alpha, "source_preflight",
                        lambda unused: {"source": {"state": "VERIFIED"}})
    monkeypatch.setattr(alpha.workflow, "prepare_fixed_split",
                        lambda unused: tmp_path / "dataset.yaml")
    monkeypatch.setattr(alpha, "validate_dataset",
                        lambda *unused: {"split_seed": 42})
    monkeypatch.setattr(alpha, "audit_train_labels",
                        lambda *unused: {"status": "READY", "images": 2320})
    monkeypatch.setattr(alpha, "model_preflight",
                        lambda *unused: {"status": "MODEL_PREFLIGHT_PASSED"})
    monkeypatch.setattr(alpha, "run_one",
                        lambda *unused: pytest.fail("Live preflight must not train"))
    result = alpha.run_suite(args)
    assert result["status"] == "LIVE_PREFLIGHT_PASSED_NO_TRAINING"


def test_notebook_gate_disabled_by_default():
    notebook = Path(__file__).resolve().parents[2] / "current_live_training_notebok_LEV.ipynb"
    # If tests run in a code-only deployment, notebook may be staged separately.
    if not notebook.exists():
        pytest.skip("External LEVIR live notebook not present in this checkout")
    body = notebook.read_text(encoding="utf-8")
    assert "RUN_LEVIR_FTSC_ALPHA = False" in body
    assert "run_suite(_alpha_args, uploader=_alpha_uploader)" in body
    assert "runs/levir_ftsc_position_alpha_historical_h2_v1" in body

def test_checkpoint_alpha_identity_rejects_wrong_case():
    wrong_model = SimpleNamespace(yaml={"ftsc": {"position_alpha": 6.0, "position_alpha_mode": "fixed"}})
    ckpt = {"epoch": -1, "optimizer": None, "model": wrong_model, "train_args": {}}
    expected = {"alpha_policy": {"mode": "fixed", "alpha": 8.0},
                "train_args": {"epochs": 100}}
    with pytest.raises(RuntimeError, match="CHECKPOINT_ALPHA_POLICY_MISMATCH"):
        alpha._checkpoint_protocol(ckpt, expected, finalized=True)


def test_source_identity_detects_dirty_tree(monkeypatch):
    monkeypatch.setattr(alpha, "sha", lambda unused: "fixed-synthetic-sha")
    def git_output(argv, **kwargs):
        return "deadbeef\n" if argv[1] == "rev-parse" else "?? science_patch.py\n"
    monkeypatch.setattr(alpha.subprocess, "check_output", git_output)
    report = alpha.source_identity()
    assert report["state"] == "DIRTY"
    assert report["dirty_paths_present"] is True


def test_restore_never_overwrites_partial_local_run(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    partial = args.project / "ALPHA4_rerun/seed_42"
    partial.mkdir(parents=True)
    (partial / "run_contract.json").write_text("{}", encoding="utf-8")
    class ForbiddenUploader:
        def restore_completed_runs(self, **kwargs):
            pytest.fail("Existing local partial run must never be overwritten")
    alpha.restore_missing(args, ForbiddenUploader(), tmp_path / "dataset.yaml", {},
                          {"state": "VERIFIED"})


def test_publication_validates_locally_before_any_upload(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    run = args.project / "ALPHA4_rerun/seed_42"
    run.mkdir(parents=True)
    monkeypatch.setattr(alpha, "contract", lambda *unused: {"case": "ALPHA4_rerun"})
    monkeypatch.setattr(alpha, "validate_completed_run",
                        lambda *unused: (_ for _ in ()).throw(RuntimeError("ARTIFACT_HASH_MISMATCH")))
    class ForbiddenUploader:
        def publish_run(self, **kwargs):
            pytest.fail("Corrupt runs must not be published")
    with pytest.raises(RuntimeError, match="ARTIFACT_HASH_MISMATCH"):
        alpha.publish_completed(args, ForbiddenUploader(), "ALPHA4_rerun", 42,
                                tmp_path / "dataset.yaml", {}, {"state": "VERIFIED"})


def test_read_only_aggregate_rejects_bad_completion(tmp_path, monkeypatch):
    args = fake_args(tmp_path)
    args.aggregate_only = True
    args.confirm_run = False
    args.pretrained.write_bytes(b"dummy")
    data_yaml = args.dataset_root / "levir_ship_yolo_seed42" / "levir_ship.yaml"
    data_yaml.parent.mkdir(parents=True)
    data_yaml.write_text("path: synthetic\n", encoding="utf-8")
    monkeypatch.setattr(alpha, "source_preflight", lambda unused: {"source": {"state": "VERIFIED"}})
    monkeypatch.setattr(alpha, "validate_dataset", lambda *unused: {"split_seed": 42})
    monkeypatch.setattr(alpha, "aggregate",
                        lambda *unused: (_ for _ in ()).throw(RuntimeError("CORRUPT_COMPLETION")))
    with pytest.raises(RuntimeError, match="CORRUPT_COMPLETION"):
        alpha.run_suite(args)


def test_historical_h2_is_default_control_not_train_case():
    args = alpha.parse_args([])
    assert args.cases == ["ALPHA4_rerun", "ALPHA8"]
    assert args.seeds == [42, 43, 44]
    assert "ALPHA6" not in args.cases
    assert args.allow_alpha6_bridge is False
    assert alpha.NAMESPACE == "levir_ftsc_position_alpha_historical_h2_v1"


def test_alpha6_train_requires_explicit_bridge(tmp_path):
    args = fake_args(tmp_path)
    args.cases = ["ALPHA6_rerun"]
    with pytest.raises(RuntimeError, match="Historical H2"):
        alpha.validate_args(args)
    with pytest.raises(RuntimeError, match="Historical H2"):
        alpha.run_one(args, "ALPHA6_rerun", 42, tmp_path / "dataset.yaml", {}, {"state": "VERIFIED"})
    args.allow_alpha6_bridge = True
    alpha.validate_args(args)


def test_historical_reference_is_separate_and_three_seeds():
    reference = alpha.historical_h2_reference([42, 43, 44])
    assert [row["seed"] for row in reference] == [42, 43, 44]
    assert all(row["reference_only"] for row in reference)
    assert all(row["cohort"] == "ftsc_head_pruning_nomosaic_v1" for row in reference)
    assert all(row["case"] == "HISTORICAL_H2_ALPHA6_REFERENCE" for row in reference)
    assert sum(row["test/metrics/mAP50-95(B)"] for row in reference) / 3 == pytest.approx(.3129952615407791)


def test_historical_reference_respects_selected_seeds():
    rows = alpha.historical_h2_reference([43])
    assert len(rows) == 1 and rows[0]["seed"] == 43


def test_alpha6_yaml_parity_with_historical_source():
    import yaml
    old = yaml.safe_load(alpha.H2.read_text(encoding="utf-8"))
    new = yaml.safe_load(alpha.YAMLS["ALPHA6_rerun"].read_text(encoding="utf-8"))
    assert {k: v for k, v in old.items() if k != "ftsc"} == {
        k: v for k, v in new.items() if k != "ftsc"
    }
    historic = dict(old["ftsc"])
    fixed = dict(new["ftsc"])
    assert fixed.pop("position_alpha_mode") == "fixed"
    assert fixed == historic
