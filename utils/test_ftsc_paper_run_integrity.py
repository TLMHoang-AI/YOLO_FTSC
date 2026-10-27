"""No-training integrity regression tests for the FTSC paper completion runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from utils.ftsc_paper_run_integrity import (
    RUN_CONTRACT, TRAINING_ATTEMPT, TRAINING_COMPLETION, TRANSFER_REPORT,
    atomic_json, finalize_training, make_contract, pretrained_identity,
    record_training_attempt, validate_completion, validate_training_attempt,
    validate_transfer_report, verify_or_create_contract,
)
from utils.ftsc_paper_preflight import sha256


def _contract(tmp_path: Path):
    config = Path(__file__).resolve().parents[1] / (
        "models_related/models_config/yolov8/tinyperson/"
        "yolov8n_tinyperson_ftsc_prune_h1_p234.yaml"
    )
    data_dir = tmp_path / "seed_data"
    data_dir.mkdir()
    data = data_dir / "tinyperson.yaml"
    data.write_text("train: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
    (data_dir / "corner_manifest.json").write_text('{"train":["one"],"val":["two"]}\n', encoding="utf-8")
    for split in ("train", "val", "test"):
        for group in ("images", "labels"):
            (data_dir / group / split).mkdir(parents=True)
        (data_dir / "images" / split / "sample.jpg").write_bytes(b"synthetic-image-bytes-" + split.encode())
        (data_dir / "labels" / split / "sample.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    pretrained = tmp_path / "yolov8n.pt"
    pretrained.write_bytes(b"pretend-checkpoint-only-for-identity-tests")
    contract = make_contract(
        dataset="TinyPerson", case="TP_PRUNE_H1_P234", seed=42, config=config,
        graph_contract="c2f_structural_p234_v1", expected_strides=(4.0, 8.0, 16.0),
        kwargs={"optimizer":"AdamW", "epochs":100, "mosaic":1, "project":"ignored",
                "data":str(data), "name":"ignored", "device":"cuda", "workers":4},
        data_yaml=data, pretrained_name=pretrained.name, pretrained_sha256=sha256(pretrained))
    return contract, config, data, pretrained


def test_contract_registration_rejects_untracked_and_changed_inputs(tmp_path):
    contract, config, data, pretrained = _contract(tmp_path)
    run_dir = tmp_path / "run"
    assert verify_or_create_contract(run_dir, contract) is False
    assert verify_or_create_contract(run_dir, contract) is True
    assert (run_dir / RUN_CONTRACT).is_file()
    assert pretrained_identity(pretrained, run_dir) == sha256(pretrained)
    changed = dict(contract, seed=43)
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_or_create_contract(run_dir, changed)
    (data.parent / "corner_manifest.json").write_text('{"train":["altered"]}', encoding="utf-8")
    changed_split = make_contract(
        dataset="TinyPerson", case="TP_PRUNE_H1_P234", seed=42, config=config,
        graph_contract="c2f_structural_p234_v1", expected_strides=(4, 8, 16),
        kwargs=contract["training"], data_yaml=data,
        pretrained_name=pretrained.name, pretrained_sha256=sha256(pretrained))
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_or_create_contract(run_dir, changed_split)


def test_untracked_checkpoint_is_never_adopted(tmp_path):
    contract, *_ = _contract(tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights/best.pt").write_bytes(b"foreign")
    with pytest.raises(RuntimeError, match="nonempty"):
        verify_or_create_contract(run_dir, contract)
    assert not (run_dir / RUN_CONTRACT).exists()


def test_missing_pretrained_blocks_new_run_but_existing_contract_pins_source(tmp_path):
    contract, _, _, pretrained = _contract(tmp_path)
    fresh = tmp_path / "fresh"
    pretrained.unlink()
    with pytest.raises(RuntimeError, match="Cannot start a new run"):
        pretrained_identity(pretrained, fresh)
    verify_or_create_contract(fresh, contract)
    assert pretrained_identity(pretrained, fresh) == contract["pretrained_sha256"]


def test_transfer_provenance_required_before_skip_or_resume(tmp_path):
    contract, *_ = _contract(tmp_path)
    run_dir = tmp_path / "run"
    verify_or_create_contract(run_dir, contract)
    with pytest.raises(RuntimeError, match="transfer report"):
        validate_transfer_report(run_dir, contract)
    transfer = {"status": "PASS", "verification":"EXACT_INTERSECTED_TENSOR_EQUALITY",
                "source_sha256":contract["pretrained_sha256"], "verified_backbone_tensor_count":50}
    atomic_json(run_dir / TRANSFER_REPORT, transfer)
    validate_transfer_report(run_dir, contract)
    transfer["source_sha256"] = "bad-hash"
    atomic_json(run_dir / TRANSFER_REPORT, transfer)
    with pytest.raises(RuntimeError, match="Unverified or mismatched"):
        validate_transfer_report(run_dir, contract)


def test_training_attempt_is_contract_bound_and_amp_fallback_is_explicit(tmp_path):
    contract, *_ = _contract(tmp_path)
    run_dir = tmp_path / "run"
    verify_or_create_contract(run_dir, contract)
    record_training_attempt(run_dir, contract, case=contract["case"], seed=42,
                            effective_amp=True, retry_reason=None)
    attempt = validate_training_attempt(run_dir, contract, case=contract["case"], seed=42)
    assert attempt["effective_amp"] is True
    with pytest.raises(RuntimeError, match="case/seed"):
        validate_training_attempt(run_dir, contract, case=contract["case"], seed=43)

    recorded = json.loads((run_dir / TRAINING_ATTEMPT).read_text(encoding="utf-8"))
    recorded["contract_sha256"] = "wrong"
    atomic_json(run_dir / TRAINING_ATTEMPT, recorded)
    with pytest.raises(RuntimeError, match="not bound"):
        validate_training_attempt(run_dir, contract, case=contract["case"], seed=42)

    with pytest.raises(RuntimeError, match="failure reason"):
        record_training_attempt(run_dir, contract, case=contract["case"], seed=42,
                                effective_amp=False, retry_reason=None)
    record_training_attempt(run_dir, contract, case=contract["case"], seed=42,
                            effective_amp=False, retry_reason="autocast kernel failure")
    attempt = validate_training_attempt(run_dir, contract, case=contract["case"], seed=42)
    assert attempt["effective_amp"] is False
    assert attempt["amp_retry_reason"] == "autocast kernel failure"


def test_completion_hashes_are_immutable(tmp_path):
    run_dir = tmp_path / "run"
    contract, *_ = _contract(tmp_path)
    verify_or_create_contract(run_dir, contract)
    (run_dir / "weights").mkdir(parents=True)
    for item in ("weights/best.pt", "weights/last.pt", "config.yaml"):
        (run_dir / item).write_bytes(item.encode("utf-8"))
    (run_dir / "results.csv").write_text("epoch,loss\n0,1.0\n", encoding="utf-8")
    (run_dir / "args.yaml").write_text(
        "optimizer: AdamW\nepochs: 100\nmosaic: 1\namp: false\n", encoding="utf-8"
    )
    assert validate_completion(run_dir) is False
    record_training_attempt(run_dir, contract, case=contract["case"], seed=42,
                            effective_amp=False, retry_reason="AMP diagnostic")
    finalize_training(run_dir, effective_amp=False, retry_reason="AMP diagnostic")
    assert (run_dir / TRAINING_COMPLETION).is_file()
    assert validate_completion(run_dir) is True
    completion = json.loads((run_dir / TRAINING_COMPLETION).read_text(encoding="utf-8"))
    assert completion["effective_amp"] is False
    (run_dir / "weights/best.pt").write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="Checkpoint changed"):
        validate_completion(run_dir)


def test_completion_is_bound_to_contract_and_saved_training_args(tmp_path):
    run_dir = tmp_path / "run"
    contract, *_ = _contract(tmp_path)
    verify_or_create_contract(run_dir, contract)
    (run_dir / "weights").mkdir(parents=True)
    for item in ("weights/best.pt", "weights/last.pt", "config.yaml"):
        (run_dir / item).write_bytes(item.encode())
    (run_dir / "results.csv").write_text("epoch,loss\n0,1\n", encoding="utf-8")
    (run_dir / "args.yaml").write_text(
        "optimizer: AdamW\nepochs: 100\nmosaic: 1\namp: true\n", encoding="utf-8"
    )
    record_training_attempt(run_dir, contract, case=contract["case"], seed=42,
                            effective_amp=True, retry_reason=None)
    finalize_training(run_dir, effective_amp=True)
    recorded = json.loads((run_dir / RUN_CONTRACT).read_text(encoding="utf-8"))
    recorded["seed"] = 44
    atomic_json(run_dir / RUN_CONTRACT, recorded)
    with pytest.raises(RuntimeError, match="not bound"):
        validate_completion(run_dir)


def test_dataset_yaml_absolute_root_is_not_part_of_split_identity(tmp_path):
    config = Path(__file__).resolve().parents[1] / (
        "models_related/models_config/yolov8/tinyperson/"
        "yolov8n_tinyperson_ftsc_prune_h1_p234.yaml"
    )
    pretrained = tmp_path / "yolov8n.pt"
    pretrained.write_bytes(b"checkpoint")
    prepared = []
    for index in (1, 2):
        folder = tmp_path / f"location_{index}"
        folder.mkdir()
        (folder / "corner_manifest.json").write_text('{"same":true}\n', encoding="utf-8")
        for split in ("train", "val", "test"):
            (folder / "images" / split).mkdir(parents=True)
            (folder / "labels" / split).mkdir(parents=True)
            (folder / "images" / split / "sample.jpg").write_bytes(b"same-crop")
            (folder / "labels" / split / "sample.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        data = folder / "tinyperson.yaml"
        data.write_text(f"path: {folder}\ntrain: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
        prepared.append(make_contract(
            dataset="TinyPerson", case="CASE", seed=42, config=config, graph_contract="graph",
            expected_strides=(4., 8.), kwargs={"epochs": 100}, data_yaml=data,
            pretrained_name=pretrained.name, pretrained_sha256=sha256(pretrained),
        )["prepared_dataset"])
    assert prepared[0] == prepared[1]


def test_content_fingerprint_changes_for_same_name_same_size_image_and_label(tmp_path):
    contract, config, data, pretrained = _contract(tmp_path)
    first = contract["prepared_dataset"]
    assert first["content_file_count"] == 6
    assert (data.parent / ".ftsc_paper_content_hash_cache_v1.json").is_file()
    image = data.parent / "images/train/sample.jpg"
    original = image.read_bytes()
    image.write_bytes(b"X" + original[1:])
    altered_image = make_contract(
        dataset="TinyPerson", case="TP_PRUNE_H1_P234", seed=42, config=config,
        graph_contract="c2f_structural_p234_v1", expected_strides=(4., 8., 16.),
        kwargs=contract["training"], data_yaml=data,
        pretrained_name=pretrained.name, pretrained_sha256=sha256(pretrained))
    assert altered_image["prepared_dataset"]["content_inventory_sha256"] != first["content_inventory_sha256"]
    image.write_bytes(original)
    label = data.parent / "labels/train/sample.txt"
    label.write_text("0 0.7 0.5 0.1 0.1\n", encoding="utf-8")
    altered_label = make_contract(
        dataset="TinyPerson", case="TP_PRUNE_H1_P234", seed=42, config=config,
        graph_contract="c2f_structural_p234_v1", expected_strides=(4., 8., 16.),
        kwargs=contract["training"], data_yaml=data,
        pretrained_name=pretrained.name, pretrained_sha256=sha256(pretrained))
    assert altered_label["prepared_dataset"]["content_inventory_sha256"] != first["content_inventory_sha256"]


def test_training_runtime_is_provenance_not_portable_scientific_contract(tmp_path, monkeypatch):
    import utils.ftsc_paper_run_integrity as integrity

    first, config, data, pretrained = _contract(tmp_path)
    assert "environment" not in first
    monkeypatch.setattr(integrity, "runtime_environment", lambda: {"python": "different"})
    second = make_contract(
        dataset="TinyPerson", case="TP_PRUNE_H1_P234", seed=42, config=config,
        graph_contract="c2f_structural_p234_v1", expected_strides=(4., 8., 16.),
        kwargs=first["training"], data_yaml=data,
        pretrained_name=pretrained.name, pretrained_sha256=sha256(pretrained))
    assert first == second
    folder = tmp_path / "checkpoint_run"
    assert verify_or_create_contract(folder, first) is False
    assert verify_or_create_contract(folder, second) is True


def test_checkpoint_seed_absent_or_wrong_is_rejected(tmp_path):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.patches import torch_load
    from utils.ftsc_paper_run_integrity import verify_checkpoint_graph

    config = Path(__file__).resolve().parents[1] / (
        "models_related/models_config/yolov8/tinyperson/"
        "yolov8n_tinyperson_ftsc_prune_h3_p2.yaml"
    )
    checkpoint = tmp_path / "case.pt"
    YOLO(config).save(checkpoint)
    state = torch_load(checkpoint, map_location=torch.device("cpu"))
    state.setdefault("train_args", {})["seed"] = 43
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="seed differs"):
        verify_checkpoint_graph(checkpoint, config=config, expected_taps=(19,),
                                expected_strides=(4.,), expected_seed=42)
    state["train_args"].pop("seed")
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match="missing mandatory seed"):
        verify_checkpoint_graph(checkpoint, config=config, expected_taps=(19,),
                                expected_strides=(4.,), expected_seed=42)


def test_completion_reconciliation_never_invokes_training(tmp_path, monkeypatch):
    import utils.ftsc_paper_run_integrity as integrity
    import ultralytics.utils.patches as patches

    contract, config, _, _ = _contract(tmp_path)
    folder = tmp_path / "reconcile_run"
    verify_or_create_contract(folder, contract)
    (folder / "weights").mkdir()
    for name in ("best", "last"):
        (folder / "weights" / f"{name}.pt").write_bytes(name.encode())
    (folder / "config.yaml").write_bytes(config.read_bytes())
    (folder / "results.csv").write_text("epoch,loss\n0,1\n", encoding="utf-8")
    (folder / "args.yaml").write_text("optimizer: AdamW\nepochs: 100\nmosaic: 1\namp: true\n", encoding="utf-8")
    record_training_attempt(folder, contract, case=contract["case"], seed=42,
                            effective_amp=True, retry_reason=None)
    atomic_json(folder / TRANSFER_REPORT, {
        "status": "PASS", "verification": "EXACT_INTERSECTED_TENSOR_EQUALITY",
        "source_sha256": contract["pretrained_sha256"], "verified_backbone_tensor_count": 30})
    import torch
    monkeypatch.setattr(patches, "torch_load", lambda *_a, **_kw: {
        "epoch": -1, "optimizer": None, "ema": None, "scaler": None, "model": object()})
    calls = []
    monkeypatch.setattr(integrity, "verify_checkpoint_graph",
                        lambda *a, **kw: calls.append((a, kw)))
    from utils.ftsc_paper_run_integrity import reconcile_finalized_training
    assert reconcile_finalized_training(
        folder, contract=contract, config=config, expected_taps=(19,22,25),
        expected_strides=(4.,8.,16.))
    assert len(calls) == 2 and validate_completion(folder)
    assert json.loads((folder / "training_reconciliation.json").read_text())["status"] == "RECOVERED_NO_TRAINING"


def test_optimizer_bearing_last_stays_resumable_not_finalized(tmp_path, monkeypatch):
    import utils.ftsc_paper_run_integrity as integrity
    import ultralytics.utils.patches as patches
    contract, config, _, _ = _contract(tmp_path)
    folder = tmp_path / "unfinished_run"
    verify_or_create_contract(folder, contract)
    (folder / "weights").mkdir()
    for item in ("weights/best.pt", "weights/last.pt", "config.yaml"):
        (folder / item).write_bytes(b"stub")
    (folder / "results.csv").write_text("epoch,loss\n0,1\n", encoding="utf-8")
    (folder / "args.yaml").write_text("optimizer: AdamW\nepochs: 100\nmosaic: 1\namp: true\n", encoding="utf-8")
    record_training_attempt(folder, contract, case=contract["case"], seed=42,
                            effective_amp=True, retry_reason=None)
    atomic_json(folder / TRANSFER_REPORT, {
        "status": "PASS", "verification": "EXACT_INTERSECTED_TENSOR_EQUALITY",
        "source_sha256": contract["pretrained_sha256"], "verified_backbone_tensor_count": 30})
    monkeypatch.setattr(patches, "torch_load",
                        lambda *_a, **_kw: {"epoch": 10, "optimizer": {"state": {}}, "model": object()})
    assert not integrity.reconcile_finalized_training(
        folder, contract=contract, config=config, expected_taps=(19,22,25),
        expected_strides=(4.,8.,16.))
    assert not (folder / TRAINING_COMPLETION).exists()


def test_empty_or_header_only_results_are_not_complete(tmp_path):
    from utils.ftsc_paper_run_integrity import verify_completed_artifacts

    run_dir = tmp_path / "run"
    (run_dir / "weights").mkdir(parents=True)
    for item in ("weights/best.pt", "weights/last.pt", "args.yaml", "config.yaml"):
        (run_dir / item).write_bytes(b"x")
    (run_dir / "results.csv").write_text("epoch,loss\n", encoding="utf-8")
    assert not verify_completed_artifacts(run_dir)


def test_checkpoint_graph_reconstruction_accepts_exact_case_and_rejects_wrong_taps(tmp_path):
    from ultralytics import YOLO
    from utils.ftsc_paper_run_integrity import verify_checkpoint_graph

    root = Path(__file__).resolve().parents[1]
    config = root / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_ftsc_prune_h3_p2.yaml"
    checkpoint = tmp_path / "case.pt"
    YOLO(config).save(checkpoint)
    verify_checkpoint_graph(checkpoint, config=config, expected_taps=(19,), expected_strides=(4.,))
    with pytest.raises(RuntimeError, match="Detect.from mismatch"):
        verify_checkpoint_graph(checkpoint, config=config, expected_taps=(19, 22), expected_strides=(4., 8.))
