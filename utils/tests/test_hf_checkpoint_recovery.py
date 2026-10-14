"""Offline safety regressions: no real HF credentials, networking or training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.ftsc_completion_planner import decide, validate_six_core
from utils.hf_checkpoint_recovery import HFCheckpointRecovery, CheckpointRecoveryError


class FakeApi:
    def __init__(self, files, *, fails=False):
        self.files, self.fails = files, fails

    def repo_info(self, **kwargs):
        if self.fails:
            raise ConnectionError("HF unreachable")
        return SimpleNamespace(sha="commit-123")

    def list_repo_files(self, **kwargs):
        assert kwargs["revision"] == "commit-123"
        return list(self.files)


@pytest.fixture
def case_paths(tmp_path):
    paths = [
        "varroa/causal/V1_FTSC/seed_42/weights/best.pt",
        "varroa_ftsc_position_alpha_h2_v1/ALPHA4_rerun/seed_42/weights/best.pt",
        "varroa_ftsc_next_ablation_suite/FTSC_P34/seed_43/weights/best.pt",
        "varroa/causal/V1_FTSC/seed_42/args.yaml",
        "varroa/causal/V1_FTSC/seed_42/config.yaml",
        "tinyperson/ALPHA4_rerun/seed_42/weights/best.pt",
    ]
    for remote in paths:
        f = tmp_path / remote
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(remote.encode())
    return paths


def test_discovery_separates_namespaces_seeds_and_cases(case_paths):
    recovery = HFCheckpointRecovery("example/results", api=FakeApi(case_paths))
    inv = recovery.inventory(["varroa", "varroa_ftsc_position_alpha_h2_v1", "varroa_ftsc_next_ablation_suite"])
    assert inv["revision"] == "commit-123"
    results = recovery.candidates(inv, case="V1_FTSC", seed=42, aliases=("V1_FTSC",))
    assert len(results) == 1
    assert results[0].checkpoint_path.startswith("varroa/")
    assert len(results[0].metadata_paths) == 2
    assert not recovery.candidates(inv, case="V1_FTSC", seed=43, aliases=("V1_FTSC",))
    assert not recovery.candidates(inv, case="ALPHA4_rerun", seed=43, aliases=("ALPHA4_rerun",))
    assert recovery.candidates(inv, case="FTSC_P34", seed=43, aliases=("FTSC_P34",))


def test_restore_requires_scientific_verification_and_detects_collision(tmp_path, case_paths):
    root = tmp_path
    def fake_download(**kwargs):
        assert kwargs["revision"] == "commit-123"
        return str(root / kwargs["filename"])
    recovery = HFCheckpointRecovery("example/results", api=FakeApi(case_paths), download_fn=fake_download)
    cand = recovery.candidates(recovery.inventory(["varroa"]), case="V1_FTSC",
                               seed=42, aliases=("V1_FTSC",))[0]
    dst = tmp_path / "restored/weights/best.pt"
    with pytest.raises(CheckpointRecoveryError, match="unverified"):
        recovery.restore(cand, destination=dst, verified=False, cache_dir=tmp_path)
    assert not dst.exists()
    src = tmp_path / cand.checkpoint_path
    expected = hashlib.sha256(src.read_bytes()).hexdigest()
    found = recovery.restore(cand, destination=dst, verified=True,
                             cache_dir=tmp_path, expected_sha256=expected)
    assert found["checkpoint_sha256"] == expected
    assert dst.read_bytes() == src.read_bytes()
    with pytest.raises(CheckpointRecoveryError, match="differs"):
        recovery.restore(cand, destination=dst, verified=True,
                         cache_dir=tmp_path, expected_sha256="f"*64)
    dst.write_bytes(b"different")
    with pytest.raises(CheckpointRecoveryError, match="collision"):
        recovery.restore(cand, destination=dst, verified=True,
                         cache_dir=tmp_path, expected_sha256=expected)


def test_hf_error_never_mistaken_for_absent_checkpoint():
    recovery = HFCheckpointRecovery("example/results", api=FakeApi([], fails=True))
    with pytest.raises(ConnectionError):
        recovery.inventory(["varroa"])
    d = decide(dataset="varroa", case="V1_FTSC", seed=42,
               inventory_successful=False, candidates=[], existing_metrics=None,
               metric_keys=KEYS)
    assert d.action == "BLOCKED_HF_UNAVAILABLE"


KEYS = {"AP50": "a", "mAP50-95": "b", "AP50-Small": "c",
        "parameters": "p", "GFLOPs": "g", "inference_ms": "t"}
VALID = {"a": .91, "b": .35, "c": .88, "p": 2678471,
         "g": 11.79, "t": .79}


def test_planner_metric_and_candidate_states():
    assert not validate_six_core(VALID, KEYS)
    assert validate_six_core({**VALID, "c": -1}, KEYS) == ("AP50-Small",)
    assert "inference_ms" in validate_six_core({**VALID, "t": float("nan")}, KEYS)
    common = dict(dataset="varroa", case="V1_FTSC", seed=42,
                  inventory_successful=True, metric_keys=KEYS)
    assert decide(**common, candidates=[], existing_metrics=None).action == "TRAIN_REQUIRED"
    assert decide(**common, candidates=[{"checkpoint_path": "one", "verified": False}],
                  existing_metrics=None).action == "BLOCKED_UNVERIFIED"
    a = {"checkpoint_path": "one", "verified": True, "sha256": None}
    assert decide(**common, candidates=[a], existing_metrics=None).action == "EVAL_ONLY"
    b = {"checkpoint_path": "other", "verified": True, "sha256": None}
    assert decide(**common, candidates=[a, b], existing_metrics=None).action == "BLOCKED_CONFLICT"
    a["sha256"] = "abc"
    assert decide(**common, candidates=[a], existing_metrics={**VALID,
                  "checkpoint_sha256": "abc", "evaluator_verified": True}).action == "SKIP_COMPLETE"
    assert decide(**common, candidates=[a], existing_metrics={**VALID,
                  "checkpoint_sha256": "different", "evaluator_verified": True}).action == "EVAL_ONLY"


def test_invalid_paths_are_rejected():
    recovery = HFCheckpointRecovery("example/results", api=FakeApi(["varroa/../V1_FTSC/seed_42/weights/best.pt"]))
    with pytest.raises(CheckpointRecoveryError):
        recovery.inventory(["varroa"])


def test_renamed_remote_case_can_be_discovered_for_metadata_verification():
    files = ["varroa/legacy_upload_name/seed_42/weights/best.pt",
             "varroa/legacy_upload_name/seed_42/config.yaml",
             "varroa/legacy_upload_name/seed_42/args.yaml"]
    recovery = HFCheckpointRecovery("example/results", api=FakeApi(files))
    inv = recovery.inventory(["varroa"])
    assert not recovery.candidates(inv, case="V1_FTSC", seed=42, aliases=("V1_FTSC",))
    possible = recovery.candidates(inv, case="V1_FTSC", seed=42,
                                   aliases=("V1_FTSC",), match_case=False)
    assert len(possible) == 1
    assert possible[0].metadata_paths == (
        "varroa/legacy_upload_name/seed_42/config.yaml",
        "varroa/legacy_upload_name/seed_42/args.yaml",
    )


@pytest.mark.parametrize("dataset", ["levir", "tinyperson", "varroa"])
def test_shared_planner_is_dataset_agnostic(dataset):
    result = decide(
        dataset=dataset, case="caseA", seed=44, inventory_successful=True,
        candidates=[{"checkpoint_path": dataset + "/caseA/seed_44/weights/best.pt",
                     "verified": True, "sha256": "abc"}],
        existing_metrics=None, metric_keys=KEYS,
    )
    assert result.dataset == dataset
    assert result.action == "EVAL_ONLY"
