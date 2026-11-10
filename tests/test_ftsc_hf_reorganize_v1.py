"""Offline contract checks for the FTSC HF checkpoint migration.

No HF network, no GPU and no mutation during these tests.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from utils import ftsc_hf_reorganize_v1 as m


def test_audited_manifest_and_sha_duplicates_are_consistent():
    plan = m.read_plan(m.DEFAULT_PLAN)
    report = m.summarize(plan)
    assert report["best_checkpoint_paths"] == len(plan["entries"]) >= 456
    assert report["verified_sha_from_completion_markers"] >= 420
    assert report["historical_source_provenance_only"] == 9
    assert report["unique_known_best_sha"] >= 327
    assert report["confirmed_duplicate_aliases"] >= 102
    assert report["unverified_best"] <= 27
    assert sum(report["by_dataset"].values()) == len(plan["entries"])
    assert not plan["collision_paths"]


def _fake_entries():
    checksum = "a" * 64
    base = {
        "dataset": "Varroa", "case": "V1_FTSC", "seed": 42,
        "sha256": checksum, "best_path": "varroa/V1_FTSC/seed_42/weights/best.pt",
        "source_run_dir": "varroa/V1_FTSC/seed_42",
        "target_run_dir": "Varroa/original_runs/V1_FTSC/seed_42",
        "canonical_sha_source": "varroa/V1_FTSC/seed_42/weights/best.pt",
        "canonical_target_checkpoint": "Varroa/original_runs/V1_FTSC/seed_42/weights/best.pt",
        "action": "STAGE_PRIMARY_CHECKPOINT",
    }
    alias = {**base,
             "best_path": "varroa_ftsc_fullmetric_recovery_v1/V1_FTSC/seed_42/weights/best.pt",
             "source_run_dir": "varroa_ftsc_fullmetric_recovery_v1/V1_FTSC/seed_42",
             "target_run_dir": "Varroa/derived_evaluations/fullmetric/V1_FTSC/seed_42",
             "action": "METADATA_ONLY_ALIAS"}
    return base, alias


def test_alias_does_not_copy_best_or_last_weights():
    primary, alias = _fake_entries()
    m.assert_plan_invariants([primary, alias])
    files = {
        primary["best_path"],
        primary["source_run_dir"] + "/weights/last.pt",
        primary["source_run_dir"] + "/upload_complete.json",
        alias["best_path"],
        alias["source_run_dir"] + "/evaluation_metrics_extended.json",
        alias["source_run_dir"] + "/upload_complete.json",
    }
    jobs = m.build_copy_jobs([primary, alias], files, "b" * 40)
    dests = {dest for _, dest, _ in jobs}
    assert primary["target_run_dir"] + "/weights/best.pt" in dests
    assert primary["target_run_dir"] + "/weights/last.pt" in dests
    assert alias["target_run_dir"] + "/weights/best.pt" not in dests
    assert alias["target_run_dir"] + "/evaluation_metrics_extended.json" in dests
    assert alias["target_run_dir"] + "/checkpoint_reference.json" in dests
    assert alias["target_run_dir"] + "/legacy_upload_complete.json" in dests
    assert alias["target_run_dir"] + "/upload_complete.json" not in dests


def test_different_seed_with_same_sha_is_refused():
    primary, alias = _fake_entries()
    alias["seed"] = 43
    with pytest.raises(m.MigrationError, match="Same checkpoint SHA"):
        m.assert_plan_invariants([primary, alias])


def test_git_blob_oid_is_deterministic():
    payload = b"checkpoint_reference.json\n"
    expected = hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload).hexdigest()
    assert m.git_blob_oid(payload) == expected


def test_legacy_delete_is_explicitly_disabled():
    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "Legacy deletion is intentionally disabled" in src
    assert "CommitOperationDelete" not in src
    assert "delete_folder(" not in src
    assert "delete_file(" not in src
    assert m.DEFAULT_PLAN.is_file()


def test_all_456_case_and_seed_folder_names_are_preserved():
    plan = m.read_plan(m.DEFAULT_PLAN)
    destinations = []
    for row in plan["entries"]:
        src = Path(row["source_run_dir"]).parts
        dest = Path(row["target_run_dir"]).parts
        assert src[-2:] == dest[-2:]
        assert row["dataset"] == dest[0]
        assert "//" not in row["target_run_dir"]
        destinations.append(row["target_run_dir"])
    assert len(destinations) == len(set(destinations)) == len(plan["entries"])


def test_migration_notebook_is_valid_and_never_deletes_checkpoints():
    import ast
    notebook = Path(m.WORKSPACE) / "FTSC_HF_REORGANIZE_SAFE.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) == 10
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    stage = next("".join(c["source"]) for c in data["cells"] if c["id"] == "stage")
    assert "ENABLE_HF_STAGE = False" in stage
    assert "delete_repo" not in stage
    assert "CommitOperationDelete" not in stage
    assert "migration.execute_stage" in stage
    assert "migration.validate_remote_sources" in stage
    assert "MAX_PASSES_PER_RUN = 10" in stage
    assert "min(MAX_OPERATIONS_PER_PASS, remaining_count)" in stage


def test_journal_b1_requires_all_three_seeds_before_any_migration():
    plan = m.read_plan(m.DEFAULT_PLAN)
    incomplete = [
        row for row in plan["entries"]
        if not (
            row["source_run_dir"].startswith(m.JOURNAL_B1_PREFIX + "/")
            and row["seed"] == 44
        )
    ]
    with pytest.raises(m.MigrationError, match="three distinct seeds"):
        m.require_journal_b1_complete(incomplete)
    with pytest.raises(m.MigrationError, match="three distinct seeds"):
        m.validate_remote_sources(None, incomplete, "a" * 40, set())
    for row in plan["entries"]:
        if row["source_run_dir"].startswith(m.JOURNAL_B1_PREFIX + "/"):
            assert row["case"] == "V8_B1_P23_NOFTSC"
            assert row["target_run_dir"].endswith(
                f"/V8_B1_P23_NOFTSC/seed_{row['seed']}"
            )


def test_new_levir_ftsc_off_namespace_keeps_its_own_training_cohort():
    plan = m.read_plan(m.DEFAULT_PLAN)
    paths = {row["source_run_dir"]: row for row in plan["entries"]}
    for seed in (42, 43):
        b1_original = paths[
            f"levir_v8_p23_portability_backfill_v1/V8_B1_P23_NOFTSC/seed_{seed}"
        ]
        b1_new = paths[
            f"levir_v8_p23_ftsc_off_hf_first_v1/V8_B1_P23_NOFTSC/seed_{seed}"
        ]
        assert b1_new["dataset"] == b1_original["dataset"] == "LEVIR-Ship"
        assert b1_new["case"] == b1_original["case"] == "V8_B1_P23_NOFTSC"
        assert b1_new["seed"] == b1_original["seed"] == seed
        assert b1_new["sha256"] != b1_original["sha256"]
        assert b1_new["target_run_dir"] != b1_original["target_run_dir"]
        assert Path(b1_new["target_run_dir"]).parts[-2:] == Path(
            b1_new["source_run_dir"]
        ).parts[-2:]
        assert b1_new["action"] == "STAGE_PRIMARY_CHECKPOINT"

