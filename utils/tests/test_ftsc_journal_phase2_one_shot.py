"""Source-only Phase-2 notebook/orchestrator regression tests.

Never authenticates to Hugging Face and never trains. Runtime end-to-end HF,
checkpoint restore, official weights and GPU tests remain separate live gates.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from utils import ftsc_journal_phase2_one_shot as suite

WORKSPACE = Path(__file__).resolve().parents[3]
NOTEBOOKS = {
    "levir": WORKSPACE / "current_live_training_notebok_LEV.ipynb",
    "tinyperson": WORKSPACE / "current_live_training_notebok_TinyPerson.ipynb",
    "varroa": WORKSPACE / "current_live_training_notebok_Varroa.ipynb",
}


@pytest.mark.parametrize("dataset", tuple(NOTEBOOKS))
def test_run_all_single_journal_launcher_and_old_cross_yolo_disabled(dataset):
    data = json.loads(NOTEBOOKS[dataset].read_text(encoding="utf-8"))
    journal = [cell for cell in data["cells"]
               if cell.get("metadata", {}).get("marimo", {}).get("name") ==
               f"ftsc_paper_phase2_one_shot_{dataset}"]
    assert len(journal) == 1
    source = "".join(journal[0]["source"])
    assert "ftsc_journal_phase2_one_shot.py" in source
    assert f"'--dataset', '{dataset}'" in source
    assert "--phase" not in source
    assert "RUN_" not in source
    assert "check=True" in source
    if dataset != "levir":
        old = [cell for cell in data["cells"]
               if "cross_yolo_live_v1" in cell.get("metadata", {}).get("marimo", {}).get("name", "")]
        assert len(old) == 1
        assert "".join(old[0]["source"]).startswith("# Archived cross-YOLO launcher: disabled")


@pytest.mark.parametrize("dataset,new_count", [("levir", 2), ("tinyperson", 4), ("varroa", 4)])
def test_safe_dry_check_has_frozen_registry(dataset, new_count, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    suite.main(["--dataset", dataset, "--data-root", "/no-data", "--dataset-root",
                "/no-prepared-data", "--check"])
    record = json.loads(capsys.readouterr().out)
    assert record["mode"] == "CHECK_ONLY"
    assert len(record["new_cases"]) == new_count
    assert len(record["new_cases"]) == len(set(record["new_cases"]))


def test_historical_levir_namespaces_are_scoped_and_not_confused():
    files = [
        "levirship/H0/seed_42/weights/best.pt",
        "levirship/H0/seed_43/weights/best.pt",
        "levirship/A0_FTSC/seed_42/weights/best.pt",
        "levirship/ES1/seed_42/weights/best.pt",
        "levirship/E_H2/seed_42/weights/best.pt",
        "tinyperson/H0/seed_42/weights/best.pt",
        "levir_ftsc_position_alpha_historical_h2_v1_rerun_d3e60b1e59d2/"
        "ALPHA4_rerun/seed_42/weights/best.pt",
        "levir_ftsc_position_alpha_historical_h2_v1_rerun_d3e60b1e59d2/"
        "ALPHA8_rerun/seed_44/weights/best.pt",
    ]
    assert suite._historical_case_paths(
        files, "levir", "H0", ("H0_p2_p3_p4_p5_nomosaic", "H0"), 42
    ) == ["levirship/H0/seed_42"]
    assert suite._historical_case_paths(
        files, "levir", "ES1_LEGACY", ("ES1",), 42
    ) == ["levirship/ES1/seed_42"]
    assert suite._historical_case_paths(
        files, "levir", "ALPHA4_rerun", ("ALPHA4_rerun",), 42
    ) == ["levir_ftsc_position_alpha_historical_h2_v1_rerun_d3e60b1e59d2/"
          "ALPHA4_rerun/seed_42"]
    assert not suite._historical_case_paths(
        files, "levir", "H2", ("H2_p2_p3_nomosaic", "H2"), 42
    )


def test_tinyperson_historical_archive_namespaces_are_not_interchangeable():
    """Original H2/ES1 and matched alpha runs must never alias portal/journal."""
    files = [
        "tinyperson/H2_MOSAIC/seed_42_corner_sw640_sh512/weights/best.pt",
        "tinyperson/ES1_MOSAIC/seed_43_corner_sw640_sh512/weights/best.pt",
        "tinyperson/A0_FTSC/seed_42_corner_sw640_sh512/weights/best.pt",
        "tinyperson_ftsc_position_alpha_h2_v1/ALPHA4_rerun/"
        "seed_42_corner_sw640_sh512/weights/best.pt",
        "tinyperson_ftsc_position_alpha_h2_v1/ALPHA8_rerun/"
        "seed_44_corner_sw640_sh512/weights/best.pt",
        "tinyperson_ftsc_cross_yolo_portability_v1/H2_MOSAIC/"
        "seed_42_corner_sw640_sh512/weights/best.pt",
        "tinyperson_ftsc_journal_phase2_v1/TP_PRUNE_H0_P2345/"
        "seed_42_corner_sw640_sh512/weights/best.pt",
    ]
    assert suite._historical_roots("tinyperson") == (
        "tinyperson", "tinyperson_ftsc_position_alpha_h2_v1")
    assert suite._historical_case_paths(files, "tinyperson", "H2_MOSAIC",
                                        ("H2_MOSAIC",), 42) == [
        "tinyperson/H2_MOSAIC/seed_42_corner_sw640_sh512"]
    assert suite._historical_case_paths(files, "tinyperson", "ES1_MOSAIC_LEGACY",
                                        ("ES1_MOSAIC",), 43) == [
        "tinyperson/ES1_MOSAIC/seed_43_corner_sw640_sh512"]
    assert suite._historical_case_paths(files, "tinyperson", "ALPHA4_rerun",
                                        ("ALPHA4_rerun",), 42) == [
        "tinyperson_ftsc_position_alpha_h2_v1/ALPHA4_rerun/"
        "seed_42_corner_sw640_sh512"]
    assert not suite._historical_case_paths(files, "tinyperson", "H2_MOSAIC",
                                            ("H2_MOSAIC",), 44)


def test_varroa_historical_archive_not_structural_head_pruning():
    files = [
        "varroa/V1_FTSC/seed_42/weights/best.pt",
        "varroa/V2_EDGE025/seed_42/weights/best.pt",
        "varroa/V3_COLOR/seed_43/weights/best.pt",
        "varroa/V0_NOFTSC/seed_42/weights/best.pt",
        "varroa_ftsc_position_alpha_h2_v1/ALPHA4_rerun/seed_44/weights/best.pt",
        "varroa_ftsc_position_alpha_h2_v1/ALPHA8_rerun/seed_42/weights/best.pt",
        "varroa_ftsc_journal_phase2_v1/VR_STRUCT_H2_P23/seed_42/weights/best.pt",
        "varroa_ftsc_cross_yolo_portability_v1/V1_FTSC/seed_42/weights/best.pt",
        "varroa_ftsc_fullmetric_recovery_v1/V1_FTSC/seed_42/weights/best.pt",
    ]
    assert suite._historical_roots("varroa") == (
        "varroa", "varroa_ftsc_position_alpha_h2_v1")
    assert suite._historical_case_paths(files, "varroa", "V1_FTSC",
                                        ("V1_FTSC",), 42) == [
        "varroa/V1_FTSC/seed_42"]
    assert suite._historical_case_paths(files, "varroa", "V2_EDGE025",
                                        ("V2_EDGE025",), 42) == [
        "varroa/V2_EDGE025/seed_42"]
    assert suite._historical_case_paths(files, "varroa", "ALPHA4_rerun",
                                        ("ALPHA4_rerun",), 44) == [
        "varroa_ftsc_position_alpha_h2_v1/ALPHA4_rerun/seed_44"]
    assert not suite._historical_case_paths(files, "varroa", "V1_FTSC",
                                            ("VR_STRUCT_H2_P23",), 42)


def test_exact_remote_seed_and_case_match():
    paths = [
        "tinyperson_ftsc_journal_phase2_v1/TP_PRUNE_H1_P234/seed_42_corner_sw640_sh512/weights/best.pt",
        "tinyperson_ftsc_journal_phase2_v1/TP_PRUNE_H1_P234/seed_43_corner_sw640_sh512/weights/best.pt",
        "tinyperson_ftsc_journal_phase2_v1/TP_PRUNE_H1_P234_fake/seed_42_corner_sw640_sh512/weights/best.pt",
        "tinyperson_ftsc_journal_phase2_v1/TP_PRUNE_H1_P234/seed_420_corner_sw640_sh512/weights/best.pt",
    ]
    assert suite._case_paths(paths, ("TP_PRUNE_H1_P234",), 42) == [
        "tinyperson_ftsc_journal_phase2_v1/TP_PRUNE_H1_P234/seed_42_corner_sw640_sh512"
    ]


@pytest.mark.parametrize("dataset,extended", [("levir", True), ("varroa", True), ("tinyperson", False)])
def test_native_six_core_artifact_contract(dataset, extended):
    for historical in (True, False):
        required = suite.required_artifacts(dataset, historical=historical)
        assert ("evaluation_metrics_extended.json" in required) is extended
        assert "evaluation_metrics.json" in required
        assert "weights/best.pt" in required
        if historical:
            assert "weights/last.pt" not in required
            assert "source_provenance.json" in required
        else:
            assert "weights/last.pt" in required
            assert "run_contract.json" in required


def test_hf_unavailable_never_authorizes_training(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN missing"):
        suite._check_token()
