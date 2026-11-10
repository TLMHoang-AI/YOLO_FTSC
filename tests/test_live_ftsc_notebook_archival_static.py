"""Text-only guards for live notebooks; these tests never execute notebook cells."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEVIR = (ROOT / "current_live_training_notebok_LEV.ipynb").read_text(encoding="utf-8")
TINY = (ROOT / "current_live_training_notebok_TinyPerson.ipynb").read_text(encoding="utf-8")
VARROA = (ROOT / "current_live_training_notebok_Varroa.ipynb").read_text(encoding="utf-8")


def test_levir_v8_backfill_is_the_single_active_report_gate():
    assert "RUN_LEVIR_V8_P23_PORTABILITY_BACKFILL = True" in LEVIR
    assert "RUN_LEVIR_FTSC_ALPHA = False" in LEVIR
    assert "RUN_LEVIR_ES1_METRIC_RECOVERY = False" in LEVIR
    assert "RUN_LEVIR_FTSC_CROSS_YOLO_PORTABILITY" not in LEVIR
    for marker in (
        "V8_B1_P23_NOFTSC", "42", "43", "44",
        "FTSCResultsUploader", "levir_v8_p23_ftsc_off_hf_first_v1",
        "list_repo_files", "HF provenance mismatch",
    ):
        assert marker in LEVIR


def test_each_notebook_has_the_expected_active_run_gate():
    expected = {
        "LEVIR": (LEVIR, "RUN_LEVIR_V8_P23_PORTABILITY_BACKFILL = True"),
        "TinyPerson": (TINY, "RUN_TINYPERSON_FTSC_CAUSAL_SUITE = False"),
        "Varroa": (VARROA, "RUN_VARROA_FTSC_NEXT_ABLATION = False"),
    }
    obsolete = ("RUN_FTSC_ABLATIONS", "RUN_LEVIR_H2_COMPLETION", "RUN_TINYPERSON_H2_COMPLETION", "RUN_VARROA_H2_COMPLETION")
    for dataset, (text, assignment) in expected.items():
        assert assignment in text
        gates = set(re.findall(r"RUN_[A-Z0-9_]+\s*=", text))
        if dataset == "LEVIR":
            assert gates == {"RUN_LEVIR_V8_P23_PORTABILITY_BACKFILL =",
                             "RUN_LEVIR_FTSC_ALPHA =", "RUN_LEVIR_ES1_METRIC_RECOVERY ="}
        else:
            assert gates == {assignment.split(" =")[0] + " ="}
        assert not any(name in text for name in obsolete)
    assert "train_levir_v8_p23_portability_backfill_v1" in LEVIR


def test_deferred_notebooks_keep_current_static_runner_contracts_without_completion_flow():
    assert "train_tinyperson_ftsc_causal_suite.py" in TINY
    assert "tinyperson_causal_suite_module" in TINY
    assert "train_tinyperson_h2_generalization_completion_v1.py" not in TINY
    assert "train_varroa_ftsc_next_ablation_suite.py" in VARROA
    assert "varroa_next_suite_module" in VARROA
    assert "NOT_APPLICABLE" in VARROA and "VARROA_HISTORICAL_REFERENCES" in VARROA
    assert "train_varroa_h2_generalization_completion_v1.py" not in VARROA
    assert "h2_generalization_completion_common.py" not in VARROA
