"""Text-only guards for live notebooks; these tests never execute notebook cells."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEVIR = (ROOT / "current_live_training_notebok_LEV.ipynb").read_text(encoding="utf-8")
TINY = (ROOT / "current_live_training_notebok_TinyPerson.ipynb").read_text(encoding="utf-8")
VARROA = (ROOT / "current_live_training_notebok_Varroa.ipynb").read_text(encoding="utf-8")


def test_levir_cross_yolo_is_disabled_and_has_one_job_archive_recovery_flow():
    assert "RUN_LEVIR_FTSC_CROSS_YOLO_PORTABILITY = False" in LEVIR
    assert "FTSCResultsUploader(token=HF_TOKEN" in LEVIR
    assert "tuple(cross_yolo.CASES)" in LEVIR
    assert "restore_completed_runs(namespace=_namespace" in LEVIR
    assert "cross_yolo.dispatch(_job_args, _rows)" in LEVIR
    assert "cross_uploader.publish_run(namespace=_namespace" in LEVIR
    assert "cross_yolo.write_summaries" in LEVIR
    assert "cross_yolo.dispatch(_cross_args, _cross_rows)" not in LEVIR
    assert "cross_yolo.write_summaries(cross_yolo.parse_args([]), _cross_all_rows)" not in LEVIR
    assert "_summary_args.families = list(LEVIR_CROSS_YOLO_SELECTION['families'])" in LEVIR
    assert "_cross_terminal_rows" in LEVIR and "_cross_summary_rows" in LEVIR
    assert "_cross_download_asset(str(_expected_checkpoint))" in LEVIR
    assert "_returned_checkpoint" not in LEVIR


def test_each_notebook_has_exactly_one_current_disabled_run_gate():
    expected = {
        "LEVIR": (LEVIR, "RUN_LEVIR_FTSC_CROSS_YOLO_PORTABILITY = False"),
        "TinyPerson": (TINY, "RUN_TINYPERSON_FTSC_CAUSAL_SUITE = False"),
        "Varroa": (VARROA, "RUN_VARROA_FTSC_NEXT_ABLATION = False"),
    }
    obsolete = ("RUN_FTSC_ABLATIONS", "RUN_LEVIR_H2_COMPLETION", "RUN_TINYPERSON_H2_COMPLETION", "RUN_VARROA_H2_COMPLETION")
    for _, (text, assignment) in expected.items():
        assert assignment in text
        assert re.findall(r"RUN_[A-Z0-9_]+\s*=", text) == [assignment.split(" =")[0] + " ="]
        assert not any(name in text for name in obsolete)
    for marker in ("yolov9t_levir_b0_stock.yaml", "yolov10n_levir_b0_stock.yaml", "yolo11n_levir_b0_stock.yaml"):
        assert marker in LEVIR


def test_deferred_notebooks_keep_current_static_runner_contracts_without_completion_flow():
    assert "train_tinyperson_ftsc_causal_suite.py" in TINY
    assert "tinyperson_causal_suite_module" in TINY
    assert "train_tinyperson_h2_generalization_completion_v1.py" not in TINY
    assert "train_varroa_ftsc_next_ablation_suite.py" in VARROA
    assert "varroa_next_suite_module" in VARROA
    assert "NOT_APPLICABLE" in VARROA and "VARROA_HISTORICAL_REFERENCES" in VARROA
    assert "train_varroa_h2_generalization_completion_v1.py" not in VARROA
    assert "h2_generalization_completion_common.py" not in VARROA
