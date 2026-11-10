"""Offline source guards for LEVIR missing B1 FTSC-off notebook workflow."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "current_live_training_notebok_LEV.ipynb"


def test_notebook_json_and_last_cell_compile():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"][-1]["id"] == "levir-v8-p23-ftsc-off-hf-first"
    source = "".join(nb["cells"][-1]["source"])
    ast.parse(source, filename=str(NB), mode="exec")
    compile(source, str(NB), "exec")


def test_hf_first_b1_only_three_seed_training_contract():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    source = "".join(nb["cells"][-1]["source"])
    assert "RUN_LEVIR_V8_P23_PORTABILITY_BACKFILL = True" in source
    assert "LEVIR_V8_B1_PREFLIGHT_ONLY = False" in source
    assert "_b1_seeds = (42, 43, 44)" in source
    assert "_b1_case = \"V8_B1_P23_NOFTSC\"" in source
    assert "HfApi as _B1HfApi" in source
    assert "list_repo_files(" in source and "repo_info(" in source
    assert "if _b1_prefix is not None:" in source
    assert "missing best.pt" in source
    assert "HF provenance mismatch" in source
    assert "checkpoint training mismatch" in source
    assert "_b1_reference.train_kwargs(" in source
    assert "_b1_contract.static_graph_parity()" in source
    assert "_b1_model.train(**_b1_kwargs)" in source
    assert "_b1_evaluate(" in source and "include_size=True" in source
    assert "_b1_uploader.publish_run(" in source
    assert "ftsc_paper_phase2_one_shot.py" not in source
    assert "V8_H2_P23_FTSC" not in source  # H2 remains reuse-only; this cell never launches it
