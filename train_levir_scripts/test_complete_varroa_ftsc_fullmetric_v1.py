"""Offline Varroa recovery-registry and recipe-gate regressions.

Run in yolo_code: python -m pytest -q
    train_levir_scripts/test_complete_varroa_ftsc_fullmetric_v1.py
No GPU, HF, model forward or dataset preparation needed.
"""
from pathlib import Path

from train_levir_scripts import complete_varroa_ftsc_fullmetric_v1 as complete


def _write_args(tmp_path: Path, seed: int, **changes):
    import yaml
    args = {
        "seed": seed, "imgsz": 640, "batch": 4, "epochs": 100,
        "patience": 20, "deterministic": True,
        "amp": True, "mosaic": 1.0, "close_mosaic": 10,
        "optimizer": "auto",
        "data": f"/marimo/yolo_code/datasets/varroa_yolo_seed{seed}/varroa.yaml",
        "copy_paste_enabled": False,
    }
    args.update(changes)
    out = tmp_path / "args.yaml"
    out.write_text(yaml.safe_dump(args), encoding="utf-8")
    return out


def test_registry_counts_and_all_model_configs_exist():
    assert len(complete.CASES) == 19
    assert len(complete.HISTORICAL_CASES) == 17
    assert len(complete.ALPHA_CASES) == 2
    assert set(complete.HISTORICAL_CASES).isdisjoint(complete.ALPHA_CASES)
    assert set(complete.SEEDS) == {42, 43, 44}
    assert all(spec.yaml.is_file() for spec in complete.CASES.values())
    assert complete.CASES["FTSC_P23_NOMOSAIC"].yaml == complete.CASES["FTSC_P23_CP2"].yaml
    assert complete.CASES["V1_FTSC"].prefixes == ("varroa_ftsc_causal_suite", "varroa")
    assert complete.CASES["ALPHA4_rerun"].alpha == 4.0


def test_causal_checkpoint_recipe_verified_without_checkpoint_download(tmp_path):
    config = complete.CASES["V1_FTSC"].yaml
    metadata = {"config.yaml": config, "args.yaml": _write_args(tmp_path, 42)}
    ok, reason = complete.scientific_verification("V1_FTSC", 42, metadata)
    assert ok, reason
    assert "original split-content SHA" in reason
    ok, reason = complete.scientific_verification("V1_FTSC", 43, metadata)
    assert not ok
    assert "seed" in reason


def test_cp2_and_mosaic_treatments_are_not_interchangeable(tmp_path):
    cp2 = complete.CASES["FTSC_P23_CP2"]
    off = complete.CASES["FTSC_P23_NOMOSAIC"]
    assert cp2.yaml == off.yaml
    base = {"config.yaml": cp2.yaml}
    args = _write_args(tmp_path, 42, copy_paste_enabled=False)
    assert not complete.scientific_verification("FTSC_P23_CP2", 42, {**base, "args.yaml": args})[0]
    args = _write_args(tmp_path, 42, mosaic=0.0, close_mosaic=0)
    assert complete.scientific_verification("FTSC_P23_NOMOSAIC", 42, {**base, "args.yaml": args})[0]
    assert not complete.scientific_verification("FTSC_P23_CP2", 42, {**base, "args.yaml": args})[0]
    cp2_settings = complete.next_suite.validated_cp2_settings()
    args = _write_args(tmp_path, 42, **cp2_settings)
    assert complete.scientific_verification("FTSC_P23_CP2", 42, {**base, "args.yaml": args})[0]
    args = _write_args(tmp_path, 42, **{**cp2_settings, "copy_paste_copies": 1})
    assert not complete.scientific_verification("FTSC_P23_CP2", 42, {**base, "args.yaml": args})[0]


def test_missing_source_metadata_never_counts_as_verified(tmp_path):
    ok, message = complete.scientific_verification("V1_FTSC", 42, {})
    assert not ok
    assert "Missing" in message
    args = _write_args(tmp_path, 42, amp=False)
    ok, message = complete.scientific_verification(
        "V1_FTSC", 42,
        {"config.yaml": complete.CASES["V1_FTSC"].yaml, "args.yaml": args},
    )
    assert not ok and "amp" in message


def test_evaluate_writes_incremental_progress_for_missing_checkpoints(tmp_path, capsys):
    """No HF/GPU access: blocked seed rows must still be visible and persisted."""
    import json
    from types import SimpleNamespace

    args = SimpleNamespace(
        project=tmp_path,
        cases=("V0_NOFTSC",),
        seeds=(42, 43, 44),
    )
    report = complete.evaluate(args)
    assert report["processed"] == 3 and report["total"] == 3
    assert report["finished"] is True
    assert [row["status"] for row in report["rows"]] == [
        "BLOCKED_NO_VERIFIED_CHECKPOINT",
        "BLOCKED_NO_VERIFIED_CHECKPOINT",
        "BLOCKED_NO_VERIFIED_CHECKPOINT",
    ]
    saved = json.loads((tmp_path / "evaluation_status.json").read_text(encoding="utf-8"))
    assert saved == report
    output = capsys.readouterr().out
    assert "[EVALUATE 1/3] V0_NOFTSC/seed_42" in output
    assert "[EVALUATE 3/3] V0_NOFTSC/seed_44" in output
