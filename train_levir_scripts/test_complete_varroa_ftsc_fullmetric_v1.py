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
    assert set(complete.SEEDS) == {42, 43, 44}
    assert all(spec.yaml.is_file() for spec in complete.CASES.values())
    assert complete.CASES["FTSC_P23_NOMOSAIC"].yaml == complete.CASES["FTSC_P23_CP2"].yaml
    assert complete.CASES["V1_FTSC"].prefixes == ("varroa",)
    assert complete.CASES["ALPHA4_rerun"].alpha == 4.0


def test_causal_checkpoint_recipe_verified_without_checkpoint_download(tmp_path):
    config = complete.CASES["V1_FTSC"].yaml
    metadata = {"config.yaml": config, "args.yaml": _write_args(tmp_path, 42)}
    ok, reason = complete.scientific_verification("V1_FTSC", 42, metadata)
    assert ok, reason
    assert "split-content SHA not archived" in reason
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
