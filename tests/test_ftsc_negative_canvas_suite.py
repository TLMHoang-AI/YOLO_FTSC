from pathlib import Path

from train_levir_scripts import train_ftsc_negative_canvas_suite as suite


def test_scope_cases_seeds_and_no_negative_bank_cli():
    args = suite.parse_args([])
    assert suite.DATASETS == ("levir", "tinyperson")
    assert suite.CASES == ("r1", "r4")
    assert args.seeds == [42, 43, 44]
    assert not hasattr(args, "negative_bank") and not hasattr(args, "hard_negative_bank")


def test_r1_r4_effective_contracts_and_controls_are_metadata_only():
    from ultralytics.cfg import get_cfg

    r1, r4 = suite.augmentation_for("r1"), suite.augmentation_for("r4")
    for settings in (r1, r4):
        assert settings["mosaic"] == 0.0 and settings["close_mosaic"] == 0
        assert settings["copy_paste_enabled"] is True
        assert settings["copy_paste_mode"] == "negative_canvas"
        assert settings["negative_cp_p"] == 0.30
        assert settings["copy_paste_copies"] == 1
        assert settings["copy_paste_max_overlap"] == 0.0
    assert (r1["negative_cp_target_policy"], r1["negative_cp_donor_policy"], r1["negative_cp_degradation"]) == (
        "empirical", "matched", "none",
    )
    assert (r4["negative_cp_target_policy"], r4["negative_cp_donor_policy"], r4["negative_cp_degradation"]) == (
        "deficit", "larger", "weak_blur",
    )
    report = suite.source_preflight()
    assert report["negative_bank_required"] is False
    assert report["enabled_datasets"] == ["levir", "tinyperson"]
    assert report["disabled_datasets"] == ["varroa", "visdrone"]
    assert all(not item["training_enabled"] for item in report["control_references"].values())
    assert report["head_inputs"] == [19, 22]
    assert report["ftsc"]["policy"] == "f5"
    assert report["ftsc"]["evidence"] == ["position_gaussian", "dfl_distribution"]
    assert get_cfg(overrides=r1).copy_paste_mode == "negative_canvas"
    assert get_cfg(overrides=r4).negative_cp_degradation == "weak_blur"


def test_default_main_is_preflight_only(monkeypatch, capsys, tmp_path):
    called = []
    monkeypatch.setattr(suite, "train_one", lambda *args, **kwargs: called.append((args, kwargs)))
    suite.main(["--dataset-root", str(tmp_path)])
    assert called == []
    assert "PREPARED_NOT_RUN" in capsys.readouterr().out
