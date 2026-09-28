"""No-training tests for the isolated LEVIR FTSC H2 MuSGD CP V1 suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import train_levir_ftsc_musgd_cp_ablation_v1 as runner
import train_levir_ftsc_augmentation_suite as historical


def _changed_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {key for key in left if left[key] != right[key]}


def test_exact_case_registry_and_frozen_common_protocol():
    assert runner.CASES == (
        "MG0_H2_MUSGD_BASE", "MG1_H2_MUSGD_CP2_RANDOM", "MG2_H2_MUSGD_CP2_COLLISION",
        "MG3_H2_MUSGD_CP2_COLLISION_P030", "MG4_H2_MUSGD_CP2_COLLISION_P070",
        "MG5_H2_MUSGD_CP1_COLLISION", "MG6_H2_MUSGD_CP2_POSITIVE_ONLY",
    )
    args = runner.parse_args([])
    assert args.seeds == [42, 43, 44]
    assert (args.split_seed, args.epochs, args.imgsz, args.batch_size, args.workers, args.patience) == (42, 100, 512, 8, 4, 20)
    for case in runner.CASES:
        kwargs = runner.train_kwargs(args, case, Path("split.yaml"), 42)
        assert kwargs["optimizer"] == "MuSGD"
        assert (kwargs["lr0"], kwargs["momentum"], kwargs["cos_lr"], kwargs["lrf"]) == (0.01, 0.9, False, 0.01)
        assert kwargs["fitness_metric"] == "map50_95"
        assert kwargs["deterministic"] is True and kwargs["copy_paste"] == 0.0
        assert (kwargs["mosaic"], kwargs["close_mosaic"], kwargs["adaptive_zoom"], kwargs["mixup"], kwargs["cutmix"]) == (0.0, 0, False, 0.0, 0.0)


def test_mg_cases_are_factor_isolated_and_mg1_is_canonical_cp2():
    configs = {case: runner.augmentation_for(case) for case in runner.CASES}
    mg0, mg1, mg2 = configs[runner.CASES[0]], configs[runner.CASES[1]], configs[runner.CASES[2]]
    assert mg0["copy_paste_enabled"] is False
    # A3 is the established canonical LEVIR CP2 configuration.  Compare the actual CP settings, not a copy.
    historical_cp2 = historical.augmentation_for("A3_CP2")
    cp_keys = {key for key in mg1 if key.startswith("copy_paste_")}
    assert {key: mg1[key] for key in cp_keys} == {key: historical_cp2[key] for key in cp_keys}
    assert _changed_keys(mg1, mg2) == {"copy_paste_placement"}
    assert _changed_keys(mg2, configs["MG3_H2_MUSGD_CP2_COLLISION_P030"]) == {"copy_paste_p"}
    assert _changed_keys(mg2, configs["MG4_H2_MUSGD_CP2_COLLISION_P070"]) == {"copy_paste_p"}
    assert _changed_keys(mg2, configs["MG5_H2_MUSGD_CP1_COLLISION"]) == {"copy_paste_copies"}
    assert _changed_keys(mg2, configs["MG6_H2_MUSGD_CP2_POSITIVE_ONLY"]) == {"copy_paste_allow_empty_target"}


def test_h2_graph_dataset_and_matrix_are_explicit(tmp_path):
    args = runner.parse_args(["--dataset-root", str(tmp_path / "datasets")])
    report = runner.source_preflight(args)
    assert report["dataset"] == "LEVIR-Ship" and report["split_seed"] == 42
    assert report["head_inputs"] == [19, 22]
    assert report["ftsc"]["evidence"] == ["position_gaussian", "dfl_distribution"]
    matrix = runner.compact_matrix()
    assert len(matrix) == 7
    assert all(row["model"] == runner.H2_MODEL_CONFIG.name and row["copy_paste_enabled"] == (row["case"] != runner.CASES[0]) for row in matrix)


def test_musgd_routing_is_reported_and_identical_for_all_cases():
    routing = runner.musgd_routing_preflight()
    assert routing["optimizer_requested"] == routing["optimizer_resolved"] == "MuSGD"
    assert (routing["base_lr"], routing["momentum"], routing["lr3_multiplier"], routing["lr3"]) == (0.01, 0.9, 3, 0.03)
    assert routing["lr3_parameter_count"] == len(routing["lr3_parameter_names"])
    assert routing["muon_parameter_count"] > 0 and routing["sgd_parameter_count"] > 0
    case_routes = list(routing["by_case"].values())
    assert len(case_routes) == 7 and all(route == case_routes[0] for route in case_routes)


def test_historical_adamw_suite_contract_is_unchanged():
    # The new family imports the historical runner only for this parity assertion; it does not mutate it.
    args = historical.parse_args([])
    assert historical.EXPERIMENT == "levir_ftsc_h2_augmentation_suite"
    assert historical.train_kwargs(args, "A3_CP2", Path("split.yaml"), 42)["optimizer"] == "auto"
    assert historical.augmentation_for("A3_CP2")["copy_paste_placement"] == "random"
