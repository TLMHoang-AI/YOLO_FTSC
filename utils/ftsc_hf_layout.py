"""Canonical FTSC Hugging Face layout (post-reorganization, November 2026).

Only the three research datasets are supported here. Public old-prefix
discovery is deliberately NOT a fallback: legacy copies can belong to
different evaluation/training cohorts.

This module performs no network IO and never supplies credentials.
"""
from __future__ import annotations

import re
from pathlib import PurePosixPath

REPO_ID = "TLMHoang/FTSC_results"
DATASETS = ("LEVIR-Ship", "TinyPerson", "Varroa")
SEEDS = (42, 43, 44)

# Exact, audited checkpoint locations for the cross-team augmentation table.
# Each value is a run directory without the /weights/best.pt suffix.
AUGMENTATION_RUNS: dict[str, dict[str, str]] = {
    "LEVIR-Ship": {
        "FTSC": "historical_reconstructions/RECON_H2_V1",
        "FTSC_A0_AUG_CONTROL": "original_runs/A0_FTSC",
        "CP2": "original_runs/A3_CP2",
        "OACP": "original_runs/A1_OACP_R2",
        "NC_R4": "original_runs/A6_NEGCANVAS_R4",
    },
    "TinyPerson": {
        "FTSC": "original_runs/H2_MOSAIC",
        "CP2": "additional_ablations/tinyperson_h2_generalization_completion_v1/TP_H2_MOSAIC_CP2",
        "OACP": "additional_ablations/tinyperson_h2_generalization_completion_v1/TP_H2_MOSAIC_OACP_R2",
    },
    "Varroa": {
        "FTSC": "original_runs/V1_FTSC",
        "CP2": "additional_ablations/varroa_ftsc_next_ablation_suite/FTSC_P23_CP2",
        "OACP": "additional_ablations/varroa_h2_generalization_completion_v1/VR_H2_MOSAIC_OACP_R2",
    },
}

# Strict publication routing: no further writes to repository root namespaces.
# Unknown experiment names in known datasets are routed to new additional_ablations,
# never into an existing original_runs, historical_reconstructions, or journal cohort.
NAMESPACE_ROOTS = {
    "levir_ftsc_h2_augmentation_suite": "LEVIR-Ship/original_runs",
    "levir_ftsc_R_variants_augmentation_suite": "LEVIR-Ship/original_runs",
    "tinyperson_h2_generalization_completion_v1": "TinyPerson/additional_ablations/tinyperson_h2_generalization_completion_v1",
    "varroa_h2_generalization_completion_v1": "Varroa/additional_ablations/varroa_h2_generalization_completion_v1",
    "varroa_ftsc_next_ablation_suite": "Varroa/additional_ablations/varroa_ftsc_next_ablation_suite",
    "levir_ftsc_journal_phase2_v1": "LEVIR-Ship/journal_experiments/phase2",
    "levir_v8_p23_ftsc_off_hf_first_v1": "LEVIR-Ship/journal_experiments/ftsc_off_hf_first_v1",
    "levir_v8_p23_portability_backfill_v1": "LEVIR-Ship/journal_experiments/causal_backfill",
    "levir_ftsc_position_alpha_historical_h2_v1_rerun_d3e60b1e59d2": "LEVIR-Ship/journal_experiments/alpha_sensitivity",
    "tinyperson_ftsc_causal_suite": "TinyPerson/original_runs",
    "varroa_ftsc_causal_suite": "Varroa/original_runs",
    "tinyperson_ftsc_journal_metric_completion_v1": "TinyPerson/derived_evaluations/tinyperson_ftsc_journal_metric_completion_v1",
    "varroa_ftsc_fullmetric_recovery_v1": "Varroa/derived_evaluations/varroa_ftsc_fullmetric_recovery_v1",
    "tinyperson_ftsc_journal_phase2_v1": "TinyPerson/journal_experiments/phase2",
    "varroa_ftsc_journal_phase2_v1": "Varroa/journal_experiments/phase2",
    "levir_ftsc_position_alpha_historical_h2_v1": "LEVIR-Ship/journal_experiments/alpha_sensitivity",
    "tinyperson_ftsc_position_alpha_h2_v1": "TinyPerson/journal_experiments/alpha_sensitivity",
    "varroa_ftsc_position_alpha_h2_v1": "Varroa/journal_experiments/alpha_sensitivity",
    "levir_ftsc_cross_yolo_portability_v1": "LEVIR-Ship/cross_yolo_portability",
    "tinyperson_ftsc_cross_yolo_portability_v1": "TinyPerson/cross_yolo_portability",
    "varroa_ftsc_cross_yolo_portability_v1": "Varroa/cross_yolo_portability",
}

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SEED_NAME = re.compile(r"^seed_(42|43|44)(?:_corner_sw640_sh512)?$")


class HFLayoutError(ValueError):
    """A remote FTSC path was ambiguous or inconsistent with the new layout."""


def segment(value: str) -> str:
    if not isinstance(value, str) or not _SAFE.fullmatch(value) or value in (".", ".."):
        raise HFLayoutError(f"Unsafe HF path component: {value!r}")
    return value


def prefix(value: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise HFLayoutError(f"Unsafe HF prefix: {value!r}")
    parts = value.split("/")
    for part in parts:
        segment(part)
    if parts[0] not in DATASETS:
        raise HFLayoutError("FTSC remote paths must begin with a canonical dataset directory")
    return "/".join(parts)


def seed_dir(dataset: str, seed: int) -> str:
    if dataset not in DATASETS or type(seed) is not int or seed not in SEEDS:
        raise HFLayoutError(f"Unsupported FTSC dataset/seed: {dataset}/{seed}")
    return f"seed_{seed}" + ("_corner_sw640_sh512" if dataset == "TinyPerson" else "")


def augmentation_run(dataset: str, case: str, seed: int) -> str:
    """Exact migrated source directory, for checkpoint and prediction recovery."""
    if dataset not in AUGMENTATION_RUNS or case not in AUGMENTATION_RUNS[dataset]:
        raise HFLayoutError(f"Unknown audited augmentation case: {dataset}/{case}")
    return prefix(f"{dataset}/{AUGMENTATION_RUNS[dataset][case]}/{seed_dir(dataset, seed)}")


def augmentation_checkpoint(dataset: str, case: str, seed: int) -> str:
    return augmentation_run(dataset, case, seed) + "/weights/best.pt"


def dataset_of_namespace(namespace: str) -> str:
    segment(namespace)
    if namespace.startswith("levir_") or namespace == "levir":
        return "LEVIR-Ship"
    if namespace.startswith("tinyperson_") or namespace == "tinyperson":
        return "TinyPerson"
    if namespace.startswith("varroa_") or namespace == "varroa":
        return "Varroa"
    raise HFLayoutError(f"Unrecognized FTSC experiment namespace: {namespace}")


def training_root(namespace: str, *, family: str | None = None) -> str:
    """Directory above case/run-name for a NEW training archive.

    This never chooses a historical checkpoint by matching substring/metric.
    """
    dataset = dataset_of_namespace(namespace)
    if namespace in NAMESPACE_ROOTS:
        root = NAMESPACE_ROOTS[namespace]
    else:
        root = f"{dataset}/additional_ablations/{segment(namespace)}"
    if "cross_yolo_portability" in namespace:
        if family is None:
            raise HFLayoutError("Cross-YOLO archive requires explicit model family")
        root += "/" + segment(family)
    return prefix(root)


def evaluation_root(dataset: str, suite: str = "capstone_aug_sixcore_v1") -> str:
    if dataset not in DATASETS:
        raise HFLayoutError(f"Unknown FTSC dataset: {dataset}")
    return prefix(f"{dataset}/derived_evaluations/{segment(suite)}")
