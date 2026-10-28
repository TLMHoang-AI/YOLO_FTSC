"""Reconstruct LEVIR cases whose ORIGINAL archived best.pt is provably absent.

Strictly separate from historical checkpoints, phase2's two corrected cues,
and the three-dataset 30-run plan. Not an alias for old seed checkpoints.
Uses the original LEVIR head-pruning no-Mosaic recipe through the same hardened
paper runner. Run only after successful authenticated HF inventory, never
after network, identity or metadata errors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as heads
from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_h1_h2_nextgen as cues
from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_edge_stability as edges
from train_levir_scripts import train_levir_ftsc_cue_corrections_v2 as paper
from utils.ftsc_paper_preflight import (
    canonical_ftsc, layers, load_yaml, require, static_case_report,
)

# Only these historical cases are explicitly authorized for source-equivalent
# NEW training when no original case+seed checkpoint exists on the known HF
# historical namespace. Do not apply this fallback to TinyPerson or Varroa.
ORIGINAL_CASES = {
    "H0": "H0_p2_p3_p4_p5_nomosaic",
    "H1": "H1_p2_p3_p4_nomosaic",
    "H2": "H2_p2_p3_nomosaic",
    "H3": "H3_p2_only_nomosaic",
}
RECON_CASE_IDS = {case: f"RECON_{case}_V1" for case in (
    *ORIGINAL_CASES, "C_H2_LEGACY", "ES3_LEGACY",
)}


def _case_spec(config: Path, case: str, graph: str) -> paper.CaseSpec:
    payload = load_yaml(config)
    detect = tuple(int(x) for x in layers(payload)[-1][0])
    # Head-pruning stride definition is fixed from its original runner.
    original = case.removeprefix("RECON_").removesuffix("_V1")
    if original in ORIGINAL_CASES:
        strides = tuple(heads.EXPECTED_STRIDES[ORIGINAL_CASES[original]])
    else:
        strides = (4., 8.)
    return paper.CaseSpec(
        yaml=config,
        cue_module="P2ColorCueFusion" if original == "C_H2_LEGACY" else (
            "P2EdgeCueFusion" if original == "ES3_LEGACY" else ""),
        detect_from=detect, strides=strides,
        graph_contract=graph,
        namespace=case, legacy_yaml=config,
    )


CASES: dict[str, paper.CaseSpec] = {
    RECON_CASE_IDS[short]: _case_spec(
        heads.VARIANTS[original], RECON_CASE_IDS[short],
        f"reconstructed_levir_{short.lower()}_original_graph_nomosaic_v1",
    )
    for short, original in ORIGINAL_CASES.items()
}
CASES[RECON_CASE_IDS["C_H2_LEGACY"]] = _case_spec(
    cues.VARIANTS["C_H2"], RECON_CASE_IDS["C_H2_LEGACY"],
    "reconstructed_levir_legacy_color_h2_prefusion_v1",
)
CASES[RECON_CASE_IDS["ES3_LEGACY"]] = _case_spec(
    edges.VARIANTS["ES3"], RECON_CASE_IDS["ES3_LEGACY"],
    "reconstructed_levir_legacy_es3_uniform025_prefusion_v1",
)


def source_preflight() -> dict[str, Any]:
    reference = load_yaml(heads.VARIANTS[ORIGINAL_CASES["H2"]])
    report: dict[str, Any] = {}
    for case, spec in CASES.items():
        payload = load_yaml(spec.yaml)
        require(canonical_ftsc(payload) == canonical_ftsc(reference),
                f"{case}: original FTSC F5/alpha6 configuration changed")
        require(layers(payload)[-1][2] == "Detect" and
                tuple(layers(payload)[-1][0]) == spec.detect_from,
                f"{case}: original Detect taps changed")
        require("RepC2f" not in {row[2] for row in layers(payload)},
                f"{case}: LEVIR structural family must remain C2f")
        if case.startswith("RECON_H"):
            static_case_report(spec.yaml, reference, spec.detect_from, "C2f", case)
        else:
            # Original Color/ES3 used pre-fusion Detect routing. Reproduce that
            # for legacy attribution ONLY; do not relabel as the corrected V2.
            require(spec.detect_from == (19, 22),
                    f"{case}: legacy Detect routing was not preserved")
            cue = "P2ColorCueFusion" if "C_H2" in case else "P2EdgeCueFusion"
            require(sum(row[2] == cue for row in layers(payload)) == 1,
                    f"{case}: original auxiliary cue intervention changed")
        report[case] = {
            "yaml": str(spec.yaml), "graph_contract": spec.graph_contract,
            "detect_from": list(spec.detect_from),
            "training_lineage": "original_LEVIR_head_pruning_100ep_nomosaic",
            "provenance": "NEW_RECONSTRUCTED_COEFFICIENTS_NOT_HISTORICAL_WEIGHTS",
        }
    return report


def run_one(args: Any) -> Path:
    require(args.case in CASES, f"Unregistered reconstruction case: {args.case}")
    require(args.seed in paper.DEFAULT_SEEDS, "Only LEVIR seeds 42/43/44 are authorized")
    return paper.run_one(args, registry=CASES)
