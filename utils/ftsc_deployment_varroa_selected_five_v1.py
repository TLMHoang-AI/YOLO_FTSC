#!/usr/bin/env python3
"""Varroa locked-five paper backend parity, without changing historical runs.

Reuses the source-verified paper evaluator with the original Varroa split,
native rect YOLO.val(batch=4, imgsz=640, conf=.001, iou=.7) for main AP,
and separate YOLO.predict(batch=4, conf=.25, iou=.5) TinyBenchmark small AP.
Dynamic TRT must match the exact PT32 post-preprocessing input bytes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from utils import ftsc_deployment_paper_backend_v1 as base

ROOT = base.ROOT
CONTRACT = "ftsc_varroa_paper_selected_five_backend_parity_v1"
REVISION = "seed_specific_native_rect_dynamic_engine_v1"

# EDGE_P2 intentionally selects the paper Edge-at-P2 ablation, never Edge P3.
FIVE_CASES = (
    "FTSC",
    "FTSC_YOLOV9T",
    "FTSC_YOLO11N",
    "EDGE_P2",
    "COLOR_CUE",
)
STAGES = {
    "varroa_five_seed42_fp32": (FIVE_CASES, (42,), ("pt32", "trt32")),
    "varroa_five_three_seed_fp32": (FIVE_CASES, (42, 43, 44), ("pt32", "trt32")),
    "varroa_five_all_precision": (FIVE_CASES, (42, 43, 44), base.BACKENDS),
}
# Locked capstone Main Results and Edge/Color ablation mean in PERCENT.
# Read-only historical anchors; NEVER used to fill new backend metrics.
PAPER_PT32_MEANS = {
    "FTSC": (91.29, 34.99, 88.83),
    "FTSC_YOLOV9T": (90.50, 34.71, 86.65),
    "FTSC_YOLO11N": (91.28, 34.75, 88.36),
    "EDGE_P2": (91.36, 35.30, 87.00),
    "COLOR_CUE": (90.35, 34.97, 85.54),
}
MODEL_CONFIGS = {
    "FTSC": "models_related/models_config/yolov8/varroa/yolov8n_varroa_ftsc_v1_p2p3.yaml",
    "FTSC_YOLOV9T": "models_related/models_config/yolov9/varroa/yolov9t_p2_varroa_ftsc_h2.yaml",
    "FTSC_YOLO11N": "models_related/models_config/yolo11/varroa/yolo11n_p2_varroa_ftsc_h2.yaml",
    "EDGE_P2": "models_related/models_config/yolov8/varroa/yolov8n_varroa_ftsc_edge_p2_detect_only.yaml",
    "COLOR_CUE": "models_related/models_config/yolov8/varroa/yolov8n_varroa_ftsc_v3_p2p3_color.yaml",
}
# Freeze the inspected canonical YAMLs; refuse silent architectural edits.
MODEL_YAML_SHA256 = {
    "FTSC": "cdab99127e6172f75a5a35f1b2cc7ae1b5a08a316447fec38092f33edde6cb09",
    "FTSC_YOLOV9T": "1926bbb6579df138870693a5d5fe9f56e16f4ca8a4c0c0a5d99b74a1e2431372",
    "FTSC_YOLO11N": "b3774b373cb534389fb3a596f85c13bf89c43483cfdc8938710bf5a222f92212",
    "EDGE_P2": "f80e4a20e0a6048f9a4bed705012a802e4ac5ce615781c7b6dd3596d4b198733",
    "COLOR_CUE": "ddecbb5a6fc680b3f46399663aeb4e7b415ec5044c70535012554b6fb1ce4997",
}
CANONICAL_RUNS = {
    "FTSC": "Varroa/original_runs/V1_FTSC",
    "FTSC_YOLOV9T": "Varroa/cross_yolo_portability/yolov9t/yolov9t__FTSC_H2",
    "FTSC_YOLO11N": "Varroa/cross_yolo_portability/yolo11n/yolo11n__FTSC_H2",
    "EDGE_P2": "Varroa/additional_ablations/varroa_ftsc_edge_placement_headscale/EDGE_P2_DETECT_ONLY",
    "COLOR_CUE": "Varroa/original_runs/V3_COLOR",
}
# Exact original HF best.pt SHA for ALL 5 x 3 selected jobs (not only seed42).
CHECKPOINT_SHA256 = {
    "FTSC": {
        42: "161b6ad89d1d4ca759b9a4565ee8a710b3d1e790c52b7ad2e1f74bb33507915e",
        43: "4677fb5708d2657340d231c52069eb0116e7f25a85769abb37ec099f840982b9",
        44: "65a7a4e2251051eecd08acd8192f66679193b496dbb996cab526bd6cd0bf7f66",
    },
    "FTSC_YOLOV9T": {
        42: "bfb182f00633fc0b383f47c780a2e9d88bf3cf5c37f65934df680e040befd7b2",
        43: "5ff2acde1a01ea521ba83c544fc2187deb8b497dc60437a1439df6d71476ff15",
        44: "3c3ae667009411a71d612695a09cf9ea02f50207c631bf6c375ef3cd7e817c1e",
    },
    "FTSC_YOLO11N": {
        42: "85098d9a849962f5a5e7b0d2e749f1b3517d02f6ab4b881d8ee1e33511cc4255",
        43: "cd3bba4d70fc58cdb09161bab239e562d73b492b0dfdd7499b5584e92e94ddb8",
        44: "078834b6539e963cd3f2b3ec7181ec3e09ee5b85ce2cdc574d05384c4969252b",
    },
    "EDGE_P2": {
        42: "90ed2c900e3c7e34a1a3ef44d545b0bca611e4d486ccedbd05e223ec0fe92184",
        43: "8462ac578df5fa3190b22b05793338777dbe1377d7232d0196bf906d74648f26",
        44: "8a8304cb67a843f80b24256cb0d58f70e1b9386d7f33c4eba97d0f98371b8414",
    },
    "COLOR_CUE": {
        42: "3d4d0365dcb00eebc9f8c88a7b4d6d81d0b6d771b456ad6073f1e04c42c15528",
        43: "187462df4cd6238e32ece0fdae0b21ef0d8da2a94f0c5f11bee8197caf0b91d3",
        44: "c7b8eec985b152bd4e90315a3f1955f714be5dac1e80af7ab16b89021c96600e",
    },
}
SELF_SOURCE = "utils/ftsc_deployment_varroa_selected_five_v1.py"


def validate_selected_sources(manifest: Path) -> dict:
    """Fail closed on incorrect Varroa case, path, SHA or local architecture."""
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    inventory = data["datasets"]["Varroa"]["cases"]
    out = {}
    for case in FIVE_CASES:
        info = inventory.get(case)
        base.need(isinstance(info, dict), f"Varroa paper case absent: {case}")
        base.need(info.get("framework") == "ultralytics",
                  f"{case}: unsupported backend framework")
        base.need(info.get("source_run") == CANONICAL_RUNS[case],
                  f"{case}: wrong canonical source run")
        observed_seeds = info.get("seeds", {})
        base.need(set(observed_seeds) == {"42", "43", "44"},
                  f"{case}: missing or extra pinned seeds")
        pinned = {}
        for seed in (42, 43, 44):
            row = observed_seeds[str(seed)]
            expect_path = f"{CANONICAL_RUNS[case]}/seed_{seed}/weights/best.pt"
            expect_sha = CHECKPOINT_SHA256[case][seed]
            base.need(row.get("path") == expect_path and
                      row.get("sha256") == expect_sha,
                      f"{case}/seed_{seed}: checkpoint path/SHA changed")
            pinned[str(seed)] = {"path": expect_path, "sha256": expect_sha}
        model_yaml = ROOT / MODEL_CONFIGS[case]
        base.need(model_yaml.is_file() and
                  base.sha(model_yaml) == MODEL_YAML_SHA256[case],
                  f"{case}: canonical model YAML missing/modified")
        out[case] = {
            "source_run": CANONICAL_RUNS[case],
            "model_yaml": str(model_yaml),
            "model_yaml_sha256": MODEL_YAML_SHA256[case],
            "seeds": pinned,
        }
    return out


def run_tests() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest",
         "utils.tests.test_ftsc_deployment_varroa_selected_five_v1"],
        cwd=str(ROOT), check=True,
    )


def install_isolated_contract() -> None:
    """Only change current Python process; retain other runner/source hashes."""
    original_source_hashes = base.source_hashes

    def hashes():
        result = original_source_hashes()
        result[SELF_SOURCE] = base.sha(ROOT / SELF_SOURCE)
        for cfg in sorted(set(MODEL_CONFIGS.values())):
            result[cfg] = base.sha(ROOT / cfg)
        return result

    base.CONTRACT = CONTRACT
    base.REVISION = REVISION
    base.STAGES = STAGES
    base.PAPER_ANCHORS["Varroa"] = PAPER_PT32_MEANS.copy()
    base.source_hashes = hashes


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(x == "--dataset" or x.startswith("--dataset=") for x in arguments):
        arguments.extend(("--dataset", "Varroa"))
    if not any(x == "--stage" or x.startswith("--stage=") for x in arguments):
        arguments.extend(("--stage", "varroa_five_seed42_fp32"))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default="Varroa")
    parser.add_argument("--stage", default="varroa_five_seed42_fp32")
    parser.add_argument("--phase", default="plan")
    parser.add_argument("--report-manifest", type=Path,
                        default=base.deploy.REPORT_MANIFEST)
    parser.add_argument("--run-offline-tests", action="store_true")
    known, _ = parser.parse_known_args(arguments)
    base.need(known.dataset == "Varroa", "Varroa-only selected-five runner")
    base.need(known.stage in STAGES, "Invalid Varroa selected-five stage")
    base.need(known.phase in ("plan", "verify", "preflight",
                              "evaluate", "audit", "aggregate"),
              "Invalid phase")
    pinned = validate_selected_sources(known.report_manifest.expanduser().resolve())
    if known.phase == "evaluate" and known.run_offline_tests:
        run_tests()
    install_isolated_contract()
    result = base.main(arguments)
    if known.phase == "plan":
        print(json.dumps({
            "selection": "VARROA_FIVE_LOCKED_PAPER",
            "cases": list(FIVE_CASES),
            "paper_reference_means_pct": PAPER_PT32_MEANS,
            "pinned_sources": pinned,
            "contract": CONTRACT,
            "revision": REVISION,
            "dataset_specific_protocol": base.CONFIG["Varroa"],
        }, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
