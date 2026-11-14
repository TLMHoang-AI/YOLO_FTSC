#!/usr/bin/env python3
"""TinyPerson five-paper-case deployment evaluation, safely isolated from LEVIR.

All three accuracy metrics originate from official original-image merged
TinyBenchmark, using the *same backend's own corner detections*:
  corner sw640/sh512 -> YOLO.predict imgsz640, conf.001, IoU.5,
  max_det300, 64-image historical source lists (PT32/PT16) -> translate
  into original coords -> class-0 only -> merge NMS .5 ->
  TinyBenchmark original-image GT (IoU .50:.05:.95, Small bucket).
TensorRT dynamic batch8 mimics original 64-corner auto/rect choice while
repeating preprocessing on its own engine; all actual preprocessed
per-image uint8 RGB CHW bytes must match same-case/seed PT32.

Never train, upload, replace checkpoints, patch the original deployment
runner, or modify LEVIR's source-hash-pinned validation records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from utils import ftsc_deployment_paper_backend_v1 as base

ROOT = base.ROOT
CONTRACT = "ftsc_tinyperson_paper_selected_five_backend_parity_v1"
REVISION = "original_merged_corners_dynamic_engine_v1"
FIVE_CASES = (
    "FTSC",
    "FTSC_YOLOV9T",
    "FTSC_YOLO11N",
    "EDGE_CUE",
    "COLOR_CUE",
)
STAGES = {
    "tiny_five_seed42_fp32": (FIVE_CASES, (42,), ("pt32", "trt32")),
    "tiny_five_three_seed_fp32": (FIVE_CASES, (42, 43, 44), ("pt32", "trt32")),
    "tiny_five_all_precision": (FIVE_CASES, (42, 43, 44), base.BACKENDS),
}
# Locked capstone Main Results plus Edge/Color Ablation C. Percent units.
# Comparison is read-only; never use these values to synthesize backend AP.
PAPER_PT32_MEANS = {
    "FTSC": (57.68, 21.51, 69.65),
    "FTSC_YOLOV9T": (57.86, 21.74, 70.27),
    "FTSC_YOLO11N": (56.72, 21.05, 69.42),
    "EDGE_CUE": (57.31, 21.30, 69.43),
    "COLOR_CUE": (57.79, 21.52, 70.04),
}
# Local YAML names are authenticated against seed42 HF args.yaml and paper
# case mapping; full checkpoint byte SHA is verified by base.source_restore.
MODEL_CONFIGS = {
    "FTSC": "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_nomosaic_h2_p2_p3.yaml",
    "FTSC_YOLOV9T": "models_related/models_config/yolov9/tinyperson/yolov9t_p2_tinyperson_ftsc_h2.yaml",
    "FTSC_YOLO11N": "models_related/models_config/yolo11/tinyperson/yolo11n_p2_tinyperson_ftsc_h2.yaml",
    "EDGE_CUE": "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml",
    "COLOR_CUE": "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_ftsc_color_h2_mosaic.yaml",
}
CANONICAL_RUNS = {
    "FTSC": "TinyPerson/original_runs/H2_MOSAIC",
    "FTSC_YOLOV9T": "TinyPerson/cross_yolo_portability/yolov9t/yolov9t__FTSC_H2",
    "FTSC_YOLO11N": "TinyPerson/cross_yolo_portability/yolo11n/yolo11n__FTSC_H2",
    "EDGE_CUE": "TinyPerson/original_runs/ES1_MOSAIC",
    "COLOR_CUE": "TinyPerson/journal_experiments/phase2/TP_CUE_COLOR_H2_MOSAIC",
}
SEED42_HF_SHA256 = {
    "FTSC": "16c8d182b46b976c8e3bc70299950a356901ec7f205ed78b57cef050ca61fe79",
    "FTSC_YOLOV9T": "d550cdcaa5fa3fab172ab493ff90fbadab7840c026428d40d5428ab7472d7a91",
    "FTSC_YOLO11N": "698c48374d81324e74f07d7c8619c979077f881645d5a072826e2ccea8e28b3b",
    "EDGE_CUE": "2ac7ff5d2c3dd09ea2b0d0d7acff7e50313df72606fe0849833d8ec6fb956e51",
    "COLOR_CUE": "f3ba4ba3e182c1d9a49dc6caa2813258acc4901a43d223c1656957ff30db353a",
}
SELF_SOURCE = "utils/ftsc_deployment_tinyperson_selected_five_v1.py"


def validate_selected_sources(manifest: Path) -> dict:
    """Fail before plan/HF/engine on a renamed case or wrong paper checkpoint."""
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    original = data["datasets"]["TinyPerson"]["cases"]
    result = {}
    for name in FIVE_CASES:
        case = original.get(name)
        base.need(isinstance(case, dict),
                  f"TinyPerson paper case missing from HF inventory: {name}")
        base.need(case.get("source_run") == CANONICAL_RUNS[name],
                  f"{name}: report manifest points to wrong run")
        seed_paths = {}
        for seed in (42, 43, 44):
            row = case.get("seeds", {}).get(str(seed))
            base.need(isinstance(row, dict),
                      f"{name}: source seed {seed} is absent")
            path, digest = row.get("path", ""), row.get("sha256", "")
            expected = (f"{CANONICAL_RUNS[name]}/"
                        f"seed_{seed}_corner_sw640_sh512/weights/best.pt")
            base.need(path == expected and len(digest) == 64
                      and all(c in "0123456789abcdef" for c in digest),
                      f"{name}/{seed}: source path or SHA format differs from paper")
            if seed == 42:
                base.need(digest == SEED42_HF_SHA256[name],
                          f"{name}/42: source best.pt SHA changed from verified HF")
            seed_paths[str(seed)] = {"path": path, "sha256": digest}
        model_yaml = ROOT / MODEL_CONFIGS[name]
        base.need(model_yaml.is_file(), f"{name}: canonical model YAML missing: {model_yaml}")
        result[name] = {
            "model_yaml": str(model_yaml),
            "model_yaml_sha256": base.sha(model_yaml),
            "source_run": case["source_run"],
            "seeds": seed_paths,
        }
    return result


def _run_tests():
    subprocess.run(
        [sys.executable, "-m", "unittest", "utils.tests.test_ftsc_deployment_tinyperson_selected_five_v1"],
        cwd=str(ROOT), check=True,
    )


def install_isolated_contract() -> None:
    """Patch this process only; never edit/hash-change the LEVIR V2 source."""
    old_hashes = base.source_hashes

    def source_hashes():
        refs = old_hashes()
        refs[SELF_SOURCE] = base.sha(ROOT / SELF_SOURCE)
        for cfg in sorted(set(MODEL_CONFIGS.values())):
            refs[cfg] = base.sha(ROOT / cfg)
        return refs

    base.CONTRACT = CONTRACT
    base.REVISION = REVISION
    base.STAGES = STAGES
    base.PAPER_ANCHORS["TinyPerson"] = PAPER_PT32_MEANS.copy()
    base.source_hashes = source_hashes


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(s == "--dataset" or s.startswith("--dataset=") for s in arguments):
        arguments.extend(("--dataset","TinyPerson"))
    if not any(s == "--stage" or s.startswith("--stage=") for s in arguments):
        arguments.extend(("--stage","tiny_five_seed42_fp32"))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default="TinyPerson")
    parser.add_argument("--stage", default="tiny_five_seed42_fp32")
    parser.add_argument("--report-manifest", type=Path, default=base.deploy.REPORT_MANIFEST)
    parser.add_argument("--phase", default="plan")
    parser.add_argument("--run-offline-tests", action="store_true")
    known, _ = parser.parse_known_args(arguments)
    base.need(known.dataset == "TinyPerson",
              "Tiny-only paper runner cannot evaluate LEVIR or Varroa")
    base.need(known.stage in STAGES, "Use one of the three Tiny paper-five stages")
    base.need(known.phase in ("plan", "verify", "preflight", "evaluate", "audit", "aggregate"),
              "Invalid evaluation phase")
    sources = validate_selected_sources(known.report_manifest.expanduser().resolve())
    if known.phase == "evaluate" and known.run_offline_tests:
        _run_tests()
    install_isolated_contract()
    # base.main runs the original exact merged TinyBenchmark scorer and
    # all baseline checkpoint, native input, engine, source code and audit gates.
    out = base.main(arguments)
    if known.phase == "plan":
        print(json.dumps({
            "paper_selection": "TINY_FIVE_LOCKED_REPORT",
            "selected_cases": list(FIVE_CASES),
            "checkpoint_metadata": sources,
            "main_results_ablation_anchor_pct": PAPER_PT32_MEANS,
            "result_contract": CONTRACT,
            "revision": REVISION,
        }, indent=2))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
