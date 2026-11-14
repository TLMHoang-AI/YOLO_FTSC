#!/usr/bin/env python3
"""Evaluation-only FTSC deployment accuracy matrix.

No training, no HF uploads, no reuse of source-PyTorch predictions as backend
accuracy. Uses audited canonical HF checkpoint paths for existing FTSC cases.
Run accuracy evaluation on Marimo Live GPU/CPU across backend variants.
Jetson FPS/latency from a separate benchmark is NOT measured by this runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.ftsc_hf_layout import (
    AUGMENTATION_RUNS, DATASETS, REPO_ID, SEEDS,
    augmentation_checkpoint, seed_dir,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "FTSC_DEPLOY_EVAL_20261112_REPORT_V3"
PARTNER_CONTRACT = "FTSC_DEPLOY_PARTNER_PREDICT_COCO_V1"
PARTNER_PROTOCOL = "partner_predict_coco_v1"
PARTNER_RUNTIME_REVISION = "trt_named_io_v2"
NATIVE_PROTOCOL = "native_ultralytics_v1"
NATIVE_VAL_PROTOCOL = "native_ultralytics_val_v2"
NATIVE_VAL_CONTRACT = "FTSC_DEPLOY_NATIVE_ULTRALYTICS_VAL_V2"
NATIVE_VAL_RUNTIME_REVISION = "trt_named_io_val_v2"
# LEVIR only: preserve paper YOLO.val pre-letterbox imgsz=512 and rect=True.
# Static TRT shape (544x544) is derived from the prior validator batch_shapes.
RECT_PARITY_PROTOCOL = "native_ultralytics_val_rect_parity_v3"
RECT_PARITY_CONTRACT = "FTSC_DEPLOY_LEVIR_RECT_CANVAS_PARITY_V3"
RECT_PARITY_RUNTIME_REVISION = "levir_512_padded_544_static_trt_v1"
RECT_PAPER_IMGSZ = 512
RECT_PAPER_STRIDE = 32  # PyTorchBackend enforces max(model.stride, 32)
RECT_ENGINE_CANVAS = 544  # measured 99/99 historical LEVIR seed42 batches
RECT_PT_VAL_BATCH = 8
TINY_MERGED_CONTRACT = "tinybenchmark_merged_original_iou50_95_v2"
DEFAULT_BACKENDS = ("pt32", "pt16", "trt32", "trt16")
BACKENDS = DEFAULT_BACKENDS + ("ort_cpu32", "ort_cpu16")
METRIC_KEYS = {
    "LEVIR-Ship": ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small"),
    "TinyPerson": ("test_merged/AP50", "test_merged/mAP50-95", "test_merged/AP50-Small"),
    "Varroa": ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small"),
}
EVALUATOR = {
    "LEVIR-Ship": "ultralytics_test_plus_tinybenchmark_native_small_50_75_v1",
    "TinyPerson": "tinybenchmark_merged_original_iou50_95_v2",
    "Varroa": "ultralytics_test_plus_tinybenchmark_native_small_50_75_v1",
}


PARTNER_METRIC_KEYS = {
    "LEVIR-Ship": ("test_coco/AP50", "test_coco/mAP50-95", "test_size/AP50-Small"),
    "TinyPerson": METRIC_KEYS["TinyPerson"],
    "Varroa": ("test_coco/AP50", "test_coco/mAP50-95", "test_size/AP50-Small"),
}
PARTNER_EVALUATOR = {
    "LEVIR-Ship": "partner_predict_coco_50_95_plus_tinybenchmark_native_ap50_small_v1",
    "TinyPerson": "partner_predict_tinybenchmark_merged_original_50_95_v1",
    "Varroa": "partner_predict_coco_50_95_plus_tinybenchmark_native_ap50_small_v1",
}


@lru_cache(maxsize=1)
def partner_source_hashes() -> dict[str, str]:
    """Pin all files capable of changing partner export, inference or AP logic."""
    names = (
        "utils/ftsc_deployment_backend_eval_v1.py",
        "utils/ftsc_partner_backend_metrics_v1.py",
        "models_related/ultralytics/ultralytics/nn/backends/tensorrt.py",
        "models_related/ultralytics/ultralytics/nn/backends/pytorch.py",
        "models_related/ultralytics/ultralytics/nn/modules/head.py",
        "models_related/ultralytics/ultralytics/nn/autobackend.py",
        "models_related/ultralytics/ultralytics/engine/exporter.py",
        "models_related/ultralytics/ultralytics/engine/predictor.py",
        "models_related/ultralytics/ultralytics/models/yolo/detect/predict.py",
        "models_related/ultralytics/ultralytics/utils/nms.py",
        "evaluate_test/standard_detection_metrics.py",
        "evaluate_test/size_bucket_evaluator.py",
        "train_tinyperson_scripts/train_all_tinyperson.py",
        "vendor/tinyperson_cocoeval.py",
    )
    return {name: sha256(ROOT / name) for name in names}


@lru_cache(maxsize=1)
def rect_parity_source_hashes() -> dict[str, str]:
    """Lock v3 evaluation, dataset padding, validator, exporter and backends."""
    names = (
        "utils/ftsc_deployment_backend_eval_v1.py",
        "models_related/ultralytics/ultralytics/nn/backends/tensorrt.py",
        "models_related/ultralytics/ultralytics/nn/backends/pytorch.py",
        "models_related/ultralytics/ultralytics/nn/modules/head.py",
        "models_related/ultralytics/ultralytics/nn/autobackend.py",
        "models_related/ultralytics/ultralytics/engine/validator.py",
        "models_related/ultralytics/ultralytics/engine/model.py",
        "models_related/ultralytics/ultralytics/engine/exporter.py",
        "models_related/ultralytics/ultralytics/data/base.py",
        "models_related/ultralytics/ultralytics/data/build.py",
        "models_related/ultralytics/ultralytics/data/dataset.py",
        "models_related/ultralytics/ultralytics/data/augment.py",
        "models_related/ultralytics/ultralytics/models/yolo/detect/val.py",
        "models_related/ultralytics/ultralytics/utils/nms.py",
        "evaluate_test/standard_detection_metrics.py",
        "evaluate_test/size_bucket_evaluator.py",
    )
    return {name: sha256(ROOT / name) for name in names}


def rect_parity_mode(args: argparse.Namespace) -> bool:
    return evaluation_protocol(args) == RECT_PARITY_PROTOCOL


def native_val_mode(args: argparse.Namespace) -> bool:
    return evaluation_protocol(args) in (NATIVE_VAL_PROTOCOL, RECT_PARITY_PROTOCOL)


def evaluation_protocol(args: argparse.Namespace) -> str:
    return getattr(args, "evaluation_protocol", NATIVE_PROTOCOL)


def evaluator_name(args: argparse.Namespace) -> str:
    if rect_parity_mode(args):
        return "ultralytics_paper_rect_native_val_with_static_engine_544_v3"
    if native_val_mode(args) and args.dataset != "TinyPerson":
        return "ultralytics_test_val_json_plus_tinybenchmark_native_small_v2"
    return (PARTNER_EVALUATOR if evaluation_protocol(args) == PARTNER_PROTOCOL else EVALUATOR)[args.dataset]


def metric_keys(dataset: str, protocol: str = NATIVE_PROTOCOL) -> tuple[str, str, str]:
    return (PARTNER_METRIC_KEYS if protocol == PARTNER_PROTOCOL else METRIC_KEYS)[dataset]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_for(project: Path, dataset: str, case: str, seed: int, backend: str) -> Path:
    return project / dataset / case / seed_dir(dataset, seed) / backend


def source_path(project: Path, dataset: str, case: str, seed: int) -> Path:
    return project / "_checkpoints" / dataset / case / seed_dir(dataset, seed) / "weights" / "best.pt"



_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def manifest_entries(path: Path | None, dataset: str) -> dict[str, dict[int, dict[str, str]]]:
    """Optional explicit HF mapping for other checkpoint variants (never guessed)."""
    if path is None:
        return {}
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    require(data.get("repo_id") == REPO_ID, "Manifest must use the audited FTSC dataset repo")
    cases = data.get("datasets", {}).get(dataset, {})
    require(isinstance(cases, dict), "Malformed deployment manifest datasets/cases")
    output: dict[str, dict[int, dict[str, str]]] = {}
    for case, per_seed in cases.items():
        require(_SEGMENT.fullmatch(case) and case not in AUGMENTATION_RUNS[dataset],
                f"Invalid/colliding external case label: {case}")
        require(isinstance(per_seed, dict), f"Invalid seed mapping: {case}")
        output[case] = {}
        for seed_text, entry in per_seed.items():
            require(seed_text in ("42", "43", "44") and isinstance(entry, dict),
                    f"Invalid manifest seed: {case}/{seed_text}")
            remote = entry.get("path")
            digest = entry.get("sha256")
            require(isinstance(remote, str) and remote.startswith(dataset + "/")
                    and remote.endswith("/weights/best.pt") and
                    all(_SEGMENT.fullmatch(part) for part in remote.split("/")),
                    f"Unsafe/invalid HF checkpoint path: {case}/{seed_text}")
            require(isinstance(digest, str) and _SHA256.fullmatch(digest),
                    f"Missing exact checkpoint SHA-256: {case}/{seed_text}")
            output[case][int(seed_text)] = {"path": remote, "sha256": digest}
    return output



REPORT_MANIFEST = ROOT / "utils/ftsc_report_deployment_case_manifest_v1.json"
REPORT_SCHEMA = "ftsc_report_driven_deployment_case_manifest_v1"


def load_report_scope(path: Path, dataset: str) -> tuple[dict[str, dict[int, dict[str, str]]], dict[str, list[str]], list[dict[str, str]]]:
    """Strict, report-curated checkpoint identities; fail before any GPU work."""
    path = path.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data.get("schema") == REPORT_SCHEMA and data.get("repo_id") == REPO_ID,
            "Report manifest schema/repository mismatch")
    require(data.get("seeds") == [42, 43, 44], "Unexpected report seed contract")
    selected = data.get("datasets", {}).get(dataset)
    require(isinstance(selected, dict), f"Dataset missing from report manifest: {dataset}")
    cases = selected.get("cases")
    require(isinstance(cases, dict) and cases, "Report case selection cannot be empty")
    output: dict[str, dict[int, dict[str, str]]] = {}
    sections: dict[str, list[str]] = {}
    encountered = set()
    for case, descriptor in cases.items():
        require(bool(_SEGMENT.fullmatch(case)), f"Unsafe case alias in report: {case!r}")
        require(isinstance(descriptor, dict) and descriptor.get("framework") == "ultralytics",
                f"Case requires a different evaluator/adapter: {case}")
        source_run = descriptor.get("source_run")
        tags = descriptor.get("sections")
        require(isinstance(source_run, str) and source_run.startswith(dataset + "/")
                and all(_SEGMENT.fullmatch(part) for part in source_run.split("/")),
                f"Invalid report source run: {case}")
        require(isinstance(tags, list) and tags and all(isinstance(t, str) for t in tags),
                f"Report section tags missing: {case}")
        per_seed = descriptor.get("seeds")
        require(isinstance(per_seed, dict) and set(per_seed) == {"42","43","44"},
                f"Report case lacks exact 42/43/44 seed entries: {case}")
        output[case] = {}
        sections[case] = tags
        for seed in SEEDS:
            record = per_seed[str(seed)]
            require(isinstance(record, dict), f"Invalid report reference: {case}/{seed}")
            expected = f"{source_run}/{seed_dir(dataset, seed)}/weights/best.pt"
            digest = record.get("sha256")
            require(record.get("path") == expected and isinstance(digest, str)
                    and bool(_SHA256.fullmatch(digest)),
                    f"Invalid or unverified report source/hash: {case}/{seed}")
            require(expected not in encountered,
                    f"Duplicate checkpoint selected as two distinct cases: {expected}")
            encountered.add(expected)
            if case in AUGMENTATION_RUNS[dataset]:
                require(augmentation_checkpoint(dataset, case, seed) == expected,
                        f"Report alias collides with different canonical case: {case}/{seed}")
            output[case][seed] = {"path": expected, "sha256": digest}
    blockers = selected.get("unresolved_main_results", [])
    require(isinstance(blockers, list) and all(isinstance(b, dict)
            and isinstance(b.get("model"), str) and isinstance(b.get("reason"), str)
            for b in blockers), "Invalid unresolved Main Results inventory")
    return output, sections, blockers


def report_protocol_warnings(path: Path, dataset: str, cases: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    raw = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    entries = raw["datasets"][dataset]["cases"]
    return [{"case": alias, "warning": entries[alias]["protocol_note"]}
            for alias in cases if entries[alias].get("protocol_note")]


def checkpoint_source(args: argparse.Namespace, case: str, seed: int) -> str:
    if case in args.manifest_cases:
        return args.manifest_cases[case][seed]["path"]
    if case in AUGMENTATION_RUNS[args.dataset]:
        return augmentation_checkpoint(args.dataset, case, seed)
    raise RuntimeError(f"Checkpoint case missing from selection: {case}")


def verify_sources(args: argparse.Namespace, archive: Any) -> list[dict[str, Any]]:
    from utils.ftsc_capstone_aug_reeval_v1 import verify_hf_sources, planned_rows
    canonical = [case for case in args.cases
                 if case in AUGMENTATION_RUNS[args.dataset] and case not in args.manifest_cases]
    verified: list[dict[str, Any]] = []
    if canonical:
        verified.extend(verify_hf_sources(archive, planned_rows(
            args.dataset, tuple(canonical), tuple(args.seeds), args.project)))
    external = [case for case in args.cases if case in args.manifest_cases]
    if external:
        api = archive._get_api()
        revision = archive._retry(lambda: api.repo_info(repo_id=REPO_ID, repo_type="dataset")).sha
        require(isinstance(revision, str) and len(revision) >= 30,
                "Cannot pin HF revision for external manifest")
        available = set(archive._retry(lambda: api.list_repo_files(
            repo_id=REPO_ID, repo_type="dataset", revision=revision)))
        args.manifest_revision = revision
        for case in external:
            for seed in args.seeds:
                spec = args.manifest_cases[case][seed]
                require(spec["path"] in available, f"Missing HF file: {spec['path']}")
                verified.append({"case": case, "seed": seed, "source": spec["path"],
                                 "expected_sha256": spec["sha256"], "revision": revision,
                                 "status": "MANIFEST_PATH_PRESENT_BYTES_NOT_YET_VERIFIED"})
    return verified


def planned(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "dataset": args.dataset, "case": case, "seed": seed, "backend": backend,
            "source_repo": REPO_ID, "source": checkpoint_source(args, case, seed),
            "output": str(path_for(args.project, args.dataset, case, seed, backend) / "metrics.json"),
            "report_sections": args.report_sections.get(case, []),
        }
        for case in args.cases for seed in args.seeds for backend in args.backends
    ]


def source_restore(args: argparse.Namespace, archive: Any, case: str, seed: int) -> dict[str, Any]:
    # Canonical transport verifies checkpoint reference and SHA, and never replaces
    # different existing bytes. DO NOT fetch old TinyPerson predictions for accuracy.
    src = source_path(args.project, args.dataset, case, seed)
    if case in AUGMENTATION_RUNS[args.dataset] and case not in args.manifest_cases:
        return archive.restore_augmentation_source(
            dataset=args.dataset, case=case, seed=seed, destination=src, include_predictions=False
        )
    # External cases are ONLY accepted with an explicit, pinned SHA manifest.
    from huggingface_hub import hf_hub_download
    spec = args.manifest_cases[case][seed]
    revision = getattr(args, "manifest_revision", None)
    require(isinstance(revision, str) and len(revision) >= 30,
            "Verify pinned HF revision before external checkpoint download")
    downloaded = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                     revision=revision, filename=spec["path"],
                                     token=archive.token))
    digest = sha256(downloaded)
    require(digest == spec["sha256"], f"HF checkpoint SHA mismatch: {spec['path']}")
    if src.is_file():
        require(sha256(src) == digest, f"Local checkpoint collision: {src}")
    else:
        src.parent.mkdir(parents=True, exist_ok=True)
        staging = src.with_suffix(".download")
        require(not staging.exists(), f"Interrupted checkpoint staging exists: {staging}")
        try:
            shutil.copyfile(downloaded, staging)
            require(sha256(staging) == digest, "Checkpoint copy hash mismatch")
            staging.replace(src)
        finally:
            staging.unlink(missing_ok=True)
    return {"checkpoint": str(src), "sha256": digest, "source": spec["path"],
            "revision": revision}


def backend_device(backend: str, requested: str) -> str:
    return "cpu" if backend.startswith("ort_cpu") else requested


def hardware_identity(device: str) -> dict[str, str]:
    """Resolve the actual machine identity; never infer Jetson from a label."""
    import torch
    gpu = "CPU"
    if torch.cuda.is_available():
        index = (int(device.split(":", 1)[1]) if device.startswith("cuda:")
                 else int(device) if device.isdigit() else 0)
        gpu = torch.cuda.get_device_name(index)
    return {
        "host_machine": platform.machine(),
        "host_system": platform.platform(),
        "gpu_name": gpu,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or "none"),
    }


def cache_export(pt: Path, digest: str, backend: str, output_dir: Path, imgsz: int,
                 device: str, protocol: str = NATIVE_PROTOCOL) -> Path:
    """Export to isolated cache, never modifying the HF checkpoint."""
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from ultralytics import YOLO
    import torch

    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    staged = export_dir / "source.pt"
    if staged.is_file():
        require(sha256(staged) == digest, f"Stale export input, inspect: {staged}")
    else:
        shutil.copyfile(pt, staged)
        require(sha256(staged) == digest, "Staged checkpoint hash mismatch")

    is_trt = backend.startswith("trt")
    expected = staged.with_suffix(".engine" if is_trt else ".onnx")
    metadata = export_dir / "export_manifest.json"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(int(device.split(":")[-1]) if device.startswith("cuda:") else int(device) if device.isdigit() else 0)
    else:
        gpu_name = "CPU"
    # TensorRT engine artifacts are bound to device/toolchain. Re-export on Jetson.
    fingerprint = {"source_sha256": digest, "backend": backend, "imgsz": imgsz,
                   "gpu_name": gpu_name, "device": device, "torch": str(torch.__version__)}
    if protocol in (PARTNER_PROTOCOL, NATIVE_VAL_PROTOCOL, RECT_PARITY_PROTOCOL):
        fingerprint["export_protocol"] = protocol
        fingerprint["simplify"] = False
        sources = (rect_parity_source_hashes() if protocol == RECT_PARITY_PROTOCOL
                   else partner_source_hashes())
        fingerprint["export_source_sha256"] = {
            name: source_hash for name, source_hash in sources.items()
            if name in (
                "utils/ftsc_deployment_backend_eval_v1.py",
                "models_related/ultralytics/ultralytics/nn/backends/tensorrt.py",
                "models_related/ultralytics/ultralytics/nn/modules/head.py",
                "models_related/ultralytics/ultralytics/engine/exporter.py",
            )
        }
    if protocol == RECT_PARITY_PROTOCOL:
        require(is_trt and imgsz == RECT_ENGINE_CANVAS,
                "LEVIR V3 only exports a static TensorRT 544x544 canvas")
        fingerprint["paper_preprocess_imgsz"] = RECT_PAPER_IMGSZ
        fingerprint["paper_rect"] = True
        fingerprint["paper_stride"] = RECT_PAPER_STRIDE
    if is_trt:
        require(torch.cuda.is_available() and device != "cpu",
                "TensorRT export requires CUDA on the target machine")
        import tensorrt as trt
        fingerprint["tensorrt"] = str(trt.__version__)
    if expected.is_file() or metadata.is_file():
        require(expected.is_file() and metadata.is_file(),
                f"Incomplete export cache at {export_dir}; inspect before re-export")
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        require(previous.get("fingerprint") == fingerprint
                and previous.get("artifact_sha256") == sha256(expected),
                f"Export fingerprint/hash mismatch: {expected}")
        return expected

    # Preserve source data. Backend precision is baked into export, not inferred
    # from PyTorch checkpoint dtype. Disable embedded NMS to keep predictor NMS.
    model = YOLO(str(staged), task="detect")
    partner_kwargs = {"simplify": False, "int8": False, "verbose": False} if protocol in (PARTNER_PROTOCOL, NATIVE_VAL_PROTOCOL, RECT_PARITY_PROTOCOL) else {}
    exported = Path(model.export(
        format="engine" if is_trt else "onnx",
        imgsz=imgsz, batch=1, half=backend.endswith("16"), device=device,
        dynamic=False, nms=False, **partner_kwargs,
    )).resolve()
    require(exported == expected.resolve() and expected.is_file(),
            f"Export returned an unexpected artifact: {exported} != {expected}")
    metadata.write_text(json.dumps({
        "fingerprint": fingerprint, "artifact_sha256": sha256(expected)
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return expected


def prepare_backend(args: argparse.Namespace, provenance: dict[str, Any],
                    backend: str, output_dir: Path, imgsz: int) -> Path:
    pt = Path(provenance["checkpoint"])
    require(pt.is_file() and sha256(pt) == provenance["sha256"], "HF source checkpoint SHA mismatch")
    if backend.startswith("pt"):
        return pt
    export_device = args.device
    if backend == "ort_cpu32":
        export_device = "cpu"
    elif backend == "ort_cpu16":
        # ONNX FP16 export needs CUDA; CPU EP may not implement all FP16 ops.
        require(args.device != "cpu", "ORT CPU FP16 export requires CUDA; CPU execution may still be unsupported")
    if rect_parity_mode(args):
        require(args.dataset == "LEVIR-Ship" and backend.startswith("trt"),
                "Rect parity V3 only supports LEVIR TensorRT exports")
        imgsz = RECT_ENGINE_CANVAS
    return cache_export(pt, provenance["sha256"], backend, output_dir, imgsz, export_device,
                        evaluation_protocol(args))


def val_detections_to_size_predictions(
    detections: list[dict[str, Any]], images: list[Path],
) -> list[dict[str, Any]]:
    """Convert detections produced by YOLO.val() into the original-image
    TinyBenchmark size-bucket format, without a second inference pass.

    The fork's DetectionValidator.pred_to_json() records file_name, image_id,
    category_id, bbox (original-image xywh) and score. Fail if any filename
    is ambiguous or if the validator emitted a foreign image.
    """
    names = [p.name for p in images]
    require(len(names) == len(set(names)),
            "Native test image names are not unique; cannot safely join val JSON by basename")
    ids = {name: index for index, name in enumerate(names, 1)}
    predictions: list[dict[str, Any]] = []
    for detection in detections:
        name = detection.get("file_name")
        require(isinstance(name, str) and name in ids,
                f"YOLO.val JSON references missing/ambiguous test image: {name!r}")
        bbox = detection.get("bbox")
        score = detection.get("score")
        require(isinstance(bbox, list) and len(bbox) == 4 and
                all(type(v) in (int, float) and math.isfinite(v) for v in bbox) and
                all(v >= 0 for v in bbox[2:]) and
                type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
                f"Malformed YOLO.val detection: {detection!r}")
        predictions.append({
            "image_id": ids[name],
            "category_id": 1,  # paper's existing class-agnostic size-bucket contract
            "bbox": [float(v) for v in bbox],
            "score": float(score),
        })
    return predictions


def paper_rect_static_validator():
    """Local validator class: TRT static 544 input with *paper* LEVIR 512 preprocessor.

    Model.val() passes the custom validator class as validator=... and uses the
    unmodified upstream BaseValidator loop. For static engines that loop forces
    rect=False before calling get_dataloader; restore rect only *here*. The
    original local variable 'imgsz=544' in BaseValidator.__call__ is retained,
    so engine warmup uses the correct 544 tensor, while the dataset loads
    512-pixel content and performs historical rectangular 16-pixel padding.
    """
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class LevirPaperRectTensorRTValidator(DetectionValidator):
        def get_dataloader(self, dataset_path: str, batch_size: int):
            require(int(self.args.imgsz) == RECT_ENGINE_CANVAS,
                    "TRT static 544 engine must be initialized with imgsz=544")
            require(int(batch_size) == 1 and int(self.args.batch) == 1,
                    "Static TRT engine requires batch 1")
            require(not self.args.rect, "Unexpected validator initialization ordering")
            # PyTorchBackend uses max(int(model.stride.max()), 32), but TensorRT
            # export metadata can contain stride=8. Restore paper PT stride=32.
            self.stride = RECT_PAPER_STRIDE
            self.args.imgsz = RECT_PAPER_IMGSZ
            self.args.rect = True
            loader = super().get_dataloader(dataset_path, batch_size)
            dataset = loader.dataset
            shapes = getattr(dataset, "batch_shapes", None)
            require(shapes is not None and len(shapes) == len(loader),
                    "Rectangular LEVIR validator has missing/incomplete batch shapes")
            require(all(tuple(int(x) for x in shape) ==
                        (RECT_ENGINE_CANVAS, RECT_ENGINE_CANVAS)
                        for shape in shapes),
                    "LEVIR has non-544 rect batch: cannot feed a static TRT engine")
            require(bool(getattr(dataset, "rect", False)) and
                    int(getattr(dataset, "imgsz", -1)) == RECT_PAPER_IMGSZ and
                    int(getattr(dataset, "stride", -1)) == RECT_PAPER_STRIDE,
                    "TensorRT dataloader does not reproduce paper rect/imgsz/stride")
            return loader

    return LevirPaperRectTensorRTValidator


def capture_rect_val_inputs(validator: Any, images: list[Path],
                            output_dir: Path) -> dict[str, Any]:
    """SHA-256 of the real dataset's preprocessed CHW uint8 tensors, all images.

    On the frozen LEVIR test split this compares the exact 512-content + pad
    used by paper PT and static TRT, not simply matching a 544 canvas shape.
    """
    import numpy as np
    import torch

    dataset = getattr(getattr(validator, "dataloader", None), "dataset", None)
    require(dataset is not None, "Cannot inspect YOLO.val dataset for pixel parity")
    ordered = [str(Path(path).resolve()) for path in dataset.im_files]
    expected = {str(path.resolve()) for path in images}
    require(len(ordered) == len(expected) and set(ordered) == expected,
            "Rect parity input hashing sees a different native test image set")
    by_file = {}
    for index, name in enumerate(ordered):
        value = dataset[index]["img"]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        require(value.dtype == np.uint8 and
                tuple(value.shape) == (3, RECT_ENGINE_CANVAS, RECT_ENGINE_CANVAS),
                f"Wrong native rect input shape/dtype: {name}, {value.shape}, {value.dtype}")
        by_file[name] = hashlib.sha256(value.tobytes()).hexdigest()
    require(len(by_file) == len(expected), "Missing input SHA records")
    report = {
        "schema": "ftsc_levir_exact_paper_rect_input_pixels_v3",
        "paper_imgsz": RECT_PAPER_IMGSZ,
        "canvas": RECT_ENGINE_CANVAS,
        "minimum_pytorch_stride": RECT_PAPER_STRIDE,
        "source_images_count": len(images),
        "image_sha256": {name: by_file[name] for name in sorted(by_file)},
    }
    path = output_dir / "test_val_preprocessed_inputs_sha256.json"
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if path.is_file():
        require(path.read_text(encoding="utf-8") == encoded,
                f"Preprocessed input mismatch on resumed run: {path}")
    else:
        staging = path.with_suffix(".json.tmp")
        require(not staging.exists(), f"Incomplete prior preprocessing hash: {staging}")
        staging.write_text(encoded, encoding="utf-8")
        staging.replace(path)
    return {
        "preprocessed_input_manifest_sha256": sha256(path),
        "preprocessed_input_count": len(by_file),
        "preprocessed_input_canvas": RECT_ENGINE_CANVAS,
    }


def verify_rect_trt_pt_input_pair(args: argparse.Namespace,
                                  row: dict[str, Any],
                                  source_digest: str,
                                  trt_fingerprints: dict[str, Any]) -> dict[str, Any]:
    """Refuse to publish TRT AP until all test image tensors match this seed's PT32."""
    if not rect_parity_mode(args) or not row["backend"].startswith("trt"):
        return {}
    pt_dir = path_for(args.project, args.dataset, row["case"], row["seed"], "pt32")
    trt_dir = path_for(args.project, args.dataset, row["case"], row["seed"], row["backend"])
    pt_record_path = pt_dir / "metrics.json"
    require(pt_record_path.is_file(),
            "Rect parity requires this case/seed PT32 model.val first (select --backends pt32 trt32)")
    pt_record = json.loads(pt_record_path.read_text(encoding="utf-8"))
    expected = expected_identity(args, {
        "case": row["case"], "seed": row["seed"], "backend": "pt32",
        "source": row["source"],
    })
    require(all(pt_record.get(k) == value for k, value in expected.items()),
            "Rect parity reference PT32 record has a different evaluator/checkpoint identity")
    require(pt_record.get("source_sha256") == source_digest,
            "TRT checkpoint differs from PT32 pixel reference checkpoint")
    left = pt_dir / "test_val_preprocessed_inputs_sha256.json"
    right = trt_dir / "test_val_preprocessed_inputs_sha256.json"
    require(left.is_file() and right.is_file(), "Native paper preprocessing fingerprints missing")
    require(sha256(left) == pt_record.get("preprocessed_input_manifest_sha256"),
            "PT32 reference preprocessing manifest hash changed")
    require(sha256(right) == trt_fingerprints.get("preprocessed_input_manifest_sha256"),
            "TRT preprocessing manifest hash changed")
    require(left.read_bytes() == right.read_bytes(),
            "TRT vs PT32 input pixels differ: NOT the same paper validation protocol")
    return {
        "full_test_input_parity": True,
        "paired_pt32_preprocessed_manifest_sha256": sha256(left),
        "paired_pt32_metrics_sha256": sha256(pt_record_path),
    }


def native_metrics(args: argparse.Namespace, model_file: Path, backend: str,
                   output_dir: Path, imgsz: int) -> tuple[dict[str, Any], dict[str, str]]:
    from evaluate_test.standard_detection_metrics import (
        ensure_local_ultralytics, standard_metrics,
    )
    from evaluate_test.size_bucket_evaluator import (
        _resolve_test_images, _coco_ground_truth, _predictions, _evaluate_coco,
    )
    ensure_local_ultralytics()
    from ultralytics import YOLO

    yaml_path = Path(args.data_yaml_template.format(seed=args.current_seed)).expanduser().resolve()
    require(yaml_path.is_file(), f"Missing frozen test YAML: {yaml_path}")
    images = _resolve_test_images(yaml_path)
    require(bool(images) and all(p.is_file() for p in images),
            f"Native test image list empty/missing for {yaml_path}")
    model = YOLO(str(model_file), task="detect")
    device = backend_device(backend, args.device)
    half = backend == "pt16"
    val_v2 = native_val_mode(args)
    paper_parity = rect_parity_mode(args)
    require(not paper_parity or (args.dataset == "LEVIR-Ship" and
            backend in ("pt32", "pt16", "trt32", "trt16")),
            "Native paper rect V3 is only implemented for LEVIR PT/TRT")
    validator_state: dict[str, Any] = {}
    if val_v2:
        # The local Ultralytics Model.val() does NOT assign self.validator.
        # Capture the actual validator through its supported on_val_end callback.
        model.add_callback("on_val_end", lambda validator: validator_state.__setitem__(
            "validator", validator
        ))
    # LEVIR paper PT: imgsz=512, batch=8, rect=True -> 512 pixels + 16 pad
    # each side -> actual 544 canvas. TRT static: export 544 batch1 but use
    # an isolated validator that keeps the 512-image preprocessing and stride32.
    val_size = (RECT_ENGINE_CANVAS if paper_parity and backend.startswith("trt")
                else imgsz)
    val_batch = (RECT_PT_VAL_BATCH if paper_parity and backend.startswith("pt")
                 else 1 if paper_parity else args.batch)
    val_rect = True if paper_parity else False if val_v2 else True
    validator_class = (paper_rect_static_validator()
                       if paper_parity and backend.startswith("trt") else None)
    result = model.val(
        validator=validator_class,
        data=str(yaml_path), split="test", imgsz=val_size, batch=val_batch,
        device=device, half=half, workers=args.workers,
        conf=0.001 if val_v2 else None,
        iou=0.5 if args.dataset == "LEVIR-Ship" else 0.7,
        max_det=300, rect=val_rect,
        plots=False, save_json=val_v2, project=str(output_dir), name="validator",
        exist_ok=True, verbose=False,
    )
    metrics = standard_metrics(result, "test")
    # The existing small-AP contract is NOT COCO AP_small: it is the
    # TinyBenchmark size-bucket AP50 on original native test images.
    gt_file = output_dir / "test_size_ground_truth.json"
    pred_file = output_dir / "test_size_predictions.json"
    gt_file.write_text(json.dumps(_coco_ground_truth(images)) + "\n", encoding="utf-8")
    if val_v2:
        validator = validator_state.get("validator")
        require(validator is not None, "YOLO.val did not execute on_val_end callback")
        require(getattr(validator, "seen", None) == len(images),
                "YOLO.val did not process exactly the native test image count")
        native_paths = getattr(getattr(validator.dataloader, "dataset", None), "im_files", [])
        require(len(native_paths) == len(images) and
                {str(Path(p).resolve()) for p in native_paths} ==
                {str(p.resolve()) for p in images},
                "YOLO.val evaluated a different set of native test images")
        if paper_parity:
            require(int(validator.args.imgsz) == RECT_PAPER_IMGSZ and
                    int(validator.args.batch) == val_batch and
                    bool(validator.args.rect) and
                    int(validator.stride) == RECT_PAPER_STRIDE,
                    "V3 validator did not preserve paper imgsz/batch/rect/stride")
            pixel_fingerprints = capture_rect_val_inputs(validator, images, output_dir)
        else:
            pixel_fingerprints = {}
        prediction = val_detections_to_size_predictions(validator.jdict, images)
    else:
        # Historical V3 compatibility only (separate predict pass).
        prediction = _predictions(model, images, imgsz=imgsz, batch=args.batch,
                                  device=device, workers=args.workers,
                                  half=half if backend.startswith("pt") else None)
    pred_file.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    metrics.update(_evaluate_coco(gt_file, pred_file))
    return metrics, {"data_yaml_sha256": sha256(yaml_path),
                     "gt_sha256": sha256(gt_file), "predictions_sha256": sha256(pred_file),
                     **(pixel_fingerprints if paper_parity else {})}


def tiny_metrics(args: argparse.Namespace, model_file: Path, backend: str,
                 output_dir: Path, imgsz: int) -> tuple[dict[str, Any], dict[str, str]]:
    from train_tinyperson_scripts.train_all_tinyperson import (
        predict_merged_test, evaluate_tiny_benchmark, TEST_MERGED_JSON,
    )
    require(args.tiny_data_root is not None and args.tiny_test_out is not None,
            "TinyPerson requires --tiny-data-root and --tiny-test-out")
    root = args.tiny_data_root.expanduser().resolve()
    test_out = args.tiny_test_out.expanduser().resolve()
    merged_gt = root / TEST_MERGED_JSON
    manifest = test_out / "corner_manifest.json"
    require(merged_gt.is_file() and manifest.is_file() and (test_out / "images").is_dir(),
            "TinyPerson frozen merged-original GT or prepared test corner manifest/images missing")
    # Re-run *this backend* on the same prepared corners and merge into original
    # image IDs. Never reuse old .pt merged JSON as TensorRT accuracy.
    prediction = predict_merged_test(
        output_dir, test_out, root,
        argparse.Namespace(imgsz=imgsz, batch_size=args.batch, workers=args.workers,
                           device=backend_device(backend, args.device)),
        model_path=model_file,
        half=(backend == "pt16") if backend.startswith("pt") else None,
        predict_iou=0.7 if evaluation_protocol(args) == PARTNER_PROTOCOL else 0.5,
        predict_rect=evaluation_protocol(args) != PARTNER_PROTOCOL,
    )
    metrics = evaluate_tiny_benchmark(prediction, merged_gt)
    return metrics, {
        "gt_sha256": sha256(merged_gt), "corner_manifest_sha256": sha256(manifest),
        "predictions_sha256": sha256(prediction),
    }


def validate_metric_values(metrics: dict[str, Any], dataset: str,
                           protocol: str = NATIVE_PROTOCOL) -> None:
    if dataset == "TinyPerson":
        require(metrics.get("test_merged/evaluator_contract") == TINY_MERGED_CONTRACT,
                "TinyPerson deployment accuracy requires original-image merged TinyBenchmark evaluator")
    for key in metric_keys(dataset, protocol):
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0,
                f"Missing/invalid deployment metric {key}: {value!r}")
    # Small AP -1 indicates no valid objects in the bucket; never silently
    # substitute overall AP or write a paper number from a missing bucket.
    require(metrics[metric_keys(dataset, protocol)[2]] >= 0, "Small-object AP is unavailable")


def expected_identity(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    partner = evaluation_protocol(args) == PARTNER_PROTOCOL
    native_val_v2 = native_val_mode(args)
    identity = {"contract": (PARTNER_CONTRACT if partner else NATIVE_VAL_CONTRACT if native_val_v2 else VERSION),
                "dataset": args.dataset, "case": row["case"],
                "seed": row["seed"], "backend": row["backend"],
                "source": row["source"], "platform": args.platform}
    if partner:
        from utils.ftsc_partner_backend_metrics_v1 import PARTNER_REPO, PARTNER_COMMIT
        identity["evaluation_protocol"] = PARTNER_PROTOCOL
        identity["partner_source"] = f"{PARTNER_REPO}@{PARTNER_COMMIT}"
        identity["runtime_revision"] = PARTNER_RUNTIME_REVISION
        identity["source_code_sha256"] = partner_source_hashes()
        identity["predict_config"] = {
            "conf": 0.001, "nms_iou": 0.7, "max_det": 300,
            "batch": 1, "rect": False,
            "tiny_merge_nms_iou": 0.5 if args.dataset == "TinyPerson" else None,
        }
    if rect_parity_mode(args):
        identity["contract"] = RECT_PARITY_CONTRACT
        identity["evaluation_protocol"] = RECT_PARITY_PROTOCOL
        identity["runtime_revision"] = RECT_PARITY_RUNTIME_REVISION
        identity["source_code_sha256"] = rect_parity_source_hashes()
        identity["validation_config"] = {
            "method": "YOLO.val", "split": "test", "confidence": 0.001,
            "nms_iou": 0.5, "max_det": 300,
            "paper_preprocess_imgsz": RECT_PAPER_IMGSZ,
            "paper_preprocess_rect": True, "paper_preprocess_stride": RECT_PAPER_STRIDE,
            "static_engine_canvas": RECT_ENGINE_CANVAS,
            "pt_val_batch": RECT_PT_VAL_BATCH, "engine_val_batch": 1,
            "native_small_source": "same_validation_detections",
            "full_test_input_pixel_sha_comparison": "required_for_trt",
        }
    elif native_val_v2:
        identity["evaluation_protocol"] = NATIVE_VAL_PROTOCOL
        identity["runtime_revision"] = NATIVE_VAL_RUNTIME_REVISION
        identity["source_code_sha256"] = partner_source_hashes()
        identity["validation_config"] = {
            "method": "YOLO.val" if args.dataset != "TinyPerson" else "TinyBenchmark merged-original",
            "split": "test", "confidence": 0.001,
            "nms_iou": 0.5 if args.dataset in ("LEVIR-Ship", "TinyPerson") else 0.7,
            "max_det": 300, "rect": False, "batch": 1,
            "native_small_source": "same_validation_detections" if args.dataset != "TinyPerson"
                                   else "merged_original_predictions",
        }
    return identity


def verify_partner_artifacts(record: dict[str, Any], folder: Path, dataset: str) -> None:
    """A saved partner accuracy result is valid only with its own prediction files."""
    if dataset == "TinyPerson":
        paths = {"predictions_sha256": folder / "evaluation/test_merged_predictions.json"}
    else:
        paths = {
            "predictions_sha256": folder / "partner_test_predictions.json",
            "small_predictions_sha256": folder / "partner_test_size_predictions.json",
            "gt_sha256": folder / "partner_test_coco_ground_truth.json",
            "small_gt_sha256": folder / "partner_test_size_ground_truth.json",
        }
    for key, artifact in paths.items():
        require(artifact.is_file() and record.get(key) == sha256(artifact),
                f"Saved partner prediction/GT fingerprint invalid: {artifact}")


def verify_native_val_artifacts(record: dict[str, Any], folder: Path, dataset: str) -> None:
    if record.get("evaluation_protocol") == RECT_PARITY_PROTOCOL:
        artifact = folder / "test_val_preprocessed_inputs_sha256.json"
        require(artifact.is_file() and
                record.get("preprocessed_input_manifest_sha256") == sha256(artifact) and
                record.get("preprocessed_input_count", 0) > 0,
                "V3 full-test image preprocess SHA manifest invalid")
        if record.get("backend", "").startswith("trt"):
            require(record.get("full_test_input_parity") is True,
                    "TRT record does not certify input SHA parity against PT32")
    if dataset == "TinyPerson":
        paths = {"predictions_sha256": folder / "evaluation/test_merged_predictions.json"}
    else:
        paths = {
            "gt_sha256": folder / "test_size_ground_truth.json",
            "predictions_sha256": folder / "test_size_predictions.json",
        }
    for key, artifact in paths.items():
        require(artifact.is_file() and record.get(key) == sha256(artifact),
                f"Native val V2 saved prediction/GT fingerprint invalid: {artifact}")


def evaluate_one(args: argparse.Namespace, archive: Any, row: dict[str, Any]) -> dict[str, Any]:
    backend, case, seed = row["backend"], row["case"], row["seed"]
    output_dir = path_for(args.project, args.dataset, case, seed, backend)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(all(cached.get(k) == v for k, v in expected_identity(args, row).items()),
                f"Cached deployment result has different identity: {metrics_path}")
        validate_metric_values(cached["metrics"], args.dataset, evaluation_protocol(args))
        if evaluation_protocol(args) == PARTNER_PROTOCOL:
            verify_partner_artifacts(cached, output_dir, args.dataset)
        if native_val_mode(args):
            verify_native_val_artifacts(cached, output_dir, args.dataset)
        if rect_parity_mode(args) and backend.startswith("trt"):
            paired = verify_rect_trt_pt_input_pair(
                args, row, cached["source_sha256"], cached)
            require(all(cached.get(k) == value for k, value in paired.items()),
                    "TRT cache PT32 pairing or full-test pixel hash changed")
        require(cached.get("device") == backend_device(backend, args.device),
                f"Cached evaluation was run on a different device: {metrics_path}")
        actual_host = hardware_identity(backend_device(backend, args.device))
        require(all(cached.get(k) == v for k, v in actual_host.items()),
                f"Cached evaluation belongs to a different host/toolchain: {metrics_path}")
        artifact = (source_path(args.project, args.dataset, case, seed)
                    if backend.startswith("pt") else
                    output_dir / "export" / ("source.engine" if backend.startswith("trt")
                                            else "source.onnx"))
        require(artifact.is_file() and sha256(artifact) == cached.get("model_artifact_sha256"),
                f"Cached evaluated backend artifact changed or missing: {artifact}")
        if not backend.startswith("pt"):
            export_metadata = output_dir / "export" / "export_manifest.json"
            require(export_metadata.is_file(), f"Cached export manifest missing: {export_metadata}")
            state = json.loads(export_metadata.read_text(encoding="utf-8"))
            require(state.get("artifact_sha256") == sha256(artifact)
                    and state.get("fingerprint", {}).get("source_sha256")
                    == cached.get("source_sha256"),
                    f"Export provenance changed since evaluation: {export_metadata}")
        source = source_path(args.project, args.dataset, case, seed)
        require(source.is_file() and sha256(source) == cached.get("source_sha256"),
                f"Cached evaluation source checkpoint missing/changed: {source}")
        if args.dataset == "TinyPerson":
            from train_tinyperson_scripts.train_all_tinyperson import TEST_MERGED_JSON
            require(sha256(args.tiny_data_root / TEST_MERGED_JSON) == cached.get("gt_sha256"),
                    "TinyPerson merged GT changed since cached evaluation")
            require(sha256(args.tiny_test_out / "corner_manifest.json") == cached.get("corner_manifest_sha256"),
                    "TinyPerson crop mapping changed since cached evaluation")
        else:
            require(sha256(Path(args.data_yaml_template.format(seed=seed))) == cached.get("data_yaml_sha256"),
                    "Native test YAML changed since cached evaluation")
        return {"status": "REUSED_VERIFIED", "output": str(metrics_path)}
    provenance = source_restore(args, archive, case, seed)
    imgsz = 512 if args.dataset == "LEVIR-Ship" else 640
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = prepare_backend(args, provenance, backend, output_dir, imgsz)
    args.current_seed = seed
    if args.dataset == "TinyPerson":
        metrics, fingerprints = tiny_metrics(args, model_file, backend, output_dir, imgsz)
    elif evaluation_protocol(args) == PARTNER_PROTOCOL:
        from utils.ftsc_partner_backend_metrics_v1 import partner_native_metrics
        metrics, fingerprints = partner_native_metrics(args, model_file, backend, output_dir, imgsz)
    else:
        metrics, fingerprints = native_metrics(args, model_file, backend, output_dir, imgsz)
    validate_metric_values(metrics, args.dataset, evaluation_protocol(args))
    paired_fingerprints = verify_rect_trt_pt_input_pair(
        args, row, provenance["sha256"], fingerprints)
    record: dict[str, Any] = {
        **expected_identity(args, row), "source_sha256": provenance["sha256"],
        "hf_revision": provenance["revision"], "model_artifact_sha256": sha256(model_file),
        "evaluator": evaluator_name(args), "imgsz": imgsz,
        "device": backend_device(backend, args.device),
        **hardware_identity(backend_device(backend, args.device)),
        "metrics": metrics, **fingerprints, **paired_fingerprints,
    }
    # Write complete result last, atomically; no writes to source HF repository.
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    return {"status": "EVALUATED", "output": str(metrics_path)}


def aggregate(args: argparse.Namespace) -> list[dict[str, Any]]:
    require(tuple(sorted(args.seeds)) == SEEDS, "Paper mean ± sample STD requires seeds 42,43,44")
    rows = []
    for case in args.cases:
        for backend in args.backends:
            found = []
            for seed in args.seeds:
                path = path_for(args.project, args.dataset, case, seed, backend) / "metrics.json"
                require(path.is_file(), f"Missing backend evaluation: {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                identity = expected_identity(args, {
                    "case": case, "seed": seed, "backend": backend,
                    "source": checkpoint_source(args, case, seed),
                })
                require(all(record.get(k) == v for k,v in identity.items()),
                        f"Mismatched backend result: {path}")
                require(record.get("evaluator") == evaluator_name(args), f"Evaluator mismatch: {path}")
                require(record.get("imgsz") == (512 if args.dataset == "LEVIR-Ship" else 640),
                        f"Resolution mismatch: {path}")
                validate_metric_values(record["metrics"], args.dataset, evaluation_protocol(args))
                if evaluation_protocol(args) == PARTNER_PROTOCOL:
                    verify_partner_artifacts(record, path.parent, args.dataset)
                if native_val_mode(args):
                    verify_native_val_artifacts(record, path.parent, args.dataset)
                if rect_parity_mode(args) and backend.startswith("trt"):
                    verified_pair = verify_rect_trt_pt_input_pair(
                        args, {"case": case, "seed": seed, "backend": backend,
                               "source": checkpoint_source(args, case, seed)},
                        record["source_sha256"], record)
                    require(all(record.get(k) == value for k, value in verified_pair.items()),
                            "TRT paper-input paired PT reference changed")
                found.append(record)
            require(len({x["gt_sha256"] for x in found}) == 1 if args.dataset == "TinyPerson" else True,
                    "Different merged GT hashes across TinyPerson seeds")
            require(len({x.get("corner_manifest_sha256") for x in found}) == 1 if args.dataset == "TinyPerson" else True,
                    "Different test corner manifests across TinyPerson seeds")
            require(all(len({x[k] for x in found}) == 1
                        for k in ("device", "host_machine", "host_system",
                                  "gpu_name", "torch_version", "cuda_version")),
                    "Mixed hardware/device/toolchain in a backend aggregate")
            data: dict[str, Any] = {
                "dataset": args.dataset, "case": case, "backend": backend,
                "platform": args.platform, "device": found[0]["device"],
                "host_machine": found[0]["host_machine"],
                "gpu_name": found[0]["gpu_name"], "n": 3,
                "evaluator": evaluator_name(args),
                "imgsz": found[0]["imgsz"],
            }
            if evaluation_protocol(args) in (PARTNER_PROTOCOL, RECT_PARITY_PROTOCOL):
                data["evaluation_protocol"] = evaluation_protocol(args)
            for label, key in zip(("AP50", "mAP50_95", "AP50_small"),
                                  metric_keys(args.dataset, evaluation_protocol(args))):
                vals = [100 * x["metrics"][key] for x in found]
                data[label + "_mean_pct"] = round(statistics.fmean(vals), 6)
                data[label + "_sample_std_pct"] = round(statistics.stdev(vals), 6)
            rows.append(data)
    output = args.project / args.dataset / f"deployment_{args.platform}_accuracy_aggregate.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows



def aggregate_report(args: argparse.Namespace) -> dict[str, Any]:
    """Partial-safe report aggregation: only complete three-seed rows get numbers."""
    results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    partner_gt_by_case: dict[str, dict[str, tuple[str, ...]]] = {}
    require(tuple(sorted(args.seeds)) == SEEDS,
            "Report means require all three seeds 42/43/44")
    for case in args.cases:
        for backend in args.backends:
            records: list[dict[str, Any]] = []
            issues: list[str] = []
            for seed in args.seeds:
                folder = path_for(args.project, args.dataset, case, seed, backend)
                metric_path = folder / "metrics.json"
                if not metric_path.is_file():
                    failed = folder / "failure.json"
                    if failed.is_file():
                        try:
                            error_record = json.loads(failed.read_text(encoding="utf-8"))
                            issues.append(f"{seed}: {error_record.get('error_type', 'ERROR')}: "
                                          f"{error_record.get('error', 'No detail')}")
                        except (OSError, ValueError, TypeError):
                            issues.append(f"{seed}: unreadable failure.json")
                    else:
                        issues.append(f"{seed}: missing metrics.json")
                    continue
                try:
                    record = json.loads(metric_path.read_text(encoding="utf-8"))
                    identity = expected_identity(args, {
                        "case": case, "seed": seed, "backend": backend,
                        "source": checkpoint_source(args, case, seed),
                    })
                    require(all(record.get(k) == v for k, v in identity.items()),
                            "different checkpoint identity or platform")
                    require(record.get("evaluator") == evaluator_name(args),
                            "evaluator contract mismatch")
                    require(record.get("imgsz") == (512 if args.dataset == "LEVIR-Ship" else 640),
                            "input resolution mismatch")
                    require(record.get("source_sha256") ==
                            (args.manifest_cases[case][seed]["sha256"]
                             if case in args.manifest_cases else record.get("source_sha256")),
                            "checkpoint SHA changed from report inventory")
                    validate_metric_values(record["metrics"], args.dataset, evaluation_protocol(args))
                    if evaluation_protocol(args) == PARTNER_PROTOCOL:
                        verify_partner_artifacts(record, folder, args.dataset)
                    if native_val_mode(args):
                        verify_native_val_artifacts(record, folder, args.dataset)
                    if rect_parity_mode(args) and backend.startswith("trt"):
                        verified_pair = verify_rect_trt_pt_input_pair(
                            args, {"case": case, "seed": seed, "backend": backend,
                                   "source": checkpoint_source(args, case, seed)},
                            record["source_sha256"], record)
                        require(all(record.get(k) == value for k, value in verified_pair.items()),
                                "TRT paper-input paired PT reference changed")
                    records.append(record)
                except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
                    issues.append(f"{seed}: invalid result: {type(error).__name__}: {error}")
            if issues:
                blockers.append({
                    "case": case, "backend": backend, "report_sections": args.report_sections.get(case, []),
                    "issues": issues, "status": "INCOMPLETE_NOT_PAPER_READY",
                })
                continue
            try:
                require(len(records) == 3, "Exactly three checkpoint seeds required")
                require(all(len({str(x[k]) for x in records}) == 1
                            for k in ("device", "host_machine", "host_system",
                                      "gpu_name", "torch_version", "cuda_version", "evaluator")),
                        "Hardware/toolchain/evaluator contract differs across seeds")
                if args.dataset == "TinyPerson":
                    require(len({x["gt_sha256"] for x in records}) == 1 and
                            len({x["corner_manifest_sha256"] for x in records}) == 1,
                            "TinyPerson GT or corner mapping differs across seeds")
                row: dict[str, Any] = {
                    "dataset": args.dataset, "case": case, "backend": backend,
                    "report_sections": ";".join(args.report_sections.get(case, [])),
                    "platform": args.platform, "n": 3, "imgsz": records[0]["imgsz"],
                    "device": records[0]["device"], "gpu_name": records[0]["gpu_name"],
                    "evaluator": records[0]["evaluator"], "status": "COMPLETE",
                }
                if evaluation_protocol(args) == RECT_PARITY_PROTOCOL:
                    row["evaluation_protocol"] = RECT_PARITY_PROTOCOL
                if evaluation_protocol(args) == PARTNER_PROTOCOL:
                    row["evaluation_protocol"] = PARTNER_PROTOCOL
                    partner_gt_by_case.setdefault(case, {})[backend] = tuple(
                        x["gt_sha256"] for x in records
                    )
                for label, key in zip(("AP50", "mAP50_95", "AP50_small"),
                                      metric_keys(args.dataset, evaluation_protocol(args))):
                    values = [100.0 * float(x["metrics"][key]) for x in records]
                    row[label + "_mean_pct"] = round(statistics.fmean(values), 6)
                    row[label + "_sample_std_pct"] = round(statistics.stdev(values), 6)
                results.append(row)
            except (KeyError, ValueError, RuntimeError, TypeError) as error:
                blockers.append({
                    "case": case, "backend": backend, "report_sections": args.report_sections.get(case, []),
                    "issues": [f"Three-seed comparability failure: {type(error).__name__}: {error}"],
                    "status": "INVALID_NOT_PAPER_READY",
                })
    if evaluation_protocol(args) == PARTNER_PROTOCOL:
        # Paired backends must have evaluated exactly the same GT for EACH seed.
        for case, per_backend in partner_gt_by_case.items():
            if len(set(per_backend.values())) > 1:
                results = [row for row in results if row["case"] != case]
                blockers.append({
                    "case": case, "backend": "CROSS_BACKEND",
                    "report_sections": args.report_sections.get(case, []),
                    "issues": ["Partner test GT SHA differs across backends for at least one seed"],
                    "status": "INVALID_NOT_PAPER_READY",
                })
    destination = args.project / args.dataset
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"deployment_{args.platform}_accuracy_aggregate.csv"
    if results:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    else:
        # Do not preserve an old apparently complete aggregate when there is no valid row.
        if csv_path.is_file():
            csv_path.unlink()
    report = {
        "status": ("REPORT_PARTIAL" if blockers else
                   "SELECTED_CASES_COMPLETE_WITH_UNRESOLVED_MAIN" if args.unresolved_main else
                   "REPORT_ALL_SELECTED_COMPLETE"),
        "scope": "report", "dataset": args.dataset, "platform": args.platform,
        "evaluation_protocol": evaluation_protocol(args),
        "complete_case_backend_rows": len(results), "blocked_case_backend_rows": len(blockers),
        "unresolved_main_results": args.unresolved_main,
        "protocol_warnings": args.protocol_warnings,
        "aggregate_csv": str(csv_path) if results else None,
        "complete_rows": results, "blockers": blockers,
    }
    destination.joinpath(f"deployment_{args.platform}_report_status.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--phase", choices=("plan", "verify", "evaluate", "aggregate"), default="plan")
    p.add_argument("--cases", nargs="+", default=None,
                   help="Optional case subset. Omit in --scope report to select all report cases")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    p.add_argument("--scope", choices=("manual", "report"), default="manual")
    p.add_argument("--evaluation-protocol",
                   choices=(RECT_PARITY_PROTOCOL, NATIVE_VAL_PROTOCOL, NATIVE_PROTOCOL, PARTNER_PROTOCOL),
                   default=NATIVE_VAL_PROTOCOL,
                   help="For LEVIR paper-accuracy parity use native_ultralytics_val_rect_parity_v3; "
                        "V2/legacy/partner remain isolated for provenance.")
    p.add_argument("--report-manifest", type=Path, default=REPORT_MANIFEST,
                   help="Curated Main Results + A/B/C/D case manifest")
    p.add_argument("--platform", choices=("marimo_live", "local_server"), default="marimo_live",
                   help="Where backend accuracy is evaluated, NOT Jetson runtime benchmark provenance")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--project", type=Path, default=ROOT / "runs/ftsc_deployment_eval_v1")
    p.add_argument("--checkpoint-manifest", type=Path,
                   help="Optional JSON of exact non-canonical HF paths and SHA-256 per case/seed")
    p.add_argument("--data-yaml-template", help="Native frozen data YAML (may include {seed})")
    p.add_argument("--tiny-data-root", type=Path)
    p.add_argument("--tiny-test-out", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--contract-version")
    p.add_argument("--confirm-eval", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    require(args.contract_version in (None, VERSION),
            "Deployment runner version mismatch: sync updated yolo_code")
    args.report_sections = {}
    args.unresolved_main = []
    args.protocol_warnings = []
    if args.scope == "report":
        require(args.checkpoint_manifest is None,
                "Report scope uses its audited report manifest; do not combine manual overrides")
        args.manifest_cases, args.report_sections, args.unresolved_main = load_report_scope(
            args.report_manifest, args.dataset)
        if args.cases is None:
            args.cases = tuple(args.manifest_cases)
        allowed = set(args.manifest_cases)
        args.protocol_warnings = report_protocol_warnings(
            args.report_manifest, args.dataset, tuple(args.cases))
    else:
        args.manifest_cases = manifest_entries(args.checkpoint_manifest, args.dataset)
        if args.cases is None:
            args.cases = ("FTSC",)
        allowed = set(AUGMENTATION_RUNS[args.dataset]) | set(args.manifest_cases)
    require(args.cases and len(set(args.cases)) == len(args.cases)
            and all(case in allowed for case in args.cases),
            f"Unknown/unmapped HF case for {args.dataset}; add exact paths/SHA in manifest")
    require(args.seeds and len(set(args.seeds)) == len(args.seeds)
            and all(seed in SEEDS for seed in args.seeds), "Unsupported/duplicate seeds")
    require(args.backends and len(set(args.backends)) == len(args.backends)
            and all(b in BACKENDS for b in args.backends), "Unsupported/duplicate backend")
    require(all((case in AUGMENTATION_RUNS[args.dataset] and case not in args.manifest_cases)
                or all(seed in args.manifest_cases[case] for seed in args.seeds)
                for case in args.cases), "Manifest lacks requested seed checkpoint mapping")
    require(args.batch > 0 and args.workers >= 0, "Invalid batch/workers")
    if evaluation_protocol(args) == PARTNER_PROTOCOL:
        require(args.batch == 1, "Partner deployment accuracy protocol requires batch=1")
    if native_val_mode(args):
        require(args.batch == 1, "Native val export batch remains 1; V3 PT validation uses fixed batch 8")
    if rect_parity_mode(args):
        require(args.dataset == "LEVIR-Ship",
                "Rect/canvas V3 is verified only on LEVIR; do not apply to other datasets")
        require(all(backend in ("pt32", "pt16", "trt32", "trt16")
                    for backend in args.backends),
                "Rect/canvas V3 supports only PyTorch/TensorRT until ORT input parity is verified")
    args.project = args.project.expanduser().resolve()
    if evaluation_protocol(args) == PARTNER_PROTOCOL:
        # Separate every artifact/cache/aggregate from the previous native protocol.
        args.project = args.project / PARTNER_PROTOCOL / PARTNER_RUNTIME_REVISION
    elif rect_parity_mode(args):
        # New, isolated result namespace: do not reuse historical V2/partner low AP.
        args.project = args.project / RECT_PARITY_PROTOCOL / RECT_PARITY_RUNTIME_REVISION
    elif native_val_mode(args):
        # Never reuse historic low-AP engine/results when validating the runtime fix.
        args.project = args.project / NATIVE_VAL_PROTOCOL / NATIVE_VAL_RUNTIME_REVISION
    rows = planned(args)
    if args.phase == "plan":
        print(json.dumps({
            "status": "PLAN_ONLY", "scope": args.scope,
            "evaluation_protocol": evaluation_protocol(args),
            "evaluator": evaluator_name(args), "project": str(args.project),
            "case_count": len(args.cases), "checkpoint_count": len(args.cases) * len(args.seeds),
            "evaluation_jobs": len(rows),
            "unresolved_main_results": args.unresolved_main,
            "protocol_warnings": args.protocol_warnings,
            "rows": rows,
        }, indent=2))
        return
    if args.phase == "aggregate":
        result = aggregate_report(args) if args.scope == "report" else aggregate(args)
        print(json.dumps(result, indent=2))
        return
    # Only evaluate phase needs test material; verify only checks HF metadata.
    if args.phase == "evaluate":
        require(args.confirm_eval, "Explicit --confirm-eval required; this never trains")
        if args.dataset == "TinyPerson":
            require(args.tiny_data_root is not None and args.tiny_test_out is not None,
                    "TinyPerson requires original GT and prepared corner test directories")
            args.tiny_data_root = args.tiny_data_root.expanduser().resolve()
            args.tiny_test_out = args.tiny_test_out.expanduser().resolve()
        else:
            require(bool(args.data_yaml_template), "--data-yaml-template required")
        if any(b in ("pt16", "trt32", "trt16", "ort_cpu16") for b in args.backends):
            import torch
            require(torch.cuda.is_available(), "FP16/TensorRT export requires CUDA; no CPU fallback")
    from utils.ftsc_hf_canonical_archive import CanonicalFTSCArchive
    archive = CanonicalFTSCArchive(repo_id=REPO_ID, token=os.environ.get("HF_TOKEN") or None,
                                   cache_dir=args.project / "_hf_cache")
    verified = verify_sources(args, archive)
    if args.phase == "verify":
        print(json.dumps({"status":"HF_VERIFIED_READ_ONLY", "rows":verified}, indent=2))
        return
    completed = []
    for index, row in enumerate(rows, 1):
        try:
            outcome = evaluate_one(args, archive, row)
            folder = path_for(args.project, args.dataset, row["case"], row["seed"], row["backend"])
            if outcome.get("status") in ("EVALUATED", "REUSED_VERIFIED"):
                (folder / "failure.json").unlink(missing_ok=True)
        except Exception as error:
            if args.scope != "report":
                raise
            folder = path_for(args.project, args.dataset, row["case"], row["seed"], row["backend"])
            folder.mkdir(parents=True, exist_ok=True)
            token = os.environ.get("HF_TOKEN", "")
            error_text = str(error)
            if token:
                error_text = error_text.replace(token, "<HF_TOKEN>")
            failure = {
                "status": "BLOCKED_BACKEND_EVALUATION", "dataset": args.dataset,
                "case": row["case"], "seed": row["seed"], "backend": row["backend"],
                "source": row["source"], "error_type": type(error).__name__,
                "error": error_text,
            }
            tmp = folder / "failure.json.tmp"
            tmp.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(folder / "failure.json")
            outcome = {"status": "BLOCKED_BACKEND_EVALUATION",
                       "case": row["case"], "seed": row["seed"], "backend": row["backend"],
                       "error_type": type(error).__name__, "error": error_text}
        completed.append(outcome)
        print(f"[deployment {index}/{len(rows)}] {args.dataset}/{row['case']}/"
              f"{row['seed']}/{row['backend']}: {outcome['status']}", flush=True)
    num_blocked = sum(x.get("status") == "BLOCKED_BACKEND_EVALUATION" for x in completed)
    print(json.dumps({"status": "REPORT_PARTIAL" if num_blocked else "EVALUATION_ONLY_COMPLETE",
                      "scope": args.scope, "jobs": len(completed), "blocked": num_blocked,
                      "rows": completed}, indent=2))


if __name__ == "__main__":
    main()
