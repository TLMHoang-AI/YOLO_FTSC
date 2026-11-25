"""LEVIR-only TensorRT INT8 paper evaluation with FP16 V3 static geometry and PT32 pixel parity.

The earlier INT8 dynamic runner and TinyPerson are NOT modified. The new runner
uses two separate TensorRT INT8 engines just like the historical FP16 paper
protocol: MAIN static 544x544/batch1 + paper_rect_static_validator restoring
512-content + stride32 rectangle, and SMALL static 512x512/batch1 +
YOLO.predict default conf=.25, TinyBenchmark native 512 image GT. Both flows
are paired with THIS CHECKPOINT's PT32 preprocessing, byte-for-byte for 788
images. Main AP50/mAP50:95 comes from native YOLO.val(test) only; AP50-small
comes from independent YOLO.predict/TinyBenchmark.

PTQ must calibrate on train split only; static544 and static512 each have
their own train calibration and TensorRT calibration cache. This is not a
promise that INT8 quantization itself can retain FP16 accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import time
import traceback
from typing import Any

from utils import ftsc_hf_all_trt_int8_paper_v1 as common

ROOT = common.ROOT
DATASET = "LEVIR-Ship"
HF_REPO = common.HF_REPO
CONTRACT = "FTSC_LEVIR_INT8_STATIC_FP16_GEOMETRY_PIXEL_PARITY_V2"
DEFAULT_PROJECT = ROOT / "runs/ftsc_hf_int8_levir_static_fp16parity_v2_callfix"
ROLES = {"main": 544, "small": 512}
METRIC_KEYS = common.METRICS[DATASET]
SCORE_NAMES = common.SCORE_NAMES
SOURCE_FILES = tuple(dict.fromkeys((
    "utils/ftsc_hf_all_trt_int8_levir_fp16parity_v2.py",
    "utils/tests/test_ftsc_hf_all_trt_int8_levir_fp16parity_v2.py",
    *common.SOURCE_FILES,
)))


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def source_hashes() -> dict[str, str]:
    return {name: common.sha256(ROOT / name) for name in SOURCE_FILES}


def path_for(project: Path, row: dict) -> Path:
    return project / DATASET / "jobs" / row["job_id"]


def inventory_file(project: Path) -> Path:
    return project / DATASET / "hf_pinned_inventory_static_fp16_v2.json"


def inventory(project: Path, repo: str) -> dict:
    from huggingface_hub import HfApi

    saved = inventory_file(project)
    if saved.is_file():
        data = json.loads(saved.read_text(encoding="utf-8"))
        validate_inventory(data, repo)
        return data
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    revision = api.repo_info(repo_id=repo, repo_type="dataset").sha
    need(isinstance(revision, str) and len(revision) == 40,
         "Cannot resolve pinned Hugging Face commit SHA")
    candidates = api.list_repo_files(repo_id=repo, repo_type="dataset", revision=revision)
    candidates = common.checkpoint_paths(candidates, DATASET)
    need(candidates, f"No LEVIR source weights/*.pt at {repo}@{revision}")
    entries = []
    for src in candidates:
        seed, identity = common.checkpoint_seed(src)
        entries.append({"path": src, "job_id": common.job_id(src),
                        "checkpoint_seed": seed, "seed_resolution": identity,
                        "kind": PurePosixPath(src).name})
    data = {
        "schema": CONTRACT, "repo": repo, "dataset": DATASET,
        "hf_revision": revision, "checkpoints": entries, "count": len(entries),
        "engines": {"main": {"imgsz": 544, "batch": 1, "dynamic": False},
                    "small": {"imgsz": 512, "batch": 1, "dynamic": False}},
        "calibration_split": "train",
        "evaluator": "FP16_LEVIR_PAPER_V3_STATIC_RECT_AND_NATIVE_TINYBENCHMARK",
    }
    common.atomic_json(saved, data)
    return data


def validate_inventory(data: dict, repo: str) -> None:
    paths = [r["path"] for r in data.get("checkpoints", [])]
    need(data.get("schema") == CONTRACT and data.get("repo") == repo
         and data.get("dataset") == DATASET
         and isinstance(data.get("hf_revision"), str)
         and len(data["hf_revision"]) == 40
         and data.get("count") == len(paths) > 0
         and paths == common.checkpoint_paths(paths, DATASET)
         and data.get("calibration_split") == "train"
         and data.get("engines") == {
             "main": {"imgsz": 544, "batch": 1, "dynamic": False},
             "small": {"imgsz": 512, "batch": 1, "dynamic": False}},
         "Cannot reuse incompatible or unpinned LEVIR V2 inventory")
    need(len({r["job_id"] for r in data["checkpoints"]}) == len(paths),
         "Collision in Hugging Face checkpoint IDs")
    for row in data["checkpoints"]:
        seed, identity = common.checkpoint_seed(row["path"])
        need((row["job_id"], row["checkpoint_seed"], row["seed_resolution"], row["kind"])
             == (common.job_id(row["path"]), seed, identity,
                 PurePosixPath(row["path"]).name),
             "Checkpoint seed/path identity changed")


def static_engine_metadata(path: Path, role: str) -> dict:
    """Fail if returned engine is dynamic, FP16-only or wrong canvas/batch."""
    need(role in ROLES and path.is_file() and path.stat().st_size > 16,
         f"Invalid {role} static engine path: {path}")
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(4), "little", signed=True)
        need(4 <= length <= 1_000_000, "TensorRT engine metadata is unavailable")
        meta = json.loads(stream.read(length).decode("utf-8"))
    args = meta.get("args") or {}
    dims = meta.get("imgsz")
    image_size = ROLES[role]
    valid_dims = dims == [image_size, image_size] or dims == image_size
    need(meta.get("task") == "detect"
         and args.get("int8") is True
         and args.get("dynamic") is False
         and args.get("nms") is False
         and (meta.get("batch") == 1 or args.get("batch") == 1)
         and valid_dims,
         f"Wrong/non-INT8 static TensorRT {role} engine metadata: {meta}")
    return meta


def static_engine_for(source: Path, checkpoint_sha: str, run: Path,
                      role: str, yaml_file: Path, fingerprint: dict,
                      source_files: dict, device: str, runtime: dict) -> tuple[Path, dict]:
    """Export one single-purpose static TensorRT INT8 engine with independent PTQ."""
    need(role in ROLES, f"Unknown LEVIR export role {role}")
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from ultralytics import YOLO

    dest = run / "paper_export" / role
    dest.mkdir(parents=True, exist_ok=True)
    staged = dest / "source.pt"
    if staged.is_file():
        need(common.sha256(staged) == checkpoint_sha,
             f"{role}: staged source checkpoint changed")
    else:
        shutil.copyfile(source, staged)
        need(common.sha256(staged) == checkpoint_sha,
             f"{role}: source checkpoint staging changed bytes")
    engine = dest / "source.engine"
    manifest = dest / "engine_int8_export.json"
    spec = {
        "protocol": CONTRACT, "role": role,
        "checkpoint_sha256": checkpoint_sha,
        "imgsz": ROLES[role], "batch": 1, "dynamic": False,
        "int8": True, "half": False, "nms": False, "simplify": False,
        "calibration_data_yaml": str(yaml_file),
        "calibration_yaml_sha256": common.sha256(yaml_file),
        "calibration_inputs": fingerprint,
        "calibration_split": "train", "calibration_fraction": 1.0,
        "source_files_sha256": source_files,
        "runtime": runtime, "device": device,
    }
    if engine.exists() or manifest.exists():
        need(engine.is_file() and manifest.is_file(),
             f"Interrupted {role} export: inspect files rather than reusing engine")
        old = json.loads(manifest.read_text(encoding="utf-8"))
        need(old.get("fingerprint") == spec
             and old.get("engine_sha256") == common.sha256(engine),
             f"{role}: incompatible INT8 engine identity")
        need(old.get("metadata") == static_engine_metadata(engine, role),
             f"{role}: cached engine metadata changed")
        return engine, old

    converted = Path(YOLO(str(staged), task="detect").export(
        format="engine", imgsz=ROLES[role], batch=1,
        half=False, int8=True, dynamic=False, nms=False,
        simplify=False, device=device, data=str(yaml_file),
        split="train", fraction=1.0, verbose=False,
    )).resolve()
    need(converted == engine.resolve() and engine.is_file(),
         f"Wrong TensorRT {role} export output: {converted}")
    metadata = static_engine_metadata(engine, role)
    record = {"fingerprint": spec, "engine_sha256": common.sha256(engine),
              "metadata": metadata}
    common.atomic_json(manifest, record)
    return engine, record


def evaluate_main_pt32(checkpoint: Path, data_yaml: Path, images: list[Path],
                       destination: Path, device: str, workers: int) -> tuple[dict, dict]:
    """Exactly historical FP16-V3 PT32 batch8 paper preprocessor."""
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics, standard_metrics
    from utils.ftsc_deployment_backend_eval_v1 import capture_rect_val_inputs
    ensure_local_ultralytics()
    from ultralytics import YOLO

    reference = destination / "pt32_reference"
    reference.mkdir(parents=True, exist_ok=True)
    seen = {}
    model = YOLO(str(checkpoint), task="detect")
    model.add_callback("on_val_end", lambda v: seen.__setitem__("validator", v))
    value = model.val(
        data=str(data_yaml), split="test", imgsz=512, batch=8, rect=True,
        conf=.001, iou=.5, max_det=300, save_json=True,
        device=device, workers=workers, plots=False, verbose=False,
        project=str(reference), name="main_pt32", exist_ok=True,
    )
    validator = seen.get("validator")
    need(validator is not None and getattr(validator, "seen", None) == len(images),
         "PT32 main did not validate all 788 LEVIR native images")
    need(int(validator.args.imgsz) == 512 and int(validator.args.batch) == 8
         and bool(validator.args.rect) and int(validator.stride) == 32,
         "PT32 LEVIR validator diverged from V3/FP16 protocol")
    pixels = capture_rect_val_inputs(validator, images, reference)
    need(pixels["preprocessed_input_canvas"] == 544
         and pixels["preprocessed_input_count"] == 788,
         "PT32 V3 main input must be 788 544x544 images")
    metrics = standard_metrics(value, "test")
    return metrics, pixels


def evaluate_main_int8(engine: Path, yaml_file: Path, images: list[Path],
                       destination: Path, device: str, workers: int,
                       pt32_hashes: dict) -> tuple[dict, dict]:
    """Use exact proven static544/rect512 validator from FP16 V3, not a new imitation."""
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics, standard_metrics
    from utils.ftsc_deployment_backend_eval_v1 import (
        paper_rect_static_validator, capture_rect_val_inputs,
    )
    ensure_local_ultralytics()
    from ultralytics import YOLO

    state = {}
    model = YOLO(str(engine), task="detect")
    model.add_callback("on_val_end", lambda v: state.__setitem__("validator", v))
    value = model.val(
        validator=paper_rect_static_validator(),
        data=str(yaml_file), split="test",
        imgsz=544, batch=1, rect=True,
        conf=.001, iou=.5, max_det=300, save_json=True,
        device=device, workers=workers, plots=False, verbose=False,
        project=str(destination), name="main_static544_int8", exist_ok=True,
    )
    v = state.get("validator")
    need(v is not None and getattr(v, "seen", None) == len(images)
         and int(v.args.imgsz) == 512 and int(v.args.batch) == 1
         and bool(v.args.rect) and int(v.stride) == 32,
         "INT8 main validator did not follow proven FP16 V3 static544 contract")
    hashed = capture_rect_val_inputs(v, images, destination)
    expected = destination / "pt32_reference" / "test_val_preprocessed_inputs_sha256.json"
    actual = destination / "test_val_preprocessed_inputs_sha256.json"
    need(expected.is_file() and expected.read_bytes() == actual.read_bytes()
         and common.sha256(expected) == pt32_hashes["preprocessed_input_manifest_sha256"],
         "INT8 main 788-input PT32 pixel parity FAILED; main AP cannot be accepted")
    return standard_metrics(value, "test"), hashed


def predict_small(checkpoint: Path, images: list[Path], destination: Path,
                  device: str, workers: int, *, backend: str) -> tuple[dict, dict]:
    """Historical 512 native-image YOLO.predict/TinyBenchmark for static1 vs PT32."""
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    from evaluate_test.size_bucket_evaluator import (
        _coco_ground_truth, _evaluate_coco, MAX_SOURCE_LIST_CHUNK,
    )
    from utils import ftsc_deployment_paper_backend_v1 as paper

    ensure_local_ultralytics()
    from ultralytics import YOLO

    need(backend in ("pt32", "trt_int8"), f"Unsupported Small predictor backend {backend}")
    reference = backend == "pt32"
    out = destination / "pt32_reference" if reference else destination
    out.mkdir(parents=True, exist_ok=True)
    # The PT32 _predictions() source grouping is 100; the static TRT engine only
    # supports one input per call. For 512-square LEVIR all auto-rect pixels match.
    source_batch = MAX_SOURCE_LIST_CHUNK if reference else 1
    predict_batch = 8 if reference else 1
    recorder: dict = {}
    legacy = (paper.historical_source_auto_flags(images, MAX_SOURCE_LIST_CHUNK)
              if not reference else None)
    predictor = paper.paper_predictor(
        recorder, backend, dynamic_required=False,
        historical_auto_by_image=legacy,
    )
    model = YOLO(str(checkpoint), task="detect")
    detections = []
    for start in range(0, len(images), source_batch):
        selection = images[start:start + source_batch]
        stream = model.predict(
            source=paper.named_predictor_sources(selection), predictor=predictor,
            imgsz=512, batch=predict_batch, device=device, workers=workers,
            conf=.25, iou=.5, max_det=300, rect=True,
            stream=True, verbose=False,
        )
        count = 0
        for prediction in stream:
            need(count < len(selection), "Unexpected extra LEVIR predictor image")
            need(Path(prediction.path).resolve() == selection[count].resolve(),
                 "Wrong source image in LEVIR TinyBenchmark predictor")
            boxes = prediction.boxes
            xyxy = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
            scores = boxes.conf.detach().cpu().tolist() if boxes is not None else []
            for (x1, y1, x2, y2), score in zip(xyxy, scores, strict=True):
                detections.append({
                    "image_id": start + count + 1, "category_id": 1,
                    "bbox": [x1, y1, max(0., x2-x1), max(0., y2-y1)],
                    "score": float(score),
                })
            count += 1
        need(count == len(selection), "LEVIR Small predictor skipped images")
    hashes = paper.seal_inputs(
        out / "small_inputs.json", recorder, images,
        "YOLO.predict/imgsz/conf_default_025/rect_true",
    )
    gt = out / "test_size_ground_truth.json"
    preds = out / "test_size_predictions.json"
    paper.write_new_json(gt, _coco_ground_truth(images))
    paper.write_new_json(preds, detections)
    result = _evaluate_coco(gt, preds)
    value = result["test_size/AP50-Small"]
    need(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
         f"Invalid {backend} native TinyBenchmark AP50-small")
    return result, {**hashes, "gt_sha256": common.sha256(gt),
                    "predictions_sha256": common.sha256(preds),
                    "prediction_count": len(detections)}


def verify_small_parity(out: Path, pt32: dict, int8: dict) -> dict:
    prior = out / "pt32_reference" / "small_inputs.json"
    now = out / "small_inputs.json"
    need(prior.is_file() and now.is_file()
         and prior.read_bytes() == now.read_bytes()
         and common.sha256(prior) == pt32["input_manifest_sha256"]
         and common.sha256(now) == int8["input_manifest_sha256"]
         and pt32["input_count"] == int8["input_count"] == 788,
         "INT8 Small 512 788-image PT32 pixel parity FAILED")
    need(pt32["gt_sha256"] == int8["gt_sha256"],
         "PT32 / INT8 TinyBenchmark ground-truth SHA mismatch")
    return {
        "main_and_small_input_pixel_parity": True,
        "small_pt32_input_manifest_sha256": pt32["input_manifest_sha256"],
        "small_int8_input_manifest_sha256": int8["input_manifest_sha256"],
        "small_gt_sha256": int8["gt_sha256"],
    }


def evaluate_one(args, pinned: dict, row: dict, source_files: dict,
                 runtime: dict, context: tuple) -> dict:
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics

    where = path_for(args.project, row)
    final = where / "metrics_int8.json"
    yaml_file, images, dataset_provenance, calibration = context
    if final.is_file():
        doc = json.loads(final.read_text(encoding="utf-8"))
        verify_completed(where, row, pinned, doc, source_files,
                         dataset_provenance, calibration)
        return {"status": "REUSED_VERIFIED", "source": row["path"]}
    checkpoint, digest = common.existing_sha_source(
        row, where, pinned["repo"], pinned["hf_revision"])
    ensure_local_ultralytics()
    engines = {}
    specs = {}
    for role in ROLES:
        path, spec = static_engine_for(checkpoint, digest, where, role, yaml_file,
                                       calibration, source_files, args.device, runtime)
        engines[role], specs[role] = path, spec

    # Measure exact PT32 predictions on THIS checkpoint and verify both
    # inference preprocessors before accepting any INT8 numbers.
    pt_main, pt_main_inputs = evaluate_main_pt32(
        checkpoint, yaml_file, images, where, args.device, args.workers)
    tr_main, tr_main_inputs = evaluate_main_int8(
        engines["main"], yaml_file, images, where, args.device, args.workers,
        pt_main_inputs)
    pt_small, pt_small_inputs = predict_small(
        checkpoint, images, where, args.device, args.workers, backend="pt32")
    tr_small, tr_small_inputs = predict_small(
        engines["small"], images, where, args.device, args.workers,
        backend="trt_int8")
    parity = verify_small_parity(where, pt_small_inputs, tr_small_inputs)

    measured = {
        METRIC_KEYS[0]: tr_main[METRIC_KEYS[0]],
        METRIC_KEYS[1]: tr_main[METRIC_KEYS[1]],
        METRIC_KEYS[2]: tr_small[METRIC_KEYS[2]],
    }
    reference = {
        METRIC_KEYS[0]: pt_main[METRIC_KEYS[0]],
        METRIC_KEYS[1]: pt_main[METRIC_KEYS[1]],
        METRIC_KEYS[2]: pt_small[METRIC_KEYS[2]],
    }
    for item in (measured, reference):
        need(all(type(item[k]) in (int, float) and math.isfinite(item[k])
                 and 0 <= item[k] <= 1 for k in METRIC_KEYS),
             f"Missing/invalid LEVIR paper metric: {item}")
    main_pt_path = where / "pt32_reference/test_val_preprocessed_inputs_sha256.json"
    main_int8_path = where / "test_val_preprocessed_inputs_sha256.json"
    need(main_pt_path.read_bytes() == main_int8_path.read_bytes(),
         "Main PT32/INT8 pixel parity must hold before saving metrics")
    provenance = {
        "protocol": CONTRACT,
        "main": {"imgsz": 544, "batch": 1, "rect": True,
                 "content_imgsz": 512, "stride": 32,
                 "engine_sha256": specs["main"]["engine_sha256"],
                 "export_manifest_sha256": common.sha256(
                     where / "paper_export/main/engine_int8_export.json"),
                 "pixel_inputs_sha256": common.sha256(main_int8_path)},
        "small": {"imgsz": 512, "batch": 1, "rect": True,
                  "conf": .25, "iou": .5,
                  "engine_sha256": specs["small"]["engine_sha256"],
                  "export_manifest_sha256": common.sha256(
                      where / "paper_export/small/engine_int8_export.json"),
                  "pixel_inputs_sha256": tr_small_inputs["input_manifest_sha256"],
                  "gt_sha256": tr_small_inputs["gt_sha256"],
                  "predictions_sha256": tr_small_inputs["predictions_sha256"]},
        "pt32_reference_input_hash": {
            "main": pt_main_inputs["preprocessed_input_manifest_sha256"],
            "small": pt_small_inputs["input_manifest_sha256"],
        },
        "full_788_image_parity": parity["main_and_small_input_pixel_parity"],
    }
    report = {
        "contract": CONTRACT, "status": "COMPLETE_INT8_STATIC_PARITY",
        "dataset": DATASET, "hf_repo": pinned["repo"],
        "hf_revision": pinned["hf_revision"], "source": row["path"],
        "checkpoint_sha256": digest, "seed": row["checkpoint_seed"],
        "seed_resolution": row["seed_resolution"],
        "source_files_sha256": source_files,
        "data_provenance": dataset_provenance,
        "train_calibration_fingerprint": calibration,
        "runtime": runtime, "metrics": measured,
        "pt32_metrics": reference,
        "INT8_minus_PT32_AP_points": {
            key: 100. * (measured[key] - reference[key]) for key in METRIC_KEYS},
        "engine_provenance": provenance,
    }
    verify_completed(where, row, pinned, report, source_files,
                     dataset_provenance, calibration)
    common.atomic_json(final, report)
    return {"status": report["status"], "source": row["path"],
            "int8_AP50_pct": 100 * measured[METRIC_KEYS[0]],
            "pt32_AP50_pct": 100 * reference[METRIC_KEYS[0]],
            "delta_AP50_points": report["INT8_minus_PT32_AP_points"][METRIC_KEYS[0]]}


def verify_completed(where: Path, row: dict, pinned: dict, report: dict,
                     sources: dict, dataset: dict, calibration: dict) -> None:
    need(report.get("contract") == CONTRACT
         and report.get("status") == "COMPLETE_INT8_STATIC_PARITY"
         and report.get("source") == row["path"]
         and report.get("hf_revision") == pinned["hf_revision"]
         and report.get("dataset") == DATASET
         and report.get("source_files_sha256") == sources
         and report.get("data_provenance") == dataset
         and report.get("train_calibration_fingerprint") == calibration,
         "V2 LEVIR result source/calibration/split/contract differs")
    roles = report.get("engine_provenance", {})
    need(roles.get("full_788_image_parity") is True,
         "V2 LEVIR result has not passed PT32/INT8 input parity")
    source = where / "source" / row["kind"]
    need(source.is_file() and common.sha256(source) == report["checkpoint_sha256"],
         f"Unverified HF source weights: {source}")
    for role in ROLES:
        engine = where / "paper_export" / role / "source.engine"
        manifest = where / "paper_export" / role / "engine_int8_export.json"
        need(engine.is_file() and manifest.is_file(),
             f"Missing {role} standalone INT8 static engine")
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        meta = static_engine_metadata(engine, role)
        need(doc["fingerprint"]["protocol"] == CONTRACT
             and doc["fingerprint"]["source_files_sha256"] == sources
             and doc["fingerprint"]["calibration_inputs"] == calibration
             and doc["fingerprint"]["calibration_split"] == "train"
             and doc["engine_sha256"] == common.sha256(engine)
             and doc["metadata"] == meta
             and doc["engine_sha256"] == roles[role]["engine_sha256"]
             and roles[role]["export_manifest_sha256"] == common.sha256(manifest),
             f"Stale/invalid {role} INT8 engine/export/calibration source")
    filenames = {
        "main": ("test_val_preprocessed_inputs_sha256.json",
                 "pt32_reference/test_val_preprocessed_inputs_sha256.json"),
        "small": ("small_inputs.json", "pt32_reference/small_inputs.json"),
    }
    for role, (relative, prior) in filenames.items():
        current = where / relative
        old = where / prior
        need(old.is_file() and current.is_file() and old.read_bytes() == current.read_bytes(),
             f"{role} INT8 pixels differ from PT32")
    need(roles["main"]["pixel_inputs_sha256"] ==
         common.sha256(where / "test_val_preprocessed_inputs_sha256.json")
         and roles["small"]["pixel_inputs_sha256"] ==
         common.sha256(where / "small_inputs.json")
         and roles["main"]["pixel_inputs_sha256"] ==
         roles["pt32_reference_input_hash"]["main"]
         and roles["small"]["pixel_inputs_sha256"] ==
         roles["pt32_reference_input_hash"]["small"],
         "PT32/INT8 pixel provenance mismatch")
    for key, pth in (
        ("gt_sha256", "test_size_ground_truth.json"),
        ("predictions_sha256", "test_size_predictions.json"),
    ):
        need(common.sha256(where / pth) == roles["small"][key],
             f"Invalid INT8 Small {key} artifact")
    need(common.sha256(where / "pt32_reference/test_size_ground_truth.json")
         == roles["small"]["gt_sha256"],
         "PT32 reference GT differs from INT8 GT")
    for group in ("metrics", "pt32_metrics"):
        metrics = report.get(group) or {}
        need(all(type(metrics.get(k)) in (float, int)
                 and math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1
                 for k in METRIC_KEYS),
             f"Missing/invalid {group} metrics")


def aggregate(args, pinned: dict, sources: dict,
              dataset: dict | None = None, calibration: dict | None = None) -> dict:
    rows, succeeded, failures = [], [], []
    for row in pinned["checkpoints"]:
        directory = path_for(args.project, row)
        file = directory / "metrics_int8.json"
        status = {"source": row["path"], "job_id": row["job_id"],
                  "seed": row["checkpoint_seed"], "checkpoint_kind": row["kind"],
                  "seed_resolution": row["seed_resolution"]}
        if file.is_file():
            try:
                doc = json.loads(file.read_text(encoding="utf-8"))
                need(dataset is not None and calibration is not None,
                     "Auditing V2 result requires frozen LEVIR split+calibration provenance")
                verify_completed(directory, row, pinned, doc, sources,
                                 dataset, calibration)
                status.update(status="COMPLETE_INT8_STATIC_PARITY",
                              checkpoint_sha256=doc["checkpoint_sha256"])
                for name, metric in zip(SCORE_NAMES, METRIC_KEYS):
                    status[name] = 100. * doc["metrics"][metric]
                    status["PT32_" + name] = 100. * doc["pt32_metrics"][metric]
                    status["Delta_" + name] = 100. * (
                        doc["metrics"][metric] - doc["pt32_metrics"][metric])
                succeeded.append(status)
            except Exception as error:
                status.update(status="BLOCKED_INVALID", reason=str(error))
                failures.append(status)
        else:
            failure = directory / "failure.json"
            reason = "not evaluated"
            if failure.is_file():
                try:
                    reason = json.loads(failure.read_text(encoding="utf-8")).get("error", reason)
                except Exception:
                    reason = "invalid failure.json"
            status.update(status="BLOCKED" if failure.is_file() else "PENDING",
                          reason=reason)
            failures.append(status)
        rows.append(status)
    fields = ["source","job_id","seed","checkpoint_kind","seed_resolution",
              "status","reason","checkpoint_sha256",
              *[field for n in SCORE_NAMES
                for field in (n, "PT32_"+n, "Delta_"+n)]]
    common.csv_report(args.project / DATASET / "all_checkpoint_status.csv", rows, fields)
    common.csv_report(args.project / DATASET / "measured_int8_metrics.csv", succeeded, fields)
    families = {}
    for row in succeeded:
        if row["checkpoint_kind"] != "best.pt" or row["seed_resolution"] != "checkpoint_path":
            continue
        key = common.SEED_RE.sub("/seed_{seed}/", row["source"])
        slot = families.setdefault(key, {})
        need(row["seed"] not in slot, f"Duplicate seed in HF checkpoint family: {key}")
        slot[row["seed"]] = row
    summary = []
    for key, seeds in families.items():
        if set(seeds) != {42,43,44}:
            continue
        result = {"checkpoint_family": key, "n": 3, "seeds": "42,43,44"}
        for name in SCORE_NAMES:
            values = [seeds[s][name] for s in (42,43,44)]
            result[name+"_mean"] = statistics.mean(values)
            result[name+"_sample_std"] = statistics.stdev(values)
        summary.append(result)
    cols = ["checkpoint_family","n","seeds",
            *[k+suffix for k in SCORE_NAMES
              for suffix in ("_mean","_sample_std")]]
    common.csv_report(args.project / DATASET / "complete_three_seed_best_aggregate.csv",
                      summary, cols)
    report = {"contract": CONTRACT, "dataset": DATASET,
              "status": "COMPLETE" if not failures else "INCOMPLETE",
              "hf_revision": pinned["hf_revision"],
              "total_HF_checkpoints": len(pinned["checkpoints"]),
              "measured_int8": len(succeeded),
              "blocked_or_pending": len(failures),
              "complete_three_seed_families": len(summary),
              "failure_examples": failures[:30],
              "separate_from_legacy_dynamic_int8": True,
              "main_engine": "static544_batch1_same_validator_as_FP16_V3",
              "small_engine": "static512_batch1_same_predict_protocol_as_FP16"}
    common.atomic_json(args.project / DATASET / "summary.json", report, immutable=False)
    return report


def dataset_context(args) -> tuple:
    """Shared frozen LEVIR dataset: train calibration, 788 official test images."""
    need(args.data_yaml is not None, "LEVIR needs canonical --data-yaml")
    yaml_file = args.data_yaml.expanduser().resolve()
    need(yaml_file.name == "levir_ship.yaml"
         and yaml_file.parent.name == "levir_ship_yolo_seed42"
         and yaml_file.is_file(),
         "Use only frozen LEVIR split seed42 YAML")
    images, context = common.levir_inputs(yaml_file)
    calibration = common.calibration_fingerprint(yaml_file)
    return yaml_file, images, context, calibration


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument("--phase", choices=("plan","run","aggregate"), default="plan")
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--confirm-int8-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    need(args.max_jobs >= 0 and args.start_index >= 0 and args.workers >= 0,
         "Invalid V2 job range/workers")
    pinned = inventory(args.project, args.repo)
    validate_inventory(pinned, args.repo)
    if args.phase == "plan":
        result = {"status": "PLAN_ONLY", "contract": CONTRACT,
                  "repo": pinned["repo"], "revision": pinned["hf_revision"],
                  "total_checkpoints": pinned["count"],
                  "main": "static544/batch1/FP16-V3-rect512 + 788 PT32 parity",
                  "small": "static512/batch1/native-512 + 788 PT32 parity",
                  "calibration": "TRAIN only; separate 544 and 512 INT8 calibrations",
                  "output": str(args.project / DATASET)}
        print(json.dumps(result, indent=2))
        return result
    context = dataset_context(args)
    code_hashes = source_hashes()
    if args.phase == "run":
        need(args.confirm_int8_run, "Explicit --confirm-int8-run required")
        from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
        ensure_local_ultralytics()
        device_info = common.runtime_identity(args.device)
        end = len(pinned["checkpoints"]) if args.max_jobs == 0 else min(
            args.start_index + args.max_jobs, len(pinned["checkpoints"]))
        need(args.start_index < end, "No LEVIR HF checkpoint in selected range")
        for index in range(args.start_index, end):
            row = pinned["checkpoints"][index]
            print(f"[LEVIR STATIC INT8 V2 {index+1}/{pinned['count']}] {row['path']}",
                  flush=True)
            try:
                print(json.dumps(evaluate_one(args, pinned, row, code_hashes,
                                              device_info, context)), flush=True)
            except Exception as error:
                failure = {"source": row["path"], "contract": CONTRACT,
                           "error_type": type(error).__name__, "error": str(error)}
                common.atomic_json(path_for(args.project, row) / "failure.json",
                                   failure, immutable=False)
                print(json.dumps({"status": "BLOCKED", **failure}), flush=True)
                traceback.print_exc()
                if args.stop_on_error:
                    raise
            finally:
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
    report = aggregate(args, pinned, code_hashes, context[2], context[3])
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
