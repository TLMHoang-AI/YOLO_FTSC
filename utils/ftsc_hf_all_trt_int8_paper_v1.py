"""Evaluate ALL HF YOLO checkpoint weights for LEVIR-Ship/TinyPerson with real TensorRT INT8.

Does not train, publish, or alter historical results / notebook bootstrap cells.
HF inventory is pinned by commit revision and includes all weights/*.pt entries,
including best.pt, last.pt and named epoch weights. Unsupported checkpoints are
reported BLOCKED, never quietly omitted.

LEVIR paper protocol: native YOLO.val(test, imgsz=512, rect=True, conf=.001,
iou=.5, batch8) on **544 actual preprocessed canvas**; independently,
YOLO.predict(test, imgsz=512, default conf=.25, iou=.5) -> TinyBenchmark
AP50-Small with native 512-pixel-coordinate GT.

TinyPerson paper protocol: 640 official sw640/sh512 corners, conf=.001, iou=.5,
batch8 original source group64, dynamic TRT8 with historic auto-rect decisions,
translate to original image, class-agnostic NMS .5, original-image TinyBenchmark
AP50/mAP50:95/AP50-Small. No corner-level AP presented as merged AP.

INT8 uses TensorRT PTQ with train-only calibration via split='train'.
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
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "FTSC_HF_ALL_TRT_INT8_PAPER_METRICS_V1"
HF_REPO = "TLMHoang/FTSC_results"
DATASETS = ("LEVIR-Ship", "TinyPerson")
SOURCE_FILES = (
    "utils/ftsc_hf_all_trt_int8_paper_v1.py",
    "utils/ftsc_deployment_paper_backend_v1.py",
    "utils/ftsc_deployment_backend_eval_v1.py",
    "evaluate_test/size_bucket_evaluator.py",
    "evaluate_test/standard_detection_metrics.py",
    "train_tinyperson_scripts/train_all_tinyperson.py",
    "train_levir_scripts/train_all_levir_yolov8n_p2_routing.py",
    "vendor/tinyperson_cocoeval.py",
    "models_related/ultralytics/ultralytics/engine/exporter.py",
    "models_related/ultralytics/ultralytics/utils/export/engine.py",
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
    "models_related/ultralytics/ultralytics/engine/model.py",
    "models_related/ultralytics/ultralytics/engine/predictor.py",
    "models_related/ultralytics/ultralytics/engine/validator.py",
    "models_related/ultralytics/ultralytics/models/yolo/detect/val.py",
    "models_related/ultralytics/ultralytics/models/yolo/detect/predict.py",
    "models_related/ultralytics/ultralytics/data/build.py",
    "models_related/ultralytics/ultralytics/data/base.py",
    "models_related/ultralytics/ultralytics/data/dataset.py",
    "models_related/ultralytics/ultralytics/data/augment.py",
    "models_related/ultralytics/ultralytics/utils/nms.py",
    "models_related/ultralytics/ultralytics/nn/backends/tensorrt.py",
)
SCORE_NAMES = ("AP50_pct", "mAP50_95_pct", "AP50_Small_pct")
METRICS = {
    "LEVIR-Ship": ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small"),
    "TinyPerson": ("test_merged/AP50", "test_merged/mAP50-95", "test_merged/AP50-Small"),
}
SEED_RE = re.compile(r"(?:^|/)seed[_-]?(\d+)(?:[_/-]|$)", re.I)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashes() -> dict[str, str]:
    return {p: sha256(ROOT / p) for p in SOURCE_FILES}


def atomic_json(path: Path, obj: Any, *, immutable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if immutable:
        if path.exists():
            need(path.read_text(encoding="utf-8") == encoded,
                 f"Refuse to overwrite previously recorded source/result: {path}")
            return
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        staging = path.with_name(path.name + ".new")
        need(not staging.exists(), f"Interrupted prior status update: {staging}")
        with staging.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        staging.replace(path)


def csv_report(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".new")
    need(not stage.exists(), f"Interrupted CSV write: {stage}")
    with stage.open("x", newline="", encoding="utf-8") as output:
        w = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    stage.replace(path)


def checkpoint_paths(paths: list[str], dataset: str) -> list[str]:
    """All saved .pt weights for this dataset, not only paper-selected five."""
    need(dataset in DATASETS, f"Unexpected dataset: {dataset}")
    prefix = dataset + "/"
    found = []
    for item in paths:
        if not item.startswith(prefix):
            continue
        p = PurePosixPath(item)
        if (not p.is_absolute() and ".." not in p.parts
                and len(p.parts) >= 4 and p.parent.name == "weights"
                and p.suffix.lower() == ".pt"):
            found.append(item)
    return sorted(set(found))


def checkpoint_seed(path: str) -> tuple[int, str]:
    m = SEED_RE.search(path)
    if m:
        seed = int(m.group(1))
        if seed in (42, 43, 44):
            return seed, "checkpoint_path"
        return seed, "checkpoint_path_unprepared"
    # Unknown lineage: preserve truth; use train seed42 strictly as an
    # explicitly labeled calibration fallback, not as a claimed training seed.
    return 42, "CALIBRATION_FALLBACK_42_UNVERIFIED_TRAINING_SEED"


def job_id(path: str) -> str:
    source = PurePosixPath(path)
    name = re.sub(r"[^0-9A-Za-z_.-]+", "-", source.parent.parent.name)[:48]
    return f"{name}__{source.stem}__{hashlib.sha256(path.encode()).hexdigest()[:18]}"


def inventory_path(project: Path, dataset: str) -> Path:
    return project / dataset / "hf_pinned_inventory.json"


def build_inventory(repo: str, dataset: str, project: Path) -> dict:
    from huggingface_hub import HfApi

    path = inventory_path(project, dataset)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        validate_inventory(stored, repo, dataset)
        return stored
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    revision = api.repo_info(repo_id=repo, repo_type="dataset").sha
    need(isinstance(revision, str) and len(revision) == 40,
         "Cannot pin HF repo commit SHA")
    paths = api.list_repo_files(repo_id=repo, repo_type="dataset", revision=revision)
    sources = checkpoint_paths(paths, dataset)
    need(sources, f"No /weights/*.pt files found at {repo}/{dataset} @ {revision}")
    entries = []
    for src in sources:
        seed, identity = checkpoint_seed(src)
        entries.append({"path": src, "job_id": job_id(src),
                        "checkpoint_seed": seed,
                        "seed_resolution": identity,
                        "kind": PurePosixPath(src).name})
    result = {"schema": CONTRACT, "repo": repo, "dataset": dataset,
              "hf_revision": revision, "checkpoints": entries,
              "checkpoint_count": len(entries),
              "int8": {"engine": "TensorRT", "calibration_split": "train",
                       "dynamic": True, "calibration_fraction": 1.0},
              "excluded": "No checkpoints under other dataset prefixes"}
    atomic_json(path, result)
    return result


def validate_inventory(data: dict, repo: str, dataset: str) -> None:
    entries = data.get("checkpoints") or []
    need(data.get("schema") == CONTRACT and data.get("repo") == repo
         and data.get("dataset") == dataset
         and isinstance(data.get("hf_revision"), str)
         and len(data["hf_revision"]) == 40
         and data.get("checkpoint_count") == len(entries) > 0,
         "Pinned inventory contract differs; use a new project path for a new HF snapshot")
    expected_paths = [e["path"] for e in entries]
    need(expected_paths == checkpoint_paths(expected_paths, dataset)
         and len({e["job_id"] for e in entries}) == len(entries),
         "Inventory includes duplicate, non-weights or wrongly-scoped checkpoint")
    for e in entries:
        seed, reason = checkpoint_seed(e["path"])
        need(e["job_id"] == job_id(e["path"])
             and e["checkpoint_seed"] == seed
             and e["kind"] == PurePosixPath(e["path"]).name
             and e["seed_resolution"] == reason,
             f"Source/seed provenance altered: {e['path']}")


def job_dir(project: Path, dataset: str, row: dict) -> Path:
    return project / dataset / "jobs" / row["job_id"]


def existing_sha_source(row: dict, run: Path, repo: str, revision: str) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download
    source = run / "source" / row["kind"]
    provenance = run / "source" / "source.json"
    if source.is_file() and provenance.is_file():
        meta = json.loads(provenance.read_text(encoding="utf-8"))
        digest = sha256(source)
        need(meta == {"repo": repo, "revision": revision, "path": row["path"],
                      "checkpoint_sha256": digest},
             f"Staged HF weight has unexpected provenance: {source}")
        return source, digest
    need(not source.exists() and not provenance.exists(),
         f"Incomplete or unverified staged model at {source}")
    downloaded = Path(hf_hub_download(repo_id=repo, repo_type="dataset",
                                     filename=row["path"], revision=revision,
                                     token=os.environ.get("HF_TOKEN") or None))
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(downloaded, source)
    digest = sha256(source)
    need(digest == sha256(downloaded), f"Failed to stage original checkpoint: {row['path']}")
    atomic_json(provenance, {"repo": repo, "revision": revision, "path": row["path"],
                             "checkpoint_sha256": digest})
    return source, digest


def runtime_identity(device: str) -> dict[str, Any]:
    import torch
    import tensorrt as trt

    need(torch.cuda.is_available(), "TensorRT INT8 export needs an available NVIDIA GPU")
    idx = int(device.split(":")[-1]) if ":" in device else int(device) if device.isdigit() else 0
    need(0 <= idx < torch.cuda.device_count(), f"CUDA device index outside available GPUs: {device}")
    return {"gpu": torch.cuda.get_device_name(idx),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "tensorrt": str(trt.__version__),
            "device": device}


def read_engine_metadata(engine_file: Path) -> dict:
    """Ultralytics writes length-prefixed JSON engine metadata; fail closed."""
    need(engine_file.is_file() and engine_file.stat().st_size > 16,
         f"Missing/empty TensorRT engine {engine_file}")
    with engine_file.open("rb") as stream:
        length = int.from_bytes(stream.read(4), "little", signed=True)
        need(4 <= length <= 1_000_000, "Engine missing valid Ultralytics metadata prefix")
        meta = json.loads(stream.read(length).decode("utf-8"))
    opt = meta.get("args", {})
    need(meta.get("task") == "detect"
         and opt.get("int8") is True
         and opt.get("dynamic") is True
         and opt.get("nms") is False,
         f"Engine metadata does not verify real INT8 export request: {engine_file}")
    return meta


def engine_for(source: Path, digest: str, run: Path, dataset: str,
               calib_yaml: Path, calib_seed: int, calib_identity: dict,
               tools_hashes: dict, device: str, hardware: dict) -> tuple[Path, dict]:
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics

    ensure_local_ultralytics()
    from ultralytics import YOLO
    model_file = source.with_suffix(".engine")
    manifest = run / "engine_int8_export.json"
    settings = {
        "contract": CONTRACT, "source_checkpoint_sha256": digest,
        "dataset": dataset, "imgsz": 512 if dataset == "LEVIR-Ship" else 640,
        "batch": 8, "dynamic": True, "int8": True, "half": False,
        "nms": False, "simplify": False, "data_sha256": sha256(calib_yaml),
        "data": str(calib_yaml), "calibration_seed": calib_seed,
        "calibration_fingerprint": calib_identity,
        "calibration_split": "train", "calibration_fraction": 1.0,
        "source_files_sha256": tools_hashes, "runtime": hardware,
    }
    if manifest.is_file() or model_file.is_file():
        need(manifest.is_file() and model_file.is_file(),
             f"Partial prior engine without provenance: {model_file}")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        need(record.get("settings") == settings
             and record.get("engine_sha256") == sha256(model_file),
             f"Incompatible INT8 cached engine; never silently reuse: {model_file}")
        meta = read_engine_metadata(model_file)
        need(record.get("engine_metadata") == meta, "Cached INT8 metadata changed")
        return model_file, record
    # The fork builds genuine INT8 TensorRT with explicit train calibration.
    exported = Path(YOLO(str(source), task="detect").export(
        format="engine", imgsz=settings["imgsz"], batch=8, device=device,
        int8=True, half=False, dynamic=True, nms=False, simplify=False,
        data=str(calib_yaml), split="train", fraction=1.0, verbose=False,
    )).resolve()
    need(exported == model_file.resolve() and exported.is_file(),
         f"Exporter returned unexpected engine: {exported}")
    metadata = read_engine_metadata(exported)
    record = {"settings": settings, "engine_sha256": sha256(exported),
              "engine_metadata": metadata}
    atomic_json(manifest, record)
    return exported, record


def calibration_fingerprint(data_yaml: Path) -> dict[str, Any]:
    """SHA all REAL train images/labels; YAML digest alone cannot prove PTQ inputs."""
    import yaml
    from evaluate_test.size_bucket_evaluator import _label_path

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    item = Path(cfg["train"])
    source = item if item.is_absolute() else root / item
    need(source.exists(), f"Missing train-only calibration input: {source}")
    if source.is_file():
        paths = [Path(s.strip()) for s in source.read_text(encoding="utf-8").splitlines()
                 if s.strip()]
        paths = [p if p.is_absolute() else (source.parent / p).resolve() for p in paths]
    else:
        paths = sorted(q for q in source.rglob("*")
                       if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"})
    need(len(paths) >= 300 and len({str(p.resolve()) for p in paths}) == len(paths),
         f"Insufficient or duplicate train-only calibration images: {len(paths)}")
    image_hash = hashlib.sha256()
    labels_hash = hashlib.sha256()
    for path in paths:
        need(path.is_file(), f"Calibration image missing: {path}")
        label = _label_path(path)
        need(label.is_file(), f"Calibration label missing: {label}")
        image_hash.update(str(path.resolve()).encode("utf-8") + b"\0")
        image_hash.update(sha256(path).encode("ascii") + b"\n")
        labels_hash.update(str(label.resolve()).encode("utf-8") + b"\0")
        labels_hash.update(sha256(label).encode("ascii") + b"\n")
    return {
        "calibration_split": "train", "calibration_images": len(paths),
        "train_image_content_sha256": image_hash.hexdigest(),
        "train_label_content_sha256": labels_hash.hexdigest(),
    }


def levir_inputs(data_yaml: Path) -> tuple[list[Path], dict]:
    from evaluate_test.size_bucket_evaluator import _resolve_test_images, _label_path
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    need(cfg["train"] != cfg["test"] and cfg["val"] != cfg["test"],
         "LEVIR train/calibration overlaps evaluation test split in YAML")
    manifest = data_yaml.parent / "manifest.json"
    need(manifest.is_file(), "Missing canonical LEVIR fixed-split manifest")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    need(m.get("seed") == 42 and
         m.get("splits", {}).get("train", {}).get("images") == 2320 and
         m.get("splits", {}).get("test", {}).get("images") == 788,
         "Wrong LEVIR frozen split policy")
    images = _resolve_test_images(data_yaml)
    need(len(images) == 788 and all(p.is_file() and _label_path(p).is_file()
                                     for p in images),
         "LEVIR-Ship test needs 788 native images + labels")
    from PIL import Image
    for p in images:
        with Image.open(p) as im:
            need(im.size == (512, 512),
                 f"LEVIR small/native evaluator requires 512x512 source image: {p}: {im.size}")
    return images, {"dataset_yaml_sha256": sha256(data_yaml),
                    "split_manifest_sha256": sha256(manifest),
                    "source_image_count": len(images),
                    "native_image_size": [512, 512],
                    "main_input_canvas": [544, 544],
                    "main_rect_stride": 32,
                    "native_size_AP_coordinate_space": "512x512 source pixels"}


def levir_evaluate(engine: Path, data_yaml: Path, images: list[Path],
                   out: Path, device: str, workers: int) -> tuple[dict, dict]:
    """Reuses both historical native and size evaluators, with exact shape guards."""
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics, standard_metrics
    from evaluate_test.size_bucket_evaluator import _coco_ground_truth, _evaluate_coco, MAX_SOURCE_LIST_CHUNK
    from utils import ftsc_deployment_paper_backend_v1 as paper
    from utils.ftsc_deployment_backend_eval_v1 import capture_rect_val_inputs
    ensure_local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator

    seen = {}
    class Rect544Validator(DetectionValidator):
        def get_dataloader(self, dataset_path, batch_size):
            need(int(self.args.imgsz) == 512 and bool(self.args.rect),
                 "LEVIR INT8 must retain native 512 rect dataset preprocessor")
            self.stride = 32  # match PT effective min stride and 544-paper canvas
            return super().get_dataloader(dataset_path, batch_size)

    detector = YOLO(str(engine), task="detect")
    detector.add_callback("on_val_end", lambda val: seen.__setitem__("validator", val))
    native = detector.val(
        validator=Rect544Validator, data=str(data_yaml), split="test",
        imgsz=512, batch=8, rect=True, conf=.001, iou=.5,
        max_det=300, save_json=False,
        device=device, workers=workers, plots=False, verbose=False,
        project=str(out), name="native_rect_test", exist_ok=True,
    )
    validator = seen.get("validator")
    need(validator is not None and int(validator.seen) == 788,
         "INT8 LEVIR native validation did not evaluate all 788 images")
    main_inputs = capture_rect_val_inputs(validator, images, out)
    need(main_inputs.get("preprocessed_input_canvas") == 544,
         "LEVIR native padded canvas is not 544")
    metrics = standard_metrics(native, "test")
    need(all(0 <= metrics[k] <= 1 for k in METRICS["LEVIR-Ship"][:2]),
         "Invalid native LEVIR AP50/mAP50:95")

    recorder = {}
    modes = paper.historical_source_auto_flags(images, MAX_SOURCE_LIST_CHUNK)
    predictor = paper.paper_predictor(recorder, "trt_int8", dynamic_required=True,
                                      historical_auto_by_image=modes)
    dets = []
    for start in range(0, len(images), 8):
        subset = images[start:start + 8]
        iterator = YOLO(str(engine), task="detect").predict(
            source=paper.named_predictor_sources(subset),
            predictor=predictor, imgsz=512, batch=8, rect=True,
            device=device, workers=workers, conf=.25, iou=.5, max_det=300,
            verbose=False, stream=True,
        )
        count = 0
        for result in iterator:
            need(count < len(subset), "LEVIR predictor emitted excess image")
            need(str(Path(result.path).resolve()) == str(subset[count].resolve()),
                 "LEVIR predictor output order/path diverged")
            boxes = result.boxes
            coords = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
            confs = boxes.conf.detach().cpu().tolist() if boxes is not None else []
            for (x1,y1,x2,y2), score in zip(coords, confs, strict=True):
                dets.append({"image_id": start + count + 1, "category_id": 1,
                             "bbox": [x1,y1,max(0.,x2-x1),max(0.,y2-y1)],
                             "score": float(score)})
            count += 1
        need(count == len(subset), "LEVIR predictor skipped images")
    small_in = paper.seal_inputs(out / "small_inputs.json", recorder, images,
                                 "YOLO.predict/imgsz512/conf025/rectTrue/native512")
    gt = out / "test_size_ground_truth.json"
    preds = out / "test_size_predictions.json"
    paper.write_new_json(gt, _coco_ground_truth(images))
    paper.write_new_json(preds, dets)
    small = _evaluate_coco(gt, preds)
    metrics[METRICS["LEVIR-Ship"][2]] = small["test_size/AP50-Small"]
    need(0 <= metrics[METRICS["LEVIR-Ship"][2]] <= 1, "LEVIR AP50-Small absent")
    return metrics, {
        "main_input": main_inputs, "small_input": small_in,
        "gt_sha256": sha256(gt), "predictions_sha256": sha256(preds),
        "prediction_count": len(dets),
        "native_val_speed_ms": dict(getattr(native, "speed", {}) or {}),
        "small_source_coordinate_space": "512px, Small area [20²,32²)",
    }


def tinyperson_evaluate(engine: Path, source_root: Path, test_out: Path,
                        out: Path, device: str, workers: int) -> tuple[dict, dict]:
    from argparse import Namespace
    from utils import ftsc_deployment_paper_backend_v1 as paper
    from train_tinyperson_scripts.train_all_tinyperson import TEST_MERGED_JSON
    import json as _json
    need((source_root / TEST_MERGED_JSON).is_file()
         and (test_out / "corner_manifest.json").is_file(),
         "TinyPerson original merged GT/corner manifest absent")
    raw = _json.loads((test_out / "corner_manifest.json").read_text())
    need(isinstance(raw.get("images"), list) and raw["images"], "TinyPerson corner inventory empty")
    start = time.perf_counter()
    result, small_meta = paper._predict_tiny(
        engine, "trt_int8",
        Namespace(data_root=source_root, tiny_test_out=test_out,
                  device=device, workers=workers), out)
    elapsed = time.perf_counter() - start
    scores = {k: result[k] for k in METRICS["TinyPerson"]}
    need(result.get("test_merged/evaluator_contract") ==
         "tinybenchmark_merged_original_iou50_95_v2",
         "TinyPerson merged original-image benchmark contract changed")
    for k in METRICS["TinyPerson"]:
        need(type(scores[k]) in (int,float) and math.isfinite(scores[k])
             and 0 <= scores[k] <= 1,
             f"Missing/invalid merged-original TinyPerson score: {k}={scores[k]}")
    return scores, {**small_meta, "merged_original_GT_sha256": sha256(source_root / TEST_MERGED_JSON),
                    "merged_original_evaluator": result["test_merged/evaluator_contract"],
                    "corner_count": len(raw["images"]),
                    "merged_e2e_wall_seconds": elapsed,
                    "coordinate_space": "original_image_after_corner_merge_NMS",
                    "corner_AP_not_used": True}


def validate_data(args, row: dict) -> tuple[Path, Any, dict, Path | None]:
    """Train-only calibration YAML; test-only metric inputs, never shared by design."""
    seed = row["checkpoint_seed"]
    if args.dataset == "LEVIR-Ship":
        need(args.data_yaml is not None, "LEVIR --data-yaml required")
        yaml_file = args.data_yaml.expanduser().resolve()
        need(yaml_file.parent.name == "levir_ship_yolo_seed42"
             and yaml_file.name == "levir_ship.yaml" and yaml_file.is_file(),
             "LEVIR requires canonical fixed split-seed42 levir_ship.yaml")
        imgs, context = levir_inputs(yaml_file)
        context["train_calibration"] = calibration_fingerprint(yaml_file)
        return yaml_file, imgs, context, None
    need(args.data_root is not None and args.prepared_root is not None,
         "TinyPerson requires --data-root and --prepared-root")
    need(seed in (42,43,44),
         f"Unknown TinyPerson training seed {seed}; no matching calibration train split")
    from train_tinyperson_scripts import train_all_tinyperson as native
    data_root = args.data_root.expanduser().resolve()
    prepared = args.prepared_root.expanduser().resolve()
    test_out = native.prepare_test_set(data_root, prepared)
    seed_dir = native.prepare_seed_dataset(data_root, prepared, test_out, seed)
    yaml_file = (seed_dir / "tinyperson.yaml").resolve()
    need(yaml_file.is_file(), f"TinyPerson calibration YAML missing {yaml_file}")
    import yaml
    config = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    need(config.get("train") != config.get("test") and
         config.get("val") != config.get("test"),
         "TinyPerson train calibration and original test are not independent")
    return yaml_file, test_out, {
        "train_calibration": calibration_fingerprint(yaml_file),
        "test_corner_manifest_sha256": sha256(test_out / "corner_manifest.json"),
        "original_test_gt_sha256": sha256(data_root / native.TEST_MERGED_JSON),
        "calibration_seed": seed,
        "calibration_seed_resolution": row["seed_resolution"],
        "calibration_yaml_sha256": sha256(yaml_file),
        "paper_metrics_coordinate_space": "merged_original_image",
    }, data_root


def validate_evaluation_evidence(where: Path, dataset: str, record: dict) -> None:
    """Rehash GT, actual input tensors and predictions; do not trust metrics alone."""
    evidence = record.get("evaluation_evidence") or {}
    if dataset == "LEVIR-Ship":
        mapping = (
            ("test_val_preprocessed_inputs_sha256.json",
             evidence.get("main_input", {}).get("preprocessed_input_manifest_sha256")),
            ("small_inputs.json",
             evidence.get("small_input", {}).get("input_manifest_sha256")),
            ("test_size_ground_truth.json", evidence.get("gt_sha256")),
            ("test_size_predictions.json", evidence.get("predictions_sha256")),
        )
        need(evidence.get("main_input", {}).get("preprocessed_input_canvas") == 544
             and evidence.get("small_source_coordinate_space", "").startswith("512px"),
             "LEVIR main 544 / native Small512 protocol evidence missing")
    else:
        mapping = (
            ("small_inputs.json", evidence.get("small_input_manifest_sha256")),
            ("evaluation/test_merged_predictions.json",
             evidence.get("small_predictions_sha256")),
        )
        need(evidence.get("corner_AP_not_used") is True
             and evidence.get("merged_original_evaluator") ==
             "tinybenchmark_merged_original_iou50_95_v2",
             "TinyPerson 3-metric original merge provenance missing")
        need(evidence.get("merged_original_GT_sha256") ==
             evidence.get("small_gt_sha256"),
             "TinyPerson recorded original merged GT identity differs")
    for rel, expected_hash in mapping:
        artifact = where / rel
        need(isinstance(expected_hash, str) and len(expected_hash) == 64
             and artifact.is_file() and sha256(artifact) == expected_hash,
             f"Missing/tampered paper input/GT/predictions: {artifact}")


def evaluate_job(args, inventory: dict, row: dict, src_hashes: dict,
                 hardware: dict, data_cache: dict | None = None) -> dict:
    where = job_dir(args.project, args.dataset, row)
    result_path = where / "metrics_int8.json"
    # Always check current calibration and split identities BEFORE reusing a cached engine.
    if data_cache is not None:
        if row["checkpoint_seed"] not in data_cache:
            data_cache[row["checkpoint_seed"]] = validate_data(args, row)
        calib_yaml, eval_input, context, tiny_root = data_cache[row["checkpoint_seed"]]
    else:
        calib_yaml, eval_input, context, tiny_root = validate_data(args, row)
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        need(saved.get("contract") == CONTRACT and
             saved.get("data_provenance") == context and
             saved.get("dataset") == args.dataset and
             saved.get("hf_revision") == inventory["hf_revision"] and
             saved.get("source") == row["path"] and
             saved.get("source_files_sha256") == src_hashes and
             saved.get("runtime") == hardware and
             saved.get("status") == "COMPLETE_INT8",
             f"Cached INT8 job source/protocol mismatch: {result_path}")
        need(all(type(saved["metrics"].get(k)) in (float,int)
                 and 0 <= saved["metrics"][k] <= 1
                 for k in METRICS[args.dataset]), "Stored result metrics invalid")
        ex = json.loads((where / "engine_int8_export.json").read_text())
        need(ex.get("settings", {}).get("calibration_fingerprint") == context["train_calibration"],
             "Cached engine calibration image/label bytes have changed")
        model = where / "source" / PurePosixPath(row["path"]).name
        need(model.is_file() and sha256(model) == ex["settings"]["source_checkpoint_sha256"],
             "Cached original weights differ")
        engine = model.with_suffix(".engine")
        need(engine.is_file() and sha256(engine) == ex["engine_sha256"],
             "Cached INT8 engine differs")
        read_engine_metadata(engine)
        validate_evaluation_evidence(where, args.dataset, saved)
        return {"status": "REUSED_IDENTICAL", "source": row["path"], "job_id": row["job_id"]}
    ckpt, ckpt_sha = existing_sha_source(
        row, where, inventory["repo"], inventory["hf_revision"])
    engine, manifest = engine_for(ckpt, ckpt_sha, where, args.dataset, calib_yaml,
                                  row["checkpoint_seed"], context["train_calibration"],
                                  src_hashes, args.device, hardware)
    if args.dataset == "LEVIR-Ship":
        score, evidence = levir_evaluate(engine, calib_yaml, eval_input,
                                         where, args.device, args.workers)
    else:
        score, evidence = tinyperson_evaluate(
            engine, tiny_root, eval_input, where, args.device, args.workers)
    for metric in METRICS[args.dataset]:
        v = score.get(metric)
        need(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1,
             f"Invalid {args.dataset}/{row['path']} INT8 {metric}={v}")
    record = {
        "status": "COMPLETE_INT8", "contract": CONTRACT,
        "dataset": args.dataset, "source": row["path"],
        "hf_repo": inventory["repo"], "hf_revision": inventory["hf_revision"],
        "checkpoint_sha256": ckpt_sha, "checkpoint_seed": row["checkpoint_seed"],
        "calibration_seed_resolution": row["seed_resolution"],
        "engine_sha256": manifest["engine_sha256"],
        "engine_export_manifest_sha256": sha256(where / "engine_int8_export.json"),
        "source_files_sha256": src_hashes,
        "runtime": hardware, "data_provenance": context,
        "evaluation_evidence": evidence,
        "metrics": {key: float(score[key]) for key in METRICS[args.dataset]},
        "metrics_are_test": True,
        "quantization": "TensorRT INT8 PTQ requested, calibrated on TRAIN only; layers may fall back to higher precision",
    }
    validate_evaluation_evidence(where, args.dataset, record)
    atomic_json(result_path, record)
    return {"status": "EVALUATED_INT8", "source": row["path"], "job_id": row["job_id"]}


def aggregate(args, inventory: dict, src_hashes: dict) -> dict:
    rows = []
    blocks = []
    completed = []
    for row in inventory["checkpoints"]:
        dest = job_dir(args.project, args.dataset, row)
        record = dest / "metrics_int8.json"
        status = {"source": row["path"], "job_id": row["job_id"],
                  "seed": row["checkpoint_seed"], "checkpoint_kind": row["kind"],
                  "seed_resolution": row["seed_resolution"]}
        if record.exists():
            try:
                saved = json.loads(record.read_text(encoding="utf-8"))
                need(saved.get("status") == "COMPLETE_INT8"
                     and saved.get("contract") == CONTRACT
                     and saved.get("dataset") == args.dataset
                     and saved.get("source") == row["path"]
                     and saved.get("hf_revision") == inventory["hf_revision"]
                     and saved.get("source_files_sha256") == src_hashes,
                     "Stored provenance diverged")
                ex = json.loads((dest / "engine_int8_export.json").read_text())
                need(ex["engine_sha256"] == saved["engine_sha256"],
                     "Engine changed since metric evaluation")
                weights = dest / "source" / row["kind"]
                engine = weights.with_suffix(".engine")
                need(sha256(engine) == ex["engine_sha256"]
                     and sha256(weights) == saved["checkpoint_sha256"],
                     "Source or engine artifact bytes changed")
                read_engine_metadata(engine)
                validate_evaluation_evidence(dest, args.dataset, saved)
                vals = [saved["metrics"][key] for key in METRICS[args.dataset]]
                need(all(type(v) in (int,float) and math.isfinite(v)
                         and 0 <= v <= 1 for v in vals),
                     "Invalid measured metric")
                status.update(status="COMPLETE_INT8", checkpoint_sha256=saved["checkpoint_sha256"],
                              engine_sha256=saved["engine_sha256"])
                status.update({key: 100*value for key,value in zip(SCORE_NAMES, vals)})
                completed.append(status)
            except Exception as exc:
                status.update(status="BLOCKED_INVALID", reason=str(exc))
                blocks.append(status)
        else:
            failure = dest / "failure.json"
            if failure.is_file():
                try:
                    fail = json.loads(failure.read_text(encoding="utf-8"))
                    status.update(status="BLOCKED", reason=fail.get("error","unreadable failure"))
                except Exception:
                    status.update(status="BLOCKED",reason="unreadable failure.json")
            else:
                status.update(status="PENDING", reason="not evaluated")
            blocks.append(status)
        rows.append(status)
    cols = ["source", "job_id", "seed", "checkpoint_kind", "seed_resolution",
            "status", "reason", "checkpoint_sha256", "engine_sha256", *SCORE_NAMES]
    csv_report(args.project / args.dataset / "all_checkpoint_status.csv", rows, cols)
    csv_report(args.project / args.dataset / "measured_int8_metrics.csv", completed, cols)
    # Group only genuine best.pt / 3 distinct seeds under an identical HF path lineage;
    # do not manufacture missing seed results or merge incompatible run families.
    groups: dict[str, dict[int, dict]] = {}
    for row in completed:
        if row["checkpoint_kind"] != "best.pt" or row["seed_resolution"] != "checkpoint_path":
            continue
        key = SEED_RE.sub("/seed_{seed}/", row["source"])
        slot = groups.setdefault(key, {})
        need(row["seed"] not in slot, f"Ambiguous duplicate seed for checkpoint family: {key}")
        slot[row["seed"]] = row
    agg = []
    for key, seeds in groups.items():
        if set(seeds) != {42,43,44}:
            continue
        out = {"checkpoint_family": key, "n": 3, "seeds": "42,43,44"}
        for name in SCORE_NAMES:
            values = [seeds[s][name] for s in (42,43,44)]
            out[name+"_mean"] = statistics.mean(values)
            out[name+"_sample_std"] = statistics.stdev(values)
        agg.append(out)
    csv_report(args.project / args.dataset / "complete_three_seed_best_aggregate.csv", agg,
               ["checkpoint_family", "n", "seeds",
                *[k+s for k in SCORE_NAMES for s in ("_mean","_sample_std")]])
    report = {"contract": CONTRACT, "dataset": args.dataset,
              "status": "COMPLETE" if not blocks else "INCOMPLETE",
              "total_HF_checkpoints": len(inventory["checkpoints"]),
              "measured_int8": len(completed), "blocked_or_pending": len(blocks),
              "three_seed_families": len(agg),
              "hf_revision": inventory["hf_revision"],
              "evaluator": list(METRICS[args.dataset]),
              "failure_examples": blocks[:30],
              "report_root": str(args.project / args.dataset)}
    atomic_json(args.project / args.dataset / "summary.json", report, immutable=False)
    return report


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--phase", choices=("plan","run","aggregate"), default="plan")
    p.add_argument("--repo", default=HF_REPO)
    p.add_argument("--project", type=Path, default=ROOT / "runs/ftsc_hf_int8_all_v1")
    p.add_argument("--data-yaml", type=Path, help="LEVIR canonical seed42 YAML")
    p.add_argument("--data-root", type=Path, help="TinyPerson original archive root")
    p.add_argument("--prepared-root", type=Path, help="TinyPerson cached prepared crop datasets")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-jobs", type=int, default=0,
                   help="For a smoke run only; 0 evaluates all HF checkpoints")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--confirm-int8-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args(argv)
    args.project = args.project.expanduser().resolve()
    need(args.max_jobs >= 0 and args.start_index >= 0 and args.workers >= 0,
         "Invalid INT8 job selection/workers")
    inv = build_inventory(args.repo, args.dataset, args.project)
    validate_inventory(inv, args.repo, args.dataset)
    if args.phase == "plan":
        summary = {"status": "PINNED_HF_PLAN", "dataset": args.dataset,
                   "repo": args.repo, "revision": inv["hf_revision"],
                   "total_checkpoints": len(inv["checkpoints"]),
                   "inventory": str(inventory_path(args.project, args.dataset)),
                   "sample": inv["checkpoints"][:12],
                   "unknown_seed_fallback_count": sum(
                       r["seed_resolution"].startswith("CALIBRATION_FALLBACK")
                       for r in inv["checkpoints"]),
                   "unsupported_seed_count": sum(
                       r["seed_resolution"] == "checkpoint_path_unprepared"
                       for r in inv["checkpoints"])}
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    source_hash = hashes()
    if args.phase == "run":
        need(args.confirm_int8_run, "INT8 conversion + inference requires --confirm-int8-run")
        from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
        ensure_local_ultralytics()
        runtime = runtime_identity(args.device)
        start = args.start_index
        end = len(inv["checkpoints"]) if args.max_jobs == 0 else min(
            start + args.max_jobs, len(inv["checkpoints"]))
        need(start < end, "No jobs in selected HF range")
        errors = []
        data_cache = {}
        for index in range(start, end):
            row = inv["checkpoints"][index]
            print(f"[{args.dataset} INT8 {index+1}/{len(inv['checkpoints'])}] {row['path']}",
                  flush=True)
            try:
                record = evaluate_job(args, inv, row, source_hash, runtime, data_cache)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            except Exception as exc:
                failure = {"source": row["path"], "job_id": row["job_id"],
                           "error_type": type(exc).__name__, "error": str(exc),
                           "hf_revision": inv["hf_revision"], "contract": CONTRACT}
                path = job_dir(args.project, args.dataset, row) / "failure.json"
                atomic_json(path, failure, immutable=False)
                errors.append(failure)
                print(json.dumps({"status": "BLOCKED", **failure}, ensure_ascii=False), flush=True)
                traceback.print_exc()
                if args.stop_on_error:
                    raise
            finally:
                # Serial all-checkpoint export can leave cached CUDA allocations.
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
        report = aggregate(args, inv, source_hash)
        report["current_range_failed"] = len(errors)
    else:
        report = aggregate(args, inv, source_hash)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
