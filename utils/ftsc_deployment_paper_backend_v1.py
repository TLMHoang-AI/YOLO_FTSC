#!/usr/bin/env python3
"""Strict historical paper evaluation across PyTorch/TensorRT, without retraining.

Never modifies the historical Ultralytics/TinyBenchmark evaluators or the
GPU-verified LEVIR V3 deployment runner. Every backend's predictions are new.

LEVIR: main AP comes from SHA-verified V3 native val544 results; paper small
        AP is an independent YOLO.predict 512/conf=.25/TinyBenchmark pass.
TinyPerson: historical official corner prediction + original-image merge,
        YOLO.predict(imgsz640, conf=.001, iou=.5, batch8) then TinyBenchmark.
Varroa: historical YOLO.val(imgsz640, rect=True, batch4, iou=.7),
        plus independent YOLO.predict(imgsz640, conf=.25, iou=.5, batch4)
        and the original native size-bucket TinyBenchmark evaluator.
TensorRT: LEVIR static 512 for the predictor; Tiny/Varroa dynamic TensorRT
        to preserve native rect/padding/batch shapes. Input pixels from the
        *actual* predictor/validator must match PT32 before reporting metrics.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

from utils import ftsc_deployment_backend_eval_v1 as deploy

ROOT = deploy.ROOT
CONTRACT = "ftsc_historical_paper_backend_parity_v1"
REVISION = "predict512_native_rect_dynamic_trt_v1"
BACKENDS = ("pt32", "trt32", "pt16", "trt16")
CONFIG = {
    "LEVIR-Ship": {"imgsz": 512, "batch": 8, "main": "v3_rect_544", "small": "predict512_conf025"},
    "TinyPerson": {"imgsz": 640, "batch": 8, "main": "tiny_original_merged", "small": "tiny_original_merged"},
    "Varroa": {"imgsz": 640, "batch": 4, "main": "native_rect_val", "small": "predict640_conf025"},
}
STAGES = {
    "ftsc_seed42_fp32": (("FTSC",), (42,), ("pt32", "trt32")),
    "ftsc_three_seed_fp32": (("FTSC",), (42, 43, 44), ("pt32", "trt32")),
    "ftsc_all_precision": (("FTSC",), (42, 43, 44), BACKENDS),
    "report_fp32": (None, (42, 43, 44), ("pt32", "trt32")),
    "report_all_precision": (None, (42, 43, 44), BACKENDS),
}
# Three-seed anchors from locked capstone results. Rounded paper means are
# diagnostic references, NEVER values to backfill into new backend records.
# A mismatch flags checkpoint/protocol drift and must be inspected.
PAPER_ANCHORS = {
    "LEVIR-Ship": {
        "FTSC": (81.16, 31.30, 82.77),
        "FTSC_YOLOV9T": (78.86, 29.49, 78.26),
        "FTSC_YOLO11N": (81.77, 31.20, 80.21),
        "FTSC_OFF": (81.95, 32.05, 81.41),
        "HEAD_H0": (80.63, 30.99, 79.30),
        "HEAD_H1": (80.91, 31.83, 80.05),
        "HEAD_H3": (80.88, 31.07, 80.50),
        "COLOR_CUE": (81.05, 30.51, 80.30),
        "EDGE_CUE": (82.35, 31.96, 82.86),
        "ALPHA4": (81.02, 30.88, 80.13),
        "ALPHA8": (81.03, 30.68, 82.59),
    },
    "TinyPerson": {"FTSC": (57.68, 21.51, 69.65),
                   "FTSC_OFF": (55.44, 20.52, 68.07)},
    "Varroa": {"FTSC": (91.29, 34.99, 88.83),
               "FTSC_OFF": (90.04, 35.00, 85.20)},
}
SOURCE_FILES = (
    "utils/ftsc_deployment_paper_backend_v1.py",
    "utils/ftsc_deployment_backend_eval_v1.py",
    "evaluate_test/size_bucket_evaluator.py",
    "evaluate_test/standard_detection_metrics.py",
    "train_tinyperson_scripts/train_all_tinyperson.py",
    "train_levir_scripts/train_varroa_dbss_hit.py",
    "utils/ftsc_deployment_other_dataset_preflight_v1.py",
    "utils/ftsc_report_deployment_case_manifest_v1.json",
    "vendor/tinyperson_cocoeval.py",
    "models_related/ultralytics/ultralytics/engine/model.py",
    "models_related/ultralytics/ultralytics/engine/predictor.py",
    "models_related/ultralytics/ultralytics/engine/validator.py",
    "models_related/ultralytics/ultralytics/data/build.py",
    "models_related/ultralytics/ultralytics/data/base.py",
    "models_related/ultralytics/ultralytics/data/dataset.py",
    "models_related/ultralytics/ultralytics/data/augment.py",
    "models_related/ultralytics/ultralytics/models/yolo/detect/val.py",
    "models_related/ultralytics/ultralytics/models/yolo/detect/predict.py",
    "models_related/ultralytics/ultralytics/nn/backends/tensorrt.py",
    "models_related/ultralytics/ultralytics/engine/exporter.py",
    "models_related/ultralytics/ultralytics/utils/export/engine.py",
    "models_related/ultralytics/ultralytics/utils/nms.py",
)


def need(cond, why):
    if not cond:
        raise RuntimeError(why)


def sha(path):
    return deploy.sha256(Path(path))


def source_hashes():
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if path.is_file():
        # A blocked/incomplete job may have generated the predictor artifacts
        # before an input-parity gate stopped it. Resume only when exact bytes
        # reproduce the prior artifact. NEVER silently replace predictions.
        need(path.read_text(encoding="utf-8") == encoded,
             f"Refuse to overwrite changed existing artifact: {path}")
        return sha(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    need(not tmp.exists(), f"Interrupted write already present: {tmp}")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(path)
    return sha(path)


def directory(base, dataset, case, seed, backend):
    return deploy.path_for(base, dataset, case, seed, backend)


def normalize_row(image, pixel):
    import numpy as np
    arr = np.ascontiguousarray(pixel)
    need(arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[0] == 3,
         f"Invalid uint8 RGB CHW tensor for {image}: {arr.shape} {arr.dtype}")
    return {"shape": list(map(int, arr.shape)), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def historical_source_auto_flags(images, chunk_size):
    """Record legacy YOLO.predict LetterBox(auto=...) grouping by source list.

    Fork Predictor treats a list of paths as one batch. TensorRT with a
    smaller engine batch must apply the *same* original 64/100-image
    auto/rect decision, not recompute same_shapes on 4/8-image subchunks.
    """
    import cv2
    need(chunk_size > 0 and images, "Legacy predictor received empty inventory")
    by_image = {}
    for start in range(0, len(images), chunk_size):
        chunk = images[start:start + chunk_size]
        shapes = []
        for path in chunk:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            need(image is not None, f"Cannot decode historical source image: {path}")
            shapes.append(tuple(int(x) for x in image.shape))
        auto = len(set(shapes)) == 1
        for path in chunk:
            key = str(Path(path).resolve())
            need(key not in by_image, f"Duplicate source in legacy predictor: {key}")
            by_image[key] = auto
    need(len(by_image) == len(images), "Historical auto/rect inventory incomplete")
    return by_image


def paper_predictor(recorder, backend, *, dynamic_required=False,
                    historical_auto_by_image=None):
    """Subclass the fork predictor to hash ACTUAL images from pre_transform.

    Unlike a guessed LetterBox simulation, hashes the transformed tensors
    immediately before the fork converts BGR->RGB, and forces TRT metadata
    stride to match the PyTorchBackend effective minimum stride of 32.
    """
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    class PaperInputPredictor(DetectionPredictor):
        def setup_model(self, *args, **kwargs):
            super().setup_model(*args, **kwargs)
            need(self.model.format in ("pt", "engine"),
                 "Only .pt and TensorRT .engine paper evaluation supported")
            if backend.startswith("trt"):
                need(self.model.format == "engine", "TRT job loaded a non-TRT backend")
                if dynamic_required:
                    need(bool(self.model.dynamic), "Variable-shape paper protocol requires dynamic TRT")
                else:
                    need(not bool(self.model.dynamic), "LEVIR static predictor must use static TRT512")
            else:
                need(self.model.format == "pt", "PT job loaded an export unexpectedly")
            # Metadata's stride may be the P2 model's stride 8 rather than
            # Ultralytics PyTorchBackend's max(model.stride,32)=32.
            need(int(self.model.stride) <= 32 or int(self.model.stride) == 32,
                 "Unexpected prediction stride >32; inspect historical fork")
            self.model.stride = 32
        def pre_transform(self, im):
            need(self.batch is not None, "Predictor did not provide source paths")
            names = self.batch[0]
            if backend.startswith("trt") and historical_auto_by_image is not None:
                modes = {historical_auto_by_image.get(str(Path(name).resolve()))
                         for name in names}
                need(None not in modes and len(modes) == 1,
                     "TRT source sub-batch crosses original legacy auto/rect boundary")
                legacy_auto = next(iter(modes))
                old_rect = self.args.rect
                try:
                    # Historical PT had rect=True but auto=False on mixed
                    # source lists. A TRT subchunk could look homogeneous;
                    # forcing rect=False then recreates exactly the PT pixels.
                    self.args.rect = bool(legacy_auto)
                    output = super().pre_transform(im)
                finally:
                    self.args.rect = old_rect
            else:
                output = super().pre_transform(im)
            need(len(names) == len(output), "Predictor preprocessing batch/path mismatch")
            import numpy as np
            for name, raw in zip(names, output, strict=True):
                # Historical BasePredictor.preprocess converts BGR->RGB CHW.
                chw = np.ascontiguousarray(raw[:, :, ::-1].transpose(2, 0, 1))
                key = str(Path(name).resolve())
                need(key not in recorder, f"Duplicate prediction image in paper test: {key}")
                recorder[key] = normalize_row(key, chw)
            return output
    return PaperInputPredictor


def seal_inputs(path, records, expected_paths, protocol):
    expected = {str(Path(x).resolve()) for x in expected_paths}
    need(len(expected) == len(expected_paths) and set(records) == expected,
         f"{protocol}: wrong input images or duplicate/missing file: {len(records)} != {len(expected)}")
    data = {"schema": "paper_backend_actual_uint8_input_v1",
            "protocol": protocol, "count": len(records),
            "images": {name: records[name] for name in sorted(records)}}
    value = write_new_json(path, data)
    return {"input_manifest_sha256": value, "input_count": len(records),
            "input_image_set_sha256": hashlib.sha256(
                "\n".join(sorted(expected)).encode()).hexdigest()}


def verify_pair(project, dataset, case, seed, backend, kind, report):
    """Fail before publishing TRT/PT16 if identical input SHA to paired PT32 is absent."""
    if backend == "pt32":
        return {}
    pt_dir = directory(project, dataset, case, seed, "pt32")
    path = pt_dir / (f"{kind}_inputs.json")
    record = pt_dir / "metrics.json"
    need(path.is_file() and record.is_file(),
         f"{dataset}/{case}/{seed}: execute pt32 before {backend}")
    first = json.loads(record.read_text(encoding="utf-8"))
    need(first.get("source_sha256") == report["source_sha256"],
         "Backend checkpoint differs from PT32 reference")
    need(first.get("source_files_sha256") == source_hashes(),
         "PT32 source code identity changed")
    frozen = ("small_gt_sha256","tiny_corner_sha256","tiny_corner_manifest_sha256") if kind=="small" else (
        "main_data_yaml_sha256",)
    for key_name in frozen:
        if key_name in first or key_name in report:
            need(first.get(key_name) == report.get(key_name),
                 f"{dataset}/{case}/{seed}/{backend}: frozen GT/manifest differs from PT32: {key_name}")
    key = kind + "_input_manifest_sha256"
    need(first.get(key) == sha(path) and report.get(key) == sha(
        directory(project, dataset, case, seed, backend) / f"{kind}_inputs.json"),
         f"Missing/changed {kind} input manifest hashes")
    other = directory(project, dataset, case, seed, backend) / f"{kind}_inputs.json"
    need(path.read_bytes() == other.read_bytes(),
         f"{dataset}/{case}/{seed}/{backend}: {kind} pixels/shape differ from PT32")
    return {kind + "_full_test_input_parity": True,
            kind + "_paired_pt32_sha256": sha(path)}


def export_trt(pt, digest, out, dataset, backend, device, batch):
    """Export a new, independent predictor engine; Varroa/Tiny demand dynamic."""
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    import torch
    import tensorrt as trt
    from ultralytics import YOLO
    need(torch.cuda.is_available() and device != "cpu",
         "TensorRT requires real CUDA device; no CPU fallback")
    dynamic = dataset != "LEVIR-Ship"
    image_size = CONFIG[dataset]["imgsz"]
    exp = Path(out) / "paper_export"
    exp.mkdir(parents=True, exist_ok=True)
    staged = exp / "source.pt"
    if staged.is_file():
        need(sha(staged) == digest, "Staged source checkpoint SHA collision")
    else:
        shutil.copyfile(pt, staged)
        need(sha(staged) == digest, "Staged checkpoint copy changed bytes")
    engine = exp / "source.engine"
    manifest = exp / "export_manifest.json"
    import torch
    gpu_name = torch.cuda.get_device_name(
        int(device.split(":")[-1]) if device.startswith("cuda:") else 0)
    fp = {
        "source_sha256": digest, "backend": backend, "dataset": dataset,
        "imgsz": image_size, "dynamic": dynamic,
        "export_batch": batch if dynamic else 1, "half": backend == "trt16",
        "nms": False, "simplify": False,
        "gpu": gpu_name, "torch": str(torch.__version__),
        "tensorrt": str(trt.__version__),
        "source_files_sha256": source_hashes(),
    }
    if engine.exists() or manifest.exists():
        need(engine.is_file() and manifest.is_file(),
             "Interrupted/incomplete paper TensorRT engine cache")
        old = json.loads(manifest.read_text(encoding="utf-8"))
        need(old.get("fingerprint") == fp and old.get("artifact_sha256") == sha(engine),
             "Old engine has incompatible protocol/toolchain/source; do not reuse")
        return engine, sha(manifest)
    yolo = YOLO(str(staged), task="detect")
    exported = Path(yolo.export(
        format="engine", imgsz=image_size,
        batch=(batch if dynamic else 1),
        half=backend == "trt16",
        device=device, dynamic=dynamic, nms=False, int8=False,
        simplify=False, verbose=False,
    )).resolve()
    need(exported == engine.resolve() and engine.is_file(),
         "Exporter returned unexpected engine path")
    digest_export = sha(engine)
    write_new_json(manifest, {"fingerprint": fp,
                              "artifact_sha256": digest_export})
    return engine, sha(manifest)


def _predict_native_small(model_file, backend, images, args, out):
    """Clone only historical _predictions source-list batching, with input hash."""
    from ultralytics import YOLO
    from evaluate_test.size_bucket_evaluator import MAX_SOURCE_LIST_CHUNK
    from evaluate_test.size_bucket_evaluator import _coco_ground_truth, _evaluate_coco
    recorder = {}
    expected_auto = (historical_source_auto_flags(images, MAX_SOURCE_LIST_CHUNK)
                     if backend.startswith("trt") else None)
    predictor = paper_predictor(
        recorder, backend, dynamic_required=args.dataset == "Varroa",
        historical_auto_by_image=expected_auto,
    )
    model = YOLO(str(model_file), task="detect")
    detections = []
    batch = CONFIG[args.dataset]["batch"]
    # The original PyTorch legacy predictor uses batch8 on LEVIR.
    # The separately exported static TensorRT512 engine supports batch1 only.
    # Because all LEVIR native source images are square512, batching does not
    # change LetterBox pixels. Exact all-image SHA parity is mandatory below.
    if args.dataset=="LEVIR-Ship" and backend.startswith("trt"):
        batch=1
    chunk_size=min(MAX_SOURCE_LIST_CHUNK,batch) if backend.startswith("trt") else MAX_SOURCE_LIST_CHUNK
    for start in range(0, len(images), chunk_size):
        chunk = images[start:start+chunk_size]
        iterator = model.predict(
            source=[str(p) for p in chunk],
            predictor=predictor, imgsz=CONFIG[args.dataset]["imgsz"],
            batch=batch, device=args.device, workers=args.workers,
            iou=0.5, verbose=False, stream=True,
            **({"half": backend=="pt16"} if backend.startswith("pt") else {}),
        )
        seen_chunk=0
        for offset, result in enumerate(iterator):
            seen_chunk+=1
            need(offset < len(chunk) and start + offset < len(images),
                 "Predictor emitted extra native image")
            need(str(Path(result.path).resolve()) == str(chunk[offset].resolve()),
                 "Historical predictor results no longer follow input order")
            boxes = result.boxes
            xyxy = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
            scores = boxes.conf.detach().cpu().tolist() if boxes is not None else []
            for (x1,y1,x2,y2),score in zip(xyxy,scores,strict=True):
                detections.append({"image_id":start+offset+1,"category_id":1,
                    "bbox":[x1,y1,max(0,x2-x1),max(0,y2-y1)],
                    "score":float(score)})
        need(seen_chunk==len(chunk),
             f"Native predictor omitted outputs: {seen_chunk}/{len(chunk)}")
    need(len(recorder) == len(images), "Missing native historical predictor input")
    input_meta = seal_inputs(out/"small_inputs.json", recorder, images,
                             "YOLO.predict/imgsz/conf_default_025/rect_true")
    gt = out/"test_size_ground_truth.json"
    pr = out/"test_size_predictions.json"
    write_new_json(gt, _coco_ground_truth(images))
    write_new_json(pr, detections)
    metrics = _evaluate_coco(gt,pr)
    metrics["test_size/protocol"] = (
        "TinyBenchmark area buckets on native YOLO test images; "
        "class-agnostic; IoU=0.50:0.05:0.75; maxDets=200")
    metrics["test_size/dataset"] = "levir" if args.dataset == "LEVIR-Ship" else "varroa"
    return metrics, {"small_input_manifest_sha256":input_meta["input_manifest_sha256"],
                     "small_input_count":input_meta["input_count"],
                     "small_gt_sha256":sha(gt),
                     "small_predictions_sha256":sha(pr)}


def _predict_tiny(model_file, backend, args, out):
    """Original TinyPerson official corner→merge without changing NMS/GT."""
    from ultralytics import YOLO
    from train_tinyperson_scripts.train_all_tinyperson import (
        load_corner_annotations, TEST_CORNER_JSON, TEST_MERGED_JSON,
        nms_detections, decode_ultralytics_result, evaluate_tiny_benchmark,
    )
    data_root, test_out = args.data_root, args.tiny_test_out
    manifest = test_out/"corner_manifest.json"
    original_gt = data_root/TEST_MERGED_JSON
    corners = data_root/TEST_CORNER_JSON
    need(all(p.is_file() for p in (manifest,original_gt,corners)),
         "TinyPerson source GT/corners/manifest missing")
    mapping = json.loads(manifest.read_text(encoding="utf-8"))
    corner_data,_ = load_corner_annotations(corners)
    original_ids={im["file_name"]:im["id"] for im in corner_data.get("old_images",[])}
    if not original_ids:
        d=json.loads(original_gt.read_text(encoding="utf-8"))
        original_ids={im["file_name"]:im["id"] for im in d["images"]}
    rows=mapping["images"]
    image_paths=[test_out/"images"/f"test_{row['image_id']}.jpg" for row in rows]
    need(len(rows)>0 and all(p.is_file() for p in image_paths),
         "Missing official TinyPerson corner crop")
    recorder={}
    batch=CONFIG["TinyPerson"]["batch"]
    original_chunk_size=max(batch*8,32)
    expected_auto=(historical_source_auto_flags(image_paths,original_chunk_size)
                   if backend.startswith("trt") else None)
    predictor=paper_predictor(
        recorder,backend,dynamic_required=True,
        historical_auto_by_image=expected_auto)
    model=YOLO(str(model_file),task="detect")
    by_original=collections.defaultdict(list)
    # PT historical uses 64-source batches. TRT uses max 8 actual inputs,
    # but retains each original 64-source LetterBox auto decision via the
    # predictor wrapper above. Exact resulting pixels must still match PT.
    chunk_size=batch if backend.startswith("trt") else original_chunk_size
    for start in range(0,len(rows),chunk_size):
        current=rows[start:start+chunk_size]
        generator=model.predict(
            source=[str(p) for p in image_paths[start:start+chunk_size]],
            predictor=predictor, imgsz=640, batch=batch,
            device=args.device, workers=args.workers,
            conf=0.001, iou=0.5, max_det=300, rect=True,
            stream=True, verbose=False,
            **({"half":backend=="pt16"} if backend.startswith("pt") else {}),
        )
        for row,result in zip(current,generator,strict=True):
            original_id=original_ids[row["file_name"]]
            corner=row["corner"]
            xyxy,scores,classes=decode_ultralytics_result(result)
            for box,score,cls in zip(xyxy,scores,classes,strict=True):
                if int(cls)!=0: continue
                translated=[box[0]+corner[0],box[1]+corner[1],
                            box[2]+corner[0],box[3]+corner[1]]
                by_original[original_id].append({"xyxy":translated,"score":float(score)})
    detections=[]
    for original_id,items in by_original.items():
        for item in nms_detections(items,iou_threshold=0.5):
            x1,y1,x2,y2=item["xyxy"]
            detections.append({"image_id":original_id,"category_id":1,
                  "bbox":[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],
                  "score":item["score"]})
    meta=seal_inputs(out/"small_inputs.json",recorder,image_paths,
                     "TinyPerson_640_corner_predict_merge_original")
    prediction=out/"evaluation/test_merged_predictions.json"
    write_new_json(prediction,detections)
    metrics=evaluate_tiny_benchmark(prediction,original_gt)
    need(metrics.get("test_merged/evaluator_contract") == deploy.TINY_MERGED_CONTRACT,
         "Changed TinyPerson benchmark contract")
    return metrics, {
        "small_input_manifest_sha256":meta["input_manifest_sha256"],
        "small_input_count":meta["input_count"],
        "small_gt_sha256":sha(original_gt),
        "tiny_corner_sha256":sha(corners),
        "tiny_corner_manifest_sha256":sha(manifest),
        "small_predictions_sha256":sha(prediction),
    }


def _val_varroa(model_file,backend,args,out,yaml_path,images):
    """Native historical YOLO.val rect=True/batch4; dynamic TRT keeps same pixels."""
    from evaluate_test.standard_detection_metrics import standard_metrics
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics import YOLO
    capture={}
    class PaperVarroaValidator(DetectionValidator):
        def get_dataloader(self, dataset_path, batch_size):
            need(self.args.rect is True or bool(self.args.rect),
                 "Dynamic TensorRT Varroa validator lost rect=True")
            self.stride = 32
            return super().get_dataloader(dataset_path,batch_size)
    model=YOLO(str(model_file),task="detect")
    model.add_callback("on_val_end",lambda validator:capture.__setitem__("validator",validator))
    with tempfile.TemporaryDirectory(prefix="ftsc_paper_varroa_val_") as tmp:
        result=model.val(
            validator=(PaperVarroaValidator if backend.startswith("trt") else None),
            data=str(yaml_path),split="test",imgsz=640,batch=4,
            device=args.device,half=backend=="pt16",
            workers=args.workers,rect=True, iou=0.7,conf=0.001,
            max_det=300,plots=False,save_json=False,
            project=str(tmp),name="test",exist_ok=False,verbose=False,
        )
    validator=capture.get("validator")
    need(validator is not None and int(validator.seen)==len(images),
         "Missing Varroa native validator test image outputs")
    ds=validator.dataloader.dataset
    need(int(ds.imgsz)==640 and bool(ds.rect) and int(ds.stride)==32,
         "Varroa validator diverged from historical size/rect/stride")
    records={}
    import numpy as np
    for i,name in enumerate(ds.im_files):
        key=str(Path(name).resolve())
        value=ds[i]["img"]
        if hasattr(value,"detach"): value=value.detach().cpu().numpy()
        need(key not in records,"Duplicate Varroa validator image")
        records[key]=normalize_row(key,value)
    inp=seal_inputs(out/"main_inputs.json",records,images,
                    "YOLO.val/640/rect/batch4/pad0.5")
    return standard_metrics(result,"test"), {
        "main_input_manifest_sha256":inp["input_manifest_sha256"],
        "main_input_count":inp["input_count"],
        "main_data_yaml_sha256":sha(yaml_path),
    }


def _verify_v3_main(args,case,seed,backend,ckpt_sha):
    """Read existing verified LEVIR V3 metrics; DO NOT rewrite or mix with another protocol."""
    root=args.base_project/deploy.RECT_PARITY_PROTOCOL/deploy.RECT_PARITY_RUNTIME_REVISION
    folder=directory(root,args.dataset,case,seed,backend)
    path=folder/"metrics.json"
    need(path.is_file(),f"Missing LEVIR V3 main result: {path}. Run V3 first.")
    row=json.loads(path.read_text(encoding="utf-8"))
    ident=deploy.expected_identity(
        argparse.Namespace(dataset=args.dataset,platform=args.platform,
                           evaluation_protocol=deploy.RECT_PARITY_PROTOCOL),
        {"case":case,"seed":seed,"backend":backend,
         "source":args.inventory[case][seed]["path"]})
    need(all(row.get(k)==v for k,v in ident.items()),
         "LEVIR V3 main metric source or protocol identity mismatch")
    need(row.get("source_sha256")==ckpt_sha,
         "LEVIR V3 main checkpoint differs from paper small checkpoint")
    deploy.verify_native_val_artifacts(row,folder,args.dataset)
    if backend.startswith("trt"):
        evidence=deploy.verify_rect_trt_pt_input_pair(
            argparse.Namespace(dataset=args.dataset,platform=args.platform,
               evaluation_protocol=deploy.RECT_PARITY_PROTOCOL,project=root),
            {"case":case,"seed":seed,"backend":backend,
             "source":args.inventory[case][seed]["path"]},
            ckpt_sha,row,
        )
        need(evidence.get("full_test_input_parity") is True,
             "LEVIR V3 TRT main validator input parity missing")
    keys=("test/metrics/mAP50(B)","test/metrics/mAP50-95(B)")
    need(all(isinstance(row.get("metrics",{}).get(k),(int,float))
             for k in keys),"Missing verified V3 main AP")
    return {k:row["metrics"][k] for k in keys},{
        "main_v3_metrics_sha256":sha(path),
        "main_v3_input_manifest_sha256":row["preprocessed_input_manifest_sha256"],
        "main_input_count":row["preprocessed_input_count"],
        "main_v3_path":str(path),
    }


def historical_predict_canvas_shapes(images, chunk_size, imgsz):
    """Count real post-LetterBox shapes under original source-list grouping.

    This mirrors the fork predictor grouping (TinyPerson 64, native 100),
    rather than preflighting isolated images whose auto-padding may differ.
    """
    from ultralytics.data.augment import LetterBox
    import cv2
    flags = historical_source_auto_flags(images, chunk_size)
    shapes = collections.Counter()
    for path in images:
        key = str(Path(path).resolve())
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        need(decoded is not None, f"Cannot decode test image: {path}")
        padded = LetterBox(
            new_shape=(imgsz, imgsz), auto=flags[key],
            stride=32, scaleup=True,
        )(image=decoded)
        shapes[f"{padded.shape[0]}x{padded.shape[1]}"] += 1
    need(sum(shapes.values()) == len(images),
         "Historical preprocessing counted fewer test images")
    return dict(sorted(shapes.items()))


def validate_dynamic_profile_shapes(shape_counts, imgsz=640):
    """Reject data shapes the repository's dynamic TensorRT exporter cannot represent.

    The fork exporter uses min=(1,3,32,32), opt=(batch,3,imgsz,imgsz),
    max=(batch,3,2*imgsz,2*imgsz). The backend's batch capacity is set to
    the historical native batch size (Tiny 8 / Varroa 4). This is an
    *export-profile feasibility check*, not runtime pixel parity.
    """
    need(isinstance(shape_counts,dict) and shape_counts,
         "Dynamic TensorRT preflight has no observed PT canvas shapes")
    for dims,count in shape_counts.items():
        parts=str(dims).lower().split("x")
        need(len(parts)==2 and all(p.isdigit() for p in parts),
             f"Unsupported observed PT canvas shape: {dims!r}")
        h,w=map(int,parts)
        need(count>0 and all(32<=d<=imgsz*2 and d%32==0 for d in (h,w)),
             f"Historical input canvas {dims} falls outside TensorRT "
             f"dynamic profile [32,{imgsz*2}] stride32; BLOCKED before GPU")
    return {"dynamic_profile_min_hw":[32,32],
            "dynamic_profile_opt_hw":[imgsz,imgsz],
            "dynamic_profile_max_hw":[imgsz*2,imgsz*2],
            "observed_canvas_shapes":shape_counts}


def _native_test_yaml(args,seed):
    p=Path(args.data_yaml_template.format(seed=seed)).resolve()
    need(p.is_file(),f"Frozen source test YAML missing: {p}")
    from evaluate_test.size_bucket_evaluator import _resolve_test_images
    images=_resolve_test_images(p)
    need(images and all(q.is_file() for q in images),
         "Frozen native test YAML missing images")
    return p,images


def preflight_path(args):
    return (args.paper_root/args.dataset/"_preflights"/
            f"{args.stage}_historical_eval.json")


def image_inventory_digest(paths):
    """Content-address every image in its historical order, not just YAML names."""
    digest=hashlib.sha256()
    need(bool(paths),"Cannot fingerprint empty source image set")
    seen=set()
    for item in paths:
        path=Path(item).resolve()
        need(path.is_file() and str(path) not in seen,
             f"Missing/duplicate source test image: {path}")
        seen.add(str(path))
        digest.update(f"{path}\t{sha(path)}\n".encode())
    return digest.hexdigest()


def native_label_inventory_digest(images):
    """Lock native YOLO test labels, including intentionally absent background files."""
    from evaluate_test.size_bucket_evaluator import _label_path
    digest=hashlib.sha256()
    for image in images:
        label=_label_path(Path(image)).resolve()
        digest.update(f"{label}\t{sha(label) if label.is_file() else 'ABSENT'}\n".encode())
    return digest.hexdigest()


def data_fingerprint(args):
    """Lock split, GT labels and image bytes, plus source code and scoped cases."""
    result={
        "contract":CONTRACT,"revision":REVISION,
        "dataset":args.dataset,"stage":args.stage,
        "cases":list(args.cases),"seeds":list(args.seeds),
        "backends":list(args.backends),
        "report_manifest_sha256":sha(args.report_manifest),
        "source_files_sha256":source_hashes(),
    }
    if args.dataset=="TinyPerson":
        from train_tinyperson_scripts.train_all_tinyperson import (
            TEST_CORNER_JSON,TEST_MERGED_JSON)
        manifest=args.tiny_test_out/"corner_manifest.json"
        rows=json.loads(manifest.read_text(encoding="utf-8"))["images"]
        paths=[args.tiny_test_out/"images"/f"test_{r['image_id']}.jpg"
               for r in rows]
        need(paths and all(x.is_file() for x in paths),"Tiny test corner set incomplete")
        result["dataset_sha256"]={
            "corner_gt":sha(args.data_root/TEST_CORNER_JSON),
            "original_gt":sha(args.data_root/TEST_MERGED_JSON),
            "corner_manifest":sha(manifest),
            "corner_count":len(paths),
            "corner_image_bytes_sha256":image_inventory_digest(paths),
        }
    else:
        result["dataset_sha256"]={}
        for seed in args.seeds:
            yaml_path,paths=_native_test_yaml(args,seed)
            result["dataset_sha256"][str(seed)]={
                "yaml":sha(yaml_path),"images":len(paths),
                "image_names_sha256":hashlib.sha256(
                    "\n".join(str(x.resolve()) for x in paths).encode()).hexdigest(),
                "image_bytes_sha256":image_inventory_digest(paths),
                "label_bytes_sha256":native_label_inventory_digest(paths),
            }
    return result


def verify_preflight(args):
    path=preflight_path(args)
    need(path.is_file(),f"Missing stage preflight. Run --phase preflight first: {path}")
    recorded=json.loads(path.read_text(encoding="utf-8"))
    need(recorded.get("status")=="HISTORICAL_PREPROCESSING_IDENTIFIED" and
         recorded.get("fingerprint")==data_fingerprint(args) and
         bool(recorded.get("checked_shapes")),
         "Dataset/source/shape provenance changed since preflight; BLOCK evaluation")
    return recorded


def evaluate_job(args,archive,case,seed,backend):
    dataset=args.dataset
    paper=directory(args.paper_root,dataset,case,seed,backend)
    paper.mkdir(parents=True,exist_ok=True)
    record_file=paper/"metrics.json"
    expected=args.inventory[case][seed]
    identity={
        "contract":CONTRACT,"revision":REVISION,
        "dataset":dataset,"case":case,"seed":seed,"backend":backend,
        "platform":args.platform,"source":expected["path"],
        "source_sha256":expected["sha256"],
        "device":args.device,"hardware":deploy.hardware_identity(args.device),
        "source_files_sha256":source_hashes(),
        "configuration":CONFIG[dataset],
    }
    if record_file.exists():
        row=json.loads(record_file.read_text(encoding="utf-8"))
        need(all(row.get(k)==v for k,v in identity.items()),
             "Previous paper result has different identity; delete nothing automatically")
        model_path=Path(row["model_path"])
        need(model_path.is_file() and sha(model_path)==row["model_artifact_sha256"],
             "Cached source/backend artifact changed")
        for key,path in (
            ("small_input_manifest_sha256",paper/"small_inputs.json"),
            ("small_predictions_sha256",
             paper/("evaluation/test_merged_predictions.json" if dataset=="TinyPerson"
                    else "test_size_predictions.json")),
        ):
            need(path.is_file() and sha(path)==row[key],"Cached prediction/input changed")
        if dataset=="Varroa":
            path=paper/"main_inputs.json"
            need(path.is_file() and sha(path)==row["main_input_manifest_sha256"],
                 "Cached main test input changed")
        if dataset=="LEVIR-Ship":
            _,metas=_verify_v3_main(args,case,seed,backend,expected["sha256"])
            need(all(row.get(k)==v for k,v in metas.items()),
                 "Existing V3 main checkpoint/input/metrics changed")
        pairs=verify_pair(args.paper_root,dataset,case,seed,backend,"small",row)
        if dataset=="Varroa":
            pairs.update(verify_pair(args.paper_root,dataset,case,seed,backend,"main",row))
        need(all(row.get(k)==v for k,v in pairs.items()),
             "Cached cross-backend input comparison changed")
        return {"status":"REUSED_VERIFIED","path":str(record_file)}
    # Reuse HF-verified checkpoint; copy to isolated paper root, never overwrite.
    args_restore=argparse.Namespace(
        project=args.paper_root,dataset=dataset,
        manifest_cases=args.inventory,manifest_revision=args.manifest_revision,
        cases=args.cases,seeds=args.seeds,
    )
    if dataset=="LEVIR-Ship":
        existing=deploy.source_path(args.base_project/
            deploy.RECT_PARITY_PROTOCOL/deploy.RECT_PARITY_RUNTIME_REVISION,
            dataset,case,seed)
        if existing.is_file():
            need(sha(existing)==expected["sha256"],"V3 cached source SHA collision")
            dst=deploy.source_path(args.paper_root,dataset,case,seed)
            if not dst.is_file():
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(existing,dst)
            need(sha(dst)==expected["sha256"],"Restored V3 checkpoint hash mismatch")
            prov={"checkpoint":str(dst),"sha256":expected["sha256"],
                  "revision":args.manifest_revision,"source":expected["path"]}
        else:
            prov=deploy.source_restore(args_restore,archive,case,seed)
    else:
        prov=deploy.source_restore(args_restore,archive,case,seed)
    ckpt=Path(prov["checkpoint"])
    need(sha(ckpt)==expected["sha256"],"Source checkpoint mismatch after restore")
    # LEVIR V3 main-metric contract is a prerequisite, NOT a fallback.
    # Check before the expensive new static512 TensorRT export.
    verified_v3_main=(_verify_v3_main(args,case,seed,backend,expected["sha256"])
                      if dataset=="LEVIR-Ship" else None)
    engine_meta=None
    if backend.startswith("trt"):
        model_file,engine_meta=export_trt(
            ckpt,expected["sha256"],paper,dataset,backend,args.device,
            CONFIG[dataset]["batch"])
    else:
        model_file=ckpt
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    yaml_path=None
    images=None
    if dataset!="TinyPerson":
        yaml_path,images=_native_test_yaml(args,seed)
    if dataset=="LEVIR-Ship":
        need(verified_v3_main is not None,"Missing frozen V3 main reference")
        main,main_meta=verified_v3_main
        small,small_meta=_predict_native_small(model_file,backend,images,args,paper)
        metrics={**main,**small}
    elif dataset=="Varroa":
        main,main_meta=_val_varroa(model_file,backend,args,paper,yaml_path,images)
        small,small_meta=_predict_native_small(model_file,backend,images,args,paper)
        metrics={**main,**small}
    else:
        metrics,small_meta=_predict_tiny(model_file,backend,args,paper)
        main_meta={}
    for k in deploy.metric_keys(dataset):
        v=metrics.get(k)
        need(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1,
             f"Paper metric invalid {k}={v}")
    entry={
        **identity,
        "model_path":str(model_file),"model_artifact_sha256":sha(model_file),
        "engine_export_manifest_sha256":engine_meta,
        "hf_revision":prov.get("revision"),
        "metrics":metrics,"main":main_meta,"small":small_meta,
        # Denormalize the file SHA keys for simple audit checks.
        **main_meta,**small_meta,
    }
    entry.update(verify_pair(args.paper_root,dataset,case,seed,backend,"small",entry))
    if dataset=="Varroa":
        entry.update(verify_pair(args.paper_root,dataset,case,seed,backend,"main",entry))
    write_new_json(record_file,entry)
    return {"status":"EVALUATED","path":str(record_file)}


def scope(args):
    inventory,sections,blockers=deploy.load_report_scope(args.report_manifest,args.dataset)
    stage=STAGES[args.stage]
    cases=list(inventory) if stage[0] is None else list(stage[0])
    need(all(x in inventory for x in cases),
         "Stage references missing pinned checkpoint")
    return inventory,sections,blockers,cases,list(stage[1]),list(stage[2])


def audit(args):
    inv,sections,blockers,cases,seeds,backends=scope(args)
    problems=[]
    rows={}
    expected_count=len(cases)*len(seeds)*len(backends)
    for case in cases:
        for seed in seeds:
            for backend in backends:
                path=directory(args.paper_root,args.dataset,case,seed,backend)/"metrics.json"
                try:
                    need(path.is_file(),f"Missing metrics: {path}")
                    row=json.loads(path.read_text(encoding="utf-8"))
                    need(row.get("contract")==CONTRACT and row.get("revision")==REVISION
                         and row.get("source_files_sha256")==source_hashes()
                         and row.get("source_sha256")==inv[case][seed]["sha256"]
                         and row.get("source")==inv[case][seed]["path"]
                         and row.get("platform")==args.platform
                         and row.get("device")==args.device
                         and row.get("hardware")==deploy.hardware_identity(args.device),
                         "Stale/wrong source/contract/host")
                    model_path=Path(row["model_path"])
                    need(model_path.is_file() and
                         sha(model_path)==row["model_artifact_sha256"],
                         "Recorded PyTorch/engine model artifact changed")
                    if backend.startswith("trt"):
                        manifest=path.parent/"paper_export/export_manifest.json"
                        need(manifest.is_file() and
                             sha(manifest)==row.get("engine_export_manifest_sha256"),
                             "TensorRT dynamic/static export fingerprint missing or changed")
                        exported=json.loads(manifest.read_text(encoding="utf-8"))
                        need(exported.get("artifact_sha256")==row["model_artifact_sha256"]
                             and exported.get("fingerprint",{}).get("backend")==backend
                             and exported.get("fingerprint",{}).get("source_sha256")==row["source_sha256"]
                             and exported.get("fingerprint",{}).get("source_files_sha256")==source_hashes(),
                             "TRT engine was not built from this audited checkpoint/source")
                    if args.dataset=="LEVIR-Ship":
                        v3=args.base_project/deploy.RECT_PARITY_PROTOCOL/deploy.RECT_PARITY_RUNTIME_REVISION
                        original=directory(v3,args.dataset,case,seed,backend)/"metrics.json"
                        need(original.is_file() and
                             sha(original)==row.get("main_v3_metrics_sha256"),
                             "Historical V3 main validation artifact changed")
                    for kind in ("small","main") if args.dataset=="Varroa" else ("small",):
                        src=path.parent/f"{kind}_inputs.json"
                        need(src.is_file() and sha(src)==row[f"{kind}_input_manifest_sha256"],
                             "Input hash manifest missing/changed")
                        if backend!="pt32":
                            pt=directory(args.paper_root,args.dataset,case,seed,"pt32")/f"{kind}_inputs.json"
                            need(pt.is_file() and pt.read_bytes()==src.read_bytes(),
                                 "PT32 vs backend input pixel mismatch")
                    checkpoint=deploy.source_path(args.paper_root,args.dataset,case,seed)
                    need(checkpoint.is_file() and sha(checkpoint)==row["source_sha256"],
                         "Physical SHA-pinned best.pt missing/changed")
                    prediction=path.parent/("evaluation/test_merged_predictions.json"
                          if args.dataset=="TinyPerson" else "test_size_predictions.json")
                    need(prediction.is_file() and
                         sha(prediction)==row["small_predictions_sha256"],
                         "Paper-compatible backend detections missing/changed")
                    if args.dataset!="TinyPerson":
                        gt=path.parent/"test_size_ground_truth.json"
                        need(gt.is_file() and sha(gt)==row["small_gt_sha256"],
                             "Native TinyBenchmark GT artifact missing/changed")
                    for metric in deploy.metric_keys(args.dataset):
                        value=row["metrics"][metric]
                        need(isinstance(value,(float,int)) and math.isfinite(value)
                             and 0<=value<=1,f"Bad metric {metric}")
                    rows[(case,seed,backend)]=row
                except (ValueError,KeyError,TypeError,RuntimeError,OSError) as e:
                    problems.append({"case":case,"seed":seed,"backend":backend,
                                     "error":f"{type(e).__name__}: {e}"})
    flags=[]
    for case in cases:
        for seed in seeds:
            for backend in backends:
                if backend=="pt32": continue
                r=rows.get((case,seed,backend))
                ref=rows.get((case,seed,"pt16" if backend=="trt16" else "pt32"))
                if not r or not ref: continue
                diffs={
                    k:100*(r["metrics"][k]-ref["metrics"][k])
                    for k in deploy.metric_keys(args.dataset)
                }
                # Audit ALL metrics, not AP50 only. Review is nonfatal to source integrity,
                # but must never be silently labeled successful accuracy parity.
                limit=0.5 if backend=="trt32" else 1.0
                if any(abs(v)>limit for v in diffs.values()):
                    flags.append({"case":case,"seed":seed,"backend":backend,
                                  "vs":"pt16" if backend=="trt16" else "pt32",
                                  "metric_deltas_pp":diffs,"review_limit_pp":limit})
    means=[]
    historical_flags=[]
    if list(seeds)==[42,43,44]:
        for case in cases:
            for backend in backends:
                group=[rows.get((case,seed,backend)) for seed in seeds]
                if any(x is None for x in group): continue
                item={"case":case,"backend":backend,"n":3}
                for key in deploy.metric_keys(args.dataset):
                    values=[100*r["metrics"][key] for r in group]
                    item[key+"_mean_pct"]=statistics.fmean(values)
                    item[key+"_std_pp"]=statistics.stdev(values)
                means.append(item)
                if backend=="pt32" and case in PAPER_ANCHORS[args.dataset]:
                    expected_anchor=PAPER_ANCHORS[args.dataset][case]
                    keys=deploy.metric_keys(args.dataset)
                    errors={
                        key: item[key+"_mean_pct"] - expected_anchor[idx]
                        for idx,key in enumerate(keys)
                    }
                    # Paper reference values were rounded to 0.01%; allow
                    # modest numerical/fork drift but never silently accept
                    # multi-point mismatches or a different checkpoint.
                    if any(abs(delta)>0.12 for delta in errors.values()):
                        historical_flags.append({
                            "case":case,"backend":"pt32","paper_reference_pct":expected_anchor,
                            "observed_pct":[item[k+"_mean_pct"] for k in keys],
                            "metric_delta_pp":errors,
                        })
    status=("INCOMPLETE_OR_INVALID" if len(rows)!=expected_count or problems
            else "COMPLETE_NEEDS_ACCURACY_REVIEW" if flags or historical_flags
            else "PASS_COMPLETE")
    return {"schema":CONTRACT,"status":status,
            "dataset":args.dataset,"stage":args.stage,"valid_jobs":len(rows),
            "expected_jobs":expected_count,"issues":problems,
            "accuracy_review_flags":flags,
            "historical_reference_review_flags":historical_flags,
            "three_seed":means,"unresolved_main":blockers}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=tuple(CONFIG),required=True)
    p.add_argument("--stage",choices=tuple(STAGES),default="ftsc_seed42_fp32")
    p.add_argument("--phase",choices=("plan","verify","preflight","evaluate","audit","aggregate"),
                   default="plan")
    p.add_argument("--allow-evaluate",action="store_true")
    p.add_argument("--run-offline-tests",action="store_true")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--platform",default="marimo_live")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--base-project",type=Path,
                   default=ROOT/"runs/ftsc_deployment_eval_v1")
    p.add_argument("--report-manifest",type=Path,default=deploy.REPORT_MANIFEST)
    p.add_argument("--data-root",type=Path)
    p.add_argument("--prepared-root",type=Path,default=ROOT/"datasets")
    p.add_argument("--data-yaml-template",type=str)
    args=p.parse_args(argv)
    need(args.workers>=0,"Invalid worker count")
    args.base_project=args.base_project.expanduser().resolve()
    args.paper_root=args.base_project/CONTRACT/REVISION
    args.inventory,args.sections,args.unresolved,args.cases,args.seeds,args.backends=scope(args)
    output={"dataset":args.dataset,"stage":args.stage,
            "cases":args.cases,"seeds":args.seeds,"backends":args.backends,
            "jobs":len(args.cases)*len(args.seeds)*len(args.backends),
            "contract":CONTRACT,"revision":REVISION,
            "paper_root":str(args.paper_root),
            "evaluation":CONFIG[args.dataset]}
    if args.phase=="plan":
        print(json.dumps({"status":"PLAN_ONLY",**output},indent=2))
        return 0
    if args.phase in ("audit","aggregate"):
        report=audit(args)
        if args.phase=="audit":
            args.paper_root.joinpath(args.dataset,"_audits").mkdir(parents=True,exist_ok=True)
            path=args.paper_root/args.dataset/"_audits"/f"{args.stage}_audit.json"
            path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            print(json.dumps({**report,"audit_path":str(path)},indent=2))
            return 0 if report["status"]=="PASS_COMPLETE" else 2
        need(args.seeds==[42,43,44] and
             report["status"]=="PASS_COMPLETE" and not report["issues"] and
             not report["accuracy_review_flags"] and
             not report["historical_reference_review_flags"] and
             report["valid_jobs"]==report["expected_jobs"],
             "Paper-compatible aggregate blocked: incomplete/invalid jobs or "
             "unresolved backend/historical AP drift; inspect --phase audit")
        dest=args.paper_root/args.dataset
        dest.mkdir(parents=True,exist_ok=True)
        # Save both reported score and explicit review flags, never hide them.
        import csv
        names=sorted(set().union(*(set(d) for d in report["three_seed"])))
        target=dest/f"deployment_{args.platform}_{args.stage}_paper_parity.csv"
        with target.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=names)
            writer.writeheader()
            writer.writerows(report["three_seed"])
        print(json.dumps({"status":report["status"],"csv":str(target),
                          "rows":len(report["three_seed"]),
                          "accuracy_review_flags":report["accuracy_review_flags"]},indent=2))
        return 0
    if args.phase=="verify":
        from utils.ftsc_hf_canonical_archive import CanonicalFTSCArchive
        archive=CanonicalFTSCArchive(repo_id=deploy.REPO_ID,
            token=os.environ.get("HF_TOKEN") or None,
            cache_dir=args.paper_root/"_hf_cache")
        cfg=argparse.Namespace(dataset=args.dataset,project=args.paper_root,
              cases=args.cases,seeds=args.seeds,manifest_cases=args.inventory)
        verified=deploy.verify_sources(cfg,archive)
        print(json.dumps({"status":"HF_MANIFEST_PATHS_VERIFIED_BYTES_PENDING",
               "note":"Checkpoint bytes SHA-256 verified during restore before inference.",
               "count":len(verified),"rows":verified},indent=2))
        return 0
    need(args.data_root is not None or args.dataset=="LEVIR-Ship",
         "Tiny/Varroa require original data root")
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    if args.dataset=="LEVIR-Ship":
        if not args.data_yaml_template:
            args.data_yaml_template=str(ROOT/"datasets/levir_ship_yolo_seed42/levir_ship.yaml")
        for seed in args.seeds:
            _native_test_yaml(args,seed)
    elif args.dataset=="Varroa":
        from train_levir_scripts.train_varroa_dbss_hit import dataset_for_seed
        from types import SimpleNamespace
        for seed in args.seeds:
            pth=dataset_for_seed(SimpleNamespace(data_root=args.data_root,
                     dataset_root=args.prepared_root),seed)
            need(pth.is_file(),"Varroa canonical preparation missing YAML")
        args.data_yaml_template=str(args.prepared_root/"varroa_yolo_seed{seed}/varroa.yaml")
    else:
        from train_tinyperson_scripts.train_all_tinyperson import prepare_test_set
        args.tiny_test_out=prepare_test_set(args.data_root,args.prepared_root)
    if args.phase=="preflight":
        if args.dataset=="TinyPerson":
            from utils.ftsc_deployment_other_dataset_preflight_v1 import check_tiny
            summary=check_tiny(args.data_root,args.tiny_test_out)
            # Static mismatch is EXPECTED and not a failure when using dynamic TRT.
            manifest=args.tiny_test_out/"corner_manifest.json"
            records=json.loads(manifest.read_text(encoding="utf-8"))["images"]
            paths=[args.tiny_test_out/"images"/f"test_{row['image_id']}.jpg"
                   for row in records]
            count=len(paths)
            need(count==summary["source_corner_count"],
                 "TinyPerson official prepared corner inventory changed")
            shape_hint=historical_predict_canvas_shapes(
                paths,max(CONFIG["TinyPerson"]["batch"]*8,32),640)
        elif args.dataset=="Varroa":
            from utils.ftsc_deployment_other_dataset_preflight_v1 import check_varroa
            summary=check_varroa(args.data_yaml_template,args.seeds,CONFIG["Varroa"]["batch"])
            count=summary["total_images"]
            shape_hint={str(x["seed"]):x["pt_canvas_shapes"] for x in summary["seeds"]}
            small_shape_hint={}
            for seed in args.seeds:
                _yaml, _paths=_native_test_yaml(args,seed)
                small_shape_hint[str(seed)]=historical_predict_canvas_shapes(
                    _paths,100,CONFIG["Varroa"]["imgsz"])
        else:
            yaml,images=_native_test_yaml(args,args.seeds[0])
            count=len(images)
            shape_hint={"512x512":count}
        dynamic_profile=None
        if args.dataset!="LEVIR-Ship":
            if args.dataset=="Varroa":
                dynamic_profile={
                    str(seed):{
                        "main_val":validate_dynamic_profile_shapes(hint),
                        "small_predict":validate_dynamic_profile_shapes(
                            small_shape_hint[str(seed)]),
                    } for seed,hint in shape_hint.items()
                }
            else:
                dynamic_profile=validate_dynamic_profile_shapes(shape_hint)
        if args.dataset=="LEVIR-Ship":
            need(count==788,"LEVIR frozen 788-image test inventory changed")
            true_shape=historical_predict_canvas_shapes(images,100,512)
            need(true_shape=={"512x512":788},
                 f"LEVIR paper prediction must be native512, not {true_shape}")
        result={
            "status":"HISTORICAL_PREPROCESSING_IDENTIFIED",
            "fingerprint":data_fingerprint(args),
            "dataset":args.dataset,"images":count,"checked_shapes":shape_hint,
            "small_predict_canvas_shapes":small_shape_hint if args.dataset=="Varroa" else (
                shape_hint if args.dataset=="TinyPerson" else None),
            "engine_type":"dynamic" if args.dataset!="LEVIR-Ship" else "static512_small_plus_existing_v3_main",
            "trt_dynamic_profile":dynamic_profile,
            "warning":"Source/data/profile preflight only. Runtime backend "
                      "input SHA parity remains mandatory in evaluate phase.",
        }
        location=preflight_path(args)
        write_new_json(location,result)
        print(json.dumps({**result,"preflight_path":str(location)},indent=2))
        return 0
    need(args.phase=="evaluate" and args.allow_evaluate,
         "Evaluation requires explicit --allow-evaluate and cannot train")
    # A plan/verify flag alone is not permission to export or publish AP.
    # Require a recorded matching dataset-source-shape preflight per stage.
    preflight=verify_preflight(args)
    print("PAPER_PRECHECK_VERIFIED",str(preflight_path(args)),flush=True)
    if args.run_offline_tests:
        import subprocess
        subprocess.run([sys.executable,"-m","unittest",
             "utils.tests.test_ftsc_deployment_paper_backend_v1"],
             cwd=str(ROOT),check=True)
    from utils.ftsc_hf_canonical_archive import CanonicalFTSCArchive
    archive=CanonicalFTSCArchive(repo_id=deploy.REPO_ID,
            token=os.environ.get("HF_TOKEN") or None,
            cache_dir=args.paper_root/"_hf_cache")
    cfg=argparse.Namespace(dataset=args.dataset,project=args.paper_root,
          cases=args.cases,seeds=args.seeds,manifest_cases=args.inventory)
    deploy.verify_sources(cfg,archive)
    args.manifest_revision=cfg.manifest_revision
    outcomes=[]
    for case in args.cases:
        for seed in args.seeds:
            for backend in args.backends:
                # pt32 is always first; no TRT can publish without a paired reference.
                try:
                    row=evaluate_job(args,archive,case,seed,backend)
                except Exception as e:
                    out=directory(args.paper_root,args.dataset,case,seed,backend)
                    out.mkdir(parents=True,exist_ok=True)
                    report={"status":"BLOCKED","case":case,"seed":seed,
                            "backend":backend,"error":str(e),
                            "error_type":type(e).__name__}
                    (out/"failure.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
                    row=report
                outcomes.append(row)
                print(json.dumps({"case":case,"seed":seed,"backend":backend,**row}),flush=True)
    print(json.dumps({"status":"EVALUATED_PARTIAL" if any(
        r.get("status")=="BLOCKED" for r in outcomes) else "EVALUATION_COMPLETE",
        "jobs":len(outcomes),"blocked":sum(r.get("status")=="BLOCKED" for r in outcomes)},indent=2))
    return 2 if any(r.get("status")=="BLOCKED" for r in outcomes) else 0


if __name__=="__main__":
    raise SystemExit(main())
