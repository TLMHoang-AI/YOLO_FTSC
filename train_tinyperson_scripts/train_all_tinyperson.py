#!/usr/bin/env python3
"""Train/evaluate/upload TinyPerson YOLOv8n base and P2/P3 plain neck with GAP attention and FTAL."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
ULTRALYTICS = ROOT / "models_related/ultralytics"
CONFIGS = {
    "yolov8n_base": ROOT / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_base.yaml",
    "yolov8n_p2p3_plain_gap": ROOT / "models_related/models_config/yolov8/tinyperson/yolov8n_tinyperson_p2p3_plain_gap.yaml",
}

VARIANTS = {
    "yolov8n_base": {
        "factorized_tal_target": False,
    },
    "yolov8n_p2p3_plain_gap": {
        "factorized_tal_target": True,
        "factorized_tal_mode": "legacy",
        "factorized_tal_tau": 0.75,
        "factorized_tal_kappa": 1.5,
        "factorized_tal_lambda": 0.5,
        "factorized_tal_s_max": 32.0,
        "factorized_tal_warmup_start": 5,
        "factorized_tal_warmup_end": 15,
        "factorized_tal_p2_only": True,
    },
}

REQUIRED = (
    "weights/best.pt",
    "weights/last.pt",
    "results.csv",
    "args.yaml",
    "evaluation_metrics.json",
    "config.yaml",
    "experiment_manifest.json",
)

TRAIN_CORNER_JSON = Path("erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json")
TEST_CORNER_JSON = Path("annotations/corner/task/tiny_set_test_sw640_sh512_all.json")
TEST_MERGED_JSON = Path("annotations/task/tiny_set_test_all.json")


def local_ultralytics() -> None:
    if str(ULTRALYTICS) not in sys.path:
        sys.path.insert(0, str(ULTRALYTICS))


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_image(data_root: Path, image_root: Path, file_name: str) -> Path:
    """Resolve a COCO relative file name, including the official archive layout."""
    candidates = [image_root / file_name, image_root / "labeled_images" / Path(file_name).name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Image {file_name!r} was not found below {image_root}. "
        "Extract the TinyPerson train/test archive or fix --data-root."
    )


def train_image_root(data_root: Path) -> Path:
    """Prefer the official erased images, but support the original-image benchmark variant."""
    erased = data_root / "erase_with_uncertain_dataset/train"
    if erased.is_dir():
        return erased
    original = data_root / "train"
    if original.is_dir():
        print(f"Erased TinyPerson train images not found; using original images at {original}.", flush=True)
        return original
    raise FileNotFoundError(
        f"Neither {erased} nor {original} exists. Extract TinyPerson train.tar.gz or fix --data-root."
    )


def yolo_label_lines(annotations: list[dict], width: int, height: int) -> list[str]:
    """Convert crop-relative COCO xywh boxes to YOLO normalized labels."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    lines = []
    for ann in annotations:
        bx, by, bw, bh = ann["bbox"]
        cx = (bx + bw / 2.0) / width
        cy = (by + bh / 2.0) / height
        nw = bw / width
        nh = bh / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    return lines


def load_corner_annotations(path: Path) -> tuple[dict, dict[int, list[dict]]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    ann_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("ignore", False) or ann.get("uncertain", False):
            continue
        ann_by_image[ann["image_id"]].append(ann)
    return data, ann_by_image


def write_corner_crop(
    src_img_path: Path,
    img_info: dict,
    annotations: list[dict],
    dest_img_path: Path,
    dest_lbl_path: Path,
) -> None:
    """Materialize one official corner entry as a YOLO image/label pair."""
    x1, y1, x2, y2 = img_info["corner"]
    width, height = int(img_info["width"]), int(img_info["height"])
    if x2 - x1 != width or y2 - y1 != height:
        raise ValueError(f"Corner and declared size disagree for image {img_info['id']}: {img_info}")
    with Image.open(src_img_path) as image:
        cropped = image.crop((x1, y1, x2, y2)).convert("RGB")
        if cropped.size != (width, height):
            raise ValueError(f"Crop size mismatch for {src_img_path}: {cropped.size} != {(width, height)}")
        cropped.save(dest_img_path, "JPEG", quality=95)
    dest_lbl_path.write_text("".join(yolo_label_lines(annotations, width, height)), encoding="utf-8")


class Uploader:
    def __init__(self, repo_id: str) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required before this upload-required workflow starts")
        if not repo_id.strip():
            raise ValueError("--hf-repo-id is required before this upload-required workflow starts")
        from huggingface_hub import HfApi

        self.repo_id, self.api = repo_id, HfApi(token=token)
        self.api.whoami()
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    @staticmethod
    def retry(operation):
        for attempt in range(3):
            try:
                return operation()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    def upload_run(self, variant: str, seed: int, run_dir: Path) -> None:
        missing = [path for path in REQUIRED if not (run_dir / path).is_file()]
        if missing:
            raise RuntimeError(f"{variant}: refusing incomplete upload: {missing}")
        remote = f"runs/{variant}/seed_{seed}"
        self.retry(lambda: self.api.upload_folder(folder_path=str(run_dir), path_in_repo=remote, repo_id=self.repo_id, repo_type="dataset"))
        expected = {f"{remote}/{path}" for path in REQUIRED}
        uploaded = set(self.retry(lambda: self.api.list_repo_files(self.repo_id, repo_type="dataset")))
        missing = sorted(expected - uploaded)
        if missing:
            raise RuntimeError(f"{variant}: Hugging Face verification failed: {missing}")
        marker = run_dir / "upload_complete.json"
        marker.write_text(json.dumps({"repo_id": self.repo_id, "variant": variant, "seed": seed, "verified": sorted(expected)}, indent=2) + "\n", encoding="utf-8")
        self.retry(lambda: self.api.upload_file(path_or_fileobj=str(marker), path_in_repo=f"{remote}/{marker.name}", repo_id=self.repo_id, repo_type="dataset"))


def prepare_test_set(data_root: Path, output_dir: Path) -> Path:
    """Materialize the official sw640/sh512 test windows for YOLO."""
    test_out_dir = output_dir / "tinyperson_test_corner_sw640_sh512"
    test_images_dir = test_out_dir / "images"
    test_labels_dir = test_out_dir / "labels"

    # If already prepared, skip
    if test_images_dir.exists() and test_labels_dir.exists() and list(test_labels_dir.glob("*.txt")):
        print("Test set already prepared.", flush=True)
        return test_out_dir

    print("Preparing test set...", flush=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)

    test_json_path = data_root / TEST_CORNER_JSON
    if not test_json_path.is_file():
        raise FileNotFoundError(f"Test annotation file not found: {test_json_path}")

    data, ann_by_image = load_corner_annotations(test_json_path)
    records = []
    for img_info in data["images"]:
        img_id = int(img_info["id"])
        src_path = resolve_dataset_image(data_root, data_root / "test", img_info["file_name"])
        dest_name = f"test_{img_id}.jpg"
        dest_img_path = test_images_dir / dest_name
        dest_lbl_path = test_labels_dir / f"test_{img_id}.txt"
        write_corner_crop(src_path, img_info, ann_by_image.get(img_id, []), dest_img_path, dest_lbl_path)
        records.append({
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "corner": img_info["corner"],
            "width": img_info["width"],
            "height": img_info["height"],
        })

    (test_out_dir / "corner_manifest.json").write_text(
        json.dumps({"corner_annotations": str(test_json_path), "images": records}, indent=2) + "\n",
        encoding="utf-8",
    )

    return test_out_dir


def prepare_seed_dataset(data_root: Path, output_dir: Path, test_out_dir: Path, seed: int) -> Path:
    """Prepare official corner crops, splitting by original image rather than crop."""
    seed_dir = output_dir / f"tinyperson_seed_{seed}_corner_sw640_sh512"
    if (seed_dir / "images/train").exists() and (seed_dir / "labels/train").exists():
        print(f"Dataset for seed {seed} already prepared.", flush=True)
        return seed_dir

    print(f"Preparing dataset split for seed {seed}...", flush=True)
    for split in ("train", "val"):
        (seed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (seed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Link test set
    for source, target in ((test_out_dir / "images", seed_dir / "images/test"), (test_out_dir / "labels", seed_dir / "labels/test")):
        if not target.exists():
            os.symlink(source, target)

    train_json_path = data_root / TRAIN_CORNER_JSON
    if not train_json_path.is_file():
        raise FileNotFoundError(f"Train annotation file not found: {train_json_path}")

    data, ann_by_image = load_corner_annotations(train_json_path)

    # Keep every window from an original image in the same split.
    by_file_name: dict[str, list[dict]] = defaultdict(list)
    for img_info in data["images"]:
        by_file_name[img_info["file_name"]].append(img_info)
    source_names = sorted(by_file_name)
    random.Random(seed).shuffle(source_names)
    val_count = max(1, int(len(source_names) * 0.1))
    splits = {"val": source_names[:val_count], "train": source_names[val_count:]}

    split_manifest = {}
    source_root = train_image_root(data_root)
    for split_name, split_names in splits.items():
        split_manifest[split_name] = []
        print(f"  Cropping {sum(len(by_file_name[name]) for name in split_names)} windows from {len(split_names)} original images for {split_name}...", flush=True)
        for file_name in split_names:
            src_img_path = resolve_dataset_image(data_root, source_root, file_name)
            for img_info in by_file_name[file_name]:
                img_id = img_info["id"]
                dest_img_path = seed_dir / "images" / split_name / f"img_{img_id}.jpg"
                dest_lbl_path = seed_dir / "labels" / split_name / f"img_{img_id}.txt"
                write_corner_crop(src_img_path, img_info, ann_by_image.get(img_id, []), dest_img_path, dest_lbl_path)
                split_manifest[split_name].append({"image_id": img_id, "file_name": file_name, "corner": img_info["corner"]})

    (seed_dir / "corner_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    # Create dataset yaml file
    yaml_path = seed_dir / "tinyperson.yaml"
    yaml_content = f"""path: {seed_dir}
train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Dataset YAML generated for seed {seed} at: {yaml_path}", flush=True)
    return seed_dir


def training_complete(run_dir: Path, epochs: int) -> bool:
    results = run_dir / "results.csv"
    return (
        (run_dir / "weights/best.pt").is_file()
        and (run_dir / "weights/last.pt").is_file()
        and results.is_file()
        and sum(1 for _ in results.open(encoding="utf-8")) > 1
    )


def box_iou_xyxy(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-12)


def nms_detections(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Class-agnostic NMS matching TinyBenchmark's one-class merge."""
    remaining = sorted(detections, key=lambda item: item["score"], reverse=True)
    kept = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            item for item in remaining
            if box_iou_xyxy(current["xyxy"], item["xyxy"]) <= iou_threshold
        ]
    return kept


def predict_merged_test(run_dir: Path, test_out_dir: Path, data_root: Path, args: argparse.Namespace) -> Path:
    """Run on corner windows, translate boxes, NMS, and emit original-image COCO detections."""
    local_ultralytics()
    from ultralytics import YOLO

    manifest = json.loads((test_out_dir / "corner_manifest.json").read_text(encoding="utf-8"))
    corner_data, _ = load_corner_annotations(data_root / TEST_CORNER_JSON)
    original_id_by_name = {image["file_name"]: image["id"] for image in corner_data.get("old_images", [])}
    if not original_id_by_name:
        merged_data = json.loads((data_root / TEST_MERGED_JSON).read_text(encoding="utf-8"))
        original_id_by_name = {image["file_name"]: image["id"] for image in merged_data["images"]}

    records = manifest["images"]
    image_paths = [str(test_out_dir / "images" / f"test_{record['image_id']}.jpg") for record in records]
    by_original: dict[int, list[dict]] = defaultdict(list)
    model = YOLO(run_dir / "weights/best.pt")
    chunk_size = max(args.batch_size * 8, 32)
    for start in range(0, len(records), chunk_size):
        chunk_records = records[start : start + chunk_size]
        predictions = model.predict(
            source=image_paths[start : start + chunk_size],
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            conf=0.001,
            iou=0.5,
            max_det=300,
            stream=True,
            verbose=False,
        )
        for record, result in zip(chunk_records, predictions):
            original_id = original_id_by_name[record["file_name"]]
            corner = record["corner"]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            scores = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            for box, score, cls in zip(xyxy, scores, classes):
                if int(cls) != 0:
                    continue
                translated = [box[0] + corner[0], box[1] + corner[1], box[2] + corner[0], box[3] + corner[1]]
                by_original[original_id].append({"xyxy": translated, "score": float(score)})

    detections = []
    for original_id, items in by_original.items():
        for item in nms_detections(items, iou_threshold=0.5):
            x1, y1, x2, y2 = item["xyxy"]
            detections.append({
                "image_id": original_id,
                "category_id": 1,
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": item["score"],
            })

    output = run_dir / "evaluation" / "test_merged_predictions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(detections, indent=2) + "\n", encoding="utf-8")
    return output


def mean_ap50_75(result) -> float:
    """Mean AP over the six IoU thresholds 0.50, ..., 0.75."""
    all_ap = result.box.all_ap
    if getattr(all_ap, "ndim", 0) != 2 or all_ap.shape[1] < 6:
        raise ValueError(f"expected per-class AP for at least six IoU thresholds, got {getattr(all_ap, 'shape', None)}")
    return float(all_ap[:, :6].mean())


def precision_ap(precision, iou_index: int, area_index: int) -> float:
    """Average a TinyBenchmark precision slice, ignoring its -1 sentinel."""
    import numpy as np

    values = np.asarray(precision[iou_index, :, :, area_index, -1])
    values = values[values > -1]
    return float(values.mean()) if values.size else -1.0


def evaluate_tiny_benchmark(prediction_path: Path, merged_gt_path: Path) -> dict[str, float]:
    """Extract explicit metrics from TinyBenchmark's precision tensor."""
    import importlib.util
    import numpy as np
    from pycocotools.coco import COCO

    evaluator_path = ROOT / "vendor/tinyperson_cocoeval.py"
    spec = importlib.util.spec_from_file_location("tinyperson_cocoeval", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TinyBenchmark evaluator from {evaluator_path}")
    evaluator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator_module)
    COCOeval, Params = evaluator_module.COCOeval, evaluator_module.Params

    Params.EVAL_STRANDARD = "tiny"
    coco_gt = COCO(str(merged_gt_path))
    coco_dt = coco_gt.loadRes(str(prediction_path))
    # TinyBenchmark predates NumPy's integer-only linspace ``num`` argument.
    old_linspace = evaluator_module.np.linspace
    def compatible_linspace(start, stop, num=50, *args, **kwargs):
        if isinstance(num, (float, np.floating)):
            num = int(num)
        return old_linspace(start, stop, num, *args, **kwargs)
    evaluator_module.np.linspace = compatible_linspace
    try:
        evaluator = COCOeval(coco_gt, coco_dt, "bbox", True, True)
    finally:
        evaluator_module.np.linspace = old_linspace
    evaluator.params.iouThrs = np.linspace(0.50, 0.75, 6)
    had_np_float = hasattr(evaluator_module.np, "float")
    if not had_np_float:
        evaluator_module.np.float = float
    try:
        evaluator.evaluate()
        evaluator.accumulate()
    finally:
        if not had_np_float:
            del evaluator_module.np.float

    area_indices = {name: evaluator.params.areaRngLbl.index(name) for name in ("tiny1", "tiny2", "tiny3", "small", "medium")}
    all_ap = [precision_ap(evaluator.eval["precision"], index, evaluator.params.areaRngLbl.index("all")) for index in range(6)]
    valid_all_ap = [value for value in all_ap if value > -1]
    metrics = {
        "test_merged/AP50": all_ap[0],
        "test_merged/AP75": all_ap[-1],
        "test_merged/mAP50-75": float(np.mean(valid_all_ap)) if valid_all_ap else -1.0,
    }
    for name, area_index in area_indices.items():
        metrics[f"test_merged/AP50-{name.title()}"] = precision_ap(evaluator.eval["precision"], 0, area_index)
        area_ap = [precision_ap(evaluator.eval["precision"], index, area_index) for index in range(6)]
        valid_area_ap = [value for value in area_ap if value > -1]
        metrics[f"test_merged/AP-{name.title()}"] = float(np.mean(valid_area_ap)) if valid_area_ap else -1.0
    return metrics


def evaluate_merged_test(run_dir: Path, test_out_dir: Path, data_root: Path, args: argparse.Namespace) -> dict[str, float]:
    """Evaluate merged detections with TinyBenchmark when its optional stack is available."""
    prediction_path = predict_merged_test(run_dir, test_out_dir, data_root, args)
    merged_gt_path = data_root / TEST_MERGED_JSON
    metrics: dict[str, float] = {"test_merged/available": 0.0}
    benchmark_root = ROOT / "vendor"
    if not (benchmark_root / "tinyperson_cocoeval.py").is_file() or not merged_gt_path.is_file():
        print("TinyBenchmark evaluator unavailable; merged detections were still written.", flush=True)
        return metrics
    sys.path.insert(0, str(benchmark_root))
    try:
        metrics.update(evaluate_tiny_benchmark(prediction_path, merged_gt_path))
        metrics["test_merged/available"] = 1.0
    except Exception as exc:
        print(f"TinyBenchmark AP evaluator unavailable: {type(exc).__name__}: {exc}", flush=True)
    return metrics


def train(variant: str, seed: int, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_dir = args.project / variant / f"seed_{seed}_corner_sw640_sh512"
    if training_complete(run_dir, args.epochs):
        print(f"Reusing completed training: {run_dir}", flush=True)
        return run_dir
    seed_everything(seed)
    local_ultralytics()
    from ultralytics import YOLO

    model = YOLO(CONFIGS[variant])
    if args.pretrained and variant == "yolov8n_base":
        model.load(args.pretrained, smart_transfer=True)

    # Train model
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=seed,
        deterministic=True,
        amp=True,
        plots=False,
        project=str(args.project / variant),
        name=f"seed_{seed}_corner_sw640_sh512",
        exist_ok=True,
        **VARIANTS[variant],
    )
    if not training_complete(run_dir, args.epochs):
        raise RuntimeError(f"Incomplete training artifacts: {run_dir}")
    return run_dir


def evaluate(run_dir: Path, data_yaml: Path, test_out_dir: Path, data_root: Path, args: argparse.Namespace) -> dict:
    output = run_dir / "evaluation_metrics.json"
    if output.is_file():
        cached = json.loads(output.read_text(encoding="utf-8"))
        required_metrics = [
            *(f"{split}/metrics/mAP50-75(B)" for split in ("val", "test")),
            "test_merged/mAP50-75",
            "test_merged/AP-Tiny1",
            "test_merged/AP-Tiny2",
            "test_merged/AP-Tiny3",
            "test_merged/AP-Small",
            "test_merged/AP-Medium",
        ]
        if all(key in cached for key in required_metrics):
            return cached
    local_ultralytics()
    from ultralytics import YOLO

    metrics: dict[str, float | str] = {"checkpoint": "best.pt", "nms_iou": 0.5}
    for split in ("val", "test"):
        result = YOLO(run_dir / "weights/best.pt").val(
            data=str(data_yaml),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            workers=args.workers,
            plots=False,
            iou=0.5,
            project=str(run_dir / "evaluation"),
            name=split,
            exist_ok=True,
        )
        metrics.update({f"{split}/{key}": float(value) for key, value in result.results_dict.items()})
        metrics[f"{split}/metrics/mAP75(B)"] = float(result.box.map75)
        metrics[f"{split}/metrics/mAP50-75(B)"] = mean_ap50_75(result)
    metrics.update(evaluate_merged_test(run_dir, test_out_dir, data_root, args))
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_metadata(variant: str, run_dir: Path, seed: int, data_yaml: Path, args: argparse.Namespace) -> None:
    local_ultralytics()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    shutil.copy2(CONFIGS[variant], run_dir / "config.yaml")
    model = YOLO(run_dir / "weights/best.pt")
    head = model.model.model[-1]
    manifest = {
        "variant": variant,
        "seed": seed,
        "config": CONFIGS[variant].name,
        "data_yaml": str(data_yaml),
        "topology": "P2/P3 plain RepC2f -> GAP ChannelAttention -> shared Detect" if "gap" in variant else "Standard YOLOv8n Head",
        "detect_from": head.f,
        "detect_stride": head.stride.tolist(),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "nms_iou": 0.5,
        "factorized_tal": VARIANTS[variant],
        "params": sum(parameter.numel() for parameter in model.model.parameters()),
        "model_gflops_thop": get_flops(model.model, imgsz=args.imgsz),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summaries(args: argparse.Namespace) -> None:
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            path = args.project / variant / f"seed_{seed}_corner_sw640_sh512" / "evaluation_metrics.json"
            if path.is_file():
                rows.append({"variant": variant, "seed": seed, **json.loads(path.read_text(encoding="utf-8"))})
    if not rows:
        return
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"variant", "seed"}, key))
    with (args.project / "summary_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = []
    for variant in args.variants:
        group = [row for row in rows if row["variant"] == variant]
        if not group:
            continue
        record = {"variant": variant, "runs": len(group)}
        for key in sorted(set.intersection(*(set(row) for row in group)) - {"variant", "seed", "checkpoint"}):
            values = [float(row[key]) for row in group]
            record[f"{key}/mean"] = statistics.fmean(values)
            record[f"{key}/std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(record)
    (args.project / "summary_aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def complete(run_dir: Path) -> bool:
    return all((run_dir / path).is_file() for path in REQUIRED)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "TinyPerson" / "tiny_set")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/tinyperson_yolov8n_baselines")
    parser.add_argument("--pretrained", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hf-repo-id", default="duyle2408/tinyperson-yolov8n-baselines")
    parser.add_argument("--skip-upload", action="store_true", help="Do not upload runs to Hugging Face")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and validate corner datasets, then exit")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.data_root, args.dataset_root, args.project = (
        args.data_root.resolve(),
        args.dataset_root.resolve(),
        args.project.resolve(),
    )
    uploader = None if args.skip_upload or args.prepare_only else Uploader(args.hf_repo_id)

    # 1. Prepare test set (shared across seeds)
    test_out_dir = prepare_test_set(args.data_root, args.dataset_root)

    if args.prepare_only:
        for seed in args.seeds:
            prepare_seed_dataset(args.data_root, args.dataset_root, test_out_dir, seed)
        print("TinyPerson dataset preparation complete!", flush=True)
        return

    # 2. Run sequential seed experiments
    for seed in args.seeds:
        # Prepare dataset split for this seed
        seed_dir = prepare_seed_dataset(args.data_root, args.dataset_root, test_out_dir, seed)
        data_yaml = seed_dir / "tinyperson.yaml"

        for variant in args.variants:
            run_dir = train(variant, seed, data_yaml, args)
            evaluate(run_dir, data_yaml, test_out_dir, args.data_root, args)
            write_metadata(variant, run_dir, seed, data_yaml, args)
            write_summaries(args)
            if not complete(run_dir):
                raise RuntimeError(f"Required post-evaluation artifacts are incomplete: {run_dir}")
            if uploader is not None:
                uploader.upload_run(variant, seed, run_dir)

    write_summaries(args)
    print("TinyPerson training matrix complete!", flush=True)


if __name__ == "__main__":
    main()
