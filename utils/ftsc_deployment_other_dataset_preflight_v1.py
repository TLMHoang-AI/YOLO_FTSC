#!/usr/bin/env python3
"""Pre-deployment preprocessing parity gate for TinyPerson and Varroa (diagnostic only).

This is NOT an AP evaluator and never exports engines, trains, uploads, or changes
models. It computes actual uint8 BCHW preprocessing hashes using the repo's
Ultralytics LetterBox/YOLODataset implementations and stops the notebook from
running V2 static TRT if native PT and TRT would see different image tensors.

TinyPerson: official sw640/sh512 corner windows -> merged original TinyBenchmark.
Varroa: native test YAML per seed, rect=True with reference batch4 vs static
TRT V2 batch1/rectFalse. A different Varroa paper batch requires an explicit
--varroa-reference-batch, never an implicit convention change.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

VALID_DATASETS = ("TinyPerson", "Varroa")
MODEL_SIZE = 640
STRIDE = 32


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def _shape_dict(c):
    return {f"{k[0]}x{k[1]}": int(n) for k, n in sorted(c.items())}


def _pair_summary(inputs, examples=12):
    """inputs: path/name -> pair of CHW uint8 arrays built by fork data code."""
    import numpy as np
    counts_left, counts_right = collections.Counter(), collections.Counter()
    different = []
    equal = 0
    digest_left, digest_right = hashlib.sha256(), hashlib.sha256()
    for key, left, right in inputs:
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        need(left.dtype == np.uint8 and right.dtype == np.uint8,
             f"Unexpected preprocessing dtype: {key}")
        need(left.ndim == right.ndim == 3 and left.shape[0] == right.shape[0] == 3,
             f"Unexpected preprocessing channel shape: {key}")
        shape_l, shape_r = tuple(left.shape[1:]), tuple(right.shape[1:])
        counts_left[shape_l] += 1
        counts_right[shape_r] += 1
        ha, hb = hashlib.sha256(left.tobytes()).hexdigest(), hashlib.sha256(right.tobytes()).hexdigest()
        digest_left.update(f"{key}\\t{shape_l}\\t{ha}\\n".encode())
        digest_right.update(f"{key}\\t{shape_r}\\t{hb}\\n".encode())
        if shape_l == shape_r and ha == hb:
            equal += 1
        elif len(different) < examples:
            different.append({"source": str(key), "pt_shape": list(shape_l),
                              "trt_static_shape": list(shape_r),
                              "pt_pixel_sha256": ha, "trt_pixel_sha256": hb})
    total = sum(counts_left.values())
    need(total > 0 and total == sum(counts_right.values()), "Empty or mismatched input set")
    return {
        "total": total, "identical_input_count": equal,
        "all_identical": equal == total,
        "pt_canvas_shapes": _shape_dict(counts_left),
        "trt_static_canvas_shapes": _shape_dict(counts_right),
        "pt_manifest_digest": digest_left.hexdigest(),
        "trt_static_manifest_digest": digest_right.hexdigest(),
        "mismatch_examples": different,
    }


def check_tiny(data_root: Path, test_out: Path):
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox
    from train_tinyperson_scripts.train_all_tinyperson import (
        TEST_MERGED_JSON, TEST_CORNER_JSON,
    )

    merged = data_root / TEST_MERGED_JSON
    corners = data_root / TEST_CORNER_JSON
    manifest = test_out / "corner_manifest.json"
    for p in (merged, corners, manifest):
        need(p.is_file(), f"Missing official TinyPerson source: {p}")
    rows = json.loads(manifest.read_text(encoding="utf-8")).get("images", [])
    need(isinstance(rows, list) and rows, "Missing prepared corner mapping")
    ids = [str(r["image_id"]) for r in rows]
    need(len(set(ids)) == len(ids), "Duplicate corner image identifiers")
    left_lb = LetterBox(new_shape=(MODEL_SIZE, MODEL_SIZE),
                        auto=True, stride=STRIDE, scaleup=True)
    right_lb = LetterBox(new_shape=(MODEL_SIZE, MODEL_SIZE),
                         auto=False, stride=STRIDE, scaleup=True)

    def samples():
        for record in rows:
            path = test_out / "images" / f"test_{record['image_id']}.jpg"
            need(path.is_file(), f"Missing prepared corner image {path}")
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            need(img is not None, f"Cannot decode TinyPerson crop {path}")
            # Predictor pre_transform returns BGR HWC; preprocess flips BGR->RGB CHW.
            def chw(box):
                x = box(image=img)
                return np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))
            yield str(path.resolve()), chw(left_lb), chw(right_lb)

    evidence = _pair_summary(samples())
    evidence.update({
        "dataset": "TinyPerson",
        "paper_protocol": "predict_merged_test test corners sw640/sh512, batch1, rect=True, conf=.001, iou=.5, max_det=300, merge NMS=.5",
        "source_original_gt_sha256": sha(merged),
        "source_corner_annotations_sha256": sha(corners),
        "corner_manifest_sha256": sha(manifest),
        "source_corner_count": len(rows),
        "model_imgsz": MODEL_SIZE,
        "source": str(test_out.resolve()),
        "pt_preprocess": "predictor LetterBox auto=True, stride=32, scaleup=True",
        "trt_v2_preprocess": "static predictor LetterBox auto=False, stride=32, scaleup=True",
        "conclusion": "This tests image tensor parity BEFORE corner-to-original merge and evaluation, not correctness of postprocessing or AP.",
    })
    return evidence


def check_varroa(yaml_pattern: str, seeds: list[int], reference_batch: int):
    import numpy as np
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    need(reference_batch > 0, "Reference PT batch must be positive")
    by_seed = []
    for seed in seeds:
        yaml_path = Path(yaml_pattern.format(seed=seed)).expanduser().resolve()
        need(yaml_path.is_file(), f"Missing frozen Varroa seed YAML: {yaml_path}")
        data = check_det_dataset(str(yaml_path), autodownload=False)
        test = data.get("test")
        need(isinstance(test, (str, list)) and test,
             f"Missing frozen Varroa test split in {yaml_path}")
        def mk(batch, rect):
            cfg = get_cfg(overrides={
                "task": "detect", "mode": "val", "imgsz": MODEL_SIZE,
                "rect": rect, "cache": False, "single_cls": False,
            })
            return build_yolo_dataset(
                cfg, img_path=test, batch=batch, data=data,
                mode="val", rect=rect, stride=STRIDE,
            )
        pt_ds = mk(reference_batch, True)
        trt_ds = mk(1, False)
        pt_files = {str(Path(p).resolve()): i for i, p in enumerate(pt_ds.im_files)}
        trt_files = {str(Path(p).resolve()): i for i, p in enumerate(trt_ds.im_files)}
        need(pt_files and set(pt_files) == set(trt_files),
             "PT and TRT Varroa test image sets differ")
        def samples():
            for key in sorted(pt_files):
                p = pt_ds[pt_files[key]]["img"]
                t = trt_ds[trt_files[key]]["img"]
                if hasattr(p, "detach"):
                    p = p.detach().cpu().numpy()
                if hasattr(t, "detach"):
                    t = t.detach().cpu().numpy()
                yield key, np.asarray(p), np.asarray(t)
        evidence = _pair_summary(samples())
        evidence.update({
            "seed": seed, "dataset": "Varroa",
            "yaml": str(yaml_path), "yaml_sha256": sha(yaml_path),
            "paper_imgsz": MODEL_SIZE, "reference_pt_batch": reference_batch,
            "pt_protocol": "native YOLO.val(split=test, rect=True, stride32, pad=.5)",
            "trt_v2_protocol": "static YOLO.val(split=test, rect=False, batch1)",
            "pt_batch_shapes": (np.asarray(pt_ds.batch_shapes).tolist()
                                if pt_ds.batch_shapes is not None else None),
            "source_test_image_set_sha256": hashlib.sha256(
                "\\n".join(sorted(pt_files)).encode()).hexdigest(),
        })
        by_seed.append(evidence)
    return {
        "dataset": "Varroa", "seeds": by_seed,
        "all_identical": all(s["all_identical"] for s in by_seed),
        "total_images": sum(s["total"] for s in by_seed),
        "model_imgsz": MODEL_SIZE,
        "conclusion": "A different 640/672/variable rect batch shape requires its own TensorRT adapter; do not substitute LEVIR 544.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=VALID_DATASETS)
    p.add_argument("--tiny-data-root", type=Path)
    p.add_argument("--tiny-test-out", type=Path)
    p.add_argument("--varroa-yaml-template", type=str)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--varroa-reference-batch", type=int, default=4)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)

    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    if args.dataset == "TinyPerson":
        need(args.tiny_data_root is not None and args.tiny_test_out is not None,
             "TinyPerson requires original data root and prepared test windows")
        report = check_tiny(args.tiny_data_root.resolve(), args.tiny_test_out.resolve())
        parity = report["all_identical"]
    else:
        need(args.varroa_yaml_template, "Varroa needs per-seed frozen YAML template")
        report = check_varroa(args.varroa_yaml_template, args.seeds,
                              args.varroa_reference_batch)
        parity = report["all_identical"]
    root = Path(__file__).resolve().parents[1]
    relevant = [
        "utils/ftsc_deployment_other_dataset_preflight_v1.py",
        "models_related/ultralytics/ultralytics/data/augment.py",
        "models_related/ultralytics/ultralytics/data/base.py",
        "models_related/ultralytics/ultralytics/data/build.py",
        "models_related/ultralytics/ultralytics/data/dataset.py",
        "models_related/ultralytics/ultralytics/engine/predictor.py",
        "train_tinyperson_scripts/train_all_tinyperson.py",
    ]
    report.update({
        "source_code_sha256": {name: sha(root/name) for name in relevant},
        "schema": "ftsc_other_dataset_deployment_preprocess_audit_v1",
        "status": "PASS_EXACT_INPUT_PARITY" if parity else "BLOCKED_PREPROCESSING_MISMATCH",
        "evaluated_ap": False,
        "static_engine_exported": False,
        "note": "Identical preprocessing is necessary, not sufficient, for accuracy parity. No AP was computed here.",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    need(not args.output.exists(), f"Do not overwrite prior diagnostic: {args.output}")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    need(not temporary.exists(), f"Stale interrupted diagnostic: {temporary}")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "dataset": args.dataset,
                      "output": str(args.output),
                      "matched": report.get("identical_input_count"),
                      "total": report.get("total", report.get("total_images")),
                      "per_seed": [{
                          "seed": item["seed"], "matched": item["identical_input_count"],
                          "total": item["total"],
                          "pt_shapes": item["pt_canvas_shapes"],
                          "trt_shapes": item["trt_static_canvas_shapes"],
                      } for item in report.get("seeds", [])],
                      "pt_shapes": report.get("pt_canvas_shapes"),
                      "trt_shapes": report.get("trt_static_canvas_shapes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
