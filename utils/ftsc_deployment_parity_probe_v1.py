#!/usr/bin/env python3
"""Read-only PT32 vs TensorRT engine raw-output parity probe for Marimo Live.

It does not train, export, evaluate aggregate AP, or edit source checkpoints.
Compare exactly the same RGB BCHW float32 tensor before NMS. A saved report
records engine/runtime source identity so stale notebooks cannot hide as parity.
Usage (from yolo_code):
python -m utils.ftsc_deployment_parity_probe_v1 \
  --pt /path/to/best.pt --engine /path/to/source.engine \
  --data-yaml /path/to/levir.yaml --imgsz 512 --max-images 8 \
  --output /path/to/deployment_parity_seed42.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> str:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                           check=True, capture_output=True, text=True, timeout=15)
        return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def first_output(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and value:
        return first_output(value[0])
    raise RuntimeError(f"Unsupported backend prediction structure: {type(value)}")


def normalized_input(image_path: Path, imgsz: int, device, scaleup: bool):
    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read test image: {image_path}")
    # Consistent with Ultralytics predictor: BGR -> RGB, BCHW, [0,1].
    transformed = LetterBox(new_shape=(imgsz, imgsz), auto=False,
                            scaleup=scaleup, stride=32)(image=bgr)
    rgb = np.ascontiguousarray(transformed[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def summarize_outputs(pt_output, trt_output, conf: float, nms_iou: float) -> dict:
    import torch
    from ultralytics.utils.nms import non_max_suppression

    a = first_output(pt_output)
    b = first_output(trt_output)
    result = {
        "pt_raw_shape": list(a.shape),
        "trt_raw_shape": list(b.shape),
        "pt_raw_dtype": str(a.dtype),
        "trt_raw_dtype": str(b.dtype),
        "pt_all_finite": bool(torch.isfinite(a).all()),
        "trt_all_finite": bool(torch.isfinite(b).all()),
    }
    if result["pt_raw_shape"] != result["trt_raw_shape"]:
        result["shape_mismatch"] = True
        return result
    if not result["pt_all_finite"] or not result["trt_all_finite"]:
        result["nonfinite_outputs"] = True
        return result

    # Raw outputs must have a matching order for elementwise checks to be meaningful.
    delta = (a.float() - b.float()).abs()
    result["raw_mean_abs_diff"] = float(delta.mean())
    result["raw_p99_abs_diff"] = float(torch.quantile(delta.flatten(), .99))
    result["raw_max_abs_diff"] = float(delta.max())
    if a.ndim == 3 and a.shape[1] > 4:
        result["pt_raw_score_max"] = float(a[:, 4:, :].max())
        result["trt_raw_score_max"] = float(b[:, 4:, :].max())
    outputs = []
    # identical confidence/NMS for both raw tensors, independently processed
    for pred in (a, b):
        filtered = non_max_suppression(pred.clone(), conf_thres=conf, iou_thres=nms_iou,
                                       max_det=300, multi_label=True)
        if len(filtered) != 1:
            raise RuntimeError("Parity probe expected a single inference image")
        x = filtered[0]
        outputs.append({
            "count": len(x), "score_max": float(x[:, 4].max()) if len(x) else None,
            "score_mean": float(x[:, 4].mean()) if len(x) else None,
            "high_conf_count_025": int((x[:, 4] >= .25).sum()) if len(x) else 0,
        })
    result["pt_nms"], result["trt_nms"] = outputs
    return result


def main(argv=None):
    import torch
    import tensorrt as trt
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    from evaluate_test.size_bucket_evaluator import _resolve_test_images

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--data-yaml", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scaleup", choices=("predict", "val"), default="predict",
                        help="predict=True vs val=False LetterBox scaleup behavior")
    parser.add_argument("--conf", type=float, default=.001)
    parser.add_argument("--iou", type=float, default=.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PT/TRT parity; no CPU fallback")
    if args.max_images < 1:
        parser.error("--max-images must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prior parity report: {args.output}")
    if args.pt.suffix != ".pt" or args.engine.suffix != ".engine":
        raise ValueError("Expected explicit .pt and .engine files")
    for p in (args.pt, args.engine, args.data_yaml):
        if not p.is_file():
            raise FileNotFoundError(p)

    ensure_local_ultralytics()
    from ultralytics.nn.autobackend import AutoBackend

    images = _resolve_test_images(args.data_yaml)
    if not images:
        raise RuntimeError("Frozen test split contains no images")
    images = images[:args.max_images]
    device = torch.device(args.device)
    pt_backend = AutoBackend(model=str(args.pt), device=device, fp16=False)
    trt_backend = AutoBackend(model=str(args.engine), device=device, fp16=False)
    if trt_backend.format != "engine" or trt_backend.fp16:
        raise RuntimeError("Expected an actual FP32 TensorRT engine for TRT32 parity")

    report = {
        "source": "ftsc_deployment_parity_probe_v1",
        "git_revision": git_revision(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda), "tensorrt": str(trt.__version__),
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "pt_sha256": sha256(args.pt), "engine_sha256": sha256(args.engine),
        "data_yaml_sha256": sha256(args.data_yaml),
        "conf": args.conf, "nms_iou": args.iou, "imgsz": args.imgsz,
        "letterbox_scaleup": args.scaleup == "predict",
        "trt_backend_metadata": getattr(trt_backend, "metadata", {}),
        "samples": [],
    }
    pt_backend.eval()
    trt_backend.eval()
    with torch.inference_mode():
        for image_path in images:
            input_tensor = normalized_input(image_path, args.imgsz, device,
                                            scaleup=args.scaleup == "predict")
            pt_raw = pt_backend(input_tensor.clone())
            trt_raw = trt_backend(input_tensor.clone())
            entry = summarize_outputs(pt_raw, trt_raw, args.conf, args.iou)
            entry["image"] = str(image_path)
            entry["input_sha256"] = hashlib.sha256(
                input_tensor.detach().cpu().numpy().tobytes()
            ).hexdigest()
            report["samples"].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": "DIAGNOSTIC_ONLY_NOT_AN_AP_EVALUATION",
        "output": str(args.output), "git_revision": report["git_revision"],
        "samples": len(report["samples"]),
        "worst_raw_max_abs_diff": max(
            (r.get("raw_max_abs_diff", 0.0) for r in report["samples"]), default=0.0
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
