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
from pathlib import Path
from typing import Any

from utils.ftsc_hf_layout import (
    AUGMENTATION_RUNS, DATASETS, REPO_ID, SEEDS,
    augmentation_checkpoint, seed_dir,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "FTSC_DEPLOY_EVAL_20261111_V2"
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


def checkpoint_source(args: argparse.Namespace, case: str, seed: int) -> str:
    if case in AUGMENTATION_RUNS[args.dataset]:
        return augmentation_checkpoint(args.dataset, case, seed)
    return args.manifest_cases[case][seed]["path"]


def verify_sources(args: argparse.Namespace, archive: Any) -> list[dict[str, Any]]:
    from utils.ftsc_capstone_aug_reeval_v1 import verify_hf_sources, planned_rows
    canonical = [case for case in args.cases if case in AUGMENTATION_RUNS[args.dataset]]
    verified: list[dict[str, Any]] = []
    if canonical:
        verified.extend(verify_hf_sources(archive, planned_rows(
            args.dataset, tuple(canonical), tuple(args.seeds), args.project)))
    external = [case for case in args.cases if case not in AUGMENTATION_RUNS[args.dataset]]
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
        }
        for case in args.cases for seed in args.seeds for backend in args.backends
    ]


def source_restore(args: argparse.Namespace, archive: Any, case: str, seed: int) -> dict[str, Any]:
    # Canonical transport verifies checkpoint reference and SHA, and never replaces
    # different existing bytes. DO NOT fetch old TinyPerson predictions for accuracy.
    src = source_path(args.project, args.dataset, case, seed)
    if case in AUGMENTATION_RUNS[args.dataset]:
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
                 device: str) -> Path:
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
    exported = Path(model.export(
        format="engine" if is_trt else "onnx",
        imgsz=imgsz, batch=1, half=backend.endswith("16"), device=device,
        dynamic=False, nms=False,
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
    return cache_export(pt, provenance["sha256"], backend, output_dir, imgsz, export_device)


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
    result = model.val(
        data=str(yaml_path), split="test", imgsz=imgsz, batch=args.batch,
        device=device, half=half, workers=args.workers, iou=0.5 if args.dataset == "LEVIR-Ship" else 0.7,
        plots=False, save_json=False, project=str(output_dir), name="validator",
        exist_ok=True, verbose=False,
    )
    metrics = standard_metrics(result, "test")
    # The existing small-AP contract is NOT COCO AP_small: it is the
    # TinyBenchmark size-bucket AP50 on original native test images.
    gt_file = output_dir / "test_size_ground_truth.json"
    pred_file = output_dir / "test_size_predictions.json"
    gt_file.write_text(json.dumps(_coco_ground_truth(images)) + "\n", encoding="utf-8")
    prediction = _predictions(model, images, imgsz=imgsz, batch=args.batch,
                              device=device, workers=args.workers,
                              half=half if backend.startswith("pt") else None)
    pred_file.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    metrics.update(_evaluate_coco(gt_file, pred_file))
    return metrics, {"data_yaml_sha256": sha256(yaml_path),
                     "gt_sha256": sha256(gt_file), "predictions_sha256": sha256(pred_file)}


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
    )
    metrics = evaluate_tiny_benchmark(prediction, merged_gt)
    return metrics, {
        "gt_sha256": sha256(merged_gt), "corner_manifest_sha256": sha256(manifest),
        "predictions_sha256": sha256(prediction),
    }


def validate_metric_values(metrics: dict[str, Any], dataset: str) -> None:
    for key in METRIC_KEYS[dataset]:
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and 0.0 <= value <= 1.0,
                f"Missing/invalid deployment metric {key}: {value!r}")
    # Small AP -1 indicates no valid objects in the bucket; never silently
    # substitute overall AP or write a paper number from a missing bucket.
    require(metrics[METRIC_KEYS[dataset][2]] >= 0, "Small-object AP is unavailable")


def expected_identity(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    return {"contract": VERSION, "dataset": args.dataset, "case": row["case"],
            "seed": row["seed"], "backend": row["backend"],
            "source": row["source"], "platform": args.platform}


def evaluate_one(args: argparse.Namespace, archive: Any, row: dict[str, Any]) -> dict[str, Any]:
    backend, case, seed = row["backend"], row["case"], row["seed"]
    output_dir = path_for(args.project, args.dataset, case, seed, backend)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(all(cached.get(k) == v for k, v in expected_identity(args, row).items()),
                f"Cached deployment result has different identity: {metrics_path}")
        validate_metric_values(cached["metrics"], args.dataset)
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
    else:
        metrics, fingerprints = native_metrics(args, model_file, backend, output_dir, imgsz)
    validate_metric_values(metrics, args.dataset)
    record: dict[str, Any] = {
        **expected_identity(args, row), "source_sha256": provenance["sha256"],
        "hf_revision": provenance["revision"], "model_artifact_sha256": sha256(model_file),
        "evaluator": EVALUATOR[args.dataset], "imgsz": imgsz,
        "device": backend_device(backend, args.device),
        **hardware_identity(backend_device(backend, args.device)),
        "metrics": metrics, **fingerprints,
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
                require(record.get("evaluator") == EVALUATOR[args.dataset], f"Evaluator mismatch: {path}")
                require(record.get("imgsz") == (512 if args.dataset == "LEVIR-Ship" else 640),
                        f"Resolution mismatch: {path}")
                validate_metric_values(record["metrics"], args.dataset)
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
                "evaluator": EVALUATOR[args.dataset],
                "imgsz": found[0]["imgsz"],
            }
            for label, key in zip(("AP50", "mAP50_95", "AP50_small"), METRIC_KEYS[args.dataset]):
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--phase", choices=("plan", "verify", "evaluate", "aggregate"), default="plan")
    p.add_argument("--cases", nargs="+", default=["FTSC"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
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
    args.manifest_cases = manifest_entries(args.checkpoint_manifest, args.dataset)
    allowed = set(AUGMENTATION_RUNS[args.dataset]) | set(args.manifest_cases)
    require(args.cases and len(set(args.cases)) == len(args.cases)
            and all(case in allowed for case in args.cases),
            f"Unknown/unmapped HF case for {args.dataset}; add exact paths/SHA in manifest")
    require(args.seeds and len(set(args.seeds)) == len(args.seeds)
            and all(seed in SEEDS for seed in args.seeds), "Unsupported/duplicate seeds")
    require(args.backends and len(set(args.backends)) == len(args.backends)
            and all(b in BACKENDS for b in args.backends), "Unsupported/duplicate backend")
    require(all(case in AUGMENTATION_RUNS[args.dataset] or
                all(seed in args.manifest_cases[case] for seed in args.seeds)
                for case in args.cases), "Manifest lacks requested seed checkpoint mapping")
    require(args.batch > 0 and args.workers >= 0, "Invalid batch/workers")
    args.project = args.project.expanduser().resolve()
    rows = planned(args)
    if args.phase == "plan":
        print(json.dumps({"status":"PLAN_ONLY","rows":rows}, indent=2))
        return
    if args.phase == "aggregate":
        print(json.dumps(aggregate(args), indent=2))
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
    for row in rows:
        completed.append(evaluate_one(args, archive, row))
    print(json.dumps({"status":"EVALUATION_ONLY_COMPLETE", "rows":completed}, indent=2))


if __name__ == "__main__":
    main()
