#!/usr/bin/env python3
"""Read-only precision/paper-input audit for LEVIR V3 PT/TRT deployment runs.

No training, export, HF access or model inference. Does not alter metrics.json.
A derived audit report is written beneath the isolated V3 run namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from types import SimpleNamespace

from utils import ftsc_deployment_backend_eval_v1 as deploy

STAGES = {
    "ftsc_fp32": ("FTSC", ("pt32", "trt32")),
    "ftsc_all_precision": ("FTSC", ("pt32", "trt32", "pt16", "trt16")),
    "report_fp32": ("ALL", ("pt32", "trt32")),
    "report_all_precision": ("ALL", ("pt32", "trt32", "pt16", "trt16")),
}
SEEDS = (42, 43, 44)
METRICS = {
    "AP50": "test/metrics/mAP50(B)",
    "mAP50_95": "test/metrics/mAP50-95(B)",
    "AP50_Small": "test_size/AP50-Small",
}
SHA = re.compile(r"^[a-f0-9]{64}$")
REFERENCE_IMAGES = 788


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for buf in iter(lambda: stream.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def get_scope(stage: str, manifest_path: Path) -> dict:
    require(stage in STAGES, f"Unknown stage {stage!r}")
    full = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(full.get("schema") == deploy.REPORT_SCHEMA and
            full.get("repo_id") == deploy.REPO_ID,
            "Incorrect report-driven checkpoint manifest")
    manifest_cases = full["datasets"]["LEVIR-Ship"]["cases"]
    case_mode, backends = STAGES[stage]
    cases = list(manifest_cases) if case_mode == "ALL" else [case_mode]
    require(set(cases) <= set(manifest_cases),
            "Requested cases are not present in audited manifest")
    require(all(set(manifest_cases[c]["seeds"]) == {"42", "43", "44"}
                for c in cases), "Some report cases lack a complete three-seed manifest")
    return {"stage": stage, "dataset": "LEVIR-Ship", "cases": cases,
            "seeds": list(SEEDS), "backends": list(backends),
            "protocol": deploy.RECT_PARITY_PROTOCOL,
            "runtime_revision": deploy.RECT_PARITY_RUNTIME_REVISION}


def _read_metrics(folder: Path, case: str, seed: int, backend: str,
                  expected_source: dict, platform: str) -> dict:
    path = folder / "metrics.json"
    if not path.is_file():
        failure = folder / "failure.json"
        if failure.is_file():
            error = json.loads(failure.read_text(encoding="utf-8"))
            raise RuntimeError("Missing metrics.json; failure.json: " +
                               str(error.get("error", error.get("error_type", "unknown"))))
        raise RuntimeError("Missing metrics.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    args = SimpleNamespace(dataset="LEVIR-Ship", platform=platform,
                           evaluation_protocol=deploy.RECT_PARITY_PROTOCOL)
    expected = deploy.expected_identity(args, {
        "case": case, "seed": seed, "backend": backend,
        "source": expected_source["path"],
    })
    require(all(record.get(key) == value for key, value in expected.items()),
            "Protocol, backend, evaluator config, source code or platform drift")
    require(record.get("evaluator") == deploy.evaluator_name(args),
            "Evaluator mismatch")
    deploy.validate_metric_values(record["metrics"], "LEVIR-Ship",
                                  deploy.RECT_PARITY_PROTOCOL)
    require(record.get("source_sha256") == expected_source["sha256"],
            "Checkpoint SHA does not match audited manifest")
    checkpoint = deploy.source_path(
        folder.parents[3], "LEVIR-Ship", case, seed,
    )
    # folder = <V3 root>/LEVIR-Ship/<case>/seed_<n>/<backend>
    # folder.parents[3] is the isolated V3 project root.
    require(checkpoint.is_file() and file_sha(checkpoint) == record["source_sha256"],
            "Actual cached source checkpoint is missing/has changed")
    require(isinstance(record.get("model_artifact_sha256"), str) and
            SHA.fullmatch(record["model_artifact_sha256"]),
            "Evaluated backend artifact SHA missing")
    if backend.startswith("pt"):
        require(record["model_artifact_sha256"] == record["source_sha256"],
                "PT model artifact differs from source checkpoint")
    else:
        engine = folder / "export/source.engine"
        require(engine.is_file() and file_sha(engine) == record["model_artifact_sha256"],
                "Actual static TensorRT engine is missing/has changed")
        export_file = folder / "export/export_manifest.json"
        require(export_file.is_file(), "TensorRT export metadata missing")
        export = json.loads(export_file.read_text(encoding="utf-8"))
        fp = export.get("fingerprint", {})
        require(export.get("artifact_sha256") == record["model_artifact_sha256"]
                and fp.get("source_sha256") == record["source_sha256"]
                and fp.get("backend") == backend and fp.get("imgsz") == 544
                and fp.get("paper_preprocess_imgsz") == 512
                and fp.get("paper_stride") == 32 and fp.get("paper_rect") is True
                and fp.get("simplify") is False,
                "TensorRT export has different engine/paper-preprocess contract")
    require(record.get("imgsz") == 512,
            "Paper preprocessing CLI imgsz must be 512")
    require(record.get("preprocessed_input_count") == REFERENCE_IMAGES and
            record.get("preprocessed_input_canvas") == 544,
            "Missing 788 images / 544-canvas proof")
    for key in ("data_yaml_sha256", "gt_sha256", "predictions_sha256",
                "preprocessed_input_manifest_sha256"):
        require(isinstance(record.get(key), str) and SHA.fullmatch(record[key]),
                f"Invalid {key} fingerprint")
    deploy.verify_native_val_artifacts(record, folder, "LEVIR-Ship")
    inputs = folder / "test_val_preprocessed_inputs_sha256.json"
    manifests = json.loads(inputs.read_text(encoding="utf-8"))
    mapping = manifests.get("image_sha256")
    require(manifests.get("source_images_count") == REFERENCE_IMAGES and
            manifests.get("paper_imgsz") == 512 and
            manifests.get("canvas") == 544 and
            manifests.get("minimum_pytorch_stride") == 32 and
            isinstance(mapping, dict) and len(mapping) == REFERENCE_IMAGES and
            all(isinstance(value, str) and SHA.fullmatch(value)
                for value in mapping.values()),
            "Input manifest content is incomplete or not paper-aligned")
    require(file_sha(inputs) == record["preprocessed_input_manifest_sha256"],
            "The 788-image pixel manifest was changed")
    if backend.startswith("trt"):
        require(record.get("full_test_input_parity") is True,
                "TensorRT record did not pass runner's PT32 pixel parity gate")
        require(record.get("paired_pt32_preprocessed_manifest_sha256") ==
                record["preprocessed_input_manifest_sha256"],
                "TensorRT PT32 pairing digest is inconsistent")
    return record


def audit_stage(stage: str, base_project: Path, manifest_path: Path,
                platform: str = "marimo_live") -> dict:
    scope = get_scope(stage, manifest_path)
    root = base_project / scope["protocol"] / scope["runtime_revision"]
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "datasets"]["LEVIR-Ship"]["cases"]
    found = {}
    issues = []
    comparisons = []
    for case in scope["cases"]:
        for seed in SEEDS:
            for backend in scope["backends"]:
                name = f"{case}/seed_{seed}/{backend}"
                folder = deploy.path_for(root, "LEVIR-Ship", case, seed, backend)
                try:
                    found[(case, seed, backend)] = _read_metrics(
                        folder, case, seed, backend,
                        entries[case]["seeds"][str(seed)], platform)
                except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                    issues.append({"job": name, "error": f"{type(exc).__name__}: {exc}"})

    # Any metric comparison requires both complete, valid records.
    for case in scope["cases"]:
        for seed in SEEDS:
            first = found.get((case, seed, "pt32"))
            if first is None:
                continue
            for other_name in ("pt16", "trt32", "trt16"):
                other = found.get((case, seed, other_name))
                if other is None:
                    continue
                reference_name = ("pt16" if other_name == "trt16"
                                  and (case, seed, "pt16") in found else "pt32")
                reference = found[(case, seed, reference_name)]
                tag = f"{case}/seed_{seed}/{other_name} vs {reference_name}"
                try:
                    for key in ("data_yaml_sha256", "gt_sha256",
                                "preprocessed_input_manifest_sha256",
                                "source_sha256"):
                        require(reference[key] == other[key],
                                f"{tag}: different {key}")
                    for key in ("device", "gpu_name", "torch_version", "cuda_version"):
                        require(reference[key] == other[key],
                                f"{tag}: different {key}")
                    if other_name == "trt16":
                        # Its V3 runner checks PT32; independently enforce PT16 here.
                        require((case, seed, "pt16") in found,
                                f"{tag}: missing exact-precision PT16 reference")
                        require(other.get("paired_pt32_metrics_sha256") ==
                                file_sha(deploy.path_for(
                                    root, "LEVIR-Ship", case, seed, "pt32") / "metrics.json"),
                                f"{tag}: PT32 pairing file has changed")
                    if other_name == "trt32":
                        require(other.get("paired_pt32_metrics_sha256") ==
                                file_sha(deploy.path_for(
                                    root, "LEVIR-Ship", case, seed, "pt32") / "metrics.json"),
                                f"{tag}: PT32 pairing file has changed")
                    row = {"case": case, "seed": seed,
                           "reference": reference_name, "backend": other_name,
                           "identical_all_788_image_inputs": True}
                    row.update({key + "_delta_pp":
                                100.0 * (other["metrics"][metric_key] -
                                         reference["metrics"][metric_key])
                                for key, metric_key in METRICS.items()})
                    comparisons.append(row)
                except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
                    issues.append({"job": tag, "error": f"{type(exc).__name__}: {exc}"})

    # Frozen test split and GT are shared by all seeds and cases.
    for attr in ("data_yaml_sha256", "gt_sha256", "preprocessed_input_manifest_sha256"):
        unique = {item[attr] for item in found.values()}
        if len(unique) > 1:
            issues.append({"job": "CROSS_CASE_TEST_IDENTITY",
                           "error": f"V3 run set disagrees on {attr} ({len(unique)} values)"})

    aggregate_preview = []
    for case in scope["cases"]:
        for backend in scope["backends"]:
            rows = [found.get((case, seed, backend)) for seed in SEEDS]
            if any(value is None for value in rows):
                continue
            item = {"case": case, "backend": backend, "seeds": list(SEEDS)}
            for metric, key in METRICS.items():
                values = [100.0 * value["metrics"][key] for value in rows]
                item[metric + "_mean_pct"] = statistics.fmean(values)
                item[metric + "_sample_std_pp"] = statistics.stdev(values)
            aggregate_preview.append(item)

    # These are REVIEW thresholds, not fabricated expected outcomes. Preserve
    # the measured AP even when the accuracy gap is larger than anticipated.
    # Treat precision conversion as distinct from switching inference engines.
    accuracy_flags = []
    for item in comparisons:
        backend = item["backend"]
        if backend == "trt32":
            threshold = 0.5
        elif backend in ("trt16", "pt16"):
            threshold = 1.0
        else:
            continue
        if abs(item["AP50_delta_pp"]) > threshold:
            accuracy_flags.append({
                "case": item["case"], "seed": item["seed"],
                "comparison": f"{item['backend']} vs {item['reference']}",
                "AP50_delta_pp": item["AP50_delta_pp"],
                "review_threshold_pp": threshold,
                "reason": "Unexpected AP50 drift despite verified input parity",
            })
    expected = len(scope["cases"]) * len(SEEDS) * len(scope["backends"])
    require(len(found) <= expected, "Unexpected duplicate validation records")
    status = ("INCOMPLETE_OR_INVALID" if len(found) != expected or issues
              else "COMPLETE_NEEDS_ACCURACY_REVIEW" if accuracy_flags
              else "PASS_COMPLETE")
    return {
        "status": status,
        "scope": scope,
        "project": str(root), "expected_jobs": expected,
        "valid_jobs": len(found), "issues": issues,
        "accuracy_review_flags": accuracy_flags,
        "comparisons": comparisons, "three_seed_preview": aggregate_preview,
        "note": "Input parity is SHA-verified; FP32/FP16 metric differences are "
                "reported, not forced to zero. AP preview is not Jetson FPS.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=tuple(STAGES))
    p.add_argument("--base-project", type=Path,
                   default=deploy.ROOT / "runs/ftsc_deployment_eval_v1")
    p.add_argument("--report-manifest", type=Path, default=deploy.REPORT_MANIFEST)
    p.add_argument("--platform", default="marimo_live")
    p.add_argument("--describe-stage", action="store_true")
    args = p.parse_args(argv)
    scope = get_scope(args.stage, args.report_manifest)
    if args.describe_stage:
        print(json.dumps(scope, indent=2))
        return 0
    report = audit_stage(args.stage, args.base_project, args.report_manifest,
                         platform=args.platform)
    audit_dir = (args.base_project / scope["protocol"] / scope["runtime_revision"] /
                 "LEVIR-Ship" / "_audits")
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / (args.stage + "_audit.json")
    tmp = path.with_suffix(".json.tmp")
    require(not tmp.exists(), f"Incomplete prior audit report: {tmp}")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    print(json.dumps({
        "status": report["status"], "stage": args.stage,
        "expected_jobs": report["expected_jobs"], "valid_jobs": report["valid_jobs"],
        "issues": report["issues"],
        "accuracy_review_flags": report["accuracy_review_flags"],
        "pairwise_comparisons": report["comparisons"],
        "three_seed_preview": report["three_seed_preview"],
        "audit_report": str(path),
    }, indent=2))
    return 0 if report["status"] == "PASS_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
