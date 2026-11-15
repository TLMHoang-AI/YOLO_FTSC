"""Paper-scope LEVIR validation re-evaluation; never trains or evaluates test split.

Exactly 11 distinct checkpoints configurations (33 seed jobs). Both main AP metrics
are native Ultralytics YOLO.val(split="val", imgsz=512, batch=8, rect=True,
conf=.001, iou=.5); Small is independent YOLO.predict on the SAME val images
with historical defaults conf=.25, rect=True, iou=.5 and TinyBenchmark
Small=20^2..32^2 area. Test results and HF source artifacts are never written.

Usage (from yolo_code):
  python -m utils.ftsc_levir_validation_paper_v1 --phase plan
  python -m utils.ftsc_levir_validation_paper_v1 --phase complete \
      --data-yaml datasets/levir_ship_yolo_seed42/levir_ship.yaml \
      --confirm-evaluate --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "utils/ftsc_levir_validation_cases_v1.json"
DEFAULT_PROJECT = ROOT / "runs/ftsc_levir_validation_paper_v1"
CONTRACT = "FTSC_LEVIR_PAPER_VAL_3METRICS_20261115_V1"
METRICS = ("val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)", "val_size/AP50-Small")
PERCENT_LABELS = ("Val_AP50_pct", "Val_mAP50_95_pct", "Val_AP50_small_pct")
ALL_SEEDS = (42, 43, 44)
EXPECTED_CASES = (
    "FTSC", "FTSC_YOLOV9T", "FTSC_YOLO11N", "FTSC_OFF", "HEAD_H0",
    "HEAD_H1", "HEAD_H3", "EDGE_CUE", "COLOR_CUE", "ALPHA4", "ALPHA8",
)
EVAL_SOURCE_FILES = (
    "utils/ftsc_levir_validation_paper_v1.py",
    "utils/ftsc_levir_validation_cases_v1.json",
    "evaluate_test/standard_detection_metrics.py",
    "evaluate_test/size_bucket_evaluator.py",
    "vendor/tinyperson_cocoeval.py",
    "models_related/ultralytics/ultralytics/engine/model.py",
    "models_related/ultralytics/ultralytics/engine/predictor.py",
    "models_related/ultralytics/ultralytics/models/yolo/detect/val.py",
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: Path, value: Any) -> None:
    """One-shot immutable result: no overwrites of prior evaluated outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def replace_derived(path: Path, value: Any) -> None:
    """Only aggregate/report files are replaceable, not per-checkpoint metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    require(not temporary.exists(), f"Unresolved interrupted write: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def write_csv_derived(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    require(not temporary.exists(), f"Unresolved interrupted write: {temporary}")
    with temporary.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    m = json.loads(path.read_text(encoding="utf-8"))
    require(m.get("schema") == "ftsc_levir_validation_all_paper_cases_v1"
            and m.get("dataset") == "LEVIR-Ship"
            and m.get("seeds") == list(ALL_SEEDS), "Unexpected validation manifest schema/seeds")
    cases = m.get("cases")
    require(isinstance(cases, dict) and set(cases) == set(EXPECTED_CASES),
            "Validation manifest must contain exactly 11 canonical paper cases")
    seen = set()
    for name in EXPECTED_CASES:
        spec = cases[name]
        require(set(spec["checkpoints"]) == {"42", "43", "44"},
                f"Missing seeds in {name}")
        for seed in ALL_SEEDS:
            ref = spec["checkpoints"][str(seed)]
            expected = f'{spec["source_run"]}/seed_{seed}/weights/best.pt'
            require(ref["path"] == expected, f"{name}/{seed}: unexpected HF source path")
            digest = ref["sha256"]
            require(isinstance(digest, str) and len(digest) == 64
                    and all(ch in "0123456789abcdef" for ch in digest),
                    f"{name}/{seed}: invalid SHA256")
            require(ref["path"].startswith("LEVIR-Ship/")
                    and ref["path"] not in seen, f"Duplicate/unsafe checkpoint: {ref['path']}")
            seen.add(ref["path"])
    require(m.get("scope", "").find("excludes augmentation") >= 0,
            "The fixed paper validation scope must exclude augmentation compatibility")
    require(cases["COLOR_CUE"]["source_run"].endswith("/C_H2_POSTFUSION_V2"),
            "Legacy Color V1 is prohibited")
    return m


def selected_cases(value: list[str] | None) -> tuple[str, ...]:
    if not value:
        return EXPECTED_CASES
    require(len(value) == len(set(value)) and set(value) <= set(EXPECTED_CASES),
            f"Invalid duplicate/unknown case selection: {value}")
    return tuple(k for k in EXPECTED_CASES if k in value)


def selected_seeds(value: list[int] | None) -> tuple[int, ...]:
    if not value:
        return ALL_SEEDS
    require(len(value) == len(set(value)) and set(value) <= set(ALL_SEEDS),
            f"Invalid duplicate/unknown seed selection: {value}")
    return tuple(s for s in ALL_SEEDS if s in value)


def job_dir(project: Path, case: str, seed: int) -> Path:
    return project / case / f"seed_{seed}"


def source_sha256() -> dict[str, str]:
    return {rel: sha256(ROOT / rel) for rel in EVAL_SOURCE_FILES}


def resolve_val_images(data_yaml: Path) -> list[Path]:
    """Mirror the repository's native test-image resolver, on YAML 'val'."""
    import yaml

    y = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    require(isinstance(y, dict) and isinstance(y.get("val"), str),
            "LEVIR val path must be a single resolved YAML path/list file")
    root = Path(y.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    item = Path(y["val"])
    target = item if item.is_absolute() else root / item
    require(target.exists(), f"Missing validation split path: {target}")
    if target.is_file():
        lines = [line.strip() for line in target.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        paths = [Path(line) if Path(line).is_absolute()
                 else (target.parent / line).resolve() for line in lines]
    else:
        paths = sorted(p for p in target.rglob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    require(paths and len(paths) == len({str(p.resolve()) for p in paths}),
            "Validation split empty or contains duplicate image paths")
    missing = [str(p) for p in paths if not p.is_file()]
    require(not missing, f"Validation images missing, first paths: {missing[:5]}")
    return paths


def split_provenance(data_yaml: Path, images: list[Path]) -> dict[str, Any]:
    from evaluate_test.size_bucket_evaluator import _label_path

    require(images, "Empty validation split")
    paths = hashlib.sha256()
    label_hash = hashlib.sha256()
    num_labels = 0
    for path in images:
        path_text = str(path.resolve())
        paths.update(path_text.encode("utf-8") + b"\n")
        label = _label_path(path)
        require(label.is_file(), f"Missing validation label file: {label}")
        num_labels += 1
        label_hash.update(str(label.resolve()).encode("utf-8") + b"\0")
        label_hash.update(sha256(label).encode("ascii") + b"\n")
    return {
        "data_yaml_sha256": sha256(data_yaml),
        "ordered_image_paths_sha256": paths.hexdigest(),
        "ordered_label_contents_sha256": label_hash.hexdigest(),
        "image_count": len(images),
        "label_file_count": num_labels,
        "split": "val", "frozen_split_seed": 42,
    }


def source_checkpoint(spec: dict[str, Any], seed: int, repo: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    row = spec["checkpoints"][str(seed)]
    downloaded = Path(hf_hub_download(
        repo_id=repo, repo_type="dataset", revision=revision,
        filename=row["path"],
    ))
    digest = sha256(downloaded)
    require(digest == row["sha256"],
            f"Checkpoint SHA mismatch for {row['path']}: expected {row['sha256']}, got {digest}")
    return downloaded


def pinned_revision(repo: str) -> str:
    from huggingface_hub import HfApi
    revision = HfApi().repo_info(repo_id=repo, repo_type="dataset").sha
    require(isinstance(revision, str) and len(revision) >= 40,
            f"Could not pin source HF repository revision: {repo}")
    return revision


def run_one(args: argparse.Namespace, manifest: dict[str, Any],
            case: str, seed: int, revision: str,
            split: dict[str, Any], images: list[Path],
            hashes: dict[str, str]) -> dict[str, Any]:
    spec = manifest["cases"][case]
    saved = job_dir(args.project, case, seed) / "val_metrics.json"
    if saved.exists():
        old = json.loads(saved.read_text(encoding="utf-8"))
        validate_result(old, manifest, case, seed, split, hashes)
        require(old.get("source/hf_revision") == revision,
                f"HF revision changed since this job: {saved}; do not reuse unpinned artifacts")
        return {"case": case, "seed": seed, "status": "REUSED_IDENTICAL", "metrics": str(saved)}
    checkpoint = source_checkpoint(spec, seed, manifest["repo_id"], revision)
    from evaluate_test.standard_detection_metrics import (
        ensure_local_ultralytics, standard_metrics,
    )
    fork_path = ensure_local_ultralytics()
    from ultralytics import YOLO
    from evaluate_test.size_bucket_evaluator import _coco_ground_truth, _predictions, _evaluate_coco

    run = job_dir(args.project, case, seed)
    run.mkdir(parents=True, exist_ok=True)
    # Existing incomplete artifacts are never silently reused.
    for name in ("val_gt.json", "val_predictions.json"):
        require(not (run / name).exists(),
                f"Partial uncommitted evaluation artifacts at {run / name}; inspect before retrying")

    model = YOLO(str(checkpoint))
    native = model.val(
        data=str(args.data_yaml), split="val", imgsz=args.imgsz,
        batch=args.batch, rect=True, conf=0.001, iou=0.5,
        device=args.device, workers=args.workers,
        plots=False, verbose=False,
        project=str(run), name="ultralytics_val", exist_ok=True,
    )
    scores = standard_metrics(native, "val")
    # Independent original paper predictor. Do not reuse validator.jdict or change
    # predictor default score threshold (.25) / NMS / auto-letterbox.
    small_model = YOLO(str(checkpoint))
    detections = _predictions(small_model, images, imgsz=args.imgsz,
                              batch=args.batch, device=args.device,
                              workers=args.workers)
    ground_truth = _coco_ground_truth(images)
    gt_path = run / "val_gt.json"
    det_path = run / "val_predictions.json"
    write_new(gt_path, ground_truth)
    write_new(det_path, detections)
    tiny_scores = _evaluate_coco(gt_path, det_path)
    scores[METRICS[2]] = tiny_scores["test_size/AP50-Small"]
    for key in METRICS:
        v = scores.get(key)
        require(isinstance(v, (float, int)) and math.isfinite(float(v))
                and 0 <= float(v) <= 1, f"{case}/{seed}: invalid {key}={v}")
    require(scores[METRICS[2]] >= 0,
            f"{case}/{seed}: Small AP unavailable (no qualifying objects)")

    record = {
        "contract": CONTRACT, "dataset": "LEVIR-Ship", "case": case,
        "seed": seed, "sections": spec["sections"],
        "source/hf_repo": manifest["repo_id"],
        "source/hf_revision": revision,
        "source/checkpoint": spec["checkpoints"][str(seed)]["path"],
        "source/checkpoint_sha256": spec["checkpoints"][str(seed)]["sha256"],
        "source/evaluator_sha256": hashes,
        "source/local_ultralytics": fork_path,
        "split_provenance": split,
        "evaluation": {
            "no_training": True, "split": "val",
            "main": "YOLO.val rect=True conf=.001 iou=.5 native Ultralytics metrics",
            "small": "YOLO.predict native val images default conf=.25 rect=True iou=.5"
                     " -> class-agnostic TinyBenchmark 20^2<=area<32^2",
            "imgsz": args.imgsz, "batch": args.batch, "workers": args.workers,
            "device": args.device, "gt_sha256": sha256(gt_path),
            "predictions_sha256": sha256(det_path),
            "gt_object_count": len(ground_truth["annotations"]),
            "prediction_count": len(detections),
        },
        "metrics": {key: float(scores[key]) for key in METRICS},
    }
    write_new(saved, record)
    return {"case": case, "seed": seed, "status": "EVALUATED", "metrics": str(saved)}


def validate_result(record: dict[str, Any], manifest: dict[str, Any],
                    case: str, seed: int, split: dict[str, Any],
                    hashes: dict[str, str]) -> None:
    want = manifest["cases"][case]["checkpoints"][str(seed)]
    require(record.get("contract") == CONTRACT
            and record.get("dataset") == "LEVIR-Ship"
            and record.get("case") == case and record.get("seed") == seed
            and record.get("source/checkpoint") == want["path"]
            and record.get("source/checkpoint_sha256") == want["sha256"]
            and record.get("source/hf_repo") == manifest["repo_id"]
            and record.get("source/evaluator_sha256") == hashes
            and record.get("split_provenance") == split
            and record.get("evaluation", {}).get("no_training") is True
            and record.get("evaluation", {}).get("split") in ("val", None),
            f"Stored validation result identity differs for {case}/seed_{seed}")
    scores = record.get("metrics") or {}
    for key in METRICS:
        v = scores.get(key)
        require(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1,
                f"Invalid stored {case}/seed_{seed}/{key}")
def aggregate(args: argparse.Namespace, manifest: dict[str, Any],
              cases: tuple[str, ...], seeds: tuple[int, ...],
              split: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    rows = []
    summary = []
    blockers = []
    revisions = set()
    for case in cases:
        seed_rows = []
        for seed in seeds:
            path = job_dir(args.project, case, seed) / "val_metrics.json"
            if not path.exists():
                blockers.append({"case": case, "seed": seed, "reason": "missing per-seed metrics"})
                continue
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                validate_result(rec, manifest, case, seed, split, hashes)
                e = rec["evaluation"]
                run = job_dir(args.project, case, seed)
                require(sha256(run / "val_gt.json") == e["gt_sha256"]
                        and sha256(run / "val_predictions.json") == e["predictions_sha256"],
                        f"GT/prediction artifacts changed: {run}")
                revisions.add(rec["source/hf_revision"])
                row = {"case": case, "seed": seed, "status": "VERIFIED",
                       "hf_checkpoint": rec["source/checkpoint"],
                       "checkpoint_sha256": rec["source/checkpoint_sha256"]}
                for label, key in zip(PERCENT_LABELS, METRICS):
                    row[label] = 100.0 * rec["metrics"][key]
                rows.append(row)
                seed_rows.append(row)
            except Exception as exc:
                blockers.append({"case": case, "seed": seed, "reason": str(exc)})
        if len(seed_rows) != len(seeds):
            continue
        out = {"case": case, "n": len(seed_rows),
               "seeds": ",".join(str(row["seed"]) for row in seed_rows)}
        for label in PERCENT_LABELS:
            vals = [row[label] for row in seed_rows]
            out[label + "_mean"] = statistics.mean(vals)
            out[label + "_sample_std"] = statistics.stdev(vals) if len(vals) >= 2 else None
        summary.append(out)
    if len(revisions) > 1:
        blockers.append({"reason": "mixed HF source revisions", "revisions": sorted(revisions)})
    expected = len(cases) * len(seeds)
    status = "COMPLETE" if not blockers and len(rows) == expected else "BLOCKED"
    report = {"contract": CONTRACT, "status": status, "expected_jobs": expected,
              "valid_jobs": len(rows), "blockers": blockers,
              "hf_revisions": sorted(revisions),
              "per_seed_csv": str(args.project / "levir_val_metrics_by_seed.csv"),
              "aggregate_csv": str(args.project / "levir_val_metrics_aggregate.csv")}
    if status == "COMPLETE":
        write_csv_derived(args.project / "levir_val_metrics_by_seed.csv", rows,
                          ["case", "seed", "status", "hf_checkpoint", "checkpoint_sha256", *PERCENT_LABELS])
        write_csv_derived(args.project / "levir_val_metrics_aggregate.csv", summary,
                          ["case", "n", "seeds", *[x + y for x in PERCENT_LABELS
                                              for y in ("_mean", "_sample_std")]])
    replace_derived(args.project / "levir_val_report_status.json", report)
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "evaluate", "aggregate", "complete"),
                        default="plan")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--confirm-evaluate", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    cases = selected_cases(args.cases)
    seeds = selected_seeds(args.seeds)
    args.project = args.project.expanduser().resolve()
    require(args.imgsz == 512 and args.batch == 8,
            "Paper validation contract fixes imgsz=512 and batch=8")
    require(args.workers >= 0, "Negative worker count")
    jobs = [{"case": case, "seed": seed,
             "checkpoint": manifest["cases"][case]["checkpoints"][str(seed)]["path"]}
            for case in cases for seed in seeds]
    if args.phase == "plan":
        report = {"contract": CONTRACT, "status": "PLAN_ONLY", "jobs": jobs,
                  "expected_jobs": len(jobs), "excluded": ["augmentation_compatibility"]}
        print(json.dumps(report, indent=2))
        return report
    require(args.data_yaml is not None, "--data-yaml required for any non-plan phase")
    args.data_yaml = args.data_yaml.expanduser().resolve()
    require(args.data_yaml.is_file(), f"Missing val/test YAML: {args.data_yaml}")
    require(args.data_yaml.name == "levir_ship.yaml"
            and args.data_yaml.parent.name == "levir_ship_yolo_seed42",
            "Use ONLY the frozen LEVIR split-seed-42 dataset YAML")
    frozen_manifest_path = args.data_yaml.parent / "manifest.json"
    require(frozen_manifest_path.is_file(), "Missing frozen LEVIR split manifest")
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    require(frozen.get("seed") == 42
            and frozen.get("splits", {}).get("val", {}).get("images") == 788
            and frozen.get("splits", {}).get("test", {}).get("images") == 788,
            "Frozen LEVIR split manifest does not match 42/788 validation protocol")
    # Guard against accidental validation/test leakage; no test inference is run.
    import yaml
    from evaluate_test.size_bucket_evaluator import _resolve_test_images
    y = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    require(y.get("val") and y.get("test") and y["val"] != y["test"],
            "Validation must refer to a distinct split from test")
    images = resolve_val_images(args.data_yaml)
    require(len(images) == 788, f"Expected exactly 788 LEVIR validation images; got {len(images)}")
    test_paths = _resolve_test_images(args.data_yaml)
    require(not ({p.resolve() for p in images} & {p.resolve() for p in test_paths}),
            "Validation and test splits have overlapping image files")
    split = split_provenance(args.data_yaml, images)
    hashes = source_sha256()
    if args.phase in ("evaluate", "complete"):
        require(args.confirm_evaluate,
                "GPU evaluation requires explicit --confirm-evaluate")
        # Pin once for all downloads; mixed revisions are forbidden.
        revision = pinned_revision(manifest["repo_id"])
        errors = []
        for index, job in enumerate(jobs, 1):
            case, seed = job["case"], job["seed"]
            print(f"[LEVIR VAL {index}/{len(jobs)}] {case}/seed_{seed}", flush=True)
            try:
                outcome = run_one(args, manifest, case, seed, revision,
                                  split, images, hashes)
                print(json.dumps(outcome), flush=True)
            except Exception as exc:
                failure = {"case": case, "seed": seed, "reason": str(exc)}
                errors.append(failure)
                print(json.dumps({"status": "FAILED", **failure}), flush=True)
                traceback.print_exc()
                if args.stop_on_error:
                    raise
        if args.phase == "evaluate":
            report = {"status": "EVALUATION_FINISHED_WITH_BLOCKERS" if errors else
                      "EVALUATION_FINISHED", "expected_jobs": len(jobs),
                      "failed_jobs": errors}
            print(json.dumps(report, indent=2))
            return report
    report = aggregate(args, manifest, cases, seeds, split, hashes)
    print(json.dumps(report, indent=2))
    if report["status"] != "COMPLETE":
        raise RuntimeError("LEVIR val 3-metric aggregation blocked; see levir_val_report_status.json")
    return report


if __name__ == "__main__":
    main()
