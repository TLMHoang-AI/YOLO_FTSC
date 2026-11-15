"""TinyPerson paper-scope validation: 11 FTSC configurations x seeds 42/43/44.

Validation is performed on each seed's prepared sw640/sh512 *corner crops*.
Native YOLO.val metrics and independent TinyBenchmark AP50-Small are both
computed on the SAME validation split. This does NOT claim original-image
merged validation AP, which requires distinct original-image ground truth.
The paper's merged TEST numbers remain untouched.

Run from /marimo/yolo_code after preparing the official per-seed corner data:
  python -m utils.ftsc_tinyperson_validation_paper_v1 --phase plan
  python -m utils.ftsc_tinyperson_validation_paper_v1 --phase complete \
    --confirm-evaluate --device cuda:0

No training, test evaluation, checkpoint modification or HF upload.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_MANIFEST = ROOT / "utils/ftsc_report_deployment_case_manifest_v1.json"
PROJECT = ROOT / "runs/ftsc_tinyperson_validation_paper_v1"
CONTRACT = "FTSC_TINYPERSON_PAPER_VAL_CORNER_3METRICS_20261115_V1"
SEEDS = (42, 43, 44)
CASES = (
    "FTSC", "FTSC_YOLOV9T", "FTSC_YOLO11N", "FTSC_OFF",
    "HEAD_H0", "HEAD_H1", "HEAD_H3", "EDGE_CUE",
    "COLOR_CUE", "ALPHA4", "ALPHA8",
)
METRICS = ("val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)", "val_size/AP50-Small")
LABELS = ("Val_AP50_pct", "Val_mAP50_95_pct", "Val_AP50_small_pct")
SOURCES = (
    "utils/ftsc_tinyperson_validation_paper_v1.py",
    "utils/ftsc_report_deployment_case_manifest_v1.json",
    "train_tinyperson_scripts/train_all_tinyperson.py",
    "evaluate_test/standard_detection_metrics.py",
    "evaluate_test/size_bucket_evaluator.py",
    "vendor/tinyperson_cocoeval.py",
    "models_related/ultralytics/ultralytics/engine/model.py",
    "models_related/ultralytics/ultralytics/engine/predictor.py",
    "models_related/ultralytics/ultralytics/models/yolo/detect/val.py",
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
)
EXPECTED_SOURCE = {
    "FTSC": "TinyPerson/original_runs/H2_MOSAIC",
    "FTSC_YOLOV9T": "TinyPerson/cross_yolo_portability/yolov9t/yolov9t__FTSC_H2",
    "FTSC_YOLO11N": "TinyPerson/cross_yolo_portability/yolo11n/yolo11n__FTSC_H2",
    "FTSC_OFF": "TinyPerson/original_runs/H2_NOFTSC_NM",
    "HEAD_H0": "TinyPerson/journal_experiments/phase2/TP_PRUNE_H0_P2345",
    "HEAD_H1": "TinyPerson/journal_experiments/phase2/TP_PRUNE_H1_P234",
    "HEAD_H3": "TinyPerson/journal_experiments/phase2/TP_PRUNE_H3_P2",
    "EDGE_CUE": "TinyPerson/original_runs/ES1_MOSAIC",
    "COLOR_CUE": "TinyPerson/journal_experiments/phase2/TP_CUE_COLOR_H2_MOSAIC",
    "ALPHA4": "TinyPerson/journal_experiments/alpha_sensitivity/ALPHA4_rerun",
    "ALPHA8": "TinyPerson/journal_experiments/alpha_sensitivity/ALPHA8_rerun",
}


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in SOURCES}


def immutable_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")


def derived_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".new")
    require(not temp.exists(), f"Interrupted derived write: {temp}")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, sort_keys=True, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp.replace(path)


def derived_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".new")
    require(not temp.exists(), f"Interrupted derived CSV write: {temp}")
    with temp.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def manifest_cases(path: Path = REPORT_MANIFEST) -> tuple[str, dict[str, dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    require(raw.get("schema") == "ftsc_report_driven_deployment_case_manifest_v1"
            and raw.get("repo_id") == "TLMHoang/FTSC_results"
            and raw.get("seeds") == [42, 43, 44],
            "Unexpected paper HF manifest identity")
    cases = raw["datasets"]["TinyPerson"]["cases"]
    require(set(cases) == set(CASES), "TinyPerson paper case set changed")
    seen: set[str] = set()
    for case in CASES:
        spec = cases[case]
        require(spec["source_run"] == EXPECTED_SOURCE[case], f"{case}: wrong paper checkpoint family")
        require(set(spec["seeds"]) == {"42", "43", "44"}, f"{case}: missing seed checkpoint")
        for seed in SEEDS:
            entry = spec["seeds"][str(seed)]
            expected = (f"{spec['source_run']}/seed_{seed}_corner_sw640_sh512/weights/best.pt")
            digest = entry["sha256"]
            require(entry["path"] == expected and entry["path"] not in seen,
                    f"{case}/seed{seed}: wrong/duplicate HF path")
            require(isinstance(digest, str) and len(digest) == 64
                    and all(ch in "0123456789abcdef" for ch in digest),
                    f"{case}/seed{seed}: SHA256 missing/malformed")
            seen.add(entry["path"])
    return raw["repo_id"], cases


def choose(requested: list | None, allowed: tuple, label: str) -> tuple:
    if not requested:
        return allowed
    require(len(set(requested)) == len(requested)
            and set(requested) <= set(allowed), f"Unknown/duplicate {label}: {requested}")
    return tuple(item for item in allowed if item in requested)


def data_yaml_for(template: str, seed: int) -> Path:
    require("{seed}" in template, "TinyPerson requires per-seed YAML template with {seed}")
    path = Path(template.format(seed=seed)).expanduser().resolve()
    require(path.name == "tinyperson.yaml"
            and path.parent.name == f"tinyperson_seed_{seed}_corner_sw640_sh512"
            and path.is_file(), f"Prepare TinyPerson seed {seed} corner split first: {path}")
    return path


def split_context(data_yaml: Path, seed: int) -> tuple[dict[str, Any], list[Path]]:
    import yaml
    from evaluate_test.size_bucket_evaluator import _label_path

    values = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    require(isinstance(values, dict) and all(isinstance(values.get(k), str)
            for k in ("path", "train", "val", "test")), f"Invalid TinyPerson YAML: {data_yaml}")
    base = Path(values["path"])
    base = base if base.is_absolute() else (data_yaml.parent / base).resolve()
    val = Path(values["val"])
    val = val if val.is_absolute() else base / val
    train = Path(values["train"])
    train = train if train.is_absolute() else base / train
    test = Path(values["test"])
    test = test if test.is_absolute() else base / test
    require(val.is_dir() and train.is_dir() and test.exists()
            and len({val.resolve(), train.resolve(), test.resolve()}) == 3,
            "TinyPerson train/val/test paths invalid or overlapping")
    manifest_path = data_yaml.parent / "corner_manifest.json"
    require(manifest_path.is_file(), f"Missing original-image split manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_records, train_records = manifest.get("val"), manifest.get("train")
    require(isinstance(val_records, list) and val_records
            and isinstance(train_records, list) and train_records,
            "Empty or invalid TinyPerson corner manifest")
    val_names = {record["file_name"] for record in val_records}
    train_names = {record["file_name"] for record in train_records}
    require(val_names.isdisjoint(train_names),
            "Original-image leakage between TinyPerson train and val")
    require(len({int(record["image_id"]) for record in val_records}) == len(val_records),
            "Duplicate validation corner IDs")
    required_names = {f"img_{record['image_id']}.jpg" for record in val_records}
    images = sorted(p for p in val.glob("*.jpg") if p.is_file())
    require({p.name for p in images} == required_names
            and len(images) == len(val_records),
            "Prepared TinyPerson val crops differ from original-image split manifest")
    require(not any(p.name in {t.name for t in train.glob("*.jpg")} for p in images),
            "TinyPerson train/val corner image names overlap")
    hashes, label_hashes = hashlib.sha256(), hashlib.sha256()
    for path in images:
        label = _label_path(path)
        require(label.is_file(), f"Missing TinyPerson validation label: {label}")
        hashes.update(str(path.resolve()).encode() + b"\n")
        label_hashes.update(str(label.resolve()).encode() + b"\0")
        label_hashes.update(sha256(label).encode() + b"\n")
    return {
        "seed": seed, "split": "val", "corner_geometry": "sw640_sh512",
        "coordinate_space": "corner_window",
        "data_yaml_sha256": sha256(data_yaml),
        "corner_manifest_sha256": sha256(manifest_path),
        "ordered_image_paths_sha256": hashes.hexdigest(),
        "ordered_label_contents_sha256": label_hashes.hexdigest(),
        "val_corner_count": len(images),
        "val_original_count": len(val_names),
        "train_original_count": len(train_names),
        "original_image_disjoint": True,
    }, images


def job_dir(project: Path, case: str, seed: int) -> Path:
    return project / case / f"seed_{seed}"


def pinned_hf_revision(repo: str) -> str:
    from huggingface_hub import HfApi
    revision = HfApi().repo_info(repo_id=repo, repo_type="dataset").sha
    require(isinstance(revision, str) and len(revision) >= 40, "Cannot pin Hugging Face revision")
    return revision


def restore(repo: str, spec: dict, seed: int, revision: str) -> Path:
    from huggingface_hub import hf_hub_download
    row = spec["seeds"][str(seed)]
    path = Path(hf_hub_download(repo_id=repo, repo_type="dataset",
                                revision=revision, filename=row["path"]))
    require(sha256(path) == row["sha256"], f"Checkpoint SHA mismatch: {row['path']}")
    return path


def validate_record(record: dict, cases: dict, repo: str, case: str, seed: int,
                    split: dict, hashes: dict) -> None:
    entry = cases[case]["seeds"][str(seed)]
    require(record.get("contract") == CONTRACT
            and record.get("dataset") == "TinyPerson"
            and record.get("case") == case and record.get("seed") == seed
            and record.get("source/hf_repo") == repo
            and record.get("source/checkpoint") == entry["path"]
            and record.get("source/checkpoint_sha256") == entry["sha256"]
            and record.get("source/evaluator_sha256") == hashes
            and record.get("split_provenance") == split
            and record.get("evaluation", {}).get("no_training") is True
            and record.get("evaluation", {}).get("split") == "val"
            and record.get("evaluation", {}).get("coordinate_space") == "corner_window"
            and record.get("evaluation", {}).get("imgsz") == 640
            and record.get("evaluation", {}).get("batch") == 8
            and record.get("evaluation", {}).get("native_nms_iou") == .5
            and record.get("evaluation", {}).get("small_nms_iou") == .5,
            f"{case}/seed{seed}: cached validation identity changed")
    for metric in METRICS:
        value = (record.get("metrics") or {}).get(metric)
        require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
                f"{case}/seed{seed}: invalid {metric}={value}")


def check_artifacts(project: Path, case: str, seed: int, record: dict) -> None:
    meta = record["evaluation"]
    for filename, hash_key in (("val_gt.json", "gt_sha256"),
                               ("val_predictions.json", "predictions_sha256")):
        path = job_dir(project, case, seed) / filename
        require(path.is_file() and sha256(path) == meta[hash_key],
                f"Missing/tampered TinyPerson validation artifact: {path}")


def evaluate_one(args: argparse.Namespace, cases: dict, repo: str, case: str,
                 seed: int, revision: str, context: tuple,
                 hashes: dict) -> dict:
    split, images, data_yaml = context
    run = job_dir(args.project, case, seed)
    target = run / "val_metrics.json"
    if target.exists():
        record = json.loads(target.read_text(encoding="utf-8"))
        validate_record(record, cases, repo, case, seed, split, hashes)
        check_artifacts(args.project, case, seed, record)
        require(record["source/hf_revision"] == revision,
                "HF revision differs from earlier validation batch")
        return {"case": case, "seed": seed, "status": "REUSED_VERIFIED"}
    checkpoint = restore(repo, cases[case], seed, revision)
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics, standard_metrics
    ensure_local_ultralytics()
    from ultralytics import YOLO
    from evaluate_test.size_bucket_evaluator import _predictions, _coco_ground_truth, _evaluate_coco
    run.mkdir(parents=True, exist_ok=True)
    require(not any((run / name).exists() for name in ("val_gt.json", "val_predictions.json")),
            f"Partial TinyPerson validation at {run}; inspect before retrying")
    result = YOLO(str(checkpoint)).val(
        data=str(data_yaml), split="val", imgsz=640, batch=8, rect=True,
        conf=.001, iou=.5, device=args.device, workers=args.workers,
        plots=False, verbose=False, project=str(run), name="ultralytics_val",
        exist_ok=True,
    )
    scores = standard_metrics(result, "val")
    # This uses the LEVIR/Varroa paper validation small-object protocol:
    # independent predictor defaults (conf=.25, rect/auto, NMS IoU=.5).
    predictions = _predictions(YOLO(str(checkpoint)), images,
                               imgsz=640, batch=8, device=args.device,
                               workers=args.workers)
    gt = _coco_ground_truth(images)
    gt_file, pred_file = run / "val_gt.json", run / "val_predictions.json"
    immutable_json(gt_file, gt)
    immutable_json(pred_file, predictions)
    size = _evaluate_coco(gt_file, pred_file)
    scores["val_size/AP50-Small"] = size["test_size/AP50-Small"]
    for key in METRICS:
        require(type(scores.get(key)) in (int, float) and math.isfinite(scores[key])
                and 0 <= scores[key] <= 1,
                f"{case}/seed{seed}: missing/invalid validation metric {key}={scores.get(key)}")
    record = {
        "contract": CONTRACT, "dataset": "TinyPerson", "case": case, "seed": seed,
        "sections": cases[case]["sections"],
        "source/hf_repo": repo, "source/hf_revision": revision,
        "source/checkpoint": cases[case]["seeds"][str(seed)]["path"],
        "source/checkpoint_sha256": cases[case]["seeds"][str(seed)]["sha256"],
        "source/evaluator_sha256": hashes,
        "split_provenance": split,
        "evaluation": {
            "no_training": True, "split": "val", "coordinate_space": "corner_window",
            "imgsz": 640, "batch": 8, "workers": args.workers, "device": args.device,
            "native_conf": .001, "native_rect": True, "native_nms_iou": .5,
            "small_conf": .25, "small_rect": True, "small_nms_iou": .5,
            "main": "Ultralytics YOLO.val on native seed-specific val corner crops",
            "small": "YOLO.predict on the SAME val corners -> TinyBenchmark Small 20-32px",
            "not_merged_original": True,
            "gt_sha256": sha256(gt_file), "predictions_sha256": sha256(pred_file),
            "gt_object_count": len(gt["annotations"]),
            "prediction_count": len(predictions),
        },
        "metrics": {key: float(scores[key]) for key in METRICS},
    }
    immutable_json(target, record)
    return {"case": case, "seed": seed, "status": "EVALUATED"}


def aggregate(args: argparse.Namespace, cases: dict, repo: str,
              chosen_cases: tuple, seeds: tuple,
              contexts: dict, hashes: dict) -> dict:
    rows, means, blockers, revisions = [], [], [], set()
    for case in chosen_cases:
        case_rows = []
        for seed in seeds:
            saved = job_dir(args.project, case, seed) / "val_metrics.json"
            if not saved.is_file():
                blockers.append({"case": case, "seed": seed, "reason": "Missing val_metrics.json"})
                continue
            try:
                rec = json.loads(saved.read_text(encoding="utf-8"))
                validate_record(rec, cases, repo, case, seed, contexts[seed][0], hashes)
                check_artifacts(args.project, case, seed, rec)
                revision = rec.get("source/hf_revision")
                require(isinstance(revision, str) and len(revision) >= 40,
                        "Unpinned HF source revision")
                revisions.add(revision)
                row = {"case": case, "seed": seed, "status": "VERIFIED",
                       "hf_checkpoint": rec["source/checkpoint"],
                       "checkpoint_sha256": rec["source/checkpoint_sha256"],
                       "val_split_seed": seed,
                       "val_yaml_sha256": contexts[seed][0]["data_yaml_sha256"]}
                for label, metric in zip(LABELS, METRICS):
                    row[label] = rec["metrics"][metric] * 100
                case_rows.append(row)
                rows.append(row)
            except Exception as exc:
                blockers.append({"case": case, "seed": seed, "reason": str(exc)})
        if len(case_rows) == len(seeds):
            row = {"case": case, "n": len(seeds),
                   "seeds": ",".join(str(s) for s in seeds)}
            for label in LABELS:
                values = [item[label] for item in case_rows]
                row[label + "_mean"] = statistics.mean(values)
                row[label + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else None
            means.append(row)
    if len(revisions) > 1:
        blockers.append({"reason": "Multiple HF revisions in one validation batch",
                         "revisions": sorted(revisions)})
    status = "COMPLETE" if not blockers and len(rows) == len(chosen_cases) * len(seeds) else "BLOCKED"
    report = {
        "contract": CONTRACT, "status": status,
        "expected_jobs": len(chosen_cases) * len(seeds), "valid_jobs": len(rows),
        "blockers": blockers, "hf_revisions": sorted(revisions),
        "coordinate_space": "corner_window", "split": "val",
        "per_seed_csv": str(args.project / "tinyperson_val_metrics_by_seed.csv"),
        "aggregate_csv": str(args.project / "tinyperson_val_metrics_aggregate.csv"),
        "split_provenance_by_seed": {str(seed): contexts[seed][0] for seed in seeds},
        "unmatched_ablation_warning": (
            "FTSC_OFF checkpoint used no-mosaic while canonical FTSC was trained with mosaic;"
            " comparison is not augmentation-matched"
        ),
    }
    if status == "COMPLETE":
        derived_csv(args.project / "tinyperson_val_metrics_by_seed.csv", rows,
                    ["case", "seed", "status", "hf_checkpoint", "checkpoint_sha256",
                     "val_split_seed", "val_yaml_sha256", *LABELS])
        derived_csv(args.project / "tinyperson_val_metrics_aggregate.csv", means,
                    ["case", "n", "seeds",
                     *[name + suffix for name in LABELS for suffix in ("_mean", "_sample_std")]])
    derived_json(args.project / "tinyperson_val_report_status.json", report)
    return report


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("plan", "evaluate", "aggregate", "complete"), default="plan")
    p.add_argument("--manifest", type=Path, default=REPORT_MANIFEST)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--data-yaml-template", default=str(ROOT / "datasets/tinyperson_seed_{seed}_corner_sw640_sh512/tinyperson.yaml"))
    p.add_argument("--cases", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--confirm-evaluate", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args(argv)
    require(args.imgsz == 640 and args.batch == 8,
            "TinyPerson validation contract fixes imgsz=640 and batch=8")
    require(args.workers >= 0, "workers must be nonnegative")
    repo, cases = manifest_cases(args.manifest)
    chosen = choose(args.cases, CASES, "case")
    seeds = choose(args.seeds, SEEDS, "seed")
    args.project = args.project.expanduser().resolve()
    jobs = [{"case": c, "seed": s, "checkpoint": cases[c]["seeds"][str(s)]["path"]}
            for c in chosen for s in seeds]
    if args.phase == "plan":
        report = {"contract": CONTRACT, "status": "PLAN_ONLY",
                  "coordinate_space": "corner_window",
                  "cases": list(chosen), "seeds": list(seeds),
                  "expected_jobs": len(jobs), "jobs": jobs,
                  "excluded": ["augmentation_compatibility"],
                  "warning": "Validation corner-crop metrics are not merged-image TEST metrics"}
        print(json.dumps(report, indent=2))
        return report
    contexts = {}
    for seed in seeds:
        data_yaml = data_yaml_for(args.data_yaml_template, seed)
        split, images = split_context(data_yaml, seed)
        contexts[seed] = (split, images, data_yaml)
    hashes = source_hashes()
    if args.phase in ("evaluate", "complete"):
        require(args.confirm_evaluate, "Explicit --confirm-evaluate required for GPU validation")
        revision = pinned_hf_revision(repo)
        failures = []
        for index, job in enumerate(jobs, 1):
            case, seed = job["case"], job["seed"]
            print(f"[TINYPERSON VAL {index}/{len(jobs)}] {case}/seed_{seed}", flush=True)
            try:
                print(json.dumps(evaluate_one(args, cases, repo, case, seed,
                                              revision, contexts[seed], hashes)), flush=True)
            except Exception as exc:
                failure = {"case": case, "seed": seed, "reason": str(exc)}
                failures.append(failure)
                print(json.dumps({"status": "FAILED", **failure}), flush=True)
                traceback.print_exc()
                if args.stop_on_error:
                    raise
        if args.phase == "evaluate":
            result = {"status": "EVALUATION_FINISHED" if not failures else
                      "EVALUATION_FINISHED_WITH_BLOCKERS", "failures": failures,
                      "expected_jobs": len(jobs)}
            print(json.dumps(result, indent=2))
            return result
    result = aggregate(args, cases, repo, chosen, seeds, contexts, hashes)
    print(json.dumps(result, indent=2))
    if result["status"] != "COMPLETE":
        raise RuntimeError("TinyPerson validation incomplete; see tinyperson_val_report_status.json")
    return result


if __name__ == "__main__":
    main()
