"""Varroa paper validation-only re-evaluation, 13 cases x seeds 42/43/44.

Native main AP: YOLO.val(split='val', imgsz=640, batch=4, rect=True,
conf=.001, iou=.7, max_det=300). Independent Small AP: YOLO.predict
(imgsz=640, batch=4, default conf=.25, iou=.5) evaluated class-agnostically
with the project's TinyBenchmark size evaluator (20^2<=area<32^2).

Varroa has a SEED-SPECIFIC prepared dataset for EVERY checkpoint seed.
No training, test evaluation, HF publication, or overwrite of archived metrics.
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
MANIFEST = ROOT / "utils/ftsc_varroa_validation_cases_v1.json"
DEFAULT_PROJECT = ROOT / "runs/ftsc_varroa_validation_paper_v1"
CONTRACT = "FTSC_VARROA_PAPER_VAL_3METRICS_20261115_V1"
METRICS = ("val/metrics/mAP50(B)", "val/metrics/mAP50-95(B)", "val_size/AP50-Small")
LABELS = ("Val_AP50_pct", "Val_mAP50_95_pct", "Val_AP50_small_pct")
SEEDS = (42, 43, 44)
CASES = (
    "FTSC", "FTSC_YOLOV9T", "FTSC_YOLO11N", "FTSC_OFF",
    "HEAD_H0", "HEAD_H1", "HEAD_H2", "HEAD_H3",
    "EDGE_P2", "EDGE_P3", "COLOR_CUE", "ALPHA4", "ALPHA8",
)
SOURCE_FILES = (
    "utils/ftsc_varroa_validation_paper_v1.py",
    "utils/ftsc_varroa_validation_cases_v1.json",
    "evaluate_test/standard_detection_metrics.py",
    "evaluate_test/size_bucket_evaluator.py",
    "vendor/tinyperson_cocoeval.py",
    "misc/prepare_dataset.py",
    "train_levir_scripts/train_varroa_dbss_hit.py",
    "models_related/ultralytics/ultralytics/engine/model.py",
    "models_related/ultralytics/ultralytics/engine/predictor.py",
    "models_related/ultralytics/ultralytics/models/yolo/detect/val.py",
    "models_related/ultralytics/ultralytics/nn/modules/ftsc.py",
)


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    return {rel: sha256(ROOT / rel) for rel in SOURCE_FILES}


def new_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def derived_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name + ".new")
    need(not target.exists(), f"Unresolved interrupted aggregate write: {target}")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    target.replace(path)


def derived_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.name + ".new")
    need(not target.exists(), f"Unresolved interrupted aggregate write: {target}")
    with target.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    target.replace(path)


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    need(data.get("schema") == "ftsc_varroa_validation_all_paper_cases_v1"
         and data.get("dataset") == "Varroa"
         and data.get("seeds") == list(SEEDS)
         and data.get("scope", "").endswith("excludes augmentation compatibility"),
         "Unexpected Varroa validation manifest schema/scope/seeds")
    policy = data.get("split_policy", {})
    need(policy == {
        "seed_specific": True, "gt_source": "gt_one", "only_positives": True,
        "class_policy": "map-3-to-1",
        "expected_counts": {"train": 2762, "val": 592, "test": 592},
    }, "Varroa preparation protocol changed")
    need(set(data.get("cases", {})) == set(CASES), "Paper Varroa requires exactly 13 cases")
    seen_paths = set()
    for case in CASES:
        entry = data["cases"][case]
        need(set(entry.get("checkpoints", {})) == {"42", "43", "44"},
             f"{case}: missing checkpoint seed")
        for seed in SEEDS:
            record = entry["checkpoints"][str(seed)]
            path = f'{entry["source_run"]}/seed_{seed}/weights/best.pt'
            digest = record.get("sha256")
            need(record.get("path") == path and path.startswith("Varroa/")
                 and path not in seen_paths,
                 f"Unexpected/duplicate source checkpoint: {case}/{seed}")
            need(isinstance(digest, str) and len(digest) == 64
                 and all(x in "0123456789abcdef" for x in digest),
                 f"Unpinned checkpoint: {case}/{seed}")
            seen_paths.add(path)
    need(data["cases"]["HEAD_H2"]["source_run"].endswith("/VR_STRUCT_H2_P23")
         and data["cases"]["FTSC"]["source_run"].endswith("/V1_FTSC"),
         "Varroa structural P2/P3 must not be confused with canonical FTSC")
    need(data["cases"]["EDGE_P2"]["source_run"].endswith("/EDGE_P2_DETECT_ONLY")
         and data["cases"]["EDGE_P3"]["source_run"].endswith("/EDGE_P3_DETECT_ONLY"),
         "Keep Edge P2/P3 distinct")
    return data


def select_cases(requested: list[str] | None) -> tuple[str, ...]:
    if not requested:
        return CASES
    need(len(set(requested)) == len(requested) and set(requested) <= set(CASES),
         f"Unknown/duplicate Varroa case: {requested}")
    return tuple(case for case in CASES if case in requested)


def select_seeds(requested: list[int] | None) -> tuple[int, ...]:
    if not requested:
        return SEEDS
    need(len(set(requested)) == len(requested) and set(requested) <= set(SEEDS),
         f"Unknown/duplicate Varroa seed: {requested}")
    return tuple(seed for seed in SEEDS if seed in requested)


def yaml_for_seed(template: str, seed: int) -> Path:
    need("{seed}" in template, "Varroa MUST use a seed-specific YAML template")
    path = Path(template.format(seed=seed)).expanduser().resolve()
    need(path.parent.name == f"varroa_yolo_seed{seed}"
         and path.name == "varroa.yaml" and path.is_file(),
         f"Missing/misplaced Varroa seed_{seed} YAML: {path}")
    return path


def val_images(data_yaml: Path) -> list[Path]:
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    need(isinstance(cfg, dict) and isinstance(cfg.get("val"), str),
         f"Invalid Varroa YAML val entry: {data_yaml}")
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    rel = Path(cfg["val"])
    target = rel if rel.is_absolute() else root / rel
    need(target.exists(), f"Varroa validation images missing: {target}")
    if target.is_file():
        paths = [Path(line.strip()) for line in target.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        paths = [p if p.is_absolute() else (target.parent / p).resolve() for p in paths]
    else:
        paths = sorted(p for p in target.rglob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    need(len(paths) == 592 and len({p.resolve() for p in paths}) == 592
         and all(p.is_file() for p in paths),
         f"Varroa val must contain exactly 592 valid unique images: {target}")
    return paths


def split_provenance(path: Path, seed: int) -> tuple[dict[str, Any], list[Path]]:
    import yaml
    from evaluate_test.size_bucket_evaluator import _label_path, _resolve_test_images
    from train_levir_scripts.train_varroa_dbss_hit import validate_dataset

    validate_dataset(path)  # seed-specific 2762/592/592 count guard
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    need(cfg.get("seed") == seed and cfg.get("gt_source") == "gt_one"
         and cfg.get("only_positives") is True and cfg.get("class_policy") == "map-3-to-1"
         and cfg.get("split_counts") == {"train": 2762, "val": 592, "test": 592},
         f"Varroa source/GT/seed preparation policy differs: {path}")
    need(cfg.get("val") != cfg.get("test"), f"Varroa val and test identical in YAML: {path}")
    imgs = val_images(path)
    test = _resolve_test_images(path)
    need(len(test) == 592 and not ({p.resolve() for p in imgs} & {p.resolve() for p in test}),
         f"Varroa val/test file overlap or test split invalid: {path}")
    labels_hash = hashlib.sha256()
    paths_hash = hashlib.sha256()
    images_hash = hashlib.sha256()
    for image in imgs:
        label = _label_path(image)
        need(label.is_file(), f"Missing Varroa validation YOLO labels: {label}")
        ipath = str(image.resolve()).encode("utf-8")
        labels_hash.update(str(label.resolve()).encode("utf-8") + b"\0")
        labels_hash.update(sha256(label).encode("ascii") + b"\n")
        paths_hash.update(ipath + b"\n")
        images_hash.update(ipath + b"\0" + sha256(image).encode("ascii") + b"\n")
    return {
        "seed": seed, "split": "val",
        "data_yaml_sha256": sha256(path),
        "ordered_image_paths_sha256": paths_hash.hexdigest(),
        "ordered_image_bytes_sha256": images_hash.hexdigest(),
        "ordered_label_contents_sha256": labels_hash.hexdigest(),
        "val_image_count": len(imgs), "test_image_count": len(test),
        "preparation": "gt_one / only_positives / map-3-to-1 / seed-specific",
    }, imgs


def job_dir(project: Path, case: str, seed: int) -> Path:
    return project / case / f"seed_{seed}"


def hf_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi
    revision = HfApi().repo_info(repo_id=repo_id, repo_type="dataset").sha
    need(isinstance(revision, str) and len(revision) >= 40,
         "Hugging Face dataset revision unavailable; no checkpoint restore")
    return revision


def fetch_checkpoint(manifest: dict, case: str, seed: int, revision: str) -> Path:
    from huggingface_hub import hf_hub_download
    row = manifest["cases"][case]["checkpoints"][str(seed)]
    path = Path(hf_hub_download(
        repo_id=manifest["repo_id"], repo_type="dataset", revision=revision,
        filename=row["path"]))
    need(sha256(path) == row["sha256"],
         f"Varroa checkpoint SHA256 mismatch: {case}/seed_{seed}")
    return path


def verified_record(rec: dict[str, Any], manifest: dict[str, Any],
                    case: str, seed: int, split: dict,
                    sources: dict[str, str]) -> None:
    want = manifest["cases"][case]["checkpoints"][str(seed)]
    need(rec.get("contract") == CONTRACT and rec.get("dataset") == "Varroa"
         and rec.get("case") == case and rec.get("seed") == seed
         and rec.get("source/hf_repo") == manifest["repo_id"]
         and rec.get("source/checkpoint") == want["path"]
         and rec.get("source/checkpoint_sha256") == want["sha256"]
         and rec.get("source/evaluator_sha256") == sources
         and rec.get("split_provenance") == split
         and rec.get("evaluation", {}).get("no_training") is True
         and rec.get("evaluation", {}).get("split") == "val"
         and rec.get("evaluation", {}).get("imgsz") == 640
         and rec.get("evaluation", {}).get("batch") == 4
         and rec.get("evaluation", {}).get("native_nms_iou") == .7
         and rec.get("evaluation", {}).get("small_nms_iou") == .5,
         f"Varroa cached result identity mismatch: {case}/seed_{seed}")
    for key in METRICS:
        value = (rec.get("metrics") or {}).get(key)
        need(type(value) in (int, float) and math.isfinite(value)
             and 0 <= value <= 1, f"Invalid Varroa {case}/seed_{seed}/{key}={value}")


def check_artifacts(rec: dict, project: Path, case: str, seed: int) -> None:
    run = job_dir(project, case, seed)
    meta = rec["evaluation"]
    for file, key in (("val_gt.json", "gt_sha256"),
                      ("val_predictions.json", "predictions_sha256")):
        path = run / file
        need(path.is_file() and sha256(path) == meta[key],
             f"Varroa prediction/GT artifact changed: {path}")


def evaluate_one(args: argparse.Namespace, manifest: dict,
                 case: str, seed: int, revision: str, split: dict,
                 images: list[Path], yaml_path: Path,
                 sources: dict[str, str]) -> dict[str, Any]:
    target = job_dir(args.project, case, seed) / "val_metrics.json"
    if target.exists():
        rec = json.loads(target.read_text(encoding="utf-8"))
        verified_record(rec, manifest, case, seed, split, sources)
        check_artifacts(rec, args.project, case, seed)
        need(rec["source/hf_revision"] == revision,
             f"Varroa HF revision drifted; refuse cached result: {target}")
        return {"case": case, "seed": seed, "status": "REUSED_VERIFIED"}
    checkpoint = fetch_checkpoint(manifest, case, seed, revision)
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics, standard_metrics
    fork = ensure_local_ultralytics()
    from ultralytics import YOLO
    from evaluate_test.size_bucket_evaluator import _predictions, _coco_ground_truth, _evaluate_coco

    directory = job_dir(args.project, case, seed)
    directory.mkdir(parents=True, exist_ok=True)
    need(not any((directory / f).exists() for f in ("val_gt.json", "val_predictions.json")),
         f"Uncommitted Varroa validation artifacts at {directory}; inspect before retry")
    native = YOLO(str(checkpoint)).val(
        data=str(yaml_path), split="val",
        imgsz=640, batch=4, rect=True,
        conf=.001, iou=.7, max_det=300,
        device=args.device, workers=args.workers,
        plots=False, verbose=False, project=str(directory),
        name="ultralytics_val", exist_ok=True,
    )
    scores = standard_metrics(native, "val")
    detections = _predictions(YOLO(str(checkpoint)), images,
                              imgsz=640, batch=4, device=args.device,
                              workers=args.workers)
    gt = _coco_ground_truth(images)
    gt_file = directory / "val_gt.json"
    pred_file = directory / "val_predictions.json"
    new_json(gt_file, gt)
    new_json(pred_file, detections)
    small_result = _evaluate_coco(gt_file, pred_file)
    scores[METRICS[2]] = small_result["test_size/AP50-Small"]
    for key in METRICS:
        x = scores.get(key)
        need(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1,
             f"Invalid Varroa score: {case}/seed_{seed}/{key}={x}")
    rec = {
        "contract": CONTRACT, "dataset": "Varroa", "case": case, "seed": seed,
        "sections": manifest["cases"][case]["sections"],
        "source/hf_repo": manifest["repo_id"],
        "source/hf_revision": revision,
        "source/checkpoint": manifest["cases"][case]["checkpoints"][str(seed)]["path"],
        "source/checkpoint_sha256": manifest["cases"][case]["checkpoints"][str(seed)]["sha256"],
        "source/evaluator_sha256": sources,
        "source/local_ultralytics": fork,
        "split_provenance": split,
        "evaluation": {
            "no_training": True, "split": "val", "imgsz": 640, "batch": 4,
            "workers": args.workers, "device": args.device,
            "native_nms_iou": .7, "native_conf": .001, "native_rect": True,
            "small_nms_iou": .5, "small_conf": .25, "small_rect": True,
            "main_protocol": "Ultralytics native YOLO.val on Varroa val split",
            "small_protocol": "independent YOLO.predict on Varroa val + TinyBenchmark Small 20-32px",
            "gt_sha256": sha256(gt_file), "predictions_sha256": sha256(pred_file),
            "gt_object_count": len(gt["annotations"]),
            "prediction_count": len(detections),
        },
        "metrics": {k: float(scores[k]) for k in METRICS},
    }
    new_json(target, rec)
    return {"case": case, "seed": seed, "status": "EVALUATED"}


def aggregate(args: argparse.Namespace, manifest: dict, cases: tuple[str, ...],
              seeds: tuple[int, ...], contexts: dict[int, tuple[dict, list[Path], Path]],
              sources: dict[str, str]) -> dict[str, Any]:
    rows: list[dict] = []
    means: list[dict] = []
    failures: list[dict] = []
    revisions = set()
    for case in cases:
        case_rows = []
        for seed in seeds:
            target = job_dir(args.project, case, seed) / "val_metrics.json"
            if not target.is_file():
                failures.append({"case": case, "seed": seed, "reason": "missing val_metrics.json"})
                continue
            try:
                record = json.loads(target.read_text(encoding="utf-8"))
                verified_record(record, manifest, case, seed, contexts[seed][0], sources)
                check_artifacts(record, args.project, case, seed)
                revisions.add(record["source/hf_revision"])
                row = {"case": case, "seed": seed, "status": "VERIFIED",
                       "hf_checkpoint": record["source/checkpoint"],
                       "checkpoint_sha256": record["source/checkpoint_sha256"],
                       "val_split_seed": seed,
                       "val_yaml_sha256": contexts[seed][0]["data_yaml_sha256"]}
                for label, metric in zip(LABELS, METRICS):
                    row[label] = record["metrics"][metric] * 100.0
                rows.append(row)
                case_rows.append(row)
            except Exception as exc:
                failures.append({"case": case, "seed": seed, "reason": str(exc)})
        if len(case_rows) == len(seeds):
            agg = {"case": case, "n": len(seeds),
                   "seeds": ",".join(str(s) for s in seeds)}
            for label in LABELS:
                values = [row[label] for row in case_rows]
                agg[label + "_mean"] = statistics.mean(values)
                agg[label + "_sample_std"] = (
                    statistics.stdev(values) if len(values) > 1 else None)
            means.append(agg)
    if len(revisions) > 1:
        failures.append({"reason": "Mixed HF revisions in Varroa batch",
                         "revisions": sorted(revisions)})
    status = "COMPLETE" if not failures and len(rows) == len(cases) * len(seeds) else "BLOCKED"
    report = {
        "contract": CONTRACT, "status": status, "expected_jobs": len(cases) * len(seeds),
        "valid_jobs": len(rows), "blockers": failures, "hf_revisions": sorted(revisions),
        "per_seed_csv": str(args.project / "varroa_val_metrics_by_seed.csv"),
        "aggregate_csv": str(args.project / "varroa_val_metrics_aggregate.csv"),
        "split_provenance_by_seed": {str(seed): contexts[seed][0] for seed in seeds},
    }
    if status == "COMPLETE":
        derived_csv(args.project / "varroa_val_metrics_by_seed.csv", rows,
                    ["case", "seed", "status", "hf_checkpoint", "checkpoint_sha256",
                     "val_split_seed", "val_yaml_sha256", *LABELS])
        derived_csv(args.project / "varroa_val_metrics_aggregate.csv", means,
                    ["case", "n", "seeds",
                     *[label + suffix for label in LABELS
                       for suffix in ("_mean", "_sample_std")]])
    derived_json(args.project / "varroa_val_report_status.json", report)
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "evaluate", "aggregate", "complete"),
                        default="plan")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--data-yaml-template", default="")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--confirm-evaluate", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    cases = select_cases(args.cases)
    seeds = select_seeds(args.seeds)
    args.project = args.project.expanduser().resolve()
    need(args.imgsz == 640 and args.batch == 4,
         "Varroa native validation contract fixes imgsz=640 batch=4")
    need(args.workers >= 0, "Invalid negative workers count")
    jobs = [{"case": c, "seed": s,
             "checkpoint": manifest["cases"][c]["checkpoints"][str(s)]["path"]}
            for c in cases for s in seeds]
    if args.phase == "plan":
        report = {"contract": CONTRACT, "status": "PLAN_ONLY",
                  "cases": list(cases), "seeds": list(seeds),
                  "expected_jobs": len(jobs), "jobs": jobs,
                  "excluded": ["augmentation_compatibility"]}
        print(json.dumps(report, indent=2))
        return report
    need(args.data_yaml_template,
         "Varroa requires --data-yaml-template with {seed} placeholder")
    contexts = {}
    for seed in seeds:
        data_yaml = yaml_for_seed(args.data_yaml_template, seed)
        split, images = split_provenance(data_yaml, seed)
        contexts[seed] = (split, images, data_yaml)
    hashes = source_hashes()
    if args.phase in ("evaluate", "complete"):
        need(args.confirm_evaluate,
             "GPU validation requires explicit --confirm-evaluate")
        revision = hf_revision(manifest["repo_id"])
        errors = []
        for index, job in enumerate(jobs, 1):
            case, seed = job["case"], job["seed"]
            print(f"[VARROA VAL {index}/{len(jobs)}] {case}/seed_{seed}", flush=True)
            try:
                split, images, yaml_path = contexts[seed]
                out = evaluate_one(args, manifest, case, seed, revision,
                                   split, images, yaml_path, hashes)
                print(json.dumps(out), flush=True)
            except Exception as exc:
                failure = {"case": case, "seed": seed, "reason": str(exc)}
                errors.append(failure)
                print(json.dumps({"status": "FAILED", **failure}), flush=True)
                traceback.print_exc()
                if args.stop_on_error:
                    raise
        if args.phase == "evaluate":
            status = "EVALUATION_FINISHED" if not errors else "EVALUATION_FINISHED_WITH_BLOCKERS"
            report = {"status": status, "expected_jobs": len(jobs), "failures": errors}
            print(json.dumps(report, indent=2))
            return report
    report = aggregate(args, manifest, cases, seeds, contexts, hashes)
    print(json.dumps(report, indent=2))
    if report["status"] != "COMPLETE":
        raise RuntimeError("Varroa val re-evaluation incomplete: see varroa_val_report_status.json")
    return report


if __name__ == "__main__":
    main()
