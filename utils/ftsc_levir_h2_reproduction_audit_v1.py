"""LEVIR H2 reproduction audit: read-only source inspection and evaluation-only parity.

Does NOT train, resume, upload, overwrite or modify archived source artifacts.
Output: runs/levir_h2_reproduction_audit_v1/<cohort>/seed_N/... .
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/levir_h2_reproduction_audit_v1"
PAPER_MANIFEST = ROOT / "utils/ftsc_levir_validation_cases_v1.json"
HIST = ROOT / "reference/levir_h2_historical_head_pruning_seed42_43_44.csv"
REPO = "TLMHoang/FTSC_results"
SEEDS = (42, 43, 44)
SOURCES = {
    "RECON_H2_V1": "LEVIR-Ship/historical_reconstructions/RECON_H2_V1",
    "FACTORIAL_P23_FTSC": "LEVIR-Ship/journal_experiments/head_factorial_v1/P23_FTSC",
    "FACTORIAL_P23_NOFTSC": "LEVIR-Ship/journal_experiments/head_factorial_v1/P23_NOFTSC",
}
METRIC_KEYS = ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)", "test_size/AP50-Small")
METRIC_LABELS = ("AP50", "mAP50_95", "AP50_small")
METADATA = ("args.yaml", "config.yaml", "factorial_contract.json", "experiment_manifest.json",
            "training_completion.json", "factorial_completion.json",
            "evaluation_metrics.json", "evaluation_metrics_extended.json",
            "checkpoint_reference.json", "results.csv")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def immutable_json(path, obj):
    # Compare the JSON representation: integer keys become strings on disk.
    obj = json.loads(json.dumps(obj, sort_keys=True, default=str))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        require(prior == obj, f"Existing output differs; preserve it: {path}")
        return
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def canonical_source_sha():
    manifest = json.loads(PAPER_MANIFEST.read_text())
    return {int(s): v["sha256"] for s, v in manifest["cases"]["FTSC"]["checkpoints"].items()}


def historical():
    rows = {}
    with HIST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            require(row["variant"] == "H2_p2_p3_nomosaic", "Historical case mismatch")
            s = int(row["seed"])
            rows[s] = {label: 100.0 * float(row[key])
                       for label, key in zip(METRIC_LABELS[:2], METRIC_KEYS[:2])}
    require(set(rows) == set(SEEDS), "Historical source must contain exactly 42/43/44")
    return rows


def selection_id(args):
    payload = json.dumps({"cases": args.cases, "seeds": args.seeds}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def inventory(args):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    rev = api.repo_info(repo_id=REPO, repo_type="dataset").sha
    all_paths = set(api.list_repo_files(repo_id=REPO, repo_type="dataset", revision=rev))
    expected = canonical_source_sha()
    ref = historical()
    data = {"schema": "levir_h2_repro_audit_v1", "repo": REPO, "revision": rev,
            "historical_reference_sha256": sha(HIST), "historical_per_seed_pct": ref,
            "cases": {}}
    for cohort in args.cases:
        data["cases"][cohort] = {}
        for seed in args.seeds:
            base = f"{SOURCES[cohort]}/seed_{seed}"
            pt = f"{base}/weights/best.pt"
            row = {"checkpoint": pt, "exists": pt in all_paths, "metadata": {},
                   "status": "FOUND" if pt in all_paths else "MISSING"}
            for name in METADATA:
                source = f"{base}/{name}"
                if source not in all_paths:
                    continue
                downloaded = Path(hf_hub_download(
                    repo_id=REPO, repo_type="dataset", filename=source, revision=rev))
                if downloaded.stat().st_size > 5_000_000:
                    row["metadata"][name] = {"sha256": sha(downloaded), "large": True}
                    continue
                text = downloaded.read_text(encoding="utf-8", errors="replace")
                if name.endswith(".json"):
                    try:
                        parsed = json.loads(text)
                    except ValueError:
                        parsed = {"unparsed": text[:4000]}
                elif name.endswith(".yaml"):
                    import yaml
                    parsed = yaml.safe_load(text)
                else:
                    parsed = {"lines": text.splitlines()[-4:]}
                row["metadata"][name] = {"sha256": sha(downloaded), "data": parsed}
            if cohort == "RECON_H2_V1":
                row["expected_sha256"] = expected[seed]
            data["cases"][cohort][str(seed)] = row
    inventory_path = args.project / f"inventory_{rev[:12]}_{selection_id(args)}.json"
    immutable_json(inventory_path, data)
    print(json.dumps({"phase": "inventory", "revision": rev,
                      "path": str(inventory_path),
                      "summary": {c: {s: v["status"] for s, v in rows.items()}
                                  for c, rows in data["cases"].items()}}, indent=2))
    return data


def test_images(data_yaml):
    from evaluate_test.size_bucket_evaluator import _resolve_test_images
    images = _resolve_test_images(data_yaml)
    require(len(images) == 788, f"Expected historical 788 test images, got {len(images)}")
    return images


def evaluate(args, info):
    from huggingface_hub import hf_hub_download
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    from evaluate_test.size_bucket_evaluator import _coco_ground_truth, _predictions, _evaluate_coco
    ensure_local_ultralytics()
    from ultralytics import YOLO
    require(args.data_yaml.is_file(), f"Missing prepared LEVIR YAML: {args.data_yaml}")
    images = test_images(args.data_yaml)
    ref = historical()
    output_rows = []
    for cohort in args.cases:
        for seed in args.seeds:
            item = info["cases"][cohort][str(seed)]
            require(item["exists"], f"Source checkpoint absent: {item['checkpoint']}")
            out = args.project / cohort / f"seed_{seed}"
            filename = out / "evaluation_parity.json"
            if filename.exists():
                prev = json.loads(filename.read_text())
                require(prev["source"]["hf_revision"] == info["revision"] and
                        prev["source"]["checkpoint"] == item["checkpoint"],
                        f"Existing parity result has different source: {filename}")
                record = prev
            else:
                pt = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
                                          filename=item["checkpoint"], revision=info["revision"]))
                digest = sha(pt)
                if cohort == "RECON_H2_V1":
                    require(digest == item["expected_sha256"], "RECON H2 pinned checkpoint SHA mismatch")
                # Report SHA for new factorial too (not pre-certified as historical).
                model = YOLO(str(pt))
                native = model.val(data=str(args.data_yaml), split="test", imgsz=512,
                                   batch=8, rect=True, conf=.001, iou=.5,
                                   device=args.device, workers=args.workers,
                                   plots=False, verbose=False,
                                   project=str(out), name="native_test", exist_ok=True)
                ap50 = float(native.results_dict["metrics/mAP50(B)"])
                map95 = float(native.results_dict["metrics/mAP50-95(B)"])
                # Historical AP50-small: independent predict path (default conf=.25).
                detections = _predictions(YOLO(str(pt)), images, imgsz=512, batch=8,
                                           device=args.device, workers=args.workers)
                gt = _coco_ground_truth(images)
                gt_path, pred_path = out / "small_gt.json", out / "small_predictions.json"
                immutable_json(gt_path, gt)
                immutable_json(pred_path, detections)
                bucket = _evaluate_coco(gt_path, pred_path)
                small = float(bucket["test_size/AP50-Small"])
                require(all(math.isfinite(v) and 0 <= v <= 1 for v in (ap50, map95, small)),
                        "Invalid metrics from evaluation")
                record = {
                    "source": {"repo": REPO, "hf_revision": info["revision"],
                               "checkpoint": item["checkpoint"], "checkpoint_sha256": digest},
                    "data_yaml_sha256": sha(args.data_yaml), "test_image_count": len(images),
                    "small_gt_sha256": sha(gt_path), "small_predictions_sha256": sha(pred_path),
                    "evaluation_contract": {
                        "native": "YOLO.val test imgsz512 batch8 rect=True conf=.001 iou=.5",
                        "small": "YOLO.predict native test imgsz512 batch8 conf default=.25 iou=.5; TinyBenchmark",
                        "no_training": True},
                    "metrics_pct": {"AP50": 100*ap50, "mAP50_95": 100*map95, "AP50_small": 100*small},
                    "historical_H2_reference_pct": ref[seed] if cohort == "RECON_H2_V1" else None,
                }
                immutable_json(filename, record)
            output_rows.append({"cohort": cohort, "seed": seed, **record["metrics_pct"],
                                "checkpoint_sha256": record["source"]["checkpoint_sha256"]})
            print(f"[EVALUATED/REUSED] {cohort}/{seed}: {record['metrics_pct']}", flush=True)
    summaries = []
    for cohort in args.cases:
        subset = [r for r in output_rows if r["cohort"] == cohort]
        item = {"cohort": cohort, "n": len(subset)}
        for metric in METRIC_LABELS:
            vals = [r[metric] for r in subset]
            item[f"{metric}_mean"] = statistics.fmean(vals)
            item[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
        summaries.append(item)
    lookup = {(row["cohort"], row["seed"]): row for row in output_rows}
    paired_effects = []
    if all((cohort, seed) in lookup for cohort in
           ("FACTORIAL_P23_FTSC", "FACTORIAL_P23_NOFTSC") for seed in args.seeds):
        for metric in METRIC_LABELS:
            deltas = [lookup[("FACTORIAL_P23_FTSC", seed)][metric] -
                      lookup[("FACTORIAL_P23_NOFTSC", seed)][metric] for seed in args.seeds]
            paired_effects.append({"metric": metric, "mean_delta_pp": statistics.fmean(deltas),
                                   "sample_std_pp": statistics.stdev(deltas) if len(deltas)>1 else None,
                                   "per_seed_pp": dict(zip(map(str, args.seeds), deltas))})
    result = {"rows": output_rows, "aggregate": summaries,
              "factorial_p23_matched_effects": paired_effects,
              "note": "RECON_H2 is NEVER substituted as a factorial arm."}
    immutable_json(args.project / f"summary_{info['revision'][:12]}_{selection_id(args)}.json", result)
    print(json.dumps({"aggregate": summaries, "paired_effects": paired_effects}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("inventory", "evaluate"), default="inventory")
    p.add_argument("--confirm-evaluate", action="store_true")
    p.add_argument("--cases", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--data-yaml", type=Path,
                   default=ROOT/"datasets/levir_ship_yolo_seed42/levir_ship.yaml")
    p.add_argument("--project", type=Path, default=OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args(argv)
    require(len(args.cases) == len(set(args.cases)), "Duplicate cases")
    require(len(args.seeds) == len(set(args.seeds)) and set(args.seeds) <= set(SEEDS),
            "Invalid seed selection")
    require(args.phase != "evaluate" or args.confirm_evaluate, "Evaluation requires --confirm-evaluate")
    info = inventory(args)
    if args.phase == "evaluate":
        evaluate(args, info)


if __name__ == "__main__":
    main()
