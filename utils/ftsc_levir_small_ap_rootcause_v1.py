#!/usr/bin/env python3
"""Investigate LEVIR-Ship paper AP50-Small vs deployment V3 without changing V3.

Scientific controls:
  replay: filter the SAME saved V3 validator detections across confidence
          thresholds. GT and TinyBenchmark evaluator remain bitwise identical.
  historical: reproduce historical evaluate_native_test_size_buckets() using
          YOLO.predict(imgsz=512,batch=8,rect=default,conf=default 0.25) on
          SHA-pinned PT32 checkpoints; additionally predict(conf=0.001) with
          every other argument unchanged.
This distinguishes effects of score threshold from predictor vs validator
input/preprocessing and residual NMS/box-serialization differences. It does
NOT establish the last cause without comparing the decomposed measurements.

Only writes timestamped analysis files, never overwrites deployment metrics,
checkpoints, original ground truth or paper values. No training/HF upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils import ftsc_deployment_backend_eval_v1 as deploy

DEFAULT_CASES = ("FTSC", "FTSC_OFF", "FTSC_YOLOV9T", "FTSC_YOLO11N")
DEFAULT_THRESHOLDS = (0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
LEGACY_CONF = 0.25
PAPER_REFERENCE_MEANS = {
    # These values are report snapshots used for context, NOT a replay ground truth.
    "FTSC": 82.77, "FTSC_OFF": 81.41,
    "FTSC_YOLOV9T": 78.26, "FTSC_YOLO11N": 80.21,
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metric_records(root: Path, case: str, seed: int, expected: dict) -> tuple[dict, dict, list]:
    folder = deploy.path_for(root, "LEVIR-Ship", case, seed, "pt32")
    metric_file = folder / "metrics.json"
    gt_file = folder / "test_size_ground_truth.json"
    pred_file = folder / "test_size_predictions.json"
    for path in (metric_file, gt_file, pred_file):
        require(path.is_file(), f"V3 source artifact missing: {path}")
    record = json.loads(metric_file.read_text(encoding="utf-8"))
    require(record.get("source_sha256") == expected["sha256"],
            "PT32 best.pt differs from manifest-pinned expected SHA")
    require(record.get("evaluation_protocol") == deploy.RECT_PARITY_PROTOCOL and
            record.get("runtime_revision") == deploy.RECT_PARITY_RUNTIME_REVISION and
            record.get("backend") == "pt32" and record.get("case") == case and
            int(record.get("seed", -1)) == seed,
            "Wrong protocol/case/seed in native V3 metrics record")
    require(sha256(gt_file) == record["gt_sha256"] and
            sha256(pred_file) == record["predictions_sha256"],
            "Saved V3 size evaluator GT/predictions were modified")
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    predictions = json.loads(pred_file.read_text(encoding="utf-8"))
    images = gt.get("images", [])
    require(len(images) == 788 and len({x["id"] for x in images}) == 788,
            "LEVIR frozen test must contain exactly 788 unique images")
    require(len(gt.get("annotations", [])) > 0 and
            isinstance(predictions, list) and predictions,
            "Saved V3 GT or predictions incomplete")
    known = {x["id"] for x in images}
    for index, det in enumerate(predictions):
        require(det.get("image_id") in known and
                det.get("category_id") == 1 and
                len(det.get("bbox", [])) == 4 and
                all(isinstance(v, (int,float)) and math.isfinite(v) for v in det["bbox"]) and
                isinstance(det.get("score"), (int,float)) and
                0 <= det["score"] <= 1,
                f"Invalid V3 detection record #{index}")
    return record, gt, predictions


def _score(gt: dict, preds: list[dict], target_dir: Path,
           evaluate_coco: Any) -> dict:
    # Keep GT *exactly* the same bytes in replay/historical calls.
    require(len(preds) > 0, "No predictions at this confidence threshold")
    tmp_gt = target_dir / "ground_truth.json"
    tmp_pr = target_dir / "predictions.json"
    tmp_gt.write_text(json.dumps(gt, sort_keys=True) + "\n", encoding="utf-8")
    tmp_pr.write_text(json.dumps(preds, sort_keys=True) + "\n", encoding="utf-8")
    return evaluate_coco(tmp_gt, tmp_pr)


def _fraction(metrics: dict) -> float:
    return float(metrics["test_size/AP50-Small"])


def _replay(gt: dict, preds: list[dict], thresholds: tuple[float, ...],
            evaluate_coco: Any, baseline: float) -> list[dict]:
    runs = []
    for conf in thresholds:
        filtered = [d for d in preds if d["score"] >= conf]
        with tempfile.TemporaryDirectory(prefix="ftsc_small_replay_") as t:
            m = _score(gt, filtered, Path(t), evaluate_coco)
        runs.append({
            "confidence": conf,
            "retained_predictions": len(filtered),
            "removed_predictions": len(preds)-len(filtered),
            "AP50_Small_pct": 100.0 * _fraction(m),
            "AP50_All_pct": 100.0 * float(m["test_size/AP50"]),
            "mAP50_75_Small_pct": 100.0 * float(m["test_size/AP-Small"]),
        })
        if conf == 0.001:
            require(abs(_fraction(m) - baseline) < 1e-9,
                    "REPLAY FAILED: saved V3 AP50-Small is not reproduced "
                    "from exact GT and saved predictions; investigate evaluator/JSON first")
    return runs


def _predict(model: Any, paths: list[Path], batch: int, device: str,
             workers: int, conf: float | None) -> list[dict]:
    """Mimic historical _predictions; explicit conf only in controlled probe."""
    from evaluate_test.size_bucket_evaluator import _predictions
    if conf is None:
        # THIS IS THE ORIGINAL PAPER FUNCTION, including default conf=.25.
        return _predictions(model, paths, imgsz=512, batch=batch,
                            device=device, workers=workers)
    output = []
    for start in range(0,len(paths),100):
        chunk = paths[start:start+100]
        results = model.predict(
            source=[str(path) for path in chunk], imgsz=512, batch=batch,
            device=device, workers=workers, iou=0.5,
            conf=conf, verbose=False, stream=True,
        )
        for offset, result in enumerate(results):
            boxes = result.boxes
            xyxy = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
            scores = boxes.conf.detach().cpu().tolist() if boxes is not None else []
            for (x1,y1,x2,y2), score in zip(xyxy,scores):
                output.append({
                    "image_id":start+offset+1, "category_id":1,
                    "bbox":[x1,y1,max(0.0,x2-x1),max(0.0,y2-y1)],
                    "score":float(score),
                })
    return output


def one_case(root: Path, case: str, seed: int, expected: dict,
             thresholds: tuple[float, ...], mode: str,
             device: str, batch: int, workers: int,
             evaluate_coco: Any) -> dict:
    record,gt,raw = read_metric_records(root,case,seed,expected)
    baseline = _fraction(record["metrics"])
    output = {
        "case": case, "seed": seed,
        "source_sha256": expected["sha256"],
        "v3_protocol": record["evaluation_protocol"],
        "gt_sha256": record["gt_sha256"],
        "v3_predictions_sha256": record["predictions_sha256"],
        "v3_ap50_small_pct": 100.0 * baseline,
        "v3_prediction_count": len(raw),
        "paper_reference_mean_pct": PAPER_REFERENCE_MEANS.get(case),
        "reference_note": "paper_reference_mean_pct is 3-seed mean, NOT per-seed target",
        "replay": _replay(gt, raw, thresholds, evaluate_coco, baseline),
    }
    val_at_025 = next((x for x in output["replay"] if x["confidence"] == 0.25), None)
    require(val_at_025 is not None, "0.25 confidence control is mandatory")
    output["val_pred_threshold_effect_pp"] = (
        val_at_025["AP50_Small_pct"]-output["v3_ap50_small_pct"])
    if mode == "replay":
        return output

    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from ultralytics import YOLO

    src = deploy.source_path(root,"LEVIR-Ship",case,seed)
    require(src.is_file() and sha256(src) == expected["sha256"],
            f"Physical original PT checkpoint missing/mutated: {src}")
    images = [Path(row["file_name"]) for row in gt["images"]]
    require(all(p.is_file() for p in images),"Frozen historical source images are not present")
    require(len({str(p.resolve()) for p in images}) == 788,
            "Duplicate source test images")
    model=YOLO(str(src),task="detect")
    old = _predict(model, images, batch, device, workers, conf=None)
    with tempfile.TemporaryDirectory(prefix="ftsc_small_legacy_") as temp:
        legacy = _score(gt,old,Path(temp),evaluate_coco)
    low = _predict(model,images,batch,device,workers,conf=0.001)
    with tempfile.TemporaryDirectory(prefix="ftsc_small_predictlow_") as temp:
        controlled = _score(gt,low,Path(temp),evaluate_coco)
    filtered=[p for p in low if p["score"]>=0.25]
    with tempfile.TemporaryDirectory(prefix="ftsc_small_predictfilter_") as temp:
        controlled_filtered=_score(gt,filtered,Path(temp),evaluate_coco)

    output["historical_predictor"] = {
        "checkpoint_sha256": sha256(src),
        "imgsz": 512, "batch": batch, "rect": "predictor default=True",
        "nms_iou": 0.5, "explicit_conf": None,
        "effective_conf": "Ultralytics predictor default (expected 0.25)",
        "legacy_predict_025_count": len(old),
        "legacy_predict_025_AP50_Small_pct": 100*_fraction(legacy),
        "predict_001_count": len(low),
        "predict_001_AP50_Small_pct": 100*_fraction(controlled),
        "predict_001_filtered_025_count": len(filtered),
        "predict_001_filtered_025_AP50_Small_pct": 100*_fraction(controlled_filtered),
        "difference_legacy_minus_val_filtered_pp":
            100*_fraction(legacy) - val_at_025["AP50_Small_pct"],
        "difference_low_conf_predict_minus_val_pp":
            100*_fraction(controlled) - 100*baseline,
        "difference_legacy_minus_low_conf_predict_pp":
            100*_fraction(legacy)-100*_fraction(controlled),
        "note": "Prediction pathway and 512 canvas vs 544 canvas differ; "
                "historical and low-conf predictor share the same model source. "
                "Do not label this residual preprocessing-only without exact tensor parity.",
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("replay","historical"),default="replay")
    parser.add_argument("--cases",nargs="+",default=list(DEFAULT_CASES))
    parser.add_argument("--seeds",nargs="+",type=int,default=[42])
    parser.add_argument("--thresholds",nargs="+",type=float,
                        default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--predict-batch",type=int,default=8)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--base-project",type=Path,
                        default=deploy.ROOT / "runs/ftsc_deployment_eval_v1")
    parser.add_argument("--report-manifest",type=Path,default=deploy.REPORT_MANIFEST)
    args=parser.parse_args(argv)
    require(set(args.seeds)<= {42,43,44} and args.seeds,
            "Only audited LEVIR train seeds 42/43/44 allowed")
    require(args.predict_batch>0,"Invalid historical predictor batch")
    require(args.cases and len(set(args.cases))==len(args.cases),
            "Duplicate or empty cases")
    thresholds=tuple(sorted(set(args.thresholds)|{0.001,0.25}))
    require(all(0<c<1 for c in thresholds),"Invalid confidence thresholds")
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from evaluate_test.size_bucket_evaluator import _evaluate_coco
    inventory,_,_=deploy.load_report_scope(args.report_manifest,"LEVIR-Ship")
    require(all(case in inventory for case in args.cases),
            "Case not in the pinned deployment manifest")
    root=(args.base_project / deploy.RECT_PARITY_PROTOCOL /
          deploy.RECT_PARITY_RUNTIME_REVISION).resolve()
    rows=[]
    for case in args.cases:
        for seed in args.seeds:
            row=one_case(root,case,seed,inventory[case][seed],thresholds,args.mode,
                         args.device,args.predict_batch,args.workers,_evaluate_coco)
            rows.append(row)
            msg={
                "case":case, "seed":seed,
                "V3_val_conf001_AP50_small_pct":row["v3_ap50_small_pct"],
                "val_threshold_effect_pp":row["val_pred_threshold_effect_pp"],
            }
            if "historical_predictor" in row:
                msg["legacy_predict_conf025_AP50_small_pct"]=row[
                    "historical_predictor"]["legacy_predict_025_AP50_Small_pct"]
            print(json.dumps(msg),flush=True)
    aggregates=[]
    for case in args.cases:
        group=[row for row in rows if row["case"]==case]
        if len(group)!=3:
            continue
        summaries={"case":case,"n_seeds":3}
        for label,key in (
            ("v3_val_conf001_AP50_Small","v3_ap50_small_pct"),
            ("legacy_predict_conf025_AP50_Small","historical_predictor"),
        ):
            if key=="historical_predictor":
                if any(key not in row for row in group):
                    continue
                values=[row[key]["legacy_predict_025_AP50_Small_pct"] for row in group]
            else:
                values=[row[key] for row in group]
            summaries[label+"_mean_pct"]=statistics.fmean(values)
            summaries[label+"_sample_std_pp"]=statistics.stdev(values)
        summaries["paper_reference_mean_pct"]=PAPER_REFERENCE_MEANS.get(case)
        aggregates.append(summaries)
    report={
        "schema":"ftsc_levir_ap50_small_rootcause_diagnostic_v1",
        "mode":args.mode,
        "checked_protocol":"native_ultralytics_val_rect_parity_v3",
        "source_code_sha256":{
            "evaluate_test/size_bucket_evaluator.py":
                sha256(deploy.ROOT/"evaluate_test/size_bucket_evaluator.py"),
            "utils/ftsc_deployment_backend_eval_v1.py":
                sha256(deploy.ROOT/"utils/ftsc_deployment_backend_eval_v1.py"),
            "utils/ftsc_levir_small_ap_rootcause_v1.py":
                sha256(Path(__file__).resolve()),
        },
        "cases":args.cases,"seeds":args.seeds,"thresholds":thresholds,
        "results":rows,"three_seed_summary":aggregates,
        "conclusion":"This report isolates confidence filtering on SAVED V3 "
                     "predictions, then optionally contrasts exact legacy predictor "
                     "and matched low-confidence predictor on the SAME checkpoint. "
                     "No paper result is overwritten.",
    }
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out=(args.base_project / "_small_ap_rootcause_v1" /
         f"{args.mode}_{stamp}.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    require(not out.exists(),"Do not overwrite an existing diagnostic")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":"DIAGNOSTIC_COMPLETE",
        "mode":args.mode,"jobs":len(rows),"report":str(out),
        "three_seed_summary":aggregates,
    },indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
