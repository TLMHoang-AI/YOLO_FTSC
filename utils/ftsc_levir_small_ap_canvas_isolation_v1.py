#!/usr/bin/env python3
"""LEVIR size-AP root-cause: isolate 544-vs-512 canvas on SAME val pathway.

Reads prior historical predictor diagnostic; never re-runs or changes the legacy
paper prediction path, V3 metrics, GT, checkpoint, or deployment runner.

Control chain (each is independently TinyBenchmark-evaluated on identical GT):
  A: V3 val544 conf=.001  (saved)
  B: V3 val544 detections filtered to conf=.25 (saved replay)
  C: fresh PT32 val512 rect=False conf=.001 filtered to conf=.25
  D: historical PT32 predict512 conf=.25 (prior diagnostic)

A->B: confidence threshold with same 544 predictions.
B->C: 544 to 512 val-canvas change (same val backend, conf/NMS).
C->D: predictor-vs-validator output pathway at 512 (includes NMS, rounding).
This is an ordered decomposition, not a model-independent causal Shapley value.
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

from utils import ftsc_deployment_backend_eval_v1 as deploy
from utils import ftsc_levir_small_ap_rootcause_v1 as previous

IMG_SIZE=512
INPUT_COUNT=788


def need(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def _sha(path):
    return previous.sha256(path)


def _small(metrics):
    return 100 * float(metrics["test_size/AP50-Small"])


def _compare_dataset_to_predictor(dataset, expected_images):
    """Compare *simulated* fork predictor LetterBox to true val512 dataset tensors.

    Matches the actual predictor pre_transform(auto=True) on homogeneous
    square native LEVIR input and subsequent BGR->RGB/CHW preprocessing.
    It is a pixel parity control, not an assertion about downstream NMS.
    """
    import cv2
    import numpy as np
    from ultralytics.data.augment import LetterBox

    loader_files = {
        str(Path(path).resolve()): index
        for index, path in enumerate(dataset.im_files)
    }
    target_paths = {str(p.resolve()) for p in expected_images}
    need(len(loader_files)==INPUT_COUNT and set(loader_files)==target_paths,
         "Val512 dataloader has different LEVIR test image inventory")
    matches=0
    mismatches=[]
    shapes={}
    predictor_shapes={}
    for name in sorted(loader_files):
        val=dataset[loader_files[name]]["img"]
        if hasattr(val,"detach"):
            val=val.detach().cpu().numpy()
        val=np.ascontiguousarray(val)
        src=cv2.imread(name,cv2.IMREAD_COLOR)
        need(src is not None,f"Cannot decode source {name}")
        # Historical predictor has auto=True with homogeneous images and
        # effective PyTorch minimum stride32; validate each native image.
        letterbox=LetterBox(new_shape=(IMG_SIZE,IMG_SIZE),auto=True,
                            stride=32,scaleup=True)
        x=letterbox(image=src)
        predictor=np.ascontiguousarray(x[:,:,::-1].transpose(2,0,1))
        need(val.dtype == predictor.dtype == np.uint8,
             "Input tensors must be RGB uint8")
        need(val.shape[0]==predictor.shape[0]==3,
             "Wrong channel count in predictor/validator input")
        vshape=tuple(int(i) for i in val.shape)
        pshape=tuple(int(i) for i in predictor.shape)
        shapes[str(vshape)]=shapes.get(str(vshape),0)+1
        predictor_shapes[str(pshape)]=predictor_shapes.get(str(pshape),0)+1
        if vshape==pshape and hashlib.sha256(val.tobytes()).digest()==hashlib.sha256(predictor.tobytes()).digest():
            matches+=1
        elif len(mismatches)<10:
            mismatches.append({
                "image":name,"source_hw":list(src.shape[:2]),
                "val_chw":list(vshape),"predictor_chw":list(pshape),
            })
    return {
        "val_vs_simulated_predictor_pixel_parity": matches == INPUT_COUNT,
        "matching_image_count":matches,"total_images":INPUT_COUNT,
        "val_shapes":shapes,"simulated_predictor_shapes":predictor_shapes,
        "mismatch_examples":mismatches,
        "warning":"Simulation uses fork predictor LetterBox(auto=True), stride32, "
                  "BGR->RGB CHW. It is not a captured predictor input hook.",
    }


def _val512(case,seed,root,source,gt,report_root,device,workers):
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from ultralytics import YOLO
    # Actual converter belongs to V3 deployment adapter, not size evaluator.
    from evaluate_test.size_bucket_evaluator import _evaluate_coco

    model_path=deploy.source_path(root,"LEVIR-Ship",case,seed)
    need(model_path.is_file() and _sha(model_path)==source,
         "PT checkpoint bytes do not match historical source SHA")
    reference=deploy.path_for(root,"LEVIR-Ship",case,seed,"pt32")
    previous_record=json.loads((reference/"metrics.json").read_text(encoding="utf-8"))
    yaml=deploy.ROOT/"datasets/levir_ship_yolo_seed42/levir_ship.yaml"
    need(yaml.is_file() and _sha(yaml)==previous_record["data_yaml_sha256"],
         "Original frozen test YAML does not match V3 dataset SHA")

    paths=[Path(x["file_name"]) for x in gt["images"]]
    need(len(paths)==INPUT_COUNT and all(p.is_file() for p in paths),
         "Native LEVIR images missing for val512 test")
    from ultralytics import YOLO
    model=YOLO(str(model_path),task="detect")
    state={}
    model.add_callback("on_val_end", lambda validator: state.__setitem__("validator",validator))
    with tempfile.TemporaryDirectory(prefix="ftsc_levir_canvas512_") as tmp:
        # Does NOT modify V3 runtime, checkpoint or any historical JSON:
        # temporary standard YOLO val output disappears after this block.
        result=model.val(
            data=str(yaml), split="test", imgsz=IMG_SIZE, batch=1,
            device=device, half=False, workers=workers,
            conf=0.001, iou=0.5, max_det=300, rect=False,
            plots=False, save_json=True, project=tmp, name="val512",
            exist_ok=False, verbose=False,
        )
        validator=state.get("validator")
        need(validator is not None and int(validator.seen)==INPUT_COUNT,
             "Missing 788-image val512 validator callback")
        ds=validator.dataloader.dataset
        need(int(ds.imgsz)==IMG_SIZE and not bool(ds.rect),
             "Val512 control unexpectedly uses rectangular validation")
        require_shapes=[tuple(int(i) for i in x["img"].shape)
                        for x in (ds[i] for i in range(len(ds)))]
        need(len(require_shapes)==INPUT_COUNT and
             set(require_shapes)=={(3,IMG_SIZE,IMG_SIZE)},
             "Val512 control is not genuinely 512x512 across frozen test")
        preds=deploy.val_detections_to_size_predictions(validator.jdict,paths)
        parity=_compare_dataset_to_predictor(ds,paths)
    need(preds,"Val512 emitted no detections")
    with tempfile.TemporaryDirectory(prefix="ftsc_levir_canvas512_eval_") as tmp:
        m001=previous._score(gt,preds,Path(tmp),_evaluate_coco)
    # TemporaryDirectory was disposed; evaluation complete.
    filtered=[x for x in preds if x["score"]>=0.25]
    need(filtered,"No val512 scores survive conf=.25")
    with tempfile.TemporaryDirectory(prefix="ftsc_levir_canvas512_filter_") as tmp:
        m025=previous._score(gt,filtered,Path(tmp),_evaluate_coco)

    return {
        "val512_conf001_AP50_Small_pct":_small(m001),
        "val512_filtered_conf025_AP50_Small_pct":_small(m025),
        "val512_pred_count":len(preds),
        "val512_pred_count_conf025":len(filtered),
        "val512_native_AP50_pct":100*float(result.box.map50),
        "input_parity":parity,
        "validation_config":{
            "model":"pinned PT32","imgsz":512,"batch":1,
            "rect":False,"confidence":0.001,"nms_iou":0.5,
            "max_det":300,"save_json":True,
        },
    }


def _get_report(path):
    d=json.loads(path.read_text(encoding="utf-8"))
    need(d.get("schema")=="ftsc_levir_ap50_small_rootcause_diagnostic_v1" and
         d.get("mode")=="historical","Not a prior historical AP root-cause report")
    for name,digest in d.get("source_code_sha256",{}).items():
        p=deploy.ROOT/name
        need(p.is_file() and _sha(p)==digest,
             f"Source changed since historical control: {name}")
    return d


def isolate(historical_report,device="cuda:0",workers=2,base_project=None):
    initial=_get_report(Path(historical_report))
    base=Path(base_project) if base_project is not None else deploy.ROOT/"runs/ftsc_deployment_eval_v1"
    root=(base/deploy.RECT_PARITY_PROTOCOL/deploy.RECT_PARITY_RUNTIME_REVISION).resolve()
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    from evaluate_test.size_bucket_evaluator import _evaluate_coco
    inventory,_,_=deploy.load_report_scope(deploy.REPORT_MANIFEST,"LEVIR-Ship")
    rows=[]
    for saved in initial["results"]:
        case=saved["case"]
        seed=saved["seed"]
        need(case in inventory and seed in inventory[case],
             "Historical report case/seed outside pinned manifest")
        expected=inventory[case][seed]
        record,gt,preds=previous.read_metric_records(root,case,seed,expected)
        hist=saved["historical_predictor"]
        need(record["gt_sha256"]==saved["gt_sha256"] and
             record["predictions_sha256"]==saved["v3_predictions_sha256"] and
             hist["checkpoint_sha256"]==expected["sha256"],
             "Historical predictor and V3 artifacts are not from same checkpoint/GT")
        existing_025=next(x for x in saved["replay"]
                          if abs(x["confidence"]-0.25)<1e-12)
        val544_001= saved["v3_ap50_small_pct"]
        val544_025= existing_025["AP50_Small_pct"]
        legacy025= hist["legacy_predict_025_AP50_Small_pct"]
        interim=_val512(case,seed,root,expected["sha256"],gt,base,
                        device,workers)
        val512_025=interim["val512_filtered_conf025_AP50_Small_pct"]
        a=val544_025-val544_001
        b=val512_025-val544_025
        c=legacy025-val512_025
        need(abs((a+b+c)-(legacy025-val544_001))<1e-8,
             "Order decomposition does not sum to observed AP gap")
        rows.append({
            "case":case,"seed":seed,
            "source_sha256":expected["sha256"],
            "v3_val544_conf001_pct":val544_001,
            "v3_val544_filtered025_pct":val544_025,
            "val512_filtered025_pct":val512_025,
            "legacy_predict512_conf025_pct":legacy025,
            "delta_score_threshold_pp":a,
            "delta_val_canvas_544_to_512_pp":b,
            "delta_val512_to_predictor512_pp":c,
            "total_legacy_minus_v3_pp":legacy025-val544_001,
            "control":interim,
        })
        print(json.dumps({
            "case":case,"seed":seed,
            "score_threshold_pp":round(a,5),
            "canvas_544_to_512_pp":round(b,5),
            "predictor_residual_pp":round(c,5),
            "input_pixel_matches":
                interim["input_parity"]["matching_image_count"],
        }),flush=True)
    return {
        "schema":"ftsc_levir_small_ap_canvas_isolation_v1",
        "status":"CONTROLS_COMPLETE",
        "source_historical_report_sha256":_sha(Path(historical_report)),
        "historical_report_path":str(historical_report),
        "results":rows,
        "interpretation":"Ordered score/canvas/pathway decomposition, each scored on "
                         "identical frozen TinyBenchmark GT. Residual may include "
                         "multi_label NMS, rounding and path-specific settings.",
    }


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-report",required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--base-project",type=Path,
                        default=deploy.ROOT/"runs/ftsc_deployment_eval_v1")
    args=parser.parse_args(argv)
    d=isolate(args.historical_report,args.device,args.workers,args.base_project)
    outdir=args.base_project/"_small_ap_rootcause_v1"
    outdir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output=outdir/f"canvas512_isolation_{stamp}.json"
    need(not output.exists(),"Cannot overwrite existing result")
    output.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":d["status"],"jobs":len(d["results"]),
                      "report":str(output)},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
