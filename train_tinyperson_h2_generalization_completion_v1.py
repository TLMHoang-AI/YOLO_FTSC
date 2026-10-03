"""Dataset-specific TinyPerson H2 augmentation completion workload."""
from __future__ import annotations
import argparse
import json
import hashlib
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from h2_generalization_completion_common import SEEDS, aggregate, ensure_manifest, require_metrics, selected_cases, selected_seeds, status_fields, write_completion_marker, write_csv, write_json
from train_levir_scripts.train_ftsc_negative_canvas_suite import dataset_eligibility

ROOT=Path(__file__).resolve().parent
CASES=("TP_H2_MOSAIC","TP_H2_MOSAIC_CP2","TP_H2_MOSAIC_OACP_R2","TP_H2_NM","TP_H2_NM_R4")
DEFAULT_SEEDS=SEEDS; DEFAULT_EPOCHS=100; DEFAULT_IMGSZ=640; DEFAULT_BATCH_SIZE=8; DEFAULT_WORKERS=8; DEFAULT_PATIENCE=20; DEFAULT_OPTIMIZER="AdamW"
CANONICAL_PRETRAINED="yolov8n.pt"

def canonical_pretrained(value: str | Path) -> str:
    """Canonical artifact identity; spelling is not a scientific difference."""
    requested=Path(value)
    resolved=(ROOT/requested if not requested.is_absolute() else requested).resolve()
    canonical=(ROOT/CANONICAL_PRETRAINED).resolve()
    if resolved != canonical: raise ValueError(f"TinyPerson pretrained is locked to {canonical}")
    return str(canonical)
REQUIRED_METRICS=("test_merged/AP50","test_merged/mAP50-75","test_merged/AP75","test_merged/AP50-Tiny1","test_merged/AP50-Tiny2","test_merged/AP50-Tiny3","test_merged/AP50-Small","test_merged/AP50-Medium","test_merged/AP-Tiny1","test_merged/AP-Tiny2","test_merged/AP-Tiny3","test_merged/AP-Small","test_merged/AP-Medium","test/metrics/mAP50(B)","test/metrics/mAP50-95(B)","test/metrics/mAP75(B)","test/metrics/precision(B)","test/metrics/recall(B)","val/metrics/mAP50(B)","val/metrics/mAP50-95(B)","val/metrics/mAP75(B)","val/metrics/precision(B)","val/metrics/recall(B)","params","gflops_at_imgsz","test_speed/inference_ms_per_image")

def dataset_yaml(args: argparse.Namespace, seed: int) -> Path:
    return args.dataset_root / f"tinyperson_seed_{seed}_corner_sw640_sh512" / "tinyperson.yaml"

def base_variant(case: str) -> str:
    # H2_NM is historical-only.  R4 starts from the active H2 Mosaic graph,
    # then applies the canonical no-Mosaic R4 augmentation delta.
    return "H2_MOSAIC"

def validate_args(args: argparse.Namespace) -> None:
    frozen={"epochs":100,"imgsz":640,"batch_size":8,"patience":20}
    for key, expected in frozen.items():
        if getattr(args,key) != expected: raise ValueError(f"TinyPerson frozen protocol requires {key}={expected}")
    if args.workers < 0: raise ValueError("workers must be non-negative")
    canonical_pretrained(args.pretrained)
    selected_seeds(args); selected_cases(args,CASES)

def _historical_metrics(case: str, seed: int) -> dict[str, Any]:
    """Read only real historical rows; empty history is evaluation-backfill, never fake reuse."""
    sources={"TP_H2_MOSAIC": ROOT.parent/"results/tinyperson_ftsc_causal_suite_summary_runs.csv", "TP_H2_NM": ROOT.parent/"results/tinyperson_h2_es1_summary_runs.csv"}
    path=sources.get(case)
    if not path or not path.is_file(): return {}
    try:
        if path.suffix==".csv":
            import csv
            with path.open(newline="",encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    target="H2_MOSAIC" if case=="TP_H2_MOSAIC" else "H2"
                    if str(row.get("seed"))==str(seed) and row.get("variant")==target: return row
        else:
            payload=json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("runs",[]):
                if str(row.get("seed"))==str(seed) and row.get("variant") in {"H2","H2_NM"}: return row
    except (OSError, ValueError): return {}
    return {}

def historical_run_dir(case: str, seed: int) -> Path | None:
    roots={"TP_H2_MOSAIC":(ROOT/"runs/tinyperson_ftsc_causal_suite","H2_MOSAIC"),"TP_H2_NM":(ROOT/"runs/tinyperson_ftsc_h2_es1","H2")}
    item=roots.get(case)
    return None if item is None else item[0]/item[1]/f"seed_{seed}_corner_sw640_sh512"

def train_kwargs(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> tuple[dict[str,Any], Any]:
    from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as tiny
    from train_levir_scripts import train_levir_ftsc_augmentation_suite as augmentation
    kwargs=tiny.train_kwargs(args,base_variant(case),seed,data_yaml); context=nullcontext()
    if case=="TP_H2_MOSAIC_CP2":
        cp2=augmentation.augmentation_for("A3_CP2")
        kwargs.update({key:value for key,value in cp2.items() if key.startswith("copy_paste")})
    elif case=="TP_H2_MOSAIC_OACP_R2":
        kwargs.update(copy_paste=0.0,copy_paste_enabled=False)
        context=augmentation.configured_environment("A1_OACP_R2")
    elif case=="TP_H2_NM_R4":
        from train_levir_scripts.train_ftsc_negative_canvas_suite import augmentation_for
        kwargs.update(augmentation_for("r4"))
    kwargs.update(project=str(args.project/case),name=f"seed_{seed}_corner_sw640_sh512",exist_ok=True)
    if Path(kwargs["project"])/kwargs["name"] != _run_dir(args,case,seed): raise RuntimeError("TinyPerson completion run-dir collision")
    return kwargs,context

def preflight(args: argparse.Namespace) -> list[dict[str,Any]]:
    rows=[]
    requested=selected_seeds(args); cases=selected_cases(args,CASES)
    for seed in SEEDS:
        data=dataset_yaml(args,seed)
        audit={"status":"NOT_REQUESTED"} if seed not in requested else ({"status":"PENDING_DATA_PREPARATION"} if not data.is_file() else {"status":"READY",**dataset_eligibility(data,require_negative=False,require_eligible_small=False)})
        for case in CASES:
            if case not in cases: rows.append({"case":case,"seed":seed,"status":"NOT_REQUESTED"}); continue
            if seed not in requested: rows.append({"case":case,"seed":seed,"status":"NOT_REQUESTED"}); continue
            historical=_historical_metrics(case,seed) if case in {"TP_H2_MOSAIC","TP_H2_NM"} else {}
            status="REUSED" if historical and not require_metrics(historical,REQUIRED_METRICS) else ("EVAL_BACKFILL_REQUIRED" if case in {"TP_H2_MOSAIC","TP_H2_NM"} else "NEW_RUN_REQUIRED")
            if case=="TP_H2_NM_R4":
                status="PENDING_DATA_PREPARATION" if audit["status"]!="READY" else ("NEW_RUN_REQUIRED" if audit["original_negative_images"]>0 and audit["eligible_small_targets_le_20px"]>0 else "NOT_APPLICABLE")
            source=historical_run_dir(case,seed)
            if case in {"TP_H2_MOSAIC","TP_H2_NM"} and source is not None:
                if not (source/"weights/best.pt").is_file(): status="EVAL_BACKFILL_REQUIRED"
            rows.append({"case":case,"seed":seed,"status":status,"base":base_variant(case),"data_yaml":str(dataset_yaml(args,seed)),"source_run_dir":str(source) if source else "","source_checkpoint":str(source/"weights/best.pt") if source else "","r4_audit":json.dumps(audit,sort_keys=True),"provenance":"historical_read_only" if case in {"TP_H2_MOSAIC","TP_H2_NM"} else "h2_completion_v1",**historical})
    return status_fields(rows,requested)

def _run_dir(args: argparse.Namespace, case: str, seed: int) -> Path:
    return args.project / case / f"seed_{seed}_corner_sw640_sh512"

def _dataset_contract(data_yaml: Path, seed: int) -> dict[str, Any]:
    root=data_yaml.parent; manifest=root/"corner_manifest.json"
    if not manifest.is_file(): raise FileNotFoundError(f"TinyPerson corner manifest missing: {manifest}")
    payload=json.loads(manifest.read_text(encoding="utf-8"))
    train={item if isinstance(item,str) else item.get("file_name") for item in payload.get("train_source_images",payload.get("train",[]))}; val={item if isinstance(item,str) else item.get("file_name") for item in payload.get("val_source_images",payload.get("val",[]))}
    if not train or not val or train & val: raise RuntimeError("TinyPerson source train/val contract is empty or overlapping")
    for split, sources in (("train",train),("val",val)):
        images=list((root/"images"/split).glob("*")); labels=list((root/"labels"/split).glob("*.txt"))
        if len(images)!=len(labels) or len(images)!=len(sources): raise RuntimeError(f"TinyPerson {split} image/label/manifest counts disagree")
    return {"schema":"ftsc_tinyperson_dataset_contract_v1","seed":seed,"data_yaml":str(data_yaml.resolve()),"corner_manifest":str(manifest.resolve()),"train_sources":len(train),"val_sources":len(val)}

def _complete(metrics: dict[str,Any]) -> None:
    missing=require_metrics(metrics,REQUIRED_METRICS)
    if missing or float(metrics.get("test_merged/available",0.0)) != 1.0: raise RuntimeError(f"TinyPerson evaluation incomplete: {missing}")

def run_new_case(args: argparse.Namespace, case: str, seed: int, data_yaml: Path, test_out_dir: Path) -> dict[str,Any]:
    """Train/resume one explicitly new case, then call the native evaluator."""
    from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as tiny
    workflow=tiny._tiny_workflow(); run_dir=_run_dir(args,case,seed); metrics_path=run_dir/"evaluation_metrics.json"
    if metrics_path.is_file():
        metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
        try: _complete(metrics); return metrics
        except RuntimeError: pass  # evaluation-only completion below
    kwargs,context=train_kwargs(args,case,seed,data_yaml); tiny.local_ultralytics()
    config=tiny.VARIANTS[base_variant(case)].config; config_sha=hashlib.sha256(config.read_bytes()).hexdigest()
    scientific_kwargs={key:value for key,value in kwargs.items() if key not in {"data","project","name","device","workers","exist_ok"}}
    from train_levir_scripts import train_levir_ftsc_augmentation_suite as augmentation
    manifest={"schema":"ftsc_tinyperson_h2_completion_v1","dataset":"tinyperson","case":case,"seed":seed,"canonical_seed_universe":list(SEEDS),"requested_seed_selection":list(selected_seeds(args)),"dataset_yaml":str(data_yaml.resolve()),"dataset_preparation_seed":seed,"scientific_contract":{"model_config":str(config.resolve()),"model_config_sha256":config_sha,"pretrained_canonical":canonical_pretrained(args.pretrained),"effective_train_kwargs":scientific_kwargs,"oacp_environment":augmentation.environment_for("A1_OACP_R2") if case=="TP_H2_MOSAIC_OACP_R2" else {},"augmentation_effective_diff":{key:value for key,value in kwargs.items() if key.startswith("copy_paste") or key.startswith("hard_negative") or key in {"mosaic","close_mosaic","mosaic_policy"}}},"pretrained_requested":str(args.pretrained),"execution_context":{"default_workers":8,"effective_workers":args.workers,"device":args.device}}
    ensure_manifest(run_dir,manifest)
    if not (run_dir/"config.yaml").exists(): shutil.copy2(config,run_dir/"config.yaml")
    from ultralytics import YOLO
    last=run_dir/"weights/last.pt"
    with context:
        trained=all((run_dir/item).is_file() for item in ("weights/best.pt","weights/last.pt","results.csv","args.yaml","config.yaml"))
        if trained: pass
        elif last.is_file(): YOLO(last).train(resume=True)
        else:
            model,_=tiny.build_pretrained_model(base_variant(case),args.pretrained,seed); model.train(**kwargs)
    metrics=workflow.evaluate(run_dir,data_yaml,test_out_dir,args.data_root,args)
    # Reuse the native causal-suite complexity preflight; do not add a second
    # FLOP implementation to this completion runner.
    preflight_args=argparse.Namespace(**vars(args), variants=(base_variant(case),))
    report=tiny.model_preflight(preflight_args)["variants"][base_variant(case)]
    metrics.update({"params":report["params"],"gflops_at_imgsz":report["gflops_at_imgsz"]})
    (run_dir/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    _complete(metrics)
    write_completion_marker(run_dir,case=case,seed=seed,required=REQUIRED_METRICS,metrics=metrics)
    return metrics

def dispatch(args: argparse.Namespace, rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Prepare only requested seeds, re-audit R4, then evaluate/train selected work."""
    from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as tiny
    requested=selected_seeds(args); test_out=None
    required={int(row["seed"]) for row in rows if int(row["seed"]) in requested and row["status"] in {"NEW_RUN_REQUIRED","PENDING_DATA_PREPARATION","EVAL_BACKFILL_REQUIRED"}}
    prepared={}
    if required:
        workflow=tiny._tiny_workflow(); test_out=workflow.prepare_test_set(args.data_root,args.dataset_root)
        for seed in required:
            prepared_dir=workflow.prepare_seed_dataset(args.data_root,args.dataset_root,test_out,seed); data=prepared_dir/"tinyperson.yaml"
            if data.resolve()!=dataset_yaml(args,seed).resolve(): raise RuntimeError(f"unexpected TinyPerson seed path: {data}")
            write_json(args.project/"dataset_contracts"/f"seed_{seed}.json",_dataset_contract(data,seed)); prepared[seed]=data
            audit={"status":"READY",**dataset_eligibility(data,require_negative=False,require_eligible_small=False)}
            for candidate in rows:
                if candidate["case"]=="TP_H2_NM_R4" and int(candidate["seed"])==seed:
                    candidate["r4_audit"]=json.dumps(audit,sort_keys=True);candidate["status"]="NEW_RUN_REQUIRED" if audit["original_negative_images"]>0 and audit["eligible_small_targets_le_20px"]>0 else "NOT_APPLICABLE"
    for row in rows:
        seed=int(row["seed"])
        if row["status"]=="EVAL_BACKFILL_REQUIRED" and row["case"] in {"TP_H2_MOSAIC","TP_H2_NM"}:
            source=Path(row["source_run_dir"])
            if not (source/"weights/best.pt").is_file(): raise FileNotFoundError(f"historical TinyPerson checkpoint missing: {source}")
            metrics=workflow.evaluate(source,prepared[seed],test_out,args.data_root,args)
            preflight_args=argparse.Namespace(**vars(args),variants=("H2_MOSAIC",))
            report=tiny.model_preflight(preflight_args)["variants"]["H2_MOSAIC"]
            metrics.update({"params":report["params"],"gflops_at_imgsz":report["gflops_at_imgsz"]})
            (source/"evaluation_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8")
            _complete(metrics); row.update(metrics);row["status"]="COMPLETE";continue
        if row["status"] != "NEW_RUN_REQUIRED": continue
        metrics=run_new_case(args,row["case"],seed,prepared[seed],test_out); row.update(metrics);row["status"]="COMPLETE"
    return status_fields(rows,selected_seeds(args))

def run_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path, test_out_dir: Path) -> dict[str,Any]:
    """Public live API for one explicitly selected new case×seed."""
    if case not in {"TP_H2_MOSAIC_CP2","TP_H2_MOSAIC_OACP_R2","TP_H2_NM_R4"}: raise ValueError(f"{case} is not trainable")
    if seed not in selected_seeds(args) or case not in selected_cases(args,CASES): raise ValueError("case/seed was not selected")
    return run_new_case(args,case,seed,data_yaml,test_out_dir)

def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,default=ROOT.parent/"TinyPerson/tiny_set");p.add_argument("--dataset-root",type=Path,default=ROOT/"datasets");p.add_argument("--project",type=Path,default=ROOT/"runs/tinyperson_h2_generalization_completion_v1");p.add_argument("--pretrained",default="yolov8n.pt");p.add_argument("--epochs",type=int,default=100);p.add_argument("--imgsz",type=int,default=640);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--device",default="cuda");p.add_argument("--workers",type=int,default=8);p.add_argument("--patience",type=int,default=20);p.add_argument("--cases",nargs="+",default=list(CASES));p.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS));p.add_argument("--confirm-run",action="store_true");return p.parse_args(argv)
def build_preflight(args: argparse.Namespace) -> list[dict[str,Any]]: validate_args(args); return preflight(args)
def write_summaries(args: argparse.Namespace, rows: list[dict[str,Any]]) -> None:
    write_csv(args.project/"summary_runs.csv",rows);write_csv(args.project/"summary_aggregate.csv",aggregate(rows,REQUIRED_METRICS))
def main(argv=None):
    args=parse_args(argv);rows=build_preflight(args)
    if not args.confirm_run:
        print(json.dumps(rows,sort_keys=True));return
    write_json(args.project/"preflight.json",{"schema":"ftsc_tinyperson_h2_preflight_v1","rows":rows})
    rows=dispatch(args,rows);write_summaries(args,rows)
if __name__=="__main__":main()
