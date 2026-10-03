"""Varroa H2 completion; only the canonical Mosaic+OACP R2 case is trainable."""
from __future__ import annotations
import argparse
import hashlib
import shutil
import math
from pathlib import Path
from typing import Any
from h2_generalization_completion_common import SEEDS, aggregate, ensure_manifest, require_metrics, selected_cases, selected_seeds, status_fields, write_completion_marker, write_csv, write_json
from train_levir_scripts.train_ftsc_negative_canvas_suite import dataset_eligibility

ROOT=Path(__file__).resolve().parent
CASES=("VR_H2_MOSAIC","VR_H2_MOSAIC_CP2","VR_H2_MOSAIC_OACP_R2","VR_H2_R4")
DEFAULT_SEEDS=SEEDS; DEFAULT_EPOCHS=100; DEFAULT_IMGSZ=640; DEFAULT_BATCH_SIZE=4; DEFAULT_WORKERS=8; DEFAULT_PATIENCE=20
CANONICAL_PRETRAINED="yolov8n.pt"
REQUIRED_METRICS=("test/metrics/mAP50(B)","test/metrics/mAP50-95(B)","test/metrics/mAP75(B)","test/metrics/precision(B)","test/metrics/recall(B)","val/metrics/mAP50(B)","val/metrics/mAP50-95(B)","val/metrics/mAP75(B)","val/metrics/precision(B)","val/metrics/recall(B)","test_size/AP50-Small","model/parameters","model/GFLOPs","test_speed/inference_ms_per_image")

def oacp_kwargs(args: argparse.Namespace, seed: int, data_yaml: Path, amp: bool=True) -> dict[str,Any]:
    from train_levir_scripts import train_varroa_dbss_hit as shared
    from train_levir_scripts import train_varroa_ftsc_next_ablation_suite as suite
    kwargs=shared.train_kwargs(args,data_yaml,seed,amp)
    kwargs.update(mosaic=suite.CURRENT_VARROA_MOSAIC,close_mosaic=suite.CURRENT_VARROA_CLOSE_MOSAIC,copy_paste=0.0,copy_paste_enabled=False,hard_negative_tile=False,hard_negative_bank="")
    if kwargs["copy_paste"] != 0.0 or kwargs["copy_paste_enabled"]: raise RuntimeError("Varroa OACP must not activate CP2")
    return kwargs

def validate_args(args: argparse.Namespace) -> None:
    frozen={"epochs":100,"imgsz":640,"batch_size":4,"patience":20}
    for key, expected in frozen.items():
        if getattr(args,key) != expected: raise ValueError(f"Varroa frozen protocol requires {key}={expected}")
    if args.workers < 0: raise ValueError("workers must be non-negative")
    if getattr(args,"pretrained",CANONICAL_PRETRAINED) != CANONICAL_PRETRAINED: raise ValueError(f"Varroa pretrained is locked to {CANONICAL_PRETRAINED}")
    if getattr(args,"seed_scope","all") != "all": raise ValueError("Varroa completion requires seed_scope='all'")
    selected_seeds(args); selected_cases(args,CASES)

def _historical_metrics(case: str, seed: int) -> dict[str,Any]:
    import csv
    sources={"VR_H2_MOSAIC":(ROOT.parent/"results/varroa_ftsc_causal_suite_summary_runs.csv","V1_FTSC"),"VR_H2_MOSAIC_CP2":(ROOT.parent/"results/varroa_ftsc_next_ablation_suite_summary_runs.csv","FTSC_P23_CP2")}
    source=sources.get(case)
    if not source or not source[0].is_file(): return {}
    try:
        with source[0].open(newline="",encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("seed"))==str(seed) and row.get("variant")==source[1]: return row
    except OSError: pass
    return {}

def historical_run_dir(case: str, seed: int) -> Path | None:
    roots={"VR_H2_MOSAIC":(ROOT/"runs/varroa_ftsc_causal_suite","V1_FTSC"),"VR_H2_MOSAIC_CP2":(ROOT/"runs/varroa_ftsc_next_ablation_suite","FTSC_P23_CP2")}
    item=roots.get(case); return None if item is None else item[0]/item[1]/f"seed_{seed}"

def preflight(args: argparse.Namespace) -> list[dict[str,Any]]:
    rows=[]
    requested=selected_seeds(args); cases=selected_cases(args,CASES)
    for seed in SEEDS:
        data_yaml=args.dataset_root/f"varroa_yolo_seed{seed}"/"varroa.yaml"
        audit={"status":"NOT_REQUESTED"} if seed not in requested else ({"status":"PENDING_DATA_PREPARATION"} if not data_yaml.is_file() else {"status":"READY",**dataset_eligibility(data_yaml,require_negative=False,require_eligible_small=False)})
        for case in CASES:
            if case not in cases or seed not in requested: rows.append({"case":case,"seed":seed,"status":"NOT_REQUESTED"});continue
            historical=_historical_metrics(case,seed) if case in {"VR_H2_MOSAIC","VR_H2_MOSAIC_CP2"} else {}
            status="REUSED" if historical and not require_metrics(historical,REQUIRED_METRICS) else ("EVAL_BACKFILL_REQUIRED" if case in {"VR_H2_MOSAIC","VR_H2_MOSAIC_CP2"} else "NEW_RUN_REQUIRED")
            if case=="VR_H2_R4": status="PENDING_DATA_PREPARATION" if audit["status"]=="PENDING_DATA_PREPARATION" else "NOT_APPLICABLE"
            source=historical_run_dir(case,seed)
            rows.append({"case":case,"seed":seed,"status":status,"source_run_dir":str(source) if source else "","source_checkpoint":str(source/"weights/best.pt") if source else "","model_config":"CANONICAL_FTSC_CONFIG","copy_paste_enabled":False if case=="VR_H2_MOSAIC_OACP_R2" else "","r4_audit":str(audit),"provenance":"historical_read_only" if historical else "h2_completion_v1",**historical})
    return status_fields(rows,requested)

def _run_dir(args: argparse.Namespace, seed: int) -> Path: return args.project/"VR_H2_MOSAIC_OACP_R2"/f"seed_{seed}"
def _dataset_contract(data_yaml: Path, seed: int) -> dict[str,Any]:
    from train_levir_scripts import train_varroa_dbss_hit as shared
    import yaml
    shared.validate_dataset(data_yaml); payload=yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    for key, expected in {"gt_source":"gt_one","only_positives":True,"class_policy":"map-3-to-1","seed":seed}.items():
        if payload.get(key)!=expected: raise RuntimeError(f"Varroa canonical dataset metadata drift: {key}")
    return {"schema":"ftsc_varroa_dataset_contract_v1","seed":seed,"data_yaml":str(data_yaml.resolve()),"gt_source":"gt_one","only_positives":True,"class_policy":"map-3-to-1","expected_counts":{"train":2762,"val":592,"test":592}}
def _complete(metrics: dict[str,Any]) -> None:
    missing=require_metrics(metrics,REQUIRED_METRICS)
    if missing: raise RuntimeError(f"Varroa evaluation incomplete: {missing}")

def optimizer_provenance(args: argparse.Namespace, data_yaml: Path) -> dict[str,Any]:
    """Mirror vendored Trainer auto branch without changing its request."""
    import yaml
    payload=yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names=payload.get("names",{})
    nc=payload.get("nc",len(names) if isinstance(names,(dict,list)) else 1)
    if int(nc)!=1: raise RuntimeError(f"Varroa matched auto optimizer requires nc=1, got {nc}")
    nbs=64; iterations=math.ceil(2762/max(args.batch_size,nbs))*args.epochs
    if iterations>10000: raise RuntimeError(f"Varroa auto optimizer branch drift: iterations={iterations} > 10000")
    return {"requested":"auto","resolved":"AdamW","auto_resolution_branch":"iterations<=10000","optimizer_auto_iterations":iterations,"optimizer_auto_threshold":10000,"lr0_requested_default":.01,"lr0_resolved":.002,"momentum_requested_default":.937,"momentum_resolved":.9,"warmup_bias_lr_requested_default":.1,"warmup_bias_lr_resolved":0.0,"weight_decay":.0005,"lrf":.01,"cos_lr":False,"warmup_epochs":3.0,"warmup_momentum":.8,"fitness_metric":"map50_95"}

def run_oacp(args: argparse.Namespace, seed: int, data_yaml: Path) -> dict[str,Any]:
    from train_levir_scripts import train_varroa_dbss_hit as shared
    from train_levir_scripts import train_varroa_ftsc_next_ablation_suite as suite
    from train_levir_scripts import train_levir_ftsc_augmentation_suite as augmentation
    run_dir=_run_dir(args,seed)
    kwargs=oacp_kwargs(args,seed,data_yaml,True)
    config_sha=hashlib.sha256(suite.CANONICAL_FTSC_CONFIG.read_bytes()).hexdigest()
    optimizer=optimizer_provenance(args,data_yaml)
    manifest={"schema":"ftsc_varroa_h2_completion_v1","dataset":"varroa","case":"VR_H2_MOSAIC_OACP_R2","seed":seed,"canonical_seed_universe":list(SEEDS),"requested_seed_selection":list(selected_seeds(args)),"dataset_yaml":str(data_yaml.resolve()),"dataset_preparation_seed":seed,"scientific_contract":{"model_config":str(suite.CANONICAL_FTSC_CONFIG.resolve()),"model_config_sha256":config_sha,"pretrained":str(getattr(args,"pretrained","yolov8n.pt")),"epochs":args.epochs,"imgsz":args.imgsz,"batch":args.batch_size,"patience":args.patience,"seed_scope":"all","optimizer_provenance":optimizer,"deterministic":True,"amp":True,"oacp_environment":augmentation.environment_for("A1_OACP_R2"),"effective_train_kwargs":{key:value for key,value in kwargs.items() if key not in {"data","project","name","device","workers","exist_ok"}},"augmentation_effective_diff":{key:value for key,value in kwargs.items() if key.startswith("copy_paste") or key.startswith("hard_negative") or key in {"mosaic","close_mosaic","mosaic_policy"}}},"execution_context":{"default_workers":8,"effective_workers":args.workers,"device":args.device}}
    ensure_manifest(run_dir,manifest)
    config_target=run_dir/"config.yaml"
    if not config_target.exists(): shutil.copy2(suite.CANONICAL_FTSC_CONFIG,config_target)
    with augmentation.configured_environment("A1_OACP_R2"):
        if not shared.complete(run_dir):
            last=run_dir/"weights/last.pt"; shared.local_ultralytics()
            from ultralytics import YOLO
            if last.is_file():
                shared.seed_everything(seed); YOLO(last).train(resume=True)
            else:
                kwargs.update(project=str(args.project/"VR_H2_MOSAIC_OACP_R2"),name=f"seed_{seed}",exist_ok=True)
                shared.seed_everything(seed)
                try:
                    model=YOLO(suite.CANONICAL_FTSC_CONFIG);model.load(getattr(args,"pretrained","yolov8n.pt"),smart_transfer=True);model.train(**kwargs)
                except Exception:
                    # AMP is scientifically locked for this suite: preserve
                    # resumable artifacts and fail rather than drift to AMP=False.
                    raise
        if not shared.complete(run_dir): raise FileNotFoundError(f"incomplete Varroa run: {run_dir}")
        args.data_yaml=data_yaml; metrics=shared.evaluate(run_dir,args); _complete(metrics); write_completion_marker(run_dir,case="VR_H2_MOSAIC_OACP_R2",seed=seed,required=REQUIRED_METRICS,metrics=metrics); return metrics

def dispatch(args: argparse.Namespace, rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    from train_levir_scripts import train_varroa_dbss_hit as shared
    for seed in selected_seeds(args):
        r4=next((row for row in rows if row["case"]=="VR_H2_R4" and int(row["seed"])==seed),None)
        if r4 and r4["status"]=="PENDING_DATA_PREPARATION":
            data=shared.dataset_for_seed(args,seed); contract=_dataset_contract(data,seed)
            audit=dataset_eligibility(data,require_negative=False,require_eligible_small=False)
            r4["r4_audit"]=str(audit)
            r4["status"]="NOT_APPLICABLE" if audit["original_negative_images"]==0 else "PENDING_DATA_PREPARATION"
    for row in rows:
        if row["status"]=="EVAL_BACKFILL_REQUIRED" and row["case"] in {"VR_H2_MOSAIC","VR_H2_MOSAIC_CP2"}:
            source=Path(row["source_run_dir"])
            if not (source/"weights/best.pt").is_file(): raise FileNotFoundError(f"historical Varroa checkpoint missing: {source}")
            data=shared.dataset_for_seed(args,int(row["seed"]));_dataset_contract(data,int(row["seed"]));args.data_yaml=data;row.update(shared.evaluate(source,args));_complete(row);row["status"]="COMPLETE";continue
        if row["status"] != "NEW_RUN_REQUIRED" or row["case"] != "VR_H2_MOSAIC_OACP_R2": continue
        data=shared.dataset_for_seed(args,int(row["seed"]));write_json(args.project/"dataset_contracts"/f"seed_{row['seed']}.json",_dataset_contract(data,int(row["seed"])));row.update(run_oacp(args,int(row["seed"]),data));row["status"]="COMPLETE"
    return status_fields(rows,selected_seeds(args))

def run_one(args: argparse.Namespace, case: str, seed: int, data_yaml: Path) -> dict[str,Any]:
    if case != "VR_H2_MOSAIC_OACP_R2": raise ValueError(f"{case} is not trainable")
    if seed not in selected_seeds(args) or case not in selected_cases(args,CASES): raise ValueError("case/seed was not selected")
    return run_oacp(args,seed,data_yaml)

def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,default=ROOT.parent);p.add_argument("--dataset-root",type=Path,default=ROOT/"datasets");p.add_argument("--project",type=Path,default=ROOT/"runs/varroa_h2_generalization_completion_v1");p.add_argument("--data-yaml",type=Path,default=ROOT/"datasets/varroa_yolo_seed42/varroa.yaml");p.add_argument("--pretrained",default="yolov8n.pt");p.add_argument("--epochs",type=int,default=100);p.add_argument("--imgsz",type=int,default=640);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--device",default="cuda");p.add_argument("--workers",type=int,default=8);p.add_argument("--patience",type=int,default=20);p.add_argument("--seed-scope",default="all",choices=("all",),help=argparse.SUPPRESS);p.add_argument("--cases",nargs="+",default=list(CASES));p.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS));p.add_argument("--confirm-run",action="store_true");return p.parse_args(argv)
def build_preflight(args: argparse.Namespace) -> list[dict[str,Any]]: validate_args(args); return preflight(args)
def write_summaries(args: argparse.Namespace, rows: list[dict[str,Any]]) -> None:
    write_csv(args.project/"summary_runs.csv",rows);write_csv(args.project/"summary_aggregate.csv",aggregate(rows,REQUIRED_METRICS))
def main(argv=None):
    args=parse_args(argv);rows=build_preflight(args)
    if not args.confirm_run:
        print(rows);return
    write_json(args.project/"preflight.json",{"schema":"ftsc_varroa_h2_preflight_v1","rows":rows})
    rows=dispatch(args,rows);write_summaries(args,rows)
if __name__=="__main__":main()
