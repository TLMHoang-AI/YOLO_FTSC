#!/usr/bin/env python3
"""Read-only PT/TRT V2 metric, checkpoint and dataset audit for TinyPerson/Varroa.

Requires a full preprocessing parity preflight pass before comparing AP. This
never modifies metric/engine/checkpoint bytes; it writes only an audit JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from utils import ftsc_deployment_backend_eval_v1 as deploy

STAGES = {
    "ftsc_seed42_fp32": (["FTSC"], [42], ["pt32", "trt32"]),
    "ftsc_three_seed_fp32": (["FTSC"], [42, 43, 44], ["pt32", "trt32"]),
    "ftsc_all_precision": (["FTSC"], [42, 43, 44], ["pt32", "trt32", "pt16", "trt16"]),
    "report_fp32": (None, [42, 43, 44], ["pt32", "trt32"]),
    "report_all_precision": (None, [42, 43, 44], ["pt32", "trt32", "pt16", "trt16"]),
}

def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)

def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):
            digest.update(buf)
    return digest.hexdigest()

def scope_for(dataset,stage,manifest):
    need(dataset in ("TinyPerson","Varroa"),"Unsupported dataset")
    need(stage in STAGES, "Unknown deployment stage")
    cases, seeds, backends=STAGES[stage]
    inventory, _, _=deploy.load_report_scope(manifest,dataset)
    cases=list(inventory) if cases is None else cases
    need(all(case in inventory for case in cases),"Case missing from audited manifest")
    return dict(dataset=dataset,stage=stage,cases=cases,seeds=seeds,
                backends=backends,evaluation_protocol=deploy.NATIVE_VAL_PROTOCOL,
                expected_jobs=len(cases)*len(seeds)*len(backends))

def audit(dataset,stage,base_project,manifest,preflight,platform="marimo_live"):
    scope=scope_for(dataset,stage,manifest)
    root=Path(base_project)/deploy.NATIVE_VAL_PROTOCOL/deploy.NATIVE_VAL_RUNTIME_REVISION
    cases,_sections,_unresolved=deploy.load_report_scope(manifest,dataset)
    diagnostic=json.loads(Path(preflight).read_text(encoding="utf-8"))
    need(diagnostic.get("schema")=="ftsc_other_dataset_deployment_preprocess_audit_v1" and
         diagnostic.get("dataset")==dataset and
         diagnostic.get("status")=="PASS_EXACT_INPUT_PARITY",
         "Preflight did not verify identical native PT/static TRT inputs")
    source_hashes=diagnostic.get("source_code_sha256")
    need(isinstance(source_hashes,dict) and source_hashes,
         "Missing preprocessing source-code provenance")
    need(all((deploy.ROOT/name).is_file() and
             file_sha(deploy.ROOT/name)==digest
             for name,digest in source_hashes.items()),
         "Live preprocessing code differs from audited preflight source")
    if dataset=="TinyPerson":
        need(diagnostic.get("all_identical") is True and diagnostic.get("total",0)>0,
             "TinyPerson corner input parity incomplete")
    else:
        by_seed={x["seed"]:x for x in diagnostic.get("seeds",[])}
        need(all(seed in by_seed and by_seed[seed].get("all_identical") and
                 by_seed[seed].get("total",0)>0 for seed in scope["seeds"]),
             "Varroa preflight lacks validated image tensors for all requested test seeds")
    found={}
    issues=[]
    pairs=[]
    from utils.ftsc_deployment_backend_eval_v1 import source_path,path_for
    cfg=SimpleNamespace(dataset=dataset,platform=platform,
                        evaluation_protocol=deploy.NATIVE_VAL_PROTOCOL)
    for case in scope["cases"]:
        for seed in scope["seeds"]:
            source=cases[case][seed]
            for backend in scope["backends"]:
                tag=f"{case}/seed_{seed}/{backend}"
                folder=path_for(root,dataset,case,seed,backend)
                try:
                    path=folder/"metrics.json"
                    need(path.is_file(),"Missing metrics.json")
                    row=json.loads(path.read_text(encoding="utf-8"))
                    expected=deploy.expected_identity(cfg,dict(
                        case=case,seed=seed,backend=backend,source=source["path"]))
                    need(all(row.get(k)==v for k,v in expected.items()),
                         "Incorrect V2 evaluator/protocol source fingerprint")
                    need(row.get("evaluator")==deploy.evaluator_name(cfg),
                         "Unexpected native evaluator")
                    need(row.get("source_sha256")==source["sha256"],
                         "Checkpoint source SHA differs from manifest")
                    need(row.get("imgsz")==640,
                         "Backend validation must retain native dataset imgsz=640")
                    ckpt=source_path(root,dataset,case,seed)
                    need(ckpt.is_file() and file_sha(ckpt)==source["sha256"],
                         "Physical source checkpoint missing or changed")
                    model_file=(ckpt if backend.startswith("pt")
                                else folder/"export/source.engine")
                    need(model_file.is_file() and
                         file_sha(model_file)==row.get("model_artifact_sha256"),
                         "Physical model artifact mismatch")
                    if backend.startswith("pt"):
                        need(row["model_artifact_sha256"]==row["source_sha256"],
                             "PyTorch model artifact differs from pinned checkpoint")
                    else:
                        export_manifest=folder/"export/export_manifest.json"
                        need(export_manifest.is_file(),"Missing TensorRT export manifest")
                        exp=json.loads(export_manifest.read_text(encoding="utf-8"))
                        fp=exp.get("fingerprint",{})
                        need(exp.get("artifact_sha256")==row["model_artifact_sha256"]
                             and fp.get("backend")==backend and fp.get("imgsz")==640
                             and fp.get("source_sha256")==row["source_sha256"],
                             "TensorRT export fingerprint has an incorrect 640/source/backend contract")
                    deploy.validate_metric_values(row["metrics"],dataset,deploy.NATIVE_VAL_PROTOCOL)
                    deploy.verify_native_val_artifacts(row,folder,dataset)
                    if dataset=="TinyPerson":
                        for key in ("corner_manifest_sha256","gt_sha256"):
                            need(row.get(key)==diagnostic.get(
                                "corner_manifest_sha256" if key=="corner_manifest_sha256"
                                else "source_original_gt_sha256"),
                                "TinyPerson frozen merged GT/corner mismatch")
                    else:
                        need(row.get("data_yaml_sha256")==by_seed[seed]["yaml_sha256"],
                             "Varroa frozen test YAML differs from preflight")
                    found[(case,seed,backend)]=row
                except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
                    issues.append(dict(job=tag,error=f"{type(error).__name__}: {error}"))
    keys=("test_merged/AP50","test_merged/mAP50-95","test_merged/AP50-Small") if dataset=="TinyPerson" else (
          "test/metrics/mAP50(B)","test/metrics/mAP50-95(B)","test_size/AP50-Small")
    flags=[]
    for case in scope["cases"]:
        for seed in scope["seeds"]:
            row32=found.get((case,seed,"pt32"))
            if not row32:continue
            for backend in ("trt32","pt16","trt16"):
                row=found.get((case,seed,backend))
                if not row:continue
                reference_backend="pt16" if backend=="trt16" else "pt32"
                ref=found.get((case,seed,reference_backend))
                if not ref:
                    issues.append(dict(job=f"{case}/{seed}/{backend}",
                                       error=f"Missing {reference_backend} reference"))
                    continue
                for attr in (("gt_sha256","corner_manifest_sha256")
                             if dataset=="TinyPerson" else
                             ("data_yaml_sha256","gt_sha256")):
                    if row.get(attr)!=ref.get(attr):
                        issues.append(dict(job=f"{case}/{seed}/{backend}",
                                           error=f"Mismatched {attr} between backend and PT"))
                        break
                else:
                    diff=[100*(row["metrics"][key]-ref["metrics"][key]) for key in keys]
                    pair=dict(case=case,seed=seed,backend=backend,reference=reference_backend,
                              AP50_delta_pp=diff[0],mAP50_95_delta_pp=diff[1],
                              AP50_Small_delta_pp=diff[2])
                    pairs.append(pair)
                    threshold=0.5 if backend=="trt32" else 1.0
                    if abs(diff[0])>threshold:
                        flags.append({**pair,"review_threshold_pp":threshold})
    summaries=[]
    for case in scope["cases"]:
        for backend in scope["backends"]:
            group=[found.get((case,s,backend)) for s in (42,43,44)]
            if any(item is None for item in group):continue
            item=dict(case=case,backend=backend,seeds=[42,43,44])
            for name,key in zip(("AP50","mAP50_95","AP50_Small"),keys):
                arr=[100*v["metrics"][key] for v in group]
                item[name+"_mean_pct"]=statistics.fmean(arr)
                item[name+"_sample_std_pp"]=statistics.stdev(arr)
            summaries.append(item)
    status=("INCOMPLETE_OR_INVALID" if issues or len(found)!=scope["expected_jobs"]
            else "COMPLETE_NEEDS_ACCURACY_REVIEW" if flags else "PASS_COMPLETE")
    return dict(status=status,scope=scope,valid_jobs=len(found),issues=issues,
                accuracy_review_flags=flags,pairwise_comparisons=pairs,
                three_seed_preview=summaries,
                preflight_sha256=file_sha(preflight),
                note="Preflight certifies image tensors under observed source paths; "
                     "AP deltas compare independent backend predictions, not Jetson FPS.")

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,choices=("TinyPerson","Varroa"))
    p.add_argument("--stage",required=True,choices=tuple(STAGES))
    p.add_argument("--base-project",type=Path,default=deploy.ROOT/"runs/ftsc_deployment_eval_v1")
    p.add_argument("--report-manifest",type=Path,default=deploy.REPORT_MANIFEST)
    p.add_argument("--platform",default="marimo_live")
    p.add_argument("--preflight-report",type=Path,required=False)
    p.add_argument("--describe-stage",action="store_true")
    args=p.parse_args(argv)
    scoped=scope_for(args.dataset,args.stage,args.report_manifest)
    if args.describe_stage:
        print(json.dumps(scoped,indent=2))
        return 0
    need(args.preflight_report is not None,"--preflight-report required for audit")
    report=audit(args.dataset,args.stage,args.base_project,args.report_manifest,
                 args.preflight_report,args.platform)
    root=args.base_project/deploy.NATIVE_VAL_PROTOCOL/deploy.NATIVE_VAL_RUNTIME_REVISION/args.dataset/"_audits"
    root.mkdir(parents=True,exist_ok=True)
    path=root/(args.stage+"_audit.json")
    temp=path.with_suffix(".json.tmp")
    need(not temp.exists(),f"Interrupted audit: {temp}")
    temp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\\n",encoding="utf-8")
    temp.replace(path)
    print(json.dumps(dict(status=report["status"],stage=args.stage,
                          expected_jobs=scoped["expected_jobs"],valid_jobs=report["valid_jobs"],
                          issues=report["issues"],accuracy_review_flags=report["accuracy_review_flags"],
                          pairwise_comparisons=report["pairwise_comparisons"],
                          three_seed_preview=report["three_seed_preview"],
                          audit_report=str(path)),indent=2))
    return 0 if report["status"]=="PASS_COMPLETE" else 2

if __name__=="__main__":
    raise SystemExit(main())
