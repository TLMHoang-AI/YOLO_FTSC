#!/usr/bin/env python3
"""Dataset-specific safe deployment workflow, without editing validated LEVIR V3.

PLAN/VERIFY do not require prepared evaluation datasets. PREFLIGHT inspects
all real image tensors before GPU work. EVALUATE only proceeds after exact
PT-vs-static-TRT input parity; AUDIT checks independent backend results.
If preflight fails, do not silently switch img size, evaluation semantics
or use LEVIR's 544 static engine on these datasets.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from utils import ftsc_deployment_backend_eval_v1 as deploy
from utils import ftsc_deployment_other_dataset_audit_v1 as audit
from utils.ftsc_deployment_other_dataset_preflight_v1 import need

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_FILES=(
    "utils/ftsc_deployment_backend_eval_v1.py",
    "utils/ftsc_deployment_other_dataset_preflight_v1.py",
    "utils/ftsc_deployment_other_dataset_audit_v1.py",
    "utils/ftsc_report_deployment_case_manifest_v1.json",
)

def exec_python(module,args,env):
    argv=[sys.executable,"-m",module,*[str(x) for x in args]]
    result=subprocess.run(argv,cwd=str(ROOT),env=env,check=True)
    return result

def prepare_data(cfg,scope):
    if cfg.dataset=="TinyPerson":
        from train_tinyperson_scripts.train_all_tinyperson import prepare_test_set
        test_out=prepare_test_set(cfg.data_root,cfg.prepared_root)
        need(test_out.is_dir(),"TinyPerson test crops missing")
        return (["--tiny-data-root",str(cfg.data_root),
                 "--tiny-test-out",str(test_out)],
                ["--tiny-data-root",str(cfg.data_root),
                 "--tiny-test-out",str(test_out)])
    from train_levir_scripts.train_varroa_dbss_hit import dataset_for_seed
    from evaluate_test.standard_detection_metrics import ensure_local_ultralytics
    ensure_local_ultralytics()
    for seed in scope["seeds"]:
        yml=dataset_for_seed(SimpleNamespace(data_root=cfg.data_root,
                                              dataset_root=cfg.prepared_root),seed)
        need(yml.is_file(),f"Varroa seed {seed} has no verified YAML")
    yml_template=str(cfg.prepared_root/"varroa_yolo_seed{seed}/varroa.yaml")
    return (["--data-yaml-template",yml_template],
            ["--varroa-yaml-template",yml_template,
             "--seeds",*map(str,scope["seeds"]),
             "--varroa-reference-batch",str(cfg.varroa_reference_batch)])

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True,choices=("TinyPerson","Varroa"))
    p.add_argument("--stage",required=True,choices=tuple(audit.STAGES))
    p.add_argument("--phase",default="plan",
                   choices=("plan","verify","preflight","evaluate","audit","aggregate"))
    p.add_argument("--allow-evaluate",action="store_true")
    p.add_argument("--run-offline-tests",action="store_true")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--platform",default="marimo_live")
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--data-root",type=Path)
    p.add_argument("--prepared-root",type=Path)
    p.add_argument("--varroa-reference-batch",type=int,default=4)
    p.add_argument("--report-manifest",type=Path,default=deploy.REPORT_MANIFEST)
    p.add_argument("--base-project",type=Path,default=ROOT/"runs/ftsc_deployment_eval_v1")
    args=p.parse_args(argv)
    need(args.phase!="evaluate" or args.allow_evaluate,
         "Explicit --allow-evaluate required; never train or upload")
    need(args.workers>=0 and args.varroa_reference_batch>0,
         "Invalid workers or Varroa reference batch")
    need(all((ROOT/name).is_file() for name in REQUIRED_FILES),
         "Notebook synced without all dependency files")
    scope=audit.scope_for(args.dataset,args.stage,args.report_manifest)
    print("DEPLOYMENT_STAGE_SCOPE",json.dumps(scope,indent=2),flush=True)
    env=dict(os.environ)
    env["PYTHONPATH"]=os.pathsep.join([
        str(ROOT),str(ROOT/"models_related/ultralytics"),env.get("PYTHONPATH","")])
    if args.run_offline_tests:
        exec_python("unittest",[
            "utils.tests.test_ftsc_deployment_other_dataset_preflight_v1",
        ],env)
    runner_args=[
        "--contract-version",deploy.VERSION,
        "--scope","report","--dataset",args.dataset,
        "--phase",args.phase if args.phase in ("plan","verify","evaluate","aggregate")
                   else "plan",
        "--evaluation-protocol",deploy.NATIVE_VAL_PROTOCOL,
        "--cases",*scope["cases"],
        "--seeds",*map(str,scope["seeds"]),
        "--backends",*scope["backends"],
        "--platform",args.platform,"--device",args.device,
        "--workers",str(args.workers),"--batch","1",
        "--project",str(args.base_project),
        "--report-manifest",str(args.report_manifest),
    ]
    if args.phase in ("plan","verify"):
        exec_python("utils.ftsc_deployment_backend_eval_v1",runner_args,env)
        return 0
    need(args.data_root is not None and args.prepared_root is not None,
         "Data root and prepared root required for preflight/evaluate/audit")
    args.data_root=args.data_root.expanduser().resolve()
    args.prepared_root=args.prepared_root.expanduser().resolve()
    data_args,probe_args=prepare_data(args,scope)
    report_dir=(args.base_project/deploy.NATIVE_VAL_PROTOCOL/
                deploy.NATIVE_VAL_RUNTIME_REVISION/args.dataset/"_audits")
    report_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out=report_dir/f"preflight_{args.stage}_{stamp}.json"
    exec_python("utils.ftsc_deployment_other_dataset_preflight_v1",[
        "--dataset",args.dataset,*probe_args,"--output",str(out)],env)
    report=json.loads(out.read_text(encoding="utf-8"))
    print("PREFLIGHT_REPORT",out,"STATUS",report["status"],flush=True)
    if report.get("status")!="PASS_EXACT_INPUT_PARITY":
        raise RuntimeError(
            f"{args.dataset}: native paper preprocessing and static V2 engine do "
            f"not see identical tensors. See {out}. "
            "BLOCKED before evaluation: requires dataset-specific TRT adapter."
        )
    if args.phase=="preflight":
        print("PREFLIGHT_PASSED: no AP evaluation or engine export",flush=True)
        return 0
    audit_args=["--dataset",args.dataset,"--stage",args.stage,
                "--base-project",str(args.base_project),
                "--report-manifest",str(args.report_manifest),
                "--platform",args.platform,"--preflight-report",str(out)]
    if args.phase=="evaluate":
        verify_args=list(runner_args)
        verify_args[verify_args.index("--phase")+1]="verify"
        exec_python("utils.ftsc_deployment_backend_eval_v1",verify_args,env)
        if args.dataset=="Varroa":
            # V2 backend export may need pre-installed CUDA/ONNX dependencies;
            # import failures fail closed rather than triggering training.
            pass
        exec_python("utils.ftsc_deployment_backend_eval_v1",
                    [*runner_args,*data_args,"--confirm-eval"],env)
    if args.phase=="aggregate":
        need(scope["seeds"]==[42,43,44],
             "Aggregate requires all three seeds, not seed42 smoke")
    exec_python("utils.ftsc_deployment_other_dataset_audit_v1",audit_args,env)
    if args.phase=="aggregate":
        exec_python("utils.ftsc_deployment_backend_eval_v1",runner_args,env)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
