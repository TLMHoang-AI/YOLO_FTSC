#!/usr/bin/env python3
"""Fix LEVIR paper deployment with a STRICT two-case re-evaluation.

WHY: Previously COLOR_CUE used legacy RECON_C_H2_LEGACY_V1, not the
actual capstone C_H2_POSTFUSION_V2; EDGE_CUE ES1 was the correct checkpoint
but did not reproduce the historical three-seed paper AP. Never silently
rewrite or 'correct' historical AP to fit an engine result.

The workflow runs two independent native historical passes from the SAME
SHA-pinned checkpoint on every backend:
  (1) LEVIR V3 main: original YOLO.val nominal512/rectTrue/544 canvas,
      conf=.001, NMS=.5; PT8 and TRT static544 batch1; exact tensor SHA;
  (2) paper small: original YOLO.predict 512 conf=.25 NMS=.5
      and TinyBenchmark area Small, PT8/TRT static512, exact input SHA.
No retraining, no HF writes, no edits to old V3 or paper runtime sources.
FTSC, YOLOv9t+FTSC, and YOLO11n+FTSC already reproduced historical AP and
are NOT re-evaluated. An optional final report assembles those three
SHA-audited old cases with newly re-evaluated EDGE/COLOR, retaining flags.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from utils import ftsc_deployment_backend_eval_v1 as deploy
from utils import ftsc_deployment_paper_backend_v1 as base

ROOT = deploy.ROOT
MANIFEST = ROOT / "utils/ftsc_levir_paper_recovery_cases_v1.json"
CONTRACT = "ftsc_levir_paper_edge_color_recovery_v1"
REVISION = "correct_color_postfusion_es1_rect544_small512_v1"
DEFAULT_BASE_PROJECT = ROOT / "runs/ftsc_deployment_levir_paper_recovery_v1"
BACKENDS = ("pt32", "trt32", "pt16", "trt16")
RECOVER_CASES = ("COLOR_CUE", "EDGE_CUE")
FIVE_CASES = ("FTSC", "FTSC_YOLOV9T", "FTSC_YOLO11N", "EDGE_CUE", "COLOR_CUE")
STAGES = {
    "repair_seed42_fp32": (RECOVER_CASES, (42,), ("pt32", "trt32")),
    "repair_three_seed_fp32": (RECOVER_CASES, (42, 43, 44), ("pt32", "trt32")),
    "repair_all_precision": (RECOVER_CASES, (42, 43, 44), BACKENDS),
}
# Independently verified HF checkpoint_reference.json hashes, never paper values.
COLOR_SHA = {
    42: "f8527b339d059d5806dc13909e49ced8518518af0ffd78185c40f9d08f7cc39f",
    43: "ad0b8c0dac89aea17baa3a29798009e230d670292c8a7a900060ba46d4c21e21",
    44: "97ebbe0643eb190c5a7dc654345d0e82382b150d93c9f444a9493ce4b0eeccf2",
}
EDGE_SHA = {
    42: "32427262777cf7996b91c99ef8ec215a07da8b2215b29eadb33f0444536c4eab",
    43: "e1606a16a0246c3b1a9fb6e6edaa4c85ea26cfb29b8874d86bd38ba569fd87e1",
    44: "41fe24ecede62200c4e0b38941dde744347f3c5f1a3b26bcf368e7c04b9e2507",
}
SOURCE_FILES = (
    "utils/ftsc_deployment_levir_edge_color_recovery_v1.py",
    "utils/ftsc_levir_paper_recovery_cases_v1.json",
    "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_colorcue_postfusion_v2.yaml",
    "models_related/models_config/yolov8/levir/yolov8n_p2_levir_ftsc_h2_edge_stab_es1_scale025.yaml",
)
ORIGINAL_PAPER_CONTRACT = base.CONTRACT
# The original 132-job historical paper pass was exported under the frozen V1
# namespace. The imported base runner has since moved to a V2 revision:
# never derive a prior artifact's location from mutable base.REVISION.
ORIGINAL_PAPER_REVISION = "predict512_native_rect_dynamic_trt_v1"
THREE_VERIFIED = ("FTSC", "FTSC_YOLOV9T", "FTSC_YOLO11N")
SCORE_KEYS = deploy.metric_keys("LEVIR-Ship")
# Original ES1 per-seed native val test scores, from the frozen
# results/ftsc_edge_stability_summary_runs_full.csv. No original
# per-seed AP50-Small or training-time best.pt SHA was recorded there.
EDGE_HISTORICAL_MAIN = {
    42: (0.8260294168220195, 0.3171464125543804),
    43: (0.8246134330727788, 0.32141191051992335),
    44: (0.8198411295359872, 0.32037664548773315),
}
# Separate original-HF re-evaluation (results/es1_original_hf_metrics_by_seed.csv).
# These are comparison references only, never used to populate new metrics.
EDGE_PRIOR_HF = {
    42: (0.8153036015472848, 0.3108739792013761, 0.7926973828769855),
    43: (0.8246134330727788, 0.32141191051992335, 0.8506926952099043),
    44: (0.8198411295359872, 0.32037664548773315, 0.8087999545309826),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_recovery_manifest(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    require(document.get("schema") == deploy.REPORT_SCHEMA
            and document.get("repo_id") == deploy.REPO_ID
            and document.get("seeds") == [42, 43, 44],
            "Invalid canonical paper recovery manifest")
    only = document["datasets"]["LEVIR-Ship"]["cases"]
    require(set(only) == set(RECOVER_CASES), "Recovery must ONLY run Edge/Color")
    for case, prefix, hashes in (
        ("COLOR_CUE", "LEVIR-Ship/journal_experiments/phase2/C_H2_POSTFUSION_V2", COLOR_SHA),
        ("EDGE_CUE", "LEVIR-Ship/original_runs/ES1", EDGE_SHA),
    ):
        item = only[case]
        require(item["source_run"] == prefix and set(item["seeds"]) == {"42","43","44"},
                f"Wrong real-paper checkpoint case: {case}")
        for seed in (42,43,44):
            row = item["seeds"][str(seed)]
            require(row["path"] == f"{prefix}/seed_{seed}/weights/best.pt"
                    and row["sha256"] == hashes[seed],
                    f"{case}/seed_{seed}: checkpoint differs from paper/HF")
    for rel in SOURCE_FILES:
        require((ROOT/rel).is_file(), f"Missing recovery source/config {rel}")
    return document


def install_process_contract():
    """Process-local override; existing LEVIR V2 files stay byte-identical."""
    old_hashes = base.source_hashes
    def hashes():
        values = old_hashes()
        values.update({path: base.sha(ROOT/path) for path in SOURCE_FILES})
        return values
    base.source_hashes = hashes
    base.CONTRACT = CONTRACT
    base.REVISION = REVISION
    base.STAGES = STAGES
    # Preserve the real historical targets (81.05/30.51/80.30 and 82.35/...).
    # These are REVIEW anchors, not data to inject into the evaluation.
    return hashes


def original_report_manifest():
    path = deploy.REPORT_MANIFEST
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["datasets"]["LEVIR-Ship"]["cases"]


def parsed_options(argv):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",required=True,
                        choices=("plan","verify","preflight","evaluate","audit","aggregate","complete","diagnose-edge"))
    parser.add_argument("--stage",choices=tuple(STAGES),default="repair_seed42_fp32")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--platform",default="marimo_live")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--allow-evaluate",action="store_true")
    parser.add_argument("--run-offline-tests",action="store_true")
    parser.add_argument("--base-project",type=Path,default=DEFAULT_BASE_PROJECT)
    parser.add_argument("--data-yaml",type=Path,required=False,
                        default=ROOT/"datasets/levir_ship_yolo_seed42/levir_ship.yaml")
    parser.add_argument("--historical-es1-best42",type=Path,default=None,
                        help="Optional provenance-confirmed training seed42 best.pt; diagnostic only")
    args=parser.parse_args(argv)
    args.base_project=args.base_project.expanduser().resolve()
    args.report_manifest=MANIFEST
    args.data_yaml=args.data_yaml.expanduser().resolve()
    if args.historical_es1_best42 is not None:
        args.historical_es1_best42=args.historical_es1_best42.expanduser().resolve()
    args.dataset="LEVIR-Ship"
    args.cases=list(STAGES[args.stage][0])
    args.seeds=list(STAGES[args.stage][1])
    args.backends=list(STAGES[args.stage][2])
    args.paper_root=args.base_project/CONTRACT/REVISION
    require(args.workers>=0,"Invalid worker count")
    return args


def base_command(args,phase):
    cmd=["--dataset","LEVIR-Ship","--phase",phase,
         "--stage",args.stage,"--report-manifest",str(MANIFEST),
         "--base-project",str(args.base_project),
         "--device",args.device,"--workers",str(args.workers),
         "--platform",args.platform]
    if phase in ("preflight","evaluate"):
        cmd.extend(["--data-yaml-template",str(args.data_yaml)])
    if phase=="evaluate":
        cmd.append("--allow-evaluate")
    return cmd


def v3_main_command(args):
    return [
        sys.executable,"-m","utils.ftsc_deployment_backend_eval_v1",
        "--dataset","LEVIR-Ship","--scope","report",
        "--cases",*args.cases,"--seeds",*[str(s) for s in args.seeds],
        "--backends",*args.backends,"--phase","evaluate",
        "--evaluation-protocol",deploy.RECT_PARITY_PROTOCOL,
        "--report-manifest",str(MANIFEST),
        "--platform",args.platform,"--device",args.device,
        "--workers",str(args.workers),"--batch","1",
        "--project",str(args.base_project),
        "--data-yaml-template",str(args.data_yaml),
        "--confirm-eval",
    ]


def verify_preflight_first(args):
    require(args.data_yaml.is_file(),f"LEVIR frozen data YAML missing: {args.data_yaml}")
    _,images=base._native_test_yaml(
        argparse.Namespace(dataset="LEVIR-Ship",
                           data_yaml_template=str(args.data_yaml)),42)
    require(len(images)==788,"Expected 788 unchanged LEVIR test images")
    return base.verify_preflight(argparse.Namespace(
        dataset=args.dataset,stage=args.stage,paper_root=args.paper_root,
        report_manifest=MANIFEST,cases=args.cases,
        seeds=args.seeds,backends=args.backends,
        data_yaml_template=str(args.data_yaml)))


def verify_all_v3_main(args):
    """V3 backend runner may return zero even with per-case failures: detect all."""
    inventory, _,_ = deploy.load_report_scope(MANIFEST,"LEVIR-Ship")
    ns=argparse.Namespace(dataset="LEVIR-Ship",platform=args.platform,
         base_project=args.base_project,inventory=inventory)
    found=[]
    for case in args.cases:
        for seed in args.seeds:
            for backend in args.backends:
                result,metadata=base._verify_v3_main(
                    ns,case,seed,backend,inventory[case][seed]["sha256"])
                require(all(math.isfinite(result[k]) for k in
                         ("test/metrics/mAP50(B)","test/metrics/mAP50-95(B)")),
                        "Nonfinite native main AP")
                require(metadata.get("main_input_count")==788,
                        "Incomplete LEVIR 544-pixel validation input manifest")
                found.append((case,seed,backend))
    return found


def audit_recovery(args):
    inventory, _,_ = deploy.load_report_scope(MANIFEST,"LEVIR-Ship")
    ns=argparse.Namespace(
        dataset="LEVIR-Ship",stage=args.stage,paper_root=args.paper_root,
        base_project=args.base_project,report_manifest=MANIFEST,
        platform=args.platform,device=args.device)
    report=base.audit(ns)
    require(report["expected_jobs"]==len(args.cases)*len(args.seeds)*len(args.backends),
            "Audit expected wrong number of jobs")
    return report


def safe_write(path,value):
    """Immutable JSON or CSV: resuming cannot silently change published AP."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    if path.exists():
        require(path.read_text(encoding="utf-8")==data,
                f"Existing audit artifact changed: {path}")
        return path
    tmp=path.with_suffix(path.suffix+".tmp")
    require(not tmp.exists(),f"Incomplete artifact: {tmp}")
    tmp.write_text(data,encoding="utf-8")
    tmp.replace(path)
    return path


def require_same_artifact(left, right, label):
    """Compare exact bytes, not just individually valid stored SHA fields."""
    left, right = Path(left), Path(right)
    require(left.is_file() and right.is_file(),
            f"{label}: artifact missing ({left}, {right})")
    require(base.sha(left) == base.sha(right) and left.read_bytes() == right.read_bytes(),
            f"{label}: cross-case artifact bytes differ")


def audit_old_frozen_three(args):
    """Verify frozen V1 three-case records and exact same test inputs/GT."""
    original=original_report_manifest()
    folder=ROOT/"runs/ftsc_deployment_eval_v1"/ORIGINAL_PAPER_CONTRACT/ORIGINAL_PAPER_REVISION
    # The old three and newly fixed Color MUST be checked on identical
    # original native 512 predictor pixels and 544 main-validator pixels.
    new_v3=args.base_project/deploy.RECT_PARITY_PROTOCOL/deploy.RECT_PARITY_RUNTIME_REVISION
    for seed in (42,43,44):
        color_small=base.directory(args.paper_root,"LEVIR-Ship","COLOR_CUE",seed,"pt32")
        color_main=base.directory(new_v3,"LEVIR-Ship","COLOR_CUE",seed,"pt32")
        # Every backend must use exactly the same frozen test inputs/labels,
        # across BOTH corrected cases (not merely internally hashed files).
        for case in RECOVER_CASES:
            for backend in BACKENDS:
                small=base.directory(args.paper_root,"LEVIR-Ship",case,seed,backend)
                main=base.directory(new_v3,"LEVIR-Ship",case,seed,backend)
                for filename in ("small_inputs.json","test_size_ground_truth.json"):
                    require_same_artifact(color_small/filename,small/filename,
                                          f"{case}/{seed}/{backend} predictor {filename}")
                for filename in ("test_val_preprocessed_inputs_sha256.json",
                                 "test_size_ground_truth.json"):
                    require_same_artifact(color_main/filename,main/filename,
                                          f"{case}/{seed}/{backend} validator {filename}")
    source_code_baseline=None
    collected={}
    for case in THREE_VERIFIED:
        require(case in original,"Missing original verified FTSC/cross-YOLO case")
        for seed in (42,43,44):
            for backend in BACKENDS:
                location=base.directory(folder,"LEVIR-Ship",case,seed,backend)
                path=location/"metrics.json"
                require(path.is_file(),f"Original canonical result missing: {path}")
                item=json.loads(path.read_text(encoding="utf-8"))
                ref=original[case]["seeds"][str(seed)]
                require(item.get("contract")==ORIGINAL_PAPER_CONTRACT
                        and item.get("revision")==ORIGINAL_PAPER_REVISION
                        and item.get("case")==case and item.get("seed")==seed
                        and item.get("backend")==backend
                        and item.get("source")==ref["path"]
                        and item.get("source_sha256")==ref["sha256"]
                        and item.get("platform")==args.platform,
                        f"Old metric source/cache differs from pinned case: {case}/{seed}/{backend}")
                # Existing source hash can differ from current code if source
                # was later edited; require INTERNAL consistency, never rewrite.
                code=item.get("source_files_sha256")
                require(isinstance(code,dict) and code, "Old source provenance absent")
                if source_code_baseline is None:
                    source_code_baseline=code
                require(code==source_code_baseline,
                        "Old reference cases were produced under mixed code revisions")
                require(all(isinstance(item.get("metrics",{}).get(k),(float,int))
                            and math.isfinite(item["metrics"][k])
                            and 0<=item["metrics"][k]<=1 for k in SCORE_KEYS),
                        f"Old metric missing/out of range: {case}/{seed}/{backend}")
                model=Path(item["model_path"])
                require(model.is_file() and base.sha(model)==item["model_artifact_sha256"],
                        "Old source/engine artifact mutated")
                checkpoint=deploy.source_path(folder,"LEVIR-Ship",case,seed)
                require(checkpoint.is_file() and base.sha(checkpoint)==ref["sha256"],
                        "Old best.pt missing or hash changed")
                for kind,filename in (("small_input_manifest_sha256","small_inputs.json"),
                                      ("small_predictions_sha256","test_size_predictions.json"),
                                      ("small_gt_sha256","test_size_ground_truth.json")):
                    artifact=location/filename
                    require(artifact.is_file() and base.sha(artifact)==item.get(kind),
                            f"Old pixel/prediction/GT artifact changed: {artifact}")
                v3base=ROOT/"runs/ftsc_deployment_eval_v1"/deploy.RECT_PARITY_PROTOCOL/deploy.RECT_PARITY_RUNTIME_REVISION
                v3=base.directory(v3base,"LEVIR-Ship",case,seed,backend)/"metrics.json"
                require(v3.is_file() and base.sha(v3)==item.get("main_v3_metrics_sha256"),
                        "Old original native validator provenance changed")
                new_small=base.directory(args.paper_root,"LEVIR-Ship","COLOR_CUE",seed,"pt32")
                for filename in ("small_inputs.json", "test_size_ground_truth.json"):
                    require_same_artifact(location/filename, new_small/filename,
                                          f"Old {case}/{seed}/{backend} vs Color predictor {filename}")
                old_v3=base.directory(v3base,"LEVIR-Ship",case,seed,backend)
                new_val=base.directory(new_v3,"LEVIR-Ship","COLOR_CUE",seed,"pt32")
                for filename in ("test_val_preprocessed_inputs_sha256.json",
                                 "test_size_ground_truth.json"):
                    require_same_artifact(old_v3/filename, new_val/filename,
                                          f"Old {case}/{seed}/{backend} vs Color validator {filename}")
                if backend!="pt32":
                    a=base.directory(folder,"LEVIR-Ship",case,seed,"pt32")/"small_inputs.json"
                    b=location/"small_inputs.json"
                    require(a.is_file() and a.read_bytes()==b.read_bytes(),
                            "Old reference backend input pixels differ")
                collected[(case,seed,backend)]=item
    require(len(collected)==36,"Incomplete original 3-case/3-seed/4-backend references")
    return collected,folder


def aggregate_three_seed(case,backend,rows):
    entry={"case":case,"backend":backend,"n":3}
    for name in SCORE_KEYS:
        values=[100*rows[(case,seed,backend)]["metrics"][name] for seed in (42,43,44)]
        entry[name+"_mean_pct"]=statistics.fmean(values)
        entry[name+"_std_pp"]=statistics.stdev(values)
    return entry


def edge_drift_summary(rows):
    """Quantify ES1 historical-vs-HF differences without inventing a cause."""
    require(set(rows)=={42,43,44},"ES1 diagnosis needs PT32 for seeds 42/43/44")
    detail=[]
    for seed in (42,43,44):
        record=rows[seed]
        require(record.get("source_sha256")==EDGE_SHA[seed],
                f"ES1 seed{seed}: checkpoint SHA does not match pinned HF source")
        metrics=record["metrics"]
        observed=[float(metrics[key]) for key in SCORE_KEYS]
        require(all(math.isfinite(value) and 0<=value<=1 for value in observed),
                f"ES1 seed{seed}: invalid observed metrics")
        historic=EDGE_HISTORICAL_MAIN[seed]
        prior=EDGE_PRIOR_HF[seed]
        detail.append({
            "seed":seed,
            "checkpoint_sha256":record["source_sha256"],
            "historical_main_pct":[100*value for value in historic],
            "current_pt32_pct":[100*value for value in observed],
            "main_drift_from_historical_pp":[100*(observed[i]-historic[i]) for i in range(2)],
            "original_hf_prior_pct":[100*value for value in prior],
            "drift_from_original_hf_prior_pp":[100*(observed[i]-prior[i]) for i in range(3)],
        })
    small_mean=statistics.fmean(row["current_pt32_pct"][2] for row in detail)
    paper_small=float(base.PAPER_ANCHORS["LEVIR-Ship"]["EDGE_CUE"][2])
    return {
        "per_seed":detail,
        "paper_small_mean_pct":paper_small,
        "current_small_mean_pct":small_mean,
        "small_mean_drift_pp":small_mean-paper_small,
        "historical_small_per_seed_available":False,
        "observations":{
            "previous_hf_seed42_main_gap_pp":[100*(EDGE_PRIOR_HF[42][i]-EDGE_HISTORICAL_MAIN[42][i])
                                               for i in range(2)],
            "previous_hf_seeds43_44_main_match_historical":
                all(all(abs(EDGE_PRIOR_HF[seed][i]-EDGE_HISTORICAL_MAIN[seed][i])<1e-12
                        for i in range(2)) for seed in (43,44)),
            "hf_re_evaluation_in_prior_snapshot":
                "Both levirship/ES1 and curated LEVIR-Ship/original_runs/ES1 map "
                "to the same recorded per-seed SHA; original training-time SHA is unavailable",
        },
        "limitations":[
            "No original ES1 seed42 training-time best.pt SHA to prove training checkpoint identity",
            "No original per-seed AP50-Small detections to decompose its mean drift",
            "Original training test-input, label, and evaluator byte manifests not available",
            "A 3-seed mean drift cannot be attributed to seed42 Small AP from main AP drift alone",
        ],
    }


def diagnose_edge(args, report=None):
    """Read-only ES1 provenance analysis; diagnostic output is immutable."""
    require(args.stage in ("repair_three_seed_fp32","repair_all_precision"),
            "Run ES1 diagnosis only after a full three-seed stage")
    report=report if report is not None else audit_recovery(args)
    require(report.get("valid_jobs")==report.get("expected_jobs") and not report.get("issues"),
            "ES1 diagnostic requires full valid paper recovery stage")
    rows={}
    for seed in (42,43,44):
        path=base.directory(args.paper_root,"LEVIR-Ship","EDGE_CUE",seed,"pt32")/"metrics.json"
        require(path.is_file(),f"Missing ES1 PT32 verified metrics: {path}")
        row=json.loads(path.read_text(encoding="utf-8"))
        require(row.get("case")=="EDGE_CUE" and row.get("seed")==seed
                and row.get("backend")=="pt32" and row.get("contract")==CONTRACT
                and row.get("revision")==REVISION
                and row.get("source_files_sha256")==base.source_hashes(),
                "ES1 source/case/protocol fingerprint invalid")
        rows[seed]=row
    summary=edge_drift_summary(rows)
    backend_deltas=[]
    for seed in (42,43,44):
        for backend in args.backends:
            if backend=="pt32":
                continue
            reference_backend="pt16" if backend=="trt16" else "pt32"
            loc=base.directory(args.paper_root,"LEVIR-Ship","EDGE_CUE",seed,backend)
            ref_loc=base.directory(args.paper_root,"LEVIR-Ship","EDGE_CUE",seed,reference_backend)
            sample=json.loads((loc/"metrics.json").read_text(encoding="utf-8"))
            ref=json.loads((ref_loc/"metrics.json").read_text(encoding="utf-8"))
            require(sample.get("source_sha256")==EDGE_SHA[seed]
                    and ref.get("source_sha256")==EDGE_SHA[seed],
                    "Backend ES1 source SHA differs")
            backend_deltas.append({
                "seed":seed,"backend":backend,"versus":reference_backend,
                "delta_pp":{name:100*(sample["metrics"][name]-ref["metrics"][name])
                            for name in SCORE_KEYS},
            })
    summary["backend_deltas_pp"]=backend_deltas
    candidate=getattr(args,"historical_es1_best42",None)
    if candidate is not None:
        require(candidate.is_file(),"Supplied historical seed42 candidate best.pt missing")
        candidate_sha=base.sha(candidate)
        summary["historical_seed42_checkpoint_candidate"]={
            "path":str(candidate),"sha256":candidate_sha,
            "matches_curated_hf_source":candidate_sha==EDGE_SHA[42],
            "caution":"Only useful if this file has independent verified ORIGINAL training provenance",
        }
    else:
        summary["historical_seed42_checkpoint_candidate"]={
            "status":"NOT_PROVIDED","note":"Do not infer original training checkpoint SHA from HF alias",
        }
    historical_main_matches=all(
        all(abs(point["main_drift_from_historical_pp"][i]) <= 0.12 for i in range(2))
        for point in summary["per_seed"])
    historical_small_matches=abs(summary["small_mean_drift_pp"]) <= 0.12
    status=("HISTORICAL_METRICS_REPRODUCED_ROOT_CAUSE_UNVERIFIED"
            if historical_main_matches and historical_small_matches
            else "UNRESOLVED_HISTORICAL_ES1_PROVENANCE")
    result={
        "status":status,
        "source":"results/ftsc_edge_stability_summary_runs_full.csv + "
                 "results/es1_original_hf_metrics_by_seed.csv; pinned values above",
        "dataset":"LEVIR-Ship",
        "stage":args.stage,
        "new_paper_contract":CONTRACT,"new_paper_revision":REVISION,
        "new_source_hashes":base.source_hashes(),
        "analysis":summary,
        "interpretation":"Original-HF reevaluation previously disagreed with the historical "
                         "main test at seed42, while seeds43/44 matched. "
                         "The cause of AP50-Small mean drift is NOT established by "
                         "this evidence; do not label backend conversion or a single seed causal.",
    }
    # A later independently verified historical checkpoint is NEW evidence:
    # never overwrite a diagnostic run that did not include it.
    suffix=(base.sha(candidate)[:16] if candidate is not None else "no_original_best")
    target=(args.paper_root/"LEVIR-Ship"/"_diagnostics"/
            f"es1_historical_drift_{args.stage}_{suffix}.json")
    safe_write(target,result)
    return {"status":result["status"],"diagnostic":str(target),
            "small_mean_drift_pp":summary["small_mean_drift_pp"],
            "historical_checkpoint_candidate":summary["historical_seed42_checkpoint_candidate"]}


def publish_review_labeled(args,report):
    require(args.stage=="repair_all_precision"
            and args.seeds==[42,43,44]
            and args.backends==list(BACKENDS),
            "Only full 3seed/4backend stage can publish complete LEVIR paper report")
    require(report.get("issues")==[] and
            report.get("valid_jobs")==report.get("expected_jobs")==24,
            "Edge/Color incomplete: refuse review-labeled output")
    old,old_root=audit_old_frozen_three(args)
    old_records=[]
    for case in THREE_VERIFIED:
        for backend in BACKENDS:
            old_records.append(aggregate_three_seed(case,backend,old))
    fresh={(r["case"],r["backend"]):r for r in report["three_seed"]}
    require(len(fresh)==8,"New Edge/Color aggregate needs 8 rows")
    allrows=[]
    for case in FIVE_CASES:
        for backend in BACKENDS:
            if case in THREE_VERIFIED:
                row=next(r for r in old_records if r["case"]==case and r["backend"]==backend)
            else:
                row=fresh.get((case,backend))
                require(row is not None,"Missing fresh Edge/Color backend aggregate")
            allrows.append({
                **row,
                "source_batch":"existing_v1_sha_audited" if case in THREE_VERIFIED
                               else "new_corrected_paper_checkpoint",
                "backend_accuracy_review_flags":json.dumps(
                    [x for x in report["accuracy_review_flags"]
                     if x["case"]==case and backend in (x["backend"],x["vs"])],
                    sort_keys=True),
                "historical_reference_review_flags":json.dumps(
                    [x for x in report["historical_reference_review_flags"]
                     if x["case"]==case and backend==x["backend"]],sort_keys=True),
            })
    status=("COMPLETE_NEEDS_ACCURACY_REVIEW" if
            report["historical_reference_review_flags"] or report["accuracy_review_flags"]
            else "PASS_COMPLETE")
    outroot=args.paper_root/"LEVIR-Ship"/"_recovery_reports"
    outroot.mkdir(parents=True,exist_ok=True)
    target=outroot/"levir_paper_five_accuracy_review_labeled.csv"
    columns=sorted(set().union(*(set(x) for x in allrows)))
    buf=__import__("io").StringIO()
    writer=csv.DictWriter(buf,fieldnames=columns)
    writer.writeheader();writer.writerows(allrows)
    encoded=buf.getvalue()
    if target.exists():
        require(target.read_text(encoding="utf-8")==encoded,
                "Existing final five-case report differs; refuse overwrite")
    else:
        target.write_text(encoded,encoding="utf-8")
    summary={
        "status":status,"dataset":"LEVIR-Ship",
        "scope":"5 paper configurations, 3 seeds, four backends",
        "count":len(allrows),"new_corrected_cases":list(RECOVER_CASES),
        "original_verified_cases":list(THREE_VERIFIED),
        "new_recovery_audit":report,"old_reference_root":str(old_root),
        "new_source_manifest_sha256":base.sha(MANIFEST),
        "csv":str(target),
        "note":"Color now uses actual C_H2_POSTFUSION_V2. ES1 reference drift "
               "is a research provenance issue, not a TensorRT error. "
               "Review flags are NOT silently waived.",
    }
    reportpath=target.with_suffix(".json")
    safe_write(reportpath,summary)
    return {"status":status,"csv":str(target),"json":str(reportpath),
            "rows":len(allrows),"accuracy_flags":len(report["accuracy_review_flags"]),
            "historical_flags":len(report["historical_reference_review_flags"])}


def evaluate_stage(args, *, run_tests=False):
    require(args.allow_evaluate,"Evaluation requires explicit --allow-evaluate")
    verify_preflight_first(args)
    if run_tests:
        tests=("utils.tests.test_ftsc_deployment_paper_backend_v1",
               "utils.tests.test_ftsc_deployment_levir_edge_color_recovery_v1")
        subprocess.run([sys.executable,"-m","unittest",*tests],
                       cwd=str(ROOT),check=True)
    # The originally verified V3 validator computes ONLY the main paper AP;
    # its V3 Small AP is discarded. It can exit 0 despite per-case blockers.
    subprocess.run(v3_main_command(args),cwd=str(ROOT),check=True)
    verified=verify_all_v3_main(args)
    print(json.dumps({"v3_main_verified":len(verified),
                      "expected":len(args.cases)*len(args.seeds)*len(args.backends),
                      "all_test_pixels":788},indent=2),flush=True)
    # The separate predictor512/conf=.25 pass obtains historical AP50-Small.
    completed=base.main(base_command(args,"evaluate"))
    require(completed==0,"Independent paper Small AP evaluation has blocked jobs")
    return 0


def run_complete(args):
    """One explicit opt-in runs smoke -> 3seed FP32 -> all precision -> labeled merge."""
    require(args.allow_evaluate,
            "--phase complete requires --allow-evaluate; never automatic on Run All")
    for index,stage in enumerate(STAGES):
        args.stage=stage
        args.cases=list(STAGES[stage][0])
        args.seeds=list(STAGES[stage][1])
        args.backends=list(STAGES[stage][2])
        args.paper_root=args.base_project/CONTRACT/REVISION
        print(f"[LEVIR PAPER REPAIR {index+1}/{len(STAGES)}] {stage}",flush=True)
        # Verify pinned HF metadata, then freeze images/labels/source code in
        # an immutable preflight for exactly the current stage and seeds.
        require(base.main(base_command(args,"verify"))==0,"HF metadata verify blocked")
        require(base.main(base_command(args,"preflight"))==0,"Stage preflight blocked")
        evaluate_stage(args,run_tests=args.run_offline_tests and index==0)
        report=audit_recovery(args)
        require(report["valid_jobs"]==report["expected_jobs"]
                and not report["issues"],
                f"Stage {stage}: incomplete/invalid results; STOP before escalation")
        if stage=="repair_all_precision":
            print(json.dumps(diagnose_edge(args,report),indent=2),flush=True)
        print(json.dumps({
            "stage":stage,"audit_status":report["status"],
            "valid_jobs":report["valid_jobs"],
            "accuracy_review_flags":report["accuracy_review_flags"],
            "historical_reference_review_flags":
                report["historical_reference_review_flags"],
            "note":"Review flags are never cleared; historical ES1 mismatch "
                   "is an independent checkpoint/provenance question.",
        },indent=2),flush=True)
    result=publish_review_labeled(args,report)
    print(json.dumps(result,indent=2,sort_keys=True),flush=True)
    return 0


def main(argv=None):
    args=parsed_options(sys.argv[1:] if argv is None else argv)
    validate_recovery_manifest(MANIFEST)
    install_process_contract()
    if args.phase=="complete":
        return run_complete(args)
    if args.phase=="evaluate":
        return evaluate_stage(args,run_tests=args.run_offline_tests)
    if args.phase=="diagnose-edge":
        print(json.dumps(diagnose_edge(args),indent=2,sort_keys=True),flush=True)
        return 0
    if args.phase=="audit":
        report=audit_recovery(args)
        print(json.dumps(report,indent=2,sort_keys=True),flush=True)
        # Return zero if artifacts are complete and valid, even when the
        # separately labeled scientific/numerical review flags remain.
        return 0 if not report["issues"] and (
            report["valid_jobs"]==report["expected_jobs"]) else 2
    if args.phase=="aggregate":
        report=audit_recovery(args)
        finalized=publish_review_labeled(args,report)
        print(json.dumps(finalized,indent=2,sort_keys=True),flush=True)
        # An explicitly review-labeled artifact is not PASS_COMPLETE.
        return 0
    return base.main(base_command(args,args.phase))


if __name__=="__main__":
    raise SystemExit(main())
