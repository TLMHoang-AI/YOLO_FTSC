from argparse import Namespace
from pathlib import Path
import pytest
import train_levir_h2_generalization_completion_v1 as runner
import train_tinyperson_h2_generalization_completion_v1 as tiny_runner
import train_varroa_h2_generalization_completion_v1 as varroa_runner
from train_levir_scripts.train_ftsc_negative_canvas_suite import dataset_eligibility

def args(tmp_path): return Namespace(source_project=tmp_path/"source",top5_project=tmp_path/"top5",data_yaml=tmp_path/"data.yaml",device="cpu",workers=0)
def make_runs(tmp_path):
 for source in runner.CASES.values():
  for seed in runner.SEEDS:
   path=tmp_path/"source"/source/f"seed_{seed}"/"weights";path.mkdir(parents=True);(path/"best.pt").write_bytes(b"x")
def test_registry_disabled_and_no_train_path():
 assert tuple(runner.CASES)==("LEVIR_H2","LEVIR_H2_CP2","LEVIR_H2_OACP_R2","LEVIR_H2_R4")
 code=Path(runner.__file__).read_text();assert "model.train" not in code and "confirm-run" not in code
 assert runner.PROTOCOL["optimizer_historical_requested"]=="auto" and runner.PROTOCOL["optimizer_historical_resolved"]=="AdamW" and runner.PROTOCOL["optimizer_current_pinned"]=="AdamW" and runner.PROTOCOL["lr0"]==.002

def test_levir_r4_has_its_own_historical_suite(tmp_path):
 a=args(tmp_path);a.r_variants_project=tmp_path/"rvariants"
 assert runner.source_run_dir(a,"A6_NEGCANVAS_R4",42,"r_variants_project")==a.r_variants_project/"A6_NEGCANVAS_R4"/"seed_42"
 assert runner.SOURCE_REGISTRY["LEVIR_H2_R4"][0]=="levir_ftsc_R_variants_augmentation_suite"

def test_varroa_seed_scope_is_internal_and_locked(tmp_path):
 a=varroa_args(tmp_path); assert a.seed_scope=="all"
 values=varroa_runner.oacp_kwargs(a,43,tmp_path/"data.yaml")
 assert values["seed"]==43 and values["deterministic"] is True

def test_live_protocols_are_locked():
 assert runner.PROTOCOL["imgsz"]==512 and runner.PROTOCOL["batch"]==8 and runner.PROTOCOL["workers"]==4 and runner.PROTOCOL["patience"]==20
 assert tiny_runner.parse_args([]).workers==8 and tiny_runner.parse_args([]).imgsz==640 and tiny_runner.parse_args([]).batch_size==8
 assert varroa_runner.parse_args([]).workers==8 and varroa_runner.parse_args([]).imgsz==640 and varroa_runner.parse_args([]).batch_size==4

def test_tiny_pretrained_identity_is_canonical_and_locked(tmp_path):
 assert tiny_runner.canonical_pretrained("yolov8n.pt")==str((tiny_runner.ROOT/"yolov8n.pt").resolve())
 assert tiny_runner.canonical_pretrained(tiny_runner.ROOT/"yolov8n.pt")==str((tiny_runner.ROOT/"yolov8n.pt").resolve())
 with pytest.raises(ValueError): tiny_runner.canonical_pretrained("yolov8s.pt")
 with pytest.raises(ValueError): tiny_runner.canonical_pretrained(tmp_path/"other.pt")

def test_varroa_auto_optimizer_provenance(tmp_path):
 data=tmp_path/"data.yaml";data.write_text("nc: 1\nnames: [bee]\n")
 p=varroa_runner.optimizer_provenance(varroa_args(tmp_path),data)
 assert p["requested"]=="auto" and p["resolved"]=="AdamW" and p["lr0_resolved"]==.002 and p["momentum_resolved"]==.9 and p["warmup_bias_lr_resolved"]==0.0 and p["optimizer_auto_iterations"]<=10000

@pytest.mark.parametrize("seeds",([42],[43],[42,44],[42,43,44]))
def test_seed_subsets_and_execution_overrides_are_valid(seeds):
 from h2_generalization_completion_common import selected_seeds
 assert selected_seeds(Namespace(seeds=seeds))==tuple(seeds)
 a=tiny_runner.parse_args(["--seeds",*map(str,seeds),"--cases","TP_H2_MOSAIC_CP2","--workers","4","--device","cpu"])
 tiny_runner.validate_args(a); assert a.workers==4 and a.device=="cpu"

@pytest.mark.parametrize("seeds",([], [41], [42,42], [45]))
def test_invalid_seed_selections_fail_closed(seeds):
 from h2_generalization_completion_common import selected_seeds
 with pytest.raises(ValueError): selected_seeds(Namespace(seeds=seeds))

def test_partial_seed_status_and_single_sample_std():
 from h2_generalization_completion_common import aggregate, status_fields
 rows=status_fields([{"case":"X","seed":42,"status":"COMPLETE","m":1.},{"case":"X","seed":43,"status":"NOT_REQUESTED"},{"case":"X","seed":44,"status":"NOT_REQUESTED"}],(42,))
 assert rows[0]["requested_status"]=="COMPLETE" and rows[0]["canonical_status"]=="PARTIAL"
 result=aggregate(rows,("m",))[0]; assert result["m/mean"]==1. and result["m/std"] is None

def test_execution_context_does_not_block_resume_but_science_does(tmp_path):
 from h2_generalization_completion_common import ensure_manifest
 run=tmp_path/"run"; ensure_manifest(run,{"scientific_contract":{"batch":8},"execution_context":{"workers":8,"device":"cuda"}})
 ensure_manifest(run,{"scientific_contract":{"batch":8},"execution_context":{"workers":4,"device":"cpu"}})
 with pytest.raises(RuntimeError): ensure_manifest(run,{"scientific_contract":{"batch":4},"execution_context":{"workers":4}})
def test_default_preflight_does_not_evaluate(tmp_path,monkeypatch):
 make_runs(tmp_path);called=[];monkeypatch.setattr(runner,"evaluate_run",lambda *a,**k:called.append(a));rows=runner.preflight(args(tmp_path))
 assert not called and all(r["status"]=="EVAL_BACKFILL_REQUIRED" for r in rows)
def test_confirm_eval_calls_existing_evaluator(tmp_path,monkeypatch):
 make_runs(tmp_path);calls=[];payload={k:1. for k in runner.REQUIRED_METRICS};monkeypatch.setattr(runner,"evaluate_run",lambda *a,**k:calls.append(a) or payload)
 assert all(r["status"]=="COMPLETE" for r in runner.evaluate_backfill(args(tmp_path),runner.preflight(args(tmp_path)))) and len(calls)==12
def test_missing_best_fails_closed(tmp_path):
 with pytest.raises(FileNotFoundError):runner.preflight(args(tmp_path))
def test_fullmetric_reuse_and_metric_contract(tmp_path,monkeypatch):
 make_runs(tmp_path);top=tmp_path/"top5"/"A3_CP2"/"seed_42";(top/"weights").mkdir(parents=True);(top/"weights/best.pt").write_bytes(b"x");(top/"evaluation_metrics_extended.json").write_text("{}")
 monkeypatch.setattr(runner,"load_merged_metrics",lambda _: {k:1. for k in runner.REQUIRED_METRICS});row=next(r for r in runner.preflight(args(tmp_path)) if r["case"]=="LEVIR_H2_CP2" and r["seed"]==42)
 assert row["status"]=="REUSED" and "TOP5" in row["provenance"] and runner.metric_missing({})==list(runner.REQUIRED_METRICS)
def test_sample_std(tmp_path,monkeypatch):
 rows=[{"case":"LEVIR_H2","seed":s,"status":"REUSED",**{k:v for k in runner.REQUIRED_METRICS}} for s,v in zip(runner.SEEDS,(1.,2.,3.))]
 rows += [{"case":c,"seed":42,"status":"EVAL_BACKFILL_REQUIRED"} for c in list(runner.CASES)[1:]];runner.write_summaries(Namespace(project=tmp_path),rows)
 import csv
 assert float(list(csv.DictReader((tmp_path/"summary_aggregate.csv").open()))[0]["val/metrics/mAP50(B)/std"])==pytest.approx(1.)

def tiny_args(tmp_path):
 return Namespace(data_root=tmp_path/"raw",dataset_root=tmp_path/"datasets",project=tmp_path/"runs",pretrained="yolov8n.pt",epochs=100,imgsz=640,batch_size=8,device="cpu",workers=0,patience=20)
def test_tiny_seed_paths_and_bases(tmp_path):
 a=tiny_args(tmp_path)
 assert [tiny_runner.dataset_yaml(a,s).parent.name for s in (42,43,44)]==["tinyperson_seed_42_corner_sw640_sh512","tinyperson_seed_43_corner_sw640_sh512","tinyperson_seed_44_corner_sw640_sh512"]
 assert all(tiny_runner.base_variant(case)=="H2_MOSAIC" for case in ("TP_H2_MOSAIC_CP2","TP_H2_MOSAIC_OACP_R2","TP_H2_NM_R4"))
 assert "replace(" not in Path(tiny_runner.__file__).read_text()

def test_tiny_historical_variant_never_reaches_native_train(tmp_path,monkeypatch):
 a=tiny_args(tmp_path); seen=[]
 from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as native
 monkeypatch.setattr(native,"train_kwargs",lambda _a,variant,*_rest: seen.append(variant) or {"project":"x","name":"y"})
 tiny_runner.train_kwargs(a,"TP_H2_NM_R4",42,tiny_runner.dataset_yaml(a,42))
 assert seen==["H2_MOSAIC"]
def test_tiny_cp2_oacp_and_metric_contract(tmp_path):
 a=tiny_args(tmp_path);data=tiny_runner.dataset_yaml(a,42)
 cp2,_=tiny_runner.train_kwargs(a,"TP_H2_MOSAIC_CP2",42,data);oacp,context=tiny_runner.train_kwargs(a,"TP_H2_MOSAIC_OACP_R2",42,data)
 assert cp2["copy_paste_enabled"] is True and oacp["copy_paste_enabled"] is False
 assert context.__class__.__name__ != "nullcontext" and "test_merged/AP50" in tiny_runner.REQUIRED_METRICS
def test_tiny_dispatch_only_new_rows(tmp_path,monkeypatch):
 a=tiny_args(tmp_path);called=[]
 from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as source
 workflow=source._tiny_workflow();test_out=tmp_path/"test";seed_dir=tiny_runner.dataset_yaml(a,42).parent;seed_dir.mkdir(parents=True);tiny_runner.dataset_yaml(a,42).write_text("path: .\ntrain: images\n")
 monkeypatch.setattr(workflow,"prepare_test_set",lambda *_:test_out);monkeypatch.setattr(workflow,"prepare_seed_dataset",lambda *_:seed_dir);monkeypatch.setattr(tiny_runner,"dataset_eligibility",lambda *_ ,**__: {"original_negative_images":0,"eligible_small_targets_le_20px":0});monkeypatch.setattr(tiny_runner,"_dataset_contract",lambda *_: {});monkeypatch.setattr(tiny_runner,"run_new_case",lambda *values:called.append(values[1]) or {"test_merged/available":1.0,**{k:1.0 for k in tiny_runner.REQUIRED_METRICS}})
 rows=[{"case":"TP_H2_MOSAIC","seed":42,"status":"REUSED"},{"case":"TP_H2_MOSAIC_CP2","seed":42,"status":"NEW_RUN_REQUIRED"}]
 assert tiny_runner.dispatch(a,rows)[1]["status"]=="COMPLETE" and called==["TP_H2_MOSAIC_CP2"]

def varroa_args(tmp_path):
 return Namespace(data_root=tmp_path/"raw",dataset_root=tmp_path/"datasets",project=tmp_path/"runs",data_yaml=tmp_path/"data.yaml",epochs=100,imgsz=640,batch_size=4,device="cpu",workers=0,patience=20,seed_scope="all")
def test_varroa_oacp_uses_native_mosaic_not_cp2(tmp_path):
 values=varroa_runner.oacp_kwargs(varroa_args(tmp_path),42,tmp_path/"data.yaml")
 assert values["mosaic"]==1.0 and values["close_mosaic"]==10 and values["copy_paste"]==0.0 and values["copy_paste_enabled"] is False
 assert "CANONICAL_FTSC_CONFIG" in Path(varroa_runner.__file__).read_text() and "configured_environment(\"A1_OACP_R2\")" in Path(varroa_runner.__file__).read_text() and "shared.evaluate(run_dir,args)" in Path(varroa_runner.__file__).read_text()
def test_varroa_dispatch_only_oacp_and_evaluates(tmp_path,monkeypatch):
 a=varroa_args(tmp_path);calls=[]
 from train_levir_scripts import train_varroa_dbss_hit as shared
 monkeypatch.setattr(shared,"dataset_for_seed",lambda *_:tmp_path/"data.yaml");monkeypatch.setattr(varroa_runner,"_dataset_contract",lambda *_: {});monkeypatch.setattr(varroa_runner,"dataset_eligibility",lambda *_ ,**__: {"original_negative_images":0})
 monkeypatch.setattr(varroa_runner,"run_oacp",lambda *_:calls.append("oacp") or {k:1.0 for k in varroa_runner.REQUIRED_METRICS})
 rows=[{"case":"VR_H2_MOSAIC","seed":42,"status":"REUSED"},{"case":"VR_H2_MOSAIC_CP2","seed":42,"status":"REUSED"},{"case":"VR_H2_R4","seed":42,"status":"PENDING_DATA_PREPARATION"},{"case":"VR_H2_MOSAIC_OACP_R2","seed":42,"status":"NEW_RUN_REQUIRED"}]
 out=varroa_runner.dispatch(a,rows)
 assert calls==["oacp"] and out[-1]["status"]=="COMPLETE" and all(x["status"]!="COMPLETE" for x in out[:-1])

def test_manifest_mismatch_and_marker_fail_closed(tmp_path):
 from h2_generalization_completion_common import ensure_manifest, write_completion_marker
 run=tmp_path/"run"; ensure_manifest(run,{"case":"x"})
 with pytest.raises(RuntimeError,match="Incompatible"): ensure_manifest(run,{"case":"y"})
 with pytest.raises(RuntimeError,match="Cannot mark incomplete"): write_completion_marker(run,case="x",seed=42,required=("m",),metrics={"m":1.})

def test_summaries_are_written_inside_project_not_results(tmp_path):
 rows=[{"case":"X","seed":42,"status":"COMPLETE",**{key:1. for key in tiny_runner.REQUIRED_METRICS}}]
 tiny_runner.write_summaries(Namespace(project=tmp_path),rows)
 assert (tmp_path/"summary_runs.csv").is_file() and (tmp_path/"summary_aggregate.csv").is_file()
 assert "huggingface" not in Path(tiny_runner.__file__).read_text().lower()

def test_canonical_r4_zero_negatives_is_optional_only_when_requested(tmp_path):
 from PIL import Image
 root=tmp_path/"data";(root/"images/train").mkdir(parents=True);(root/"labels/train").mkdir(parents=True);Image.new("RGB",(100,100)).save(root/"images/train/a.jpg");(root/"labels/train/a.txt").write_text("0 .5 .5 .1 .1\n")
 yaml=tmp_path/"data.yaml";yaml.write_text("path: data\ntrain: images/train\n")
 with pytest.raises(RuntimeError,match="zero original-negative"):dataset_eligibility(yaml)
 report=dataset_eligibility(yaml,require_negative=False)
 assert report["original_negative_images"]==0 and report["original_negative_fraction"]==0.0
def test_canonical_r4_missing_and_malformed_labels_fail_closed(tmp_path):
 from PIL import Image
 root=tmp_path/"data";(root/"images/train").mkdir(parents=True);(root/"labels/train").mkdir(parents=True);Image.new("RGB",(10,10)).save(root/"images/train/a.jpg")
 yaml=tmp_path/"data.yaml";yaml.write_text("path: data\ntrain: images/train\n")
 with pytest.raises(FileNotFoundError):dataset_eligibility(yaml,require_negative=False)
 (root/"labels/train/a.txt").write_text("0 bad\n")
 with pytest.raises(ValueError):dataset_eligibility(yaml,require_negative=False)
def test_no_duplicate_r4_audit_remains():
 assert "def audit_yolo_original_negatives" not in (Path(__file__).parent/"h2_generalization_completion_common.py").read_text()
