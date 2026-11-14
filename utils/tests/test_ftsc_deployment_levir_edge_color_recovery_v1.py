"""Safe offline tests for LEVIR two-case paper checkpoint recovery."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import ftsc_deployment_levir_edge_color_recovery_v1 as recover


class LevirPaperRecoveryContracts(unittest.TestCase):
    def test_only_color_postfusion_and_edge_original_are_selected(self):
        doc=recover.validate_recovery_manifest(recover.MANIFEST)
        cases=doc["datasets"]["LEVIR-Ship"]["cases"]
        self.assertEqual(set(cases),set(recover.RECOVER_CASES))
        self.assertEqual(cases["COLOR_CUE"]["source_run"],
                         "LEVIR-Ship/journal_experiments/phase2/C_H2_POSTFUSION_V2")
        self.assertNotIn("RECON_C_H2_LEGACY_V1",
                         json.dumps(cases["COLOR_CUE"]))
        self.assertEqual(cases["EDGE_CUE"]["source_run"],
                         "LEVIR-Ship/original_runs/ES1")
        for seed in (42,43,44):
            self.assertEqual(cases["COLOR_CUE"]["seeds"][str(seed)]["sha256"],
                             recover.COLOR_SHA[seed])
            self.assertEqual(cases["EDGE_CUE"]["seeds"][str(seed)]["sha256"],
                             recover.EDGE_SHA[seed])

    def test_all_stage_dimensions_and_main_before_small(self):
        self.assertEqual(set(recover.STAGES),{
            "repair_seed42_fp32","repair_three_seed_fp32","repair_all_precision"})
        self.assertEqual(recover.STAGES["repair_all_precision"],
                         (recover.RECOVER_CASES,(42,43,44),recover.BACKENDS))
        for cases,seeds,backends in recover.STAGES.values():
            self.assertEqual(cases,recover.RECOVER_CASES)
            self.assertEqual(backends[0],"pt32")
        opts=recover.parsed_options([
            "--phase","evaluate","--stage","repair_all_precision"])
        argv=recover.v3_main_command(opts)
        self.assertEqual(argv[argv.index("--evaluation-protocol")+1],
                         recover.deploy.RECT_PARITY_PROTOCOL)
        self.assertEqual(argv[argv.index("--report-manifest")+1],
                         str(recover.MANIFEST))
        self.assertIn("--confirm-eval",argv)
        self.assertIn("COLOR_CUE",argv)
        self.assertIn("EDGE_CUE",argv)

    def test_one_shot_complete_cannot_run_without_explicit_permission(self):
        args=recover.parsed_options(["--phase","complete"])
        self.assertEqual(args.phase,"complete")
        self.assertFalse(args.allow_evaluate)
        with self.assertRaisesRegex(RuntimeError,"requires --allow-evaluate"):
            recover.run_complete(args)

    def test_original_report_manifest_has_not_been_patched(self):
        original=recover.original_report_manifest()
        self.assertIn("RECON_C_H2_LEGACY_V1",
                      original["COLOR_CUE"]["source_run"])
        self.assertNotIn("C_H2_POSTFUSION_V2",
                         original["COLOR_CUE"]["source_run"])

    def test_invalid_sha_is_rejected_before_gpu(self):
        doc=recover.validate_recovery_manifest(recover.MANIFEST)
        doc["datasets"]["LEVIR-Ship"]["cases"]["COLOR_CUE"]["seeds"]["42"]["sha256"]="0"*64
        import shutil
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"case_manifest.json"
            path.write_text(json.dumps(doc),encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"differs from paper/HF"):
                recover.validate_recovery_manifest(path)

    def test_no_paper_aggregate_from_incomplete_or_review_flagged_data(self):
        args=recover.parsed_options([
            "--phase","aggregate","--stage","repair_all_precision"])
        with self.assertRaisesRegex(RuntimeError,"incomplete"):
            recover.publish_review_labeled(args,{
                "issues":[{"case":"COLOR_CUE"}],"valid_jobs":23,"expected_jobs":24,
            })

    def test_paper_result_file_is_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"test.json"
            recover.safe_write(path,{"small_ap":0.8})
            recover.safe_write(path,{"small_ap":0.8})
            with self.assertRaisesRegex(RuntimeError,"differs"):
                recover.safe_write(path,{"small_ap":0.9})

    def test_frozen_old_paper_revision_not_inherited_from_mutable_base(self):
        self.assertEqual(recover.ORIGINAL_PAPER_CONTRACT,
                         "ftsc_historical_paper_backend_parity_v1")
        self.assertEqual(recover.ORIGINAL_PAPER_REVISION,
                         "predict512_native_rect_dynamic_trt_v1")
        self.assertNotEqual(recover.ORIGINAL_PAPER_REVISION,recover.base.REVISION)

    def test_cross_case_gt_and_input_artifacts_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            a,b=root/"old.json",root/"new.json"
            a.write_text('{"ids":[1,2]}',encoding="utf-8")
            b.write_text('{"ids":[1,2]}',encoding="utf-8")
            recover.require_same_artifact(a,b,"GT test")
            b.write_text('{"ids":[1,3]}',encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"bytes differ"):
                recover.require_same_artifact(a,b,"GT test")
            b.unlink()
            with self.assertRaisesRegex(RuntimeError,"artifact missing"):
                recover.require_same_artifact(a,b,"GT test")

    def test_es1_drift_diagnostic_never_invents_historical_small_by_seed(self):
        rows={}
        for seed,(ap50,ap5095,small) in recover.EDGE_PRIOR_HF.items():
            rows[seed]={
                "source_sha256":recover.EDGE_SHA[seed],
                "metrics":{
                    recover.SCORE_KEYS[0]:ap50,
                    recover.SCORE_KEYS[1]:ap5095,
                    recover.SCORE_KEYS[2]:small,
                },
            }
        report=recover.edge_drift_summary(rows)
        self.assertFalse(report["historical_small_per_seed_available"])
        self.assertTrue(report["observations"][
            "previous_hf_seeds43_44_main_match_historical"])
        self.assertAlmostEqual(report["per_seed"][0][
            "main_drift_from_historical_pp"][0],-1.0725815274734707)
        self.assertAlmostEqual(report["small_mean_drift_pp"],-1.120332246070916)
        rows[42]["source_sha256"]="0"*64
        with self.assertRaisesRegex(RuntimeError,"checkpoint SHA"):
            recover.edge_drift_summary(rows)

    def test_es1_diagnosis_requires_full_three_seed_stage(self):
        args=recover.parsed_options(["--phase","diagnose-edge",
                                     "--stage","repair_seed42_fp32"])
        with self.assertRaisesRegex(RuntimeError,"full three-seed"):
            recover.diagnose_edge(args)

    def test_source_provenance_adds_wrapper_yaml_manifest(self):
        baseline=lambda:{"verified_v2_source":"a"*64}
        with patch.object(recover.base,"source_hashes",baseline),\
             patch.object(recover.base,"CONTRACT","legacy"),\
             patch.object(recover.base,"REVISION","legacy"),\
             patch.object(recover.base,"STAGES",{}):
            recover.install_process_contract()
            source=recover.base.source_hashes()
            for path in recover.SOURCE_FILES:
                self.assertEqual(source[path],recover.base.sha(recover.ROOT/path))
            self.assertEqual(recover.base.CONTRACT,recover.CONTRACT)
            self.assertEqual(recover.base.REVISION,recover.REVISION)
            self.assertEqual(recover.base.STAGES,recover.STAGES)
        self.assertEqual(recover.base.CONTRACT,recover.ORIGINAL_PAPER_CONTRACT)


if __name__=="__main__":
    unittest.main()
