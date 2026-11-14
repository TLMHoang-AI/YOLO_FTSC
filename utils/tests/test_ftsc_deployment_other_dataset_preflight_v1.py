"""Offline contract checks for TinyPerson/Varroa V2 deployment preflight.

Run: python -m unittest utils.tests.test_ftsc_deployment_other_dataset_preflight_v1
Does not access HF, download checkpoints, train, export or use CUDA.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import ftsc_deployment_backend_eval_v1 as base
from utils import ftsc_deployment_other_dataset_preflight_v1 as probe
from utils import ftsc_deployment_other_dataset_audit_v1 as audit
from utils import ftsc_deployment_other_dataset_workflow_v1 as workflow


class DatasetPreflightTests(unittest.TestCase):
    def test_hash_inputs_pass_only_with_equal_pixel_bytes_and_shapes(self):
        import numpy as np
        pixel=np.full((3, 32, 32), 114, dtype=np.uint8)
        samples=[("a",pixel,pixel.copy()),("b",pixel,pixel.copy())]
        report=probe._pair_summary(samples)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["identical_input_count"],2)
        self.assertEqual(report["pt_canvas_shapes"],{"32x32":2})
        changed=pixel.copy()
        changed[1,2,3]=0
        bad=probe._pair_summary([("a",pixel,pixel),("b",pixel,changed)])
        self.assertFalse(bad["all_identical"])
        self.assertEqual(bad["identical_input_count"],1)
        self.assertEqual(len(bad["mismatch_examples"]),1)
        wrong=np.zeros((3,32,64),dtype=np.uint8)
        report=probe._pair_summary([("one",pixel,wrong)])
        self.assertFalse(report["all_identical"])
        self.assertEqual(report["trt_static_canvas_shapes"],{"32x64":1})

    def test_invalid_preprocessed_dtypes_fail_closed(self):
        import numpy as np
        pt=np.zeros((3,32,32),dtype=np.uint8)
        bad=np.zeros((3,32,32),dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError,"dtype"):
            probe._pair_summary([("first",pt,bad)])
        with self.assertRaisesRegex(RuntimeError,"Empty"):
            probe._pair_summary([])

    def test_dataset_scopes_are_explicit_and_paper_benchmark_is_not_levir(self):
        for dataset,n in (("TinyPerson",11),("Varroa",13)):
            with self.subTest(dataset=dataset):
                first=audit.scope_for(dataset,"ftsc_seed42_fp32",base.REPORT_MANIFEST)
                self.assertEqual(first["dataset"],dataset)
                self.assertEqual(first["seeds"],[42])
                self.assertEqual(first["cases"],["FTSC"])
                self.assertEqual(first["backends"],["pt32","trt32"])
                self.assertEqual(first["expected_jobs"],2)
                self.assertEqual(first["evaluation_protocol"],base.NATIVE_VAL_PROTOCOL)
                report=audit.scope_for(dataset,"report_all_precision",base.REPORT_MANIFEST)
                self.assertEqual(len(report["cases"]),n)
                self.assertEqual(report["expected_jobs"],n*3*4)
        with self.assertRaisesRegex(RuntimeError,"Unsupported"):
            audit.scope_for("LEVIR-Ship","ftsc_seed42_fp32",base.REPORT_MANIFEST)

    def test_workflow_never_bypasses_explicit_evaluate_consent(self):
        with self.assertRaisesRegex(RuntimeError,"allow-evaluate"):
            workflow.main(["--dataset","TinyPerson","--stage","ftsc_seed42_fp32",
                           "--phase","evaluate"])
        self.assertIn("preflight",workflow.__doc__.lower())

    def test_both_notebooks_safe_and_source_syntax_valid(self):
        parent=Path(__file__).resolve().parents[3]
        for filename,suffix in (
            ("current_live_training_notebok_TinyPerson.ipynb","TINY"),
            ("current_live_training_notebok_Varroa.ipynb","VARROA"),
        ):
            path=parent/filename
            if not path.is_file():
                continue  # Live clone may contain only yolo_code without notebooks.
            with self.subTest(filename=filename):
                nb=json.loads(path.read_text(encoding="utf-8"))
                cells=[cell for cell in nb["cells"]
                       if cell.get("id","").startswith("ftsc-deployment-")]
                self.assertEqual(len(cells),1)
                code="".join(cells[0]["source"])
                compile(code,str(path)+"#deployment","exec")
                self.assertIn(f"RUN_FTSC_DEPLOY_{suffix} = False",code)
                self.assertIn(f"FTSC_DEPLOY_{suffix}_ALLOW_EVALUATE = False",code)
                self.assertIn(f"FTSC_DEPLOY_{suffix}_PHASE = 'plan'",code)
                self.assertIn(f"FTSC_DEPLOY_{suffix}_STAGE = 'ftsc_seed42_fp32'",code)
                self.assertIn("ftsc_deployment_other_dataset_workflow_v1",code)
                self.assertNotIn("FTSC_DEPLOY_LEVIR_EVAL_PROTOCOL",code)
                self.assertNotIn(".train(",code)
                self.assertNotIn("upload_file",code)


if __name__=="__main__":
    unittest.main()
