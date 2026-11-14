"""Offline strict paper deployment contract checks: no training, HF, CUDA or TRT."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from utils import ftsc_deployment_paper_backend_v1 as paper


class PaperParityContracts(unittest.TestCase):
    def test_historical_protocols_do_not_conflate_main_and_small(self):
        self.assertEqual(paper.CONFIG["LEVIR-Ship"]["main"],"v3_rect_544")
        self.assertEqual(paper.CONFIG["LEVIR-Ship"]["small"],"predict512_conf025")
        self.assertEqual(paper.CONFIG["TinyPerson"]["small"],"tiny_original_merged")
        self.assertEqual(paper.CONFIG["TinyPerson"]["batch"],8)
        self.assertEqual(paper.CONFIG["Varroa"]["batch"],4)
        self.assertEqual(paper.CONFIG["Varroa"]["main"],"native_rect_val")
        self.assertNotEqual(paper.CONTRACT,paper.deploy.RECT_PARITY_PROTOCOL)
        self.assertNotEqual(paper.CONTRACT,paper.deploy.NATIVE_VAL_PROTOCOL)

    def test_only_four_allowed_backends_and_reference_is_first(self):
        self.assertEqual(paper.BACKENDS,("pt32","trt32","pt16","trt16"))
        self.assertTrue(all(stage[2][0]=="pt32" for stage in paper.STAGES.values()))
        self.assertEqual(paper.STAGES["ftsc_seed42_fp32"][1],(42,))
        self.assertEqual(paper.STAGES["report_all_precision"][1],(42,43,44))

    def test_invalid_artifact_never_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.json"
            before=paper.write_new_json(p,{"case":"FTSC","AP50":0.8})
            self.assertEqual(before,paper.sha(p))
            with self.assertRaisesRegex(RuntimeError,"Refuse to overwrite"):
                paper.write_new_json(p,{"case":"FTSC","AP50":0.9})
            self.assertEqual(json.loads(p.read_text())["AP50"],0.8)

    def test_sha_input_manifest_works_only_with_exact_image_set(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            image=folder/"test.jpg"
            image.write_bytes(b"test")
            records={str(image.resolve()):{"shape":[3,512,512],"sha256":"a"*64}}
            meta=paper.seal_inputs(folder/"input.json",records,[image],"test")
            self.assertEqual(meta["input_count"],1)
            payload=json.loads((folder/"input.json").read_text())
            self.assertEqual(payload["images"][str(image.resolve())]["shape"],[3,512,512])
            with self.assertRaisesRegex(RuntimeError,"wrong input images"):
                paper.seal_inputs(folder/"other.json",records,[folder/"wrong.jpg"],"test")

    def test_dynamic_profile_fails_closed_on_unexportable_shapes(self):
        good=paper.validate_dynamic_profile_shapes({"640x640":12,"672x640":2})
        self.assertEqual(good["dynamic_profile_max_hw"],[1280,1280])
        for cases in ({"1344x640":1},{"600x640":1},{},{"640x640":0}):
            with self.assertRaises(RuntimeError):
                paper.validate_dynamic_profile_shapes(cases)

    def test_no_trt_allowed_without_pt32_pixel_reference(self):
        with tempfile.TemporaryDirectory() as td:
            entry={"source_sha256":"a"*64,"small_input_manifest_sha256":"b"*64}
            with self.assertRaisesRegex(RuntimeError,"execute pt32 before"):
                paper.verify_pair(Path(td),"LEVIR-Ship","FTSC",42,"trt32","small",entry)

    def test_plan_does_not_need_datasets_or_hf(self):
        # Real pinned report manifest is a local source-only file; no GPU/HF.
        self.assertEqual(paper.main([
            "--dataset","TinyPerson","--stage","ftsc_seed42_fp32",
            "--phase","plan",
        ]),0)

    def test_source_manifest_includes_backends_and_evaluator(self):
        files=paper.SOURCE_FILES
        for required in ("utils/ftsc_deployment_paper_backend_v1.py",
                         "evaluate_test/size_bucket_evaluator.py",
                         "train_tinyperson_scripts/train_all_tinyperson.py",
                         "models_related/ultralytics/ultralytics/nn/backends/tensorrt.py"):
            self.assertIn(required,files)
        self.assertEqual(set(paper.source_hashes()),set(files))


if __name__=="__main__":
    unittest.main()
