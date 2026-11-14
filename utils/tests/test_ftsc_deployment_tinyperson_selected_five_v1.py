"""TinyPerson selected-five paper backend contract checks (CPU/source only)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import ftsc_deployment_tinyperson_selected_five_v1 as tiny


class TinyPaperSelectedFive(unittest.TestCase):
    def test_exact_five_and_order_matches_paper(self):
        self.assertEqual(tiny.FIVE_CASES,(
            "FTSC","FTSC_YOLOV9T","FTSC_YOLO11N",
            "EDGE_CUE","COLOR_CUE",
        ))
        self.assertEqual(set(tiny.MODEL_CONFIGS),set(tiny.FIVE_CASES))
        self.assertEqual(set(tiny.CANONICAL_RUNS),set(tiny.FIVE_CASES))
        self.assertEqual(set(tiny.PAPER_PT32_MEANS),set(tiny.FIVE_CASES))
        self.assertEqual(
            tiny.PAPER_PT32_MEANS["EDGE_CUE"],(57.31,21.30,69.43))
        self.assertEqual(
            tiny.PAPER_PT32_MEANS["COLOR_CUE"],(57.79,21.52,70.04))

    def test_reference_manifest_exact_case_checkpoint_sha_and_config(self):
        src=tiny.validate_selected_sources(tiny.base.deploy.REPORT_MANIFEST)
        self.assertEqual(list(src),list(tiny.FIVE_CASES))
        for name,record in src.items():
            self.assertEqual(record["source_run"],tiny.CANONICAL_RUNS[name])
            self.assertEqual(
                record["seeds"]["42"]["sha256"],tiny.SEED42_HF_SHA256[name])
            self.assertEqual(
                set(record["seeds"]),{"42","43","44"})
            self.assertTrue(Path(record["model_yaml"]).is_file())
            self.assertEqual(
                tiny.base.sha(record["model_yaml"]),record["model_yaml_sha256"])

    def test_wrong_checkpoint_alias_is_rejected(self):
        m=tiny.base.deploy.REPORT_MANIFEST
        data=json.loads(Path(m).read_text(encoding="utf-8"))
        data["datasets"]["TinyPerson"]["cases"]["COLOR_CUE"]["source_run"]="TinyPerson/incorrect"
        with self.subTest("wrong-run-id"):
            # No actual I/O or GPU beyond a test JSON document.
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                test=Path(directory)/"manifest.json"
                test.write_text(json.dumps(data),encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError,"wrong run"):
                    tiny.validate_selected_sources(test)

    def test_stages_are_pinned_to_five_cases_and_pt32_first(self):
        self.assertEqual(set(tiny.STAGES),{
            "tiny_five_seed42_fp32","tiny_five_three_seed_fp32",
            "tiny_five_all_precision",
        })
        for cases,seeds,backends in tiny.STAGES.values():
            self.assertEqual(cases,tiny.FIVE_CASES)
            self.assertEqual(backends[0],"pt32")
            self.assertTrue(set(seeds).issubset({42,43,44}))
        self.assertEqual(tiny.STAGES["tiny_five_all_precision"][2],tiny.base.BACKENDS)

    def test_source_hash_includes_wrapper_and_model_yaml(self):
        # Isolate monkeypatch. This test must not affect LEVIR or subsequent tests.
        baseline=lambda: {"base.py":"a"*64}
        with patch.object(tiny.base,"source_hashes",baseline),\
             patch.object(tiny.base,"CONTRACT","original"),\
             patch.object(tiny.base,"REVISION","original"),\
             patch.object(tiny.base,"STAGES",{}),\
             patch.object(tiny.base,"PAPER_ANCHORS",{"TinyPerson":{}}):
            tiny.install_isolated_contract()
            h=tiny.base.source_hashes()
            self.assertIn(tiny.SELF_SOURCE,h)
            self.assertEqual(set(tiny.MODEL_CONFIGS.values())|{"base.py",tiny.SELF_SOURCE},
                             set(h))
            self.assertEqual(tiny.base.CONTRACT,tiny.CONTRACT)
            self.assertEqual(tiny.base.REVISION,tiny.REVISION)
            self.assertEqual(tiny.base.PAPER_ANCHORS["TinyPerson"],tiny.PAPER_PT32_MEANS)
        self.assertEqual(tiny.base.CONTRACT,
                         "ftsc_historical_paper_backend_parity_v1")


if __name__=="__main__":
    unittest.main()
