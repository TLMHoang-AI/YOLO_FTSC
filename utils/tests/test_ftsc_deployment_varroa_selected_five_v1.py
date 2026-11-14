"""CPU-only locked five Varroa deployment source/protocol checks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import ftsc_deployment_varroa_selected_five_v1 as varroa


class VarroaPaperSelectedFive(unittest.TestCase):
    def test_exact_paper_five_and_edge_identity(self):
        self.assertEqual(
            varroa.FIVE_CASES,
            ("FTSC", "FTSC_YOLOV9T", "FTSC_YOLO11N", "EDGE_P2", "COLOR_CUE"),
        )
        self.assertEqual(set(varroa.MODEL_CONFIGS), set(varroa.FIVE_CASES))
        self.assertEqual(set(varroa.CANONICAL_RUNS), set(varroa.FIVE_CASES))
        self.assertEqual(set(varroa.PAPER_PT32_MEANS), set(varroa.FIVE_CASES))
        self.assertEqual(varroa.PAPER_PT32_MEANS["EDGE_P2"],
                         (91.36, 35.30, 87.00))
        self.assertEqual(varroa.PAPER_PT32_MEANS["COLOR_CUE"],
                         (90.35, 34.97, 85.54))
        self.assertNotIn("EDGE_P3", varroa.FIVE_CASES)

    def test_all_fifteen_manifest_sources_and_model_yaml_are_pinned(self):
        result = varroa.validate_selected_sources(
            varroa.base.deploy.REPORT_MANIFEST)
        self.assertEqual(tuple(result), varroa.FIVE_CASES)
        for case in varroa.FIVE_CASES:
            row = result[case]
            self.assertEqual(row["source_run"], varroa.CANONICAL_RUNS[case])
            self.assertEqual(set(row["seeds"]), {"42", "43", "44"})
            self.assertEqual(row["model_yaml_sha256"],
                             varroa.MODEL_YAML_SHA256[case])
            self.assertEqual(varroa.base.sha(Path(row["model_yaml"])),
                             varroa.MODEL_YAML_SHA256[case])
            for seed in (42, 43, 44):
                self.assertEqual(row["seeds"][str(seed)]["sha256"],
                                 varroa.CHECKPOINT_SHA256[case][seed])

    def test_wrong_hf_alias_is_blocked(self):
        data = json.loads(Path(
            varroa.base.deploy.REPORT_MANIFEST).read_text(encoding="utf-8"))
        data["datasets"]["Varroa"]["cases"]["EDGE_P2"]["source_run"] = (
            "Varroa/original_runs/WRONG_EDGE")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "wrong canonical source"):
                varroa.validate_selected_sources(path)

    def test_wrong_seed_checkpoint_digest_is_blocked(self):
        data = json.loads(Path(
            varroa.base.deploy.REPORT_MANIFEST).read_text(encoding="utf-8"))
        data["datasets"]["Varroa"]["cases"]["FTSC"]["seeds"]["44"][
            "sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "path/SHA changed"):
                varroa.validate_selected_sources(path)

    def test_stage_counts_and_varroa_original_protocol(self):
        self.assertEqual(set(varroa.STAGES), {
            "varroa_five_seed42_fp32",
            "varroa_five_three_seed_fp32",
            "varroa_five_all_precision",
        })
        self.assertEqual([len(c) * len(s) * len(b)
                          for c, s, b in varroa.STAGES.values()],
                         [10, 30, 60])
        self.assertTrue(all(c == varroa.FIVE_CASES and b[0] == "pt32"
                            for c, _, b in varroa.STAGES.values()))
        self.assertEqual(varroa.base.CONFIG["Varroa"],
                         {"imgsz": 640, "batch": 4,
                          "main": "native_rect_val",
                          "small": "predict640_conf025"})

    def test_process_local_contract_preserves_base_source(self):
        original_hashes = lambda: {"base.py": "a" * 64}
        with patch.object(varroa.base, "source_hashes", original_hashes), \\
             patch.object(varroa.base, "CONTRACT", "original"), \\
             patch.object(varroa.base, "REVISION", "original"), \\
             patch.object(varroa.base, "STAGES", {}), \\
             patch.object(varroa.base, "PAPER_ANCHORS", {"Varroa": {}}):
            varroa.install_isolated_contract()
            observed = varroa.base.source_hashes()
            self.assertEqual(set(observed),
                             set(varroa.MODEL_CONFIGS.values())
                             | {"base.py", varroa.SELF_SOURCE})
            self.assertEqual(varroa.base.CONTRACT, varroa.CONTRACT)
            self.assertEqual(varroa.base.REVISION, varroa.REVISION)
            self.assertEqual(varroa.base.PAPER_ANCHORS["Varroa"],
                             varroa.PAPER_PT32_MEANS)
        self.assertEqual(varroa.base.CONTRACT,
                         "ftsc_historical_paper_backend_parity_v1")


if __name__ == "__main__":
    unittest.main()
