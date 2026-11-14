"""Offline file identity and no-GPU guard tests for LEVIR 512-canvas isolation."""
import json
import tempfile
import unittest
from pathlib import Path

from utils import ftsc_levir_small_ap_canvas_isolation_v1 as isolate


class CanvasIsolationContracts(unittest.TestCase):
    def test_rejects_wrong_historical_report(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"history.json"
            p.write_text(json.dumps({"schema":"unrelated","mode":"historical"}))
            with self.assertRaisesRegex(RuntimeError,"Not a prior"):
                isolate._get_report(p)
            p.write_text(json.dumps({
                "schema":"ftsc_levir_ap50_small_rootcause_diagnostic_v1",
                "mode":"replay",
            }))
            with self.assertRaisesRegex(RuntimeError,"Not a prior"):
                isolate._get_report(p)

    def test_rejects_source_code_drift(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"history.json"
            p.write_text(json.dumps({
                "schema":"ftsc_levir_ap50_small_rootcause_diagnostic_v1",
                "mode":"historical",
                "source_code_sha256":{
                    "evaluate_test/size_bucket_evaluator.py":"0"*64,
                },
            }))
            with self.assertRaisesRegex(RuntimeError,"Source changed"):
                isolate._get_report(p)

    def test_unchanged_history_header_can_be_verified(self):
        from utils import ftsc_levir_small_ap_rootcause_v1 as previous
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"history.json"
            target=Path(previous.__file__).resolve()
            digest=previous.sha256(target)
            p.write_text(json.dumps({
                "schema":"ftsc_levir_ap50_small_rootcause_diagnostic_v1",
                "mode":"historical",
                "source_code_sha256":{
                    "utils/ftsc_levir_small_ap_rootcause_v1.py":digest,
                },
                "results":[],
            }))
            saved=isolate._get_report(p)
            self.assertEqual(saved["results"],[])


if __name__=="__main__":
    unittest.main()
