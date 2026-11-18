"""Offline regression checks for ONLY two LEVIR P345+FTSC port YAMLs."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from train_levir_scripts import train_levir_ftsc_p345_v9_v11_v1 as suite
from train_levir_scripts import ftsc_p345_v9_v11_hf_v1 as hf
from utils.hf_result_archive import MARKER_FILENAME


class FakeHfApi:
    def __init__(self, paths=(), fail=False):
        self.paths = tuple(paths)
        self.fail = fail

    def repo_info(self, *, repo_id, repo_type):
        assert repo_id == "TLMHoang/FTSC_results" and repo_type == "dataset"
        if self.fail:
            raise ConnectionError("Remote HF unavailable")
        return SimpleNamespace(sha="1234567890abcdef" * 3)

    def list_repo_files(self, *, repo_id, repo_type, revision):
        assert revision == "1234567890abcdef" * 3
        return self.paths


class OnlyNewP345Tests(unittest.TestCase):
    def test_exact_two_cases(self):
        self.assertEqual(set(suite.CASES), {"V9T_P345_FTSC", "V11N_P345_FTSC"})
        self.assertEqual(suite.SEEDS, (42, 43, 44))
        defaults = suite.parse_args([])
        self.assertFalse(defaults.confirm_run)
        self.assertFalse(defaults.aggregate)
        self.assertFalse(defaults.resume)
        self.assertEqual(set(defaults.cases), set(suite.CASES))

    def test_exact_stock_graph_and_canonical_ftsc(self):
        result = suite.source_preflight()
        self.assertEqual(result["status"], "STATIC_TWO_CASE_PREFLIGHT_PASSED")
        self.assertEqual(set(result["cases"]), set(suite.CASES))
        for case, spec in suite.CASES.items():
            model, stock, h2 = [suite.load(spec[name]) for name in ("model", "stock", "h2")]
            self.assertEqual(suite.graph(model), suite.graph(stock))
            self.assertEqual(model["ftsc"], h2["ftsc"])
            self.assertTrue(model["ftsc"]["enabled"])
            self.assertEqual(model["ftsc"]["policy"], "f5")
            self.assertEqual(tuple(model["head"][-1][0]), spec["detect"])
            self.assertEqual(len(spec["detect"]), 3)

    def test_protocol_is_pinned_to_cross_yolo_adamw(self):
        args = argparse.Namespace(device="cpu", workers=0)
        for case in suite.CASES:
            train = suite.training(args, case, 42, Path("split.yaml"))
            self.assertEqual(train["seed"], 42)
            self.assertEqual(train["optimizer"], "AdamW")
            self.assertEqual(train["mosaic"], 0.0)
            self.assertEqual(train["close_mosaic"], 0)
            self.assertEqual(train["epochs"], 100)
            self.assertEqual(train["imgsz"], 512)
            self.assertEqual(train["batch"], 8)
            self.assertEqual(train["fitness_metric"], "map50_95")

    def test_reject_orphan_checkpoint_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "case" / "seed_42"
            self.assertEqual(suite.state(run, {"schema": "test"}), "NEW")
            run.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "missing scientific contract"):
                suite.state(run, {"schema": "test"})
            for name in ("experiment_manifest.json", "p345_ftsc_contract.json"):
                (run / name).write_text('{"schema":"wrong"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                suite.state(run, {"schema": "test"})

    def test_invalid_pretrained_case_cannot_enter(self):
        with self.assertRaises(KeyError):
            suite.pretrained_for(suite.parse_args([]), "V8_P345_FTSC")

    def test_hf_namespace_and_absent_runs(self):
        path = hf.remote_prefix("V9T_P345_FTSC", 42)
        self.assertEqual(path,
            "LEVIR-Ship/journal_experiments/p345_ftsc_v9_v11_v1/V9T_P345_FTSC/seed_42")
        report = hf.inventory("test-only", api=FakeHfApi())
        self.assertEqual(len(report["rows"]), 6)
        self.assertTrue(all(r["status"] == "ABSENT" for r in report["rows"]))

    def test_hf_partial_archive_fails_closed(self):
        path = hf.remote_prefix("V11N_P345_FTSC", 44)
        with self.assertRaisesRegex(RuntimeError, "Partial remote run"):
            hf.inventory("test-only", api=FakeHfApi([path + "/weights/best.pt"]))

    def test_hf_completed_marker_must_have_all_files(self):
        path = hf.remote_prefix("V11N_P345_FTSC", 44)
        files = [path + "/" + name for name in (*hf.REQUIRED, MARKER_FILENAME)]
        report = hf.inventory("test-only", api=FakeHfApi(files))
        status = {(r["case"], r["seed"]): r["status"] for r in report["rows"]}
        self.assertEqual(status[("V11N_P345_FTSC", 44)], "MARKED_PRESENT")
        self.assertEqual(status[("V9T_P345_FTSC", 42)], "ABSENT")

    def test_hf_network_failure_does_not_mean_absent(self):
        with self.assertRaises(ConnectionError):
            hf.inventory("test-only", api=FakeHfApi(fail=True))

    def test_unauthorized_hf_case_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Only the two authorized"):
            hf.remote_prefix("P23_FTSC", 42)


if __name__ == "__main__":
    unittest.main()
