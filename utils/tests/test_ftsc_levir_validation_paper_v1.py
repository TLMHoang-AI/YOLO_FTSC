"""Offline validation runner regression tests: NO checkpoint download or GPU."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from utils import ftsc_levir_validation_paper_v1 as val


class ManifestTests(unittest.TestCase):
    def test_paper_scope_is_eleven_configs_thirty_three_jobs(self):
        m = val.load_manifest()
        self.assertEqual(len(m["cases"]), 11)
        self.assertEqual(len(m["cases"]) * len(m["seeds"]), 33)
        self.assertEqual(set(m["cases"]), set(val.EXPECTED_CASES))
        self.assertFalse(set(m["cases"]) & {"CP2", "OACP", "NC_R4"})
        self.assertTrue(m["cases"]["COLOR_CUE"]["source_run"].endswith("/C_H2_POSTFUSION_V2"))
        self.assertEqual(len({c["checkpoints"][str(s)]["sha256"]
                              for c in m["cases"].values() for s in val.ALL_SEEDS}), 33)

    def test_selection_validation(self):
        self.assertEqual(val.selected_cases([]), val.EXPECTED_CASES)
        self.assertEqual(val.selected_seeds([]), val.ALL_SEEDS)
        self.assertEqual(val.selected_cases(["ALPHA4", "FTSC"]), ("FTSC", "ALPHA4"))
        for selected in (["FTSC", "FTSC"], ["CP2"]):
            with self.assertRaises(RuntimeError):
                val.selected_cases(selected)
        for selected in ([41], [42, 42]):
            with self.assertRaises(RuntimeError):
                val.selected_seeds(selected)

    def test_manifest_rejects_legacy_color(self):
        source = val.load_manifest()
        source["cases"]["COLOR_CUE"]["source_run"] = (
            "LEVIR-Ship/historical_reconstructions/RECON_C_H2_LEGACY_V1")
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "cases.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                val.load_manifest(path)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.m = val.load_manifest()
        self.split = {"frozen_split_seed": 42, "split": "val", "image_count": 3}
        self.hashes = {"runner": "a" * 64}

    def make_record(self, case: str, seed: int) -> dict:
        item = self.m["cases"][case]["checkpoints"][str(seed)]
        return {
            "contract": val.CONTRACT, "dataset": "LEVIR-Ship",
            "case": case, "seed": seed, "source/checkpoint": item["path"],
            "source/checkpoint_sha256": item["sha256"],
            "source/hf_repo": self.m["repo_id"], "source/hf_revision": "a" * 40,
            "source/evaluator_sha256": self.hashes, "split_provenance": self.split,
            "evaluation": {"no_training": True},
            "metrics": {val.METRICS[0]: .82 + seed / 10000,
                        val.METRICS[1]: .32 + seed / 10000,
                        val.METRICS[2]: .80 + seed / 10000},
        }

    def test_reject_mismatched_checkpoint_and_invalid_small(self):
        record = self.make_record("FTSC", 42)
        val.validate_result(record, self.m, "FTSC", 42, self.split, self.hashes)
        record["source/checkpoint_sha256"] = "b" * 64
        with self.assertRaises(RuntimeError):
            val.validate_result(record, self.m, "FTSC", 42, self.split, self.hashes)
        record = self.make_record("FTSC", 42)
        record["metrics"][val.METRICS[2]] = -1
        with self.assertRaises(RuntimeError):
            val.validate_result(record, self.m, "FTSC", 42, self.split, self.hashes)

    def test_aggregate_all_33_sample_std_and_no_partial_csv(self):
        with tempfile.TemporaryDirectory() as t:
            project = Path(t) / "out"
            args = Namespace(project=project)
            for case in val.EXPECTED_CASES:
                for seed in val.ALL_SEEDS:
                    folder = val.job_dir(project, case, seed)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "val_gt.json").write_text("[]", encoding="utf-8")
                    (folder / "val_predictions.json").write_text("[]", encoding="utf-8")
                    rec = self.make_record(case, seed)
                    rec["evaluation"]["gt_sha256"] = val.sha256(folder / "val_gt.json")
                    rec["evaluation"]["predictions_sha256"] = val.sha256(folder / "val_predictions.json")
                    (folder / "val_metrics.json").write_text(json.dumps(rec), encoding="utf-8")
            report = val.aggregate(args, self.m, val.EXPECTED_CASES,
                                   val.ALL_SEEDS, self.split, self.hashes)
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(report["valid_jobs"], 33)
            aggregate_csv = (project / "levir_val_metrics_aggregate.csv").read_text()
            self.assertEqual(aggregate_csv.count("\n"), 12)
            self.assertIn("Val_AP50_small_pct_sample_std", aggregate_csv)
            # Removing one seed must prohibit *any* new complete aggregate CSV
            missing = val.job_dir(project, "ALPHA8", 44) / "val_metrics.json"
            missing.unlink()
            previous = (project / "levir_val_metrics_aggregate.csv").read_bytes()
            blocked = val.aggregate(args, self.m, val.EXPECTED_CASES,
                                    val.ALL_SEEDS, self.split, self.hashes)
            self.assertEqual(blocked["status"], "BLOCKED")
            self.assertEqual(blocked["valid_jobs"], 32)
            self.assertEqual((project / "levir_val_metrics_aggregate.csv").read_bytes(),
                             previous)

    def test_immutable_per_seed_files(self):
        with tempfile.TemporaryDirectory() as t:
            file = Path(t) / "out.json"
            val.write_new(file, {"score": 1})
            with self.assertRaises(FileExistsError):
                val.write_new(file, {"score": 2})
            self.assertEqual(json.loads(file.read_text()), {"score": 1})


if __name__ == "__main__":
    unittest.main()
