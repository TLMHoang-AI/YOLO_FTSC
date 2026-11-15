"""Varroa validation-only offline regression checks (no Hugging Face or GPU)."""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

from utils import ftsc_varroa_validation_paper_v1 as val


class VarroaManifestTests(unittest.TestCase):
    def test_thirteen_cases_thirty_nine_unique_weights(self):
        m = val.load_manifest()
        self.assertEqual(set(m["cases"]), set(val.CASES))
        self.assertEqual(len(m["cases"]), 13)
        refs = [item for case in m["cases"].values()
                for item in case["checkpoints"].values()]
        self.assertEqual(len(refs), 39)
        self.assertEqual(len({x["sha256"] for x in refs}), 39)
        self.assertTrue(m["cases"]["HEAD_H2"]["source_run"].endswith("/VR_STRUCT_H2_P23"))
        self.assertTrue(m["cases"]["FTSC"]["source_run"].endswith("/V1_FTSC"))
        self.assertNotIn("CP2", m["cases"])

    def test_case_and_seed_filter_reject_unknown_and_repetitions(self):
        self.assertEqual(val.select_cases(None), val.CASES)
        self.assertEqual(val.select_seeds(None), val.SEEDS)
        self.assertEqual(val.select_cases(["ALPHA8", "FTSC"]), ("FTSC", "ALPHA8"))
        self.assertEqual(val.select_seeds([44, 42]), (42, 44))
        for requested in (["FTSC", "FTSC"], ["CP2"], ["EDGE_CUE"]):
            with self.assertRaises(RuntimeError):
                val.select_cases(requested)
        for requested in ([41], [42, 42]):
            with self.assertRaises(RuntimeError):
                val.select_seeds(requested)

    def test_manifest_cannot_merge_structural_and_full_neck(self):
        m = val.load_manifest()
        m["cases"]["HEAD_H2"]["source_run"] = m["cases"]["FTSC"]["source_run"]
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "cases.json"
            path.write_text(json.dumps(m), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                val.load_manifest(path)

    def test_seed_yaml_is_nonoptional_and_seed_specific(self):
        for bad in ("some.yaml", "datasets/varroa_yolo_seed42/varroa.yaml"):
            with self.assertRaises(RuntimeError):
                val.yaml_for_seed(bad, 43)


class VarroaMetricsTests(unittest.TestCase):
    def setUp(self):
        self.m = val.load_manifest()
        self.sources = {"runner": "0" * 64}
        self.contexts = {}
        for seed in val.SEEDS:
            self.contexts[seed] = ({
                "seed": seed, "split": "val",
                "data_yaml_sha256": f"sha{seed}",
                "val_image_count": 592}, [], Path("/fake") / f"varroa_yolo_seed{seed}" / "varroa.yaml")

    def record(self, case, seed):
        source = self.m["cases"][case]["checkpoints"][str(seed)]
        return {
            "contract": val.CONTRACT, "dataset": "Varroa",
            "case": case, "seed": seed,
            "source/hf_repo": self.m["repo_id"],
            "source/hf_revision": "e" * 40,
            "source/checkpoint": source["path"],
            "source/checkpoint_sha256": source["sha256"],
            "source/evaluator_sha256": self.sources,
            "split_provenance": self.contexts[seed][0],
            "evaluation": {
                "no_training": True, "split": "val",
                "imgsz": 640, "batch": 4,
                "native_nms_iou": .7, "small_nms_iou": .5,
            },
            "metrics": {
                val.METRICS[0]: .91 + seed / 10000.0,
                val.METRICS[1]: .35 + seed / 10000.0,
                val.METRICS[2]: .88 + seed / 10000.0,
            },
        }

    def test_metrics_reject_bad_protocol_and_wrong_split_seed(self):
        r = self.record("HEAD_H2", 43)
        val.verified_record(r, self.m, "HEAD_H2", 43,
                            self.contexts[43][0], self.sources)
        with self.assertRaises(RuntimeError):
            val.verified_record(r, self.m, "HEAD_H2", 43,
                                self.contexts[42][0], self.sources)
        r["evaluation"]["native_nms_iou"] = .5
        with self.assertRaises(RuntimeError):
            val.verified_record(r, self.m, "HEAD_H2", 43,
                                self.contexts[43][0], self.sources)

    def test_full_39_aggregate_sample_std_and_block_partial(self):
        with tempfile.TemporaryDirectory() as td:
            project = Path(td) / "eval"
            args = Namespace(project=project)
            for case in val.CASES:
                for seed in val.SEEDS:
                    run = val.job_dir(project, case, seed)
                    run.mkdir(parents=True)
                    (run / "val_gt.json").write_text("[]", encoding="utf-8")
                    (run / "val_predictions.json").write_text("[]", encoding="utf-8")
                    data = self.record(case, seed)
                    data["evaluation"]["gt_sha256"] = val.sha256(run / "val_gt.json")
                    data["evaluation"]["predictions_sha256"] = val.sha256(run / "val_predictions.json")
                    (run / "val_metrics.json").write_text(json.dumps(data), encoding="utf-8")
            complete = val.aggregate(args, self.m, val.CASES, val.SEEDS,
                                     self.contexts, self.sources)
            self.assertEqual(complete["status"], "COMPLETE")
            self.assertEqual(complete["valid_jobs"], 39)
            data = (project / "varroa_val_metrics_aggregate.csv").read_text()
            self.assertEqual(len(data.strip().splitlines()), 14)
            self.assertIn("Val_AP50_small_pct_sample_std", data)
            # One missing input must not regenerate an apparently complete CSV.
            (val.job_dir(project, "EDGE_P3", 44) / "val_metrics.json").unlink()
            old_csv = (project / "varroa_val_metrics_aggregate.csv").read_bytes()
            blocked = val.aggregate(args, self.m, val.CASES, val.SEEDS,
                                    self.contexts, self.sources)
            self.assertEqual(blocked["status"], "BLOCKED")
            self.assertEqual(blocked["valid_jobs"], 38)
            self.assertEqual((project / "varroa_val_metrics_aggregate.csv").read_bytes(),
                             old_csv)

    def test_metrics_files_are_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "record.json"
            val.new_json(path, {"v": 1})
            with self.assertRaises(FileExistsError):
                val.new_json(path, {"v": 2})


if __name__ == "__main__":
    unittest.main()
