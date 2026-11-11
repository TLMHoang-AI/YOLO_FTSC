"""Offline contract checks for FTSC deployment backend evaluation.

Run from yolo_code: python -m unittest utils.tests.test_ftsc_deployment_backend_eval_v1
No checkpoint downloads, PyTorch import, GPU execution, or training.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from utils import ftsc_deployment_backend_eval_v1 as deploy


class DeploymentContractTests(unittest.TestCase):
    def test_plan_defaults_and_checkpoint_provenance(self):
        p = deploy.parser()
        args = p.parse_args(["--dataset", "TinyPerson", "--phase", "plan"])
        self.assertEqual(args.platform, "marimo_live")
        args.manifest_cases = {}
        args.report_sections = {}
        args.cases = ("FTSC",)
        args.project = Path("/tmp/test_deployment")
        rows = deploy.planned(args)
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["backend"] for r in rows}, set(deploy.DEFAULT_BACKENDS))
        self.assertEqual({r["seed"] for r in rows}, {42, 43, 44})
        self.assertTrue(all(r["source"].startswith("TinyPerson/original_runs/H2_MOSAIC/")
                            for r in rows))
        self.assertTrue(all("/seed_42_corner_sw640_sh512/" in r["source"]
                            for r in rows if r["seed"] == 42))

    def test_manifest_requires_explicit_sha_and_safe_canonical_root(self):
        digest = "a" * 64
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cases.json"
            def run(entry):
                path.write_text(json.dumps({"repo_id": deploy.REPO_ID,
                                            "datasets": {"Varroa": {"V2_EDGE025": {"42": entry}}}}))
                return deploy.manifest_entries(path, "Varroa")
            self.assertEqual(run({"path": "Varroa/journal_experiments/edge/V2_EDGE025/seed_42/weights/best.pt",
                                  "sha256": digest})["V2_EDGE025"][42]["sha256"], digest)
            for bad in (
                {"path": "../weights/best.pt", "sha256": digest},
                {"path": "Varroa/../weights/best.pt", "sha256": digest},
                {"path": "Varroa/test/weights/best.pt", "sha256": "bogus"},
            ):
                with self.assertRaises(RuntimeError):
                    run(bad)

    def test_unmapped_case_fails_closed_before_network(self):
        with self.assertRaises(RuntimeError):
            deploy.main(["--dataset", "LEVIR-Ship", "--cases", "ALPHA4_rerun", "--phase", "plan"])
        with self.assertRaises(RuntimeError):
            deploy.main(["--dataset", "TinyPerson", "--seeds", "1", "--phase", "plan"])

    def test_backend_selection_and_metric_contract(self):
        self.assertEqual(deploy.backend_device("ort_cpu32", "cuda:0"), "cpu")
        self.assertEqual(deploy.backend_device("trt16", "cuda:0"), "cuda:0")
        m = {"test_merged/AP50": .57, "test_merged/mAP50-95": .215,
             "test_merged/AP50-Small": .69}
        deploy.validate_metric_values(m, "TinyPerson")
        for bad in (float("nan"), -1, 1.5):
            with self.assertRaises(RuntimeError):
                deploy.validate_metric_values({**m, "test_merged/AP50-Small": bad}, "TinyPerson")

    def test_manifest_no_override_audited_canonical_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cases.json"
            path.write_text(json.dumps({
                "repo_id": deploy.REPO_ID, "datasets": {"TinyPerson": {
                    "FTSC": {"42": {"path": "TinyPerson/fake/weights/best.pt", "sha256": "a" * 64}}}}}))
            with self.assertRaises(RuntimeError):
                deploy.manifest_entries(path, "TinyPerson")

    def test_aggregate_requires_all_three_seeds(self):
        args = argparse.Namespace(dataset="Varroa", seeds=(42,), cases=["FTSC"],
                                  backends=["pt32"], platform="marimo_live",
                                  project=Path("/tmp/does-not-need-to-exist"), manifest_cases={})
        with self.assertRaisesRegex(RuntimeError, "42,43,44"):
            deploy.aggregate(args)

    def test_jetson_is_not_an_accuracy_execution_environment(self):
        with self.assertRaises(SystemExit):
            deploy.parser().parse_args(["--dataset", "LEVIR-Ship", "--platform", "jetson"])
        self.assertEqual(deploy.parser().parse_args(["--dataset", "Varroa"]).platform,
                         "marimo_live")

    def test_curated_report_inventory_has_exact_reference_provenance(self):
        expected = {"LEVIR-Ship": 11, "TinyPerson": 11, "Varroa": 13}
        total = 0
        for dataset, count in expected.items():
            with self.subTest(dataset=dataset):
                cases, sections, unresolved = deploy.load_report_scope(
                    deploy.REPORT_MANIFEST, dataset)
                self.assertEqual(len(cases), count)
                self.assertEqual(len(unresolved), 16)
                self.assertEqual(set(cases), set(sections))
                self.assertTrue(all(len(rows) == 3 for rows in cases.values()))
                self.assertTrue(all(len(entry["sha256"]) == 64
                                    for per_seed in cases.values() for entry in per_seed.values()))
                self.assertTrue(all("main:" in " ".join(tags)
                                    for alias, tags in sections.items() if alias in ("FTSC", "FTSC_YOLOV9T", "FTSC_YOLO11N")))
                self.assertEqual(cases["FTSC"][42]["path"],
                                 deploy.augmentation_checkpoint(dataset, "FTSC", 42))
                total += sum(len(rows) for rows in cases.values())
        self.assertEqual(total, 105)
        # Distinguish structural Varroa pruning from the full-neck canonical model.
        cases, _, _ = deploy.load_report_scope(deploy.REPORT_MANIFEST, "Varroa")
        self.assertNotEqual(cases["FTSC"][42]["path"], cases["HEAD_H2"][42]["path"])

    def test_report_plan_is_offline_and_defaults_to_all_report_cases(self):
        import contextlib
        import io
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            deploy.main(["--dataset", "LEVIR-Ship", "--scope", "report",
                         "--phase", "plan", "--backends", "pt32"])
        output = json.loads(stream.getvalue())
        self.assertEqual(output["case_count"], 11)
        self.assertEqual(output["checkpoint_count"], 33)
        self.assertEqual(output["evaluation_jobs"], 33)
        self.assertEqual(len(output["unresolved_main_results"]), 16)
        self.assertEqual(len({row["source"] for row in output["rows"]}), 33)

    def test_report_rejects_checkpoint_alias_collision_or_sha_change(self):
        with tempfile.TemporaryDirectory() as td:
            modified = json.loads(deploy.REPORT_MANIFEST.read_text(encoding="utf-8"))
            tiny = modified["datasets"]["Varroa"]["cases"]
            tiny["HEAD_H2"]["seeds"]["42"]["path"] = tiny["FTSC"]["seeds"]["42"]["path"]
            bad_path = Path(td) / "broken_report.json"
            bad_path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                deploy.load_report_scope(bad_path, "Varroa")
            tiny["HEAD_H2"]["seeds"]["42"]["path"] = (
                "Varroa/journal_experiments/phase2/VR_STRUCT_H2_P23/seed_42/weights/best.pt"
            )
            tiny["HEAD_H2"]["seeds"]["42"]["sha256"] = "not-a-hash"
            bad_path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                deploy.load_report_scope(bad_path, "Varroa")

    def test_notebooks_have_separate_safe_default_eval_cells(self):
        root = Path(__file__).resolve().parents[3]
        paths = [
            (root / "current_live_training_notebok_LEV.ipynb", "LEVIR"),
            (root / "current_live_training_notebok_TinyPerson.ipynb", "TINY"),
            (root / "current_live_training_notebok_Varroa.ipynb", "VARROA"),
        ]
        for path, suffix in paths:
            with self.subTest(path=path):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                selected = [c for c in notebook["cells"]
                            if c.get("id", "").startswith("ftsc-deployment-")]
                self.assertEqual(len(selected), 1)
                code = "".join(selected[0]["source"])
                compile(code, str(path) + "#deployment", "exec")
                self.assertIn(f"RUN_FTSC_DEPLOY_{suffix} = False", code)
                self.assertIn(f"FTSC_DEPLOY_{suffix}_ALLOW_EVALUATE = False", code)
                self.assertIn("FTSC_DEPLOY_EVAL_20261112_REPORT_V3", code)
                self.assertIn(f"FTSC_DEPLOY_{suffix}_PLATFORM = 'marimo_live'", code)
                self.assertNotIn(".train(", code)
                self.assertNotIn("upload_file", code)
                self.assertIn("'--report-manifest'", code)
                self.assertIn("'--scope', 'report'", code)
                self.assertIn(f"FTSC_DEPLOY_{suffix}_CASES = []", code)


if __name__ == "__main__":
    unittest.main()
