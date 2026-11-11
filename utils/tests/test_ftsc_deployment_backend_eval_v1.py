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
                self.assertIn("FTSC_DEPLOY_EVAL_20261111_V2", code)
                self.assertIn(f"FTSC_DEPLOY_{suffix}_PLATFORM = 'marimo_live'", code)
                self.assertNotIn(".train(", code)
                self.assertNotIn("upload_file", code)
                self.assertIn("'--checkpoint-manifest'", code)


if __name__ == "__main__":
    unittest.main()
