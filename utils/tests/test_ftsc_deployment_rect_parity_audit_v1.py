"""Offline multi-seed, multi-precision audit regression tests (no CUDA/HF)."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils import ftsc_deployment_backend_eval_v1 as deploy
from utils import ftsc_deployment_rect_parity_audit_v1 as audit


def _save(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LevirFullParityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "runs"
        self.project = (self.base / deploy.RECT_PARITY_PROTOCOL /
                        deploy.RECT_PARITY_RUNTIME_REVISION)
        self.paths = {}
        self.manifest = self.root / "report_manifest.json"
        self.meta = {"schema": deploy.REPORT_SCHEMA, "repo_id": deploy.REPO_ID,
                     "datasets": {"LEVIR-Ship": {"cases": {"FTSC": {"seeds": {}}}}}}
        self.inputs = {
            "schema": "ftsc_levir_exact_paper_rect_input_pixels_v3",
            "paper_imgsz": 512, "canvas": 544,
            "minimum_pytorch_stride": 32,
            "source_images_count": 788,
            "image_sha256": {
                f"/frozen/test/{i:04}.jpg": "a" * 64 for i in range(788)
            },
        }
        for seed in (42, 43, 44):
            raw_pt = f"seed={seed};real-source-pt".encode()
            src = deploy.source_path(self.project, "LEVIR-Ship", "FTSC", seed)
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(raw_pt)
            self.meta["datasets"]["LEVIR-Ship"]["cases"]["FTSC"]["seeds"][str(seed)] = {
                "path": f"LEVIR-Ship/fixture/FTSC/seed_{seed}/weights/best.pt",
                "sha256": _sha_bytes(raw_pt),
            }
        _save(self.manifest, json.dumps(self.meta))

    def add_run(self, seed: int, backend: str) -> None:
        source = self.meta["datasets"]["LEVIR-Ship"]["cases"]["FTSC"]["seeds"][str(seed)]
        folder = deploy.path_for(self.project, "LEVIR-Ship", "FTSC", seed, backend)
        folder.mkdir(parents=True, exist_ok=True)
        report = {
            **deploy.expected_identity(
                SimpleNamespace(dataset="LEVIR-Ship", platform="marimo_live",
                                evaluation_protocol=deploy.RECT_PARITY_PROTOCOL),
                {"case": "FTSC", "seed": seed, "backend": backend,
                 "source": source["path"]},
            ),
            "evaluator": "ultralytics_paper_rect_native_val_with_static_engine_544_v3",
            "source_sha256": source["sha256"], "imgsz": 512,
            "device": "cuda:0",
            "gpu_name": "fixture_gpu",
            "torch_version": "fixture_torch",
            "cuda_version": "fixture_cuda",
            "data_yaml_sha256": "b" * 64,
            "preprocessed_input_canvas": 544, "preprocessed_input_count": 788,
            "metrics": {
                "test/metrics/mAP50(B)": 0.812 if backend.startswith("pt") else 0.811,
                "test/metrics/mAP50-95(B)": 0.312,
                "test_size/AP50-Small": 0.89,
            },
        }
        gt = folder / "test_size_ground_truth.json"
        preds = folder / "test_size_predictions.json"
        _save(gt, '{"images": [], "annotations": []}\n')
        _save(preds, '[]\n')
        report["gt_sha256"] = audit.file_sha(gt)
        report["predictions_sha256"] = audit.file_sha(preds)
        manifest = folder / "test_val_preprocessed_inputs_sha256.json"
        _save(manifest, json.dumps(self.inputs, sort_keys=True, indent=2) + "\n")
        report["preprocessed_input_manifest_sha256"] = audit.file_sha(manifest)
        if backend.startswith("pt"):
            report["model_artifact_sha256"] = source["sha256"]
        else:
            engine = folder / "export/source.engine"
            raw = f"mock-tensorrt-{backend}-{seed}".encode()
            engine.parent.mkdir(parents=True, exist_ok=True)
            engine.write_bytes(raw)
            report["model_artifact_sha256"] = audit.file_sha(engine)
            fp = {"source_sha256": source["sha256"],
                  "backend": backend, "imgsz": 544,
                  "paper_preprocess_imgsz": 512,
                  "paper_stride": 32, "paper_rect": True,
                  "simplify": False}
            _save(folder / "export/export_manifest.json", json.dumps({
                "fingerprint": fp,
                "artifact_sha256": report["model_artifact_sha256"],
            }))
            pt_dir = deploy.path_for(self.project, "LEVIR-Ship", "FTSC", seed, "pt32")
            report.update({
                "full_test_input_parity": True,
                "paired_pt32_preprocessed_manifest_sha256":
                    report["preprocessed_input_manifest_sha256"],
                "paired_pt32_metrics_sha256":
                    audit.file_sha(pt_dir / "metrics.json"),
            })
        _save(folder / "metrics.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
        self.paths[(seed, backend)] = folder

    def populate(self, backends):
        for seed in (42, 43, 44):
            for backend in backends:
                self.add_run(seed, backend)

    def test_scope_stages_and_report_inventory(self):
        self.assertEqual(audit.get_scope("ftsc_fp32", self.manifest)["backends"],
                         ["pt32", "trt32"])
        self.assertEqual(audit.get_scope("ftsc_all_precision", self.manifest)["backends"],
                         ["pt32", "trt32", "pt16", "trt16"])
        official = audit.get_scope("report_all_precision", deploy.REPORT_MANIFEST)
        self.assertEqual(len(official["cases"]), 11)
        self.assertEqual(official["seeds"], [42, 43, 44])

    def test_fp32_three_seeds_complete_and_delta_is_reported(self):
        self.populate(("pt32", "trt32"))
        result = audit.audit_stage("ftsc_fp32", self.base, self.manifest)
        self.assertEqual(result["status"], "PASS_COMPLETE")
        self.assertEqual(result["valid_jobs"], 6)
        self.assertEqual(len(result["comparisons"]), 3)
        self.assertEqual(len(result["three_seed_preview"]), 2)
        self.assertAlmostEqual(result["comparisons"][0]["AP50_delta_pp"], -0.1)

    def test_fails_if_missing_seed_or_engine_mutated(self):
        self.populate(("pt32", "trt32"))
        bad = self.paths[(44, "trt32")] / "metrics.json"
        bad.unlink()
        report = audit.audit_stage("ftsc_fp32", self.base, self.manifest)
        self.assertEqual(report["status"], "INCOMPLETE_OR_INVALID")
        self.assertEqual(report["valid_jobs"], 5)
        self.add_run(44, "trt32")
        engine = self.paths[(43, "trt32")] / "export/source.engine"
        engine.write_bytes(b"modified-engine")
        report = audit.audit_stage("ftsc_fp32", self.base, self.manifest)
        self.assertEqual(report["status"], "INCOMPLETE_OR_INVALID")
        self.assertIn("static TensorRT engine", str(report["issues"]))

    def test_fp16_requires_exact_precision_pt16_and_all_image_sha(self):
        self.populate(("pt32", "trt32"))
        incomplete = audit.audit_stage("ftsc_all_precision", self.base, self.manifest)
        self.assertEqual(incomplete["status"], "INCOMPLETE_OR_INVALID")
        for seed in (42, 43, 44):
            self.add_run(seed, "pt16")
            self.add_run(seed, "trt16")
        result = audit.audit_stage("ftsc_all_precision", self.base, self.manifest)
        self.assertEqual(result["status"], "PASS_COMPLETE")
        self.assertEqual(result["valid_jobs"], 12)
        self.assertTrue(any(row["reference"] == "pt16" and row["backend"] == "trt16"
                            for row in result["comparisons"]))
        # A valid metrics.json does not excuse changed preprocessed pixels.
        corrupted = self.paths[(42, "pt16")] / "test_val_preprocessed_inputs_sha256.json"
        modified = json.loads(corrupted.read_text())
        modified["image_sha256"]["/frozen/test/0000.jpg"] = "f" * 64
        _save(corrupted, json.dumps(modified, sort_keys=True, indent=2) + "\n")
        bad = audit.audit_stage("ftsc_all_precision", self.base, self.manifest)
        self.assertEqual(bad["status"], "INCOMPLETE_OR_INVALID")
        self.assertTrue(bad["issues"])

    def test_large_accuracy_drop_requires_review_not_false_parity_pass(self):
        self.populate(("pt32", "trt32"))
        target = self.paths[(43, "trt32")] / "metrics.json"
        contents = json.loads(target.read_text(encoding="utf-8"))
        contents["metrics"]["test/metrics/mAP50(B)"] = 0.77
        _save(target, json.dumps(contents))
        report = audit.audit_stage("ftsc_fp32", self.base, self.manifest)
        self.assertEqual(report["status"], "COMPLETE_NEEDS_ACCURACY_REVIEW")
        self.assertEqual(report["valid_jobs"], 6)
        self.assertEqual(len(report["issues"]), 0)
        self.assertEqual(len(report["accuracy_review_flags"]), 1)
        self.assertLess(report["accuracy_review_flags"][0]["AP50_delta_pp"], -4)

    def test_audit_cannot_read_forged_source_or_mixed_eval_contract(self):
        self.populate(("pt32", "trt32"))
        ptfile = self.paths[(43, "pt32")] / "metrics.json"
        raw = json.loads(ptfile.read_text())
        raw["validation_config"]["nms_iou"] = 0.7
        _save(ptfile, json.dumps(raw))
        bad = audit.audit_stage("ftsc_fp32", self.base, self.manifest)
        self.assertEqual(bad["status"], "INCOMPLETE_OR_INVALID")
        self.assertIn("drift", str(bad["issues"]))


if __name__ == "__main__":
    unittest.main()
