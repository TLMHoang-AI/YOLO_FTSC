"""No-network/no-GPU regression tests for all-checkpoint TensorRT INT8 paper evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from utils import ftsc_hf_all_trt_int8_paper_v1 as mod


class InventoryTests(unittest.TestCase):
    def test_all_pt_checkpoint_weights_both_datasets_but_not_wrong_scope(self):
        inputs = [
            "LEVIR-Ship/original_runs/FTSC/seed_42/weights/best.pt",
            "LEVIR-Ship/original_runs/FTSC/seed_42/weights/last.pt",
            "LEVIR-Ship/original_runs/FTSC/seed_42/weights/epoch100.pt",
            "LEVIR-Ship/original_runs/FTSC/seed_42/config.yaml",
            "LEVIR-Ship/model.pt",
            "TinyPerson/original_runs/FTSC/seed_42_corner_sw640_sh512/weights/best.pt",
            "Varroa/original_runs/FTSC/seed_42/weights/best.pt",
            "../LEVIR-Ship/foo/weights/best.pt",
        ]
        result = mod.checkpoint_paths(inputs, "LEVIR-Ship")
        self.assertEqual(len(result), 3)
        self.assertTrue(any(x.endswith("/weights/last.pt") for x in result))
        self.assertFalse(any(x.startswith("Varroa/") for x in result))
        self.assertEqual(len(mod.checkpoint_paths(inputs, "TinyPerson")), 1)

    def test_unique_checkpoint_identity_and_seed_resolver(self):
        a = "TinyPerson/original_runs/FTSC/seed_42_corner_sw640_sh512/weights/best.pt"
        b = "TinyPerson/original_runs/FTSC/seed_43_corner_sw640_sh512/weights/best.pt"
        self.assertEqual(mod.checkpoint_seed(a), (42, "checkpoint_path"))
        self.assertEqual(mod.checkpoint_seed(b), (43, "checkpoint_path"))
        self.assertNotEqual(mod.job_id(a), mod.job_id(b))
        self.assertEqual(mod.checkpoint_seed("LEVIR-Ship/x/weights/best.pt"),
                         (42, "CALIBRATION_FALLBACK_42_UNVERIFIED_TRAINING_SEED"))

    def test_cannot_mutate_pinned_snapshot_inventory(self):
        path = "LEVIR-Ship/original_runs/H2/seed_42/weights/best.pt"
        data = {
            "schema": mod.CONTRACT, "repo": mod.HF_REPO, "dataset": "LEVIR-Ship",
            "hf_revision": "a" * 40, "checkpoint_count": 1,
            "checkpoints": [{"path": path, "job_id": mod.job_id(path),
                             "checkpoint_seed": 42,
                             "seed_resolution": "checkpoint_path",
                             "kind": "best.pt"}],
        }
        mod.validate_inventory(data, mod.HF_REPO, "LEVIR-Ship")
        data["checkpoints"][0]["job_id"] = "incorrect"
        with self.assertRaises(RuntimeError):
            mod.validate_inventory(data, mod.HF_REPO, "LEVIR-Ship")


class QuantizationAndPaperMetricTests(unittest.TestCase):
    def test_requires_genuine_int8_dynamic_engine_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "best.engine"
            good = {"task":"detect", "args":{"int8":True,"dynamic":True,"nms":False}}
            payload = json.dumps(good).encode()
            path.write_bytes(len(payload).to_bytes(4,"little",signed=True) + payload + b"TRT_ENGINE_BODY")
            self.assertEqual(mod.read_engine_metadata(path)["args"]["int8"], True)
            for key, value in (("int8", False), ("dynamic", False), ("nms", True)):
                bad = {"task":"detect","args":dict(good["args"])}
                bad["args"][key] = value
                encoded = json.dumps(bad).encode()
                path.write_bytes(len(encoded).to_bytes(4,"little",signed=True)+encoded+b"ENGINE_BODY")
                with self.assertRaises(RuntimeError):
                    mod.read_engine_metadata(path)

    def test_all_tiny_metrics_merged_and_levir_small_native(self):
        self.assertEqual(mod.METRICS["TinyPerson"],
                         ("test_merged/AP50", "test_merged/mAP50-95",
                          "test_merged/AP50-Small"))
        self.assertEqual(mod.METRICS["LEVIR-Ship"],
                         ("test/metrics/mAP50(B)", "test/metrics/mAP50-95(B)",
                          "test_size/AP50-Small"))

    def test_source_contract_is_calibration_train_not_evaluation(self):
        code = (mod.ROOT / "utils/ftsc_hf_all_trt_int8_paper_v1.py").read_text(encoding="utf-8")
        self.assertIn('split="train", fraction=1.0', code)
        self.assertIn('int8=True, half=False, dynamic=True', code)
        self.assertIn('paper._predict_tiny(', code)
        self.assertIn('capture_rect_val_inputs(', code)
        self.assertIn('small["test_size/AP50-Small"]', code)
        self.assertIn('native_image_size', code)
        self.assertIn('"corner_AP_not_used": True', code)


if __name__ == "__main__":
    unittest.main()
