"""Offline checks for LEVIR TensorRT INT8 static544/static512 FP16-V3 parity."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from utils import ftsc_hf_all_trt_int8_levir_fp16parity_v2 as mod
from utils import ftsc_hf_all_trt_int8_paper_v1 as legacy


class StaticVsDynamicContract(unittest.TestCase):
    def test_roles_match_historical_fp16_main_and_small(self):
        self.assertEqual(mod.ROLES, {"main": 544, "small": 512})
        self.assertNotEqual(mod.DEFAULT_PROJECT, legacy.ROOT/"runs/ftsc_hf_int8_all_v1")
        self.assertIn("STATIC_FP16", mod.CONTRACT)
        self.assertEqual(mod.METRIC_KEYS,
                         ("test/metrics/mAP50(B)",
                          "test/metrics/mAP50-95(B)",
                          "test_size/AP50-Small"))

    def test_reject_non_int8_dynamic_bad_batch_and_wrong_size(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "source.engine"
            good = {"task": "detect", "batch": 1, "imgsz": [544,544],
                    "args":{"int8": True, "dynamic": False, "nms": False}}
            def save(meta):
                encoded = json.dumps(meta).encode()
                path.write_bytes(len(encoded).to_bytes(4, "little", signed=True)
                                 + encoded + b"TRT_ENGINE_BYTES")
            save(good)
            self.assertEqual(mod.static_engine_metadata(path, "main"), good)
            with self.assertRaises(RuntimeError):
                mod.static_engine_metadata(path, "small")
            for key, value in (("int8", False), ("dynamic", True), ("nms", True)):
                altered = json.loads(json.dumps(good))
                altered["args"][key] = value
                save(altered)
                with self.assertRaises(RuntimeError):
                    mod.static_engine_metadata(path, "main")
            for key,value in (("batch",8),("imgsz",[512,512])):
                altered = json.loads(json.dumps(good))
                altered[key] = value
                save(altered)
                with self.assertRaises(RuntimeError):
                    mod.static_engine_metadata(path, "main")

    def test_pinned_scope_explicit_static_not_stale_dynamic(self):
        path = "LEVIR-Ship/historical_reconstructions/FTSC/seed_42/weights/best.pt"
        data = {
            "schema": mod.CONTRACT, "repo": mod.HF_REPO, "dataset": mod.DATASET,
            "hf_revision": "a"*40, "count":1, "calibration_split":"train",
            "engines":{"main":{"imgsz":544,"batch":1,"dynamic":False},
                       "small":{"imgsz":512,"batch":1,"dynamic":False}},
            "checkpoints":[{"path":path,"job_id":legacy.job_id(path),
                            "checkpoint_seed":42,"seed_resolution":"checkpoint_path",
                            "kind":"best.pt"}],
        }
        mod.validate_inventory(data, mod.HF_REPO)
        data["engines"]["main"]["dynamic"] = True
        with self.assertRaises(RuntimeError):
            mod.validate_inventory(data, mod.HF_REPO)


class PixelParityUnitTests(unittest.TestCase):
    def test_small_requires_788_identical_inputs_and_exact_gt(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            other = folder / "pt32_reference"
            other.mkdir()
            a = other/"small_inputs.json"
            b = folder/"small_inputs.json"
            a.write_bytes(b"same actual CHW tensor fingerprints")
            b.write_bytes(a.read_bytes())
            digest = legacy.sha256(a)
            baseline = {"input_manifest_sha256":digest, "input_count":788,
                        "gt_sha256":"a"*64}
            current = {"input_manifest_sha256":digest, "input_count":788,
                       "gt_sha256":"a"*64}
            out = mod.verify_small_parity(folder, baseline, current)
            self.assertTrue(out["main_and_small_input_pixel_parity"])
            b.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                mod.verify_small_parity(folder, baseline, current)

    def test_pt32_reference_is_mandatory_and_used_only_on_levir(self):
        src = (mod.ROOT/"utils/ftsc_hf_all_trt_int8_levir_fp16parity_v2.py").read_text()
        self.assertIn("paper_rect_static_validator()",src)
        self.assertIn("imgsz=544, batch=1, rect=True",src)
        self.assertIn("imgsz=512, batch=8, rect=True",src)
        self.assertIn("source_batch = MAX_SOURCE_LIST_CHUNK if reference else 1",src)
        self.assertIn('dynamic_required=False',src)
        self.assertIn('split="train", fraction=1.0',src)
        self.assertIn('int8=True, dynamic=False',src)
        self.assertIn("pt32_reference/test_val_preprocessed_inputs_sha256.json",src)
        self.assertIn('prior.read_bytes() == now.read_bytes()',src)
        self.assertNotIn(".train(", src)


if __name__ == "__main__":
    unittest.main()
