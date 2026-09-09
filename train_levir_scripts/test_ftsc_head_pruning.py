"""Focused structural/preflight checks for the H0-H3 head-pruning suite."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_head_pruning as suite


class HeadPruningTests(unittest.TestCase):
    def test_yaml_contract_and_detect_inputs(self):
        expected_inputs = {
            "H0_p2_p3_p4_p5_nomosaic": [19, 22, 25, 28],
            "H1_p2_p3_p4_nomosaic": [19, 22, 25],
            "H2_p2_p3_nomosaic": [19, 22],
            "H3_p2_only_nomosaic": [19],
        }
        for variant, path in suite.VARIANTS.items():
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["head"][-1][2], "Detect")
            self.assertEqual(payload["head"][-1][0], expected_inputs[variant])
            self.assertEqual(payload["ftsc"]["policy"], "f5")
            self.assertEqual(payload["ftsc"]["evidence"], ["position_gaussian", "dfl_distribution"])
            self.assertEqual(payload["ftsc"]["fixed_strengths"], {"dfl_distribution": 1.0})

    def test_protocol_contract(self):
        args = suite.parse_args(["--preflight-only"])
        kwargs = suite.train_kwargs(args, Path("data.yaml"), 42, True)
        self.assertEqual(kwargs["epochs"], 100)
        self.assertEqual(kwargs["patience"], 20)
        self.assertEqual(kwargs["imgsz"], 512)
        self.assertEqual(kwargs["batch"], 8)
        self.assertEqual(kwargs["optimizer"], "auto")
        self.assertFalse(kwargs["cos_lr"])
        self.assertEqual(kwargs["lrf"], 0.01)
        self.assertEqual(kwargs["fitness_metric"], "map50_95")
        self.assertEqual(kwargs["mosaic"], 0.0)
        self.assertEqual(kwargs["close_mosaic"], 0)

    @unittest.skipUnless(importlib.util.find_spec("cv2") is not None, "Ultralytics runtime dependency cv2 is not installed")
    def test_build_strides_and_monotonic_complexity(self):
        report = suite.model_preflight(suite.parse_args(["--preflight-only", "--device", "cpu"]))
        params = [report["complexity"][name]["params"] for name in suite.VARIANTS]
        self.assertEqual(params, sorted(params, reverse=True))
        gflops = [report["complexity"][name]["GFLOPs"] for name in suite.VARIANTS]
        if all(gflops):
            self.assertEqual(gflops, sorted(gflops, reverse=True))
        for name, strides in suite.EXPECTED_STRIDES.items():
            self.assertEqual(report["variant_reports"][name]["head_strides"], strides)


if __name__ == "__main__":
    unittest.main()
