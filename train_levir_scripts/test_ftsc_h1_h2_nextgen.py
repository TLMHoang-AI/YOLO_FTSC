"""Structural tests for the preregistered H1/H2 next-generation suite.

These tests never train and skip tensor-runtime checks when optional torch/cv2
packages are unavailable in the development environment.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_h1_h2_nextgen as suite


class NextgenContractTests(unittest.TestCase):
    def test_exact_seven_variants_and_detect_inputs(self):
        self.assertEqual(list(suite.VARIANTS), ["AZ_H1", "AZ_H2", "C_H1", "C_H2", "E_H1", "E_H2", "H4_P2_P3_P5"])
        for variant, path in suite.VARIANTS.items():
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["head"][-1][2], "Detect")
            self.assertEqual(payload["head"][-1][0], suite.EXPECTED_DETECT_INPUTS[variant])
            if variant.startswith(("C_", "E_")):
                self.assertEqual(payload["head"][10][2], "P2ColorCueFusion" if variant.startswith("C_") else "P2EdgeCueFusion")
            self.assertEqual(payload["ftsc"]["policy"], "f5")
            self.assertEqual(payload["ftsc"]["evidence"], ["position_gaussian", "dfl_distribution"])
            self.assertEqual(payload["ftsc"]["fixed_strengths"], {"dfl_distribution": 1.0})

    def test_h4_keeps_p4_intermediate_and_uses_noncontiguous_stride_contract(self):
        payload = yaml.safe_load(suite.VARIANTS["H4_P2_P3_P5"].read_text(encoding="utf-8"))
        layers = payload["backbone"] + payload["head"]
        self.assertEqual(layers[25][2], "C2f")  # P4 feature remains in graph
        self.assertEqual(layers[28][2], "C2f")  # P5 is built from the P4 path
        self.assertEqual(payload["head"][-1][0], [19, 22, 28])
        self.assertEqual(payload["ablation"]["expected_strides"], [4, 8, 32])

    def test_az_is_explicitly_blocked_and_never_uses_historical_crop(self):
        for variant in ("AZ_H1", "AZ_H2"):
            payload = yaml.safe_load(suite.VARIANTS[variant].read_text(encoding="utf-8"))
            self.assertEqual(payload["zoomdet"]["status"], "BLOCKED_EXACT_PORT")
            self.assertTrue(payload["zoomdet"]["legacy_adaptive_zoom_forbidden"])
        source = (ROOT / "models_related/ultralytics/ultralytics/nn/modules/zoomdet.py").read_text()
        self.assertIn("BLOCKED_EXACT_PORT", source)
        self.assertIn("AdaptiveZoom", source)

    def test_protocol_contract(self):
        args = suite.parse_args(["--preflight-only"])
        kwargs = suite.train_kwargs(args, Path("data.yaml"), 42, True)
        self.assertEqual((kwargs["epochs"], kwargs["patience"], kwargs["imgsz"], kwargs["batch"]), (100, 20, 512, 8))
        self.assertEqual(kwargs["workers"], 4)
        self.assertEqual(kwargs["optimizer"], "auto")
        self.assertFalse(kwargs["cos_lr"])
        self.assertEqual(kwargs["lrf"], 0.01)
        self.assertEqual(kwargs["fitness_metric"], "map50_95")
        self.assertEqual((kwargs["mosaic"], kwargs["close_mosaic"]), (0.0, 0))

    def test_source_preflight_is_read_only_metadata(self):
        report = suite.source_preflight(suite.parse_args(["--source-preflight-only"]))
        self.assertEqual(report["protocol"]["epochs"], 100)
        self.assertFalse(report["teacher_student"])
        self.assertEqual(report["configs"]["H4_P2_P3_P5"]["detect_from"], [19, 22, 28])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None and importlib.util.find_spec("cv2") is not None, "tensor runtime dependencies are not installed")
class CueTensorTests(unittest.TestCase):
    def test_color_and_edge_zero_projection_are_identity(self):
        import torch
        from ultralytics.nn.modules import P2ColorCueFusion, P2EdgeCueFusion
        torch.manual_seed(0)
        image = torch.rand(1, 3, 64, 64)
        p2 = torch.rand(1, 32, 16, 16)
        for module in (P2ColorCueFusion(32), P2EdgeCueFusion(32)):
            module.eval()
            with torch.no_grad():
                output = module(p2, image)
            self.assertTrue(torch.allclose(output, p2, atol=1e-6, rtol=0))
            self.assertTrue(torch.isfinite(output).all())

    def test_cues_are_finite_and_directional(self):
        import torch
        from ultralytics.nn.modules import oriented_edge_responses, rgb_color_cues
        image = torch.rand(2, 3, 32, 32)
        self.assertEqual(tuple(rgb_color_cues(image).shape), (2, 4, 32, 32))
        self.assertEqual(tuple(oriented_edge_responses(image).shape), (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(rgb_color_cues(image)).all())
        self.assertTrue(torch.isfinite(oriented_edge_responses(image)).all())

    def test_zoom_identity_round_trip(self):
        import torch
        from ultralytics.nn.modules.zoomdet import identity_box_transform, inverse_box_transform
        boxes = torch.tensor([[0.1, 0.2, 0.7, 0.8]])
        self.assertTrue(torch.equal(inverse_box_transform(identity_box_transform(boxes)), boxes))


if __name__ == "__main__":
    unittest.main()
