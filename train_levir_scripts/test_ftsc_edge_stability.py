"""Static and focused synthetic tests for the H2+Edge stability screen."""
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
from train_levir_scripts import train_all_levir_yolov8n_p2_ftsc_edge_stability as suite


class EdgeStabilityContractTests(unittest.TestCase):
    def test_defaults_and_explicit_edge_settings(self):
        self.assertEqual(suite.DEFAULT_VARIANTS, ["ES1", "ES2", "ES3"])
        self.assertEqual(suite.parse_args([]).seeds, [42, 43, 44])
        expected = {
            "ES1": (0.25, "constant", "learned"),
            "ES2": (0.25, "ramp", "learned"),
            "ES3": (0.25, "constant", "uniform"),
        }
        for variant, values in expected.items():
            payload = yaml.safe_load(suite.VARIANTS[variant].read_text(encoding="utf-8"))
            edge = payload["edge_stability"]
            self.assertEqual((edge["residual_scale"], edge["residual_schedule"], edge["orientation_gate_mode"]), values)
            self.assertEqual(edge["hidden"], 32)
            edge_layers = [layer for layer in payload["head"] if layer[2] == "P2EdgeCueFusion"]
            self.assertEqual(len(edge_layers), 1)
            self.assertEqual(edge_layers[0][3][0], 32)  # parser prepends P2 channels
            self.assertEqual(len(edge_layers[0][3]), 6)
            self.assertEqual(payload["head"][-1][0], [19, 22])

    def test_existing_e_h2_is_unchanged_control(self):
        payload = yaml.safe_load(suite.VARIANTS["ES0"].read_text(encoding="utf-8"))
        self.assertEqual(payload["head"][-1][0], [19, 22])
        self.assertNotIn("edge_stability", payload)
        edge_layers = [layer for layer in payload["head"] if layer[2] == "P2EdgeCueFusion"]
        self.assertEqual(edge_layers, [[-1, 1, "P2EdgeCueFusion", [32]]])
        self.assertEqual(suite.EDGE_EXPECTED["ES0"], {"residual_scale": 1.0, "residual_schedule": "constant", "orientation_gate_mode": "learned"})

    def test_protocol_has_no_unrequested_factors(self):
        args = suite.parse_args(["--preflight-only"])
        kwargs = suite.train_kwargs(args, Path("data.yaml"), 42, True)
        for key, value in suite.REQUIRED_PROTOCOL.items():
            self.assertEqual(kwargs[key], value)
        self.assertNotIn("teacher", kwargs)
        self.assertNotIn("localization_distill", kwargs)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None and importlib.util.find_spec("cv2") is not None, "tensor runtime dependencies are not installed")
class EdgeStabilityTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        from ultralytics.nn.modules import P2EdgeCueFusion

        self.torch = torch
        self.P2EdgeCueFusion = P2EdgeCueFusion

    def test_default_identity_and_backward_compatible_gate(self):
        torch = self.torch
        torch.manual_seed(0)
        module = self.P2EdgeCueFusion(32)
        self.assertEqual(module.residual_scale, 1.0)
        self.assertEqual(module.residual_schedule, "constant")
        self.assertEqual(module.orientation_gate_mode, "learned")
        self.assertIsNotNone(module.gate)
        module.eval()
        p2 = torch.randn(1, 32, 16, 16)
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            output = module(p2, image)
        self.assertTrue(torch.allclose(output, p2, atol=1e-7, rtol=0))

    def test_es1_scales_nonzero_residual_by_quarter(self):
        torch = self.torch
        torch.manual_seed(0)
        module = self.P2EdgeCueFusion(32, residual_scale=0.25)
        module.eval()
        with torch.no_grad():
            module.projection.weight.fill_(0.01)
            module.projection.bias.fill_(0.02)
        p2 = torch.randn(1, 32, 16, 16)
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            output = module(p2, image)
            # Recompute the same pre-projection feature through the public helper.
            from ultralytics.nn.modules.edge_cue import oriented_edge_responses
            responses = oriented_edge_responses(image).to(dtype=p2.dtype)
            pooled = responses.abs().mean(dim=(-2, -1))
            weights = torch.softmax(module.gate(pooled), dim=-1)
            feature = module.encoder(responses * weights.unsqueeze(-1).unsqueeze(-1))
            if feature.shape[-2:] != p2.shape[-2:]:
                feature = torch.nn.functional.interpolate(feature, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            raw_residual = module.projection(feature)
        self.assertTrue(torch.allclose(output - p2, 0.25 * raw_residual, atol=1e-6, rtol=1e-5))

    def test_es2_schedule_endpoints(self):
        module = self.P2EdgeCueFusion(32, residual_scale=0.25, residual_schedule="ramp", ramp_start_epoch=5, ramp_end_epoch=15)
        expected = {0: 0.0, 4: 0.0, 5: 0.0, 10: 0.125, 15: 0.25, 25: 0.25}
        for epoch, value in expected.items():
            self.assertAlmostEqual(module.effective_residual_scale(epoch), value, places=7)

    def test_es3_uniform_gate_is_fixed_and_gate_parameters_cannot_change_output(self):
        torch = self.torch
        module = self.P2EdgeCueFusion(32, residual_scale=0.25, orientation_gate_mode="uniform")
        self.assertIsNone(module.gate)
        module.eval()
        p2 = torch.randn(1, 32, 16, 16)
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            output = module(p2, image)
        weights = module.last_stats["orientation_gate_weights"]
        self.assertTrue(torch.allclose(weights, torch.full((4,), 0.25), atol=1e-7, rtol=0))
        self.assertAlmostEqual(float(module.last_stats["orientation_entropy"]), float(torch.log(torch.tensor(4.0))), places=6)
        self.assertEqual(tuple(output.shape), tuple(p2.shape))


if __name__ == "__main__":
    unittest.main()
