"""Local structural/regression tests for the prepared Varroa causal suite."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

from train_levir_scripts import train_varroa_ftsc_edge_placement_headscale as suite


class VarroaEdgePlacementHeadscaleTests(unittest.TestCase):
    def test_default_matrix_excludes_references_and_stage2(self):
        self.assertEqual(
            suite.STAGE1_CASES,
            ("EDGE_P2_DETECT_ONLY", "EDGE_P3_DETECT_ONLY", "HEAD_P234", "HEAD_P234_EDGE_P3"),
        )
        self.assertEqual(suite.STAGE2_CASES, ("HEAD_P34", "HEAD_P34_EDGE_P3"))
        args = suite.parse_args([])
        suite.validate_args(args)
        self.assertEqual(tuple(args.cases), suite.STAGE1_CASES)
        self.assertTrue(set(args.cases).isdisjoint(suite.HISTORICAL_REFERENCES))

    def test_stage2_requires_explicit_opt_in(self):
        args = suite.parse_args(["--cases", "HEAD_P34"])
        with self.assertRaisesRegex(RuntimeError, "require --include-stage2"):
            suite.validate_args(args)
        opted_in = suite.parse_args(["--include-stage2", "--cases", "HEAD_P34", "HEAD_P34_EDGE_P3"])
        suite.validate_args(opted_in)
        self.assertEqual(tuple(opted_in.cases), suite.STAGE2_CASES)

    def test_all_yaml_graph_contracts_and_ftsc_parity(self):
        report = suite.source_preflight()
        self.assertEqual(report["default_cases"], list(suite.STAGE1_CASES))
        self.assertEqual(report["optional_stage2_cases"], list(suite.STAGE2_CASES))
        reference_ftsc = suite._load_yaml(suite.REFERENCE_CONFIG)["ftsc"]
        for name, spec in suite.CASES.items():
            payload = suite._load_yaml(spec.config)
            self.assertEqual(payload["ftsc"], reference_ftsc, name)
            self.assertEqual(report["cases"][name]["status"], "PREPARED — NOT YET RUN")
            self.assertFalse(report["cases"][name]["edge_propagates_downstream"])

    def test_ftsc_blocks_are_text_and_value_identical(self):
        def raw_ftsc(path: Path) -> str:
            text = path.read_text(encoding="utf-8")
            return text[text.index("ftsc:\n"):text.index("nc: 1\n")]

        reference_value = suite._load_yaml(suite.REFERENCE_CONFIG)["ftsc"]
        reference_text = raw_ftsc(suite.REFERENCE_CONFIG)
        for name, spec in suite.CASES.items():
            self.assertEqual(raw_ftsc(spec.config), reference_text, name)
            self.assertEqual(suite._load_yaml(spec.config)["ftsc"], reference_value, name)

    def test_detect_taps_types_and_strides(self):
        cases = suite.source_preflight()["cases"]
        expected = {
            "EDGE_P2_DETECT_ONLY": ([19, 22], ["P2EdgeCueFusion", "RepC2f"], [4.0, 8.0]),
            "EDGE_P3_DETECT_ONLY": ([18, 22], ["RepC2f", "P3EdgeCueFusion"], [4.0, 8.0]),
            "HEAD_P234": ([18, 21, 24], ["RepC2f"] * 3, [4.0, 8.0, 16.0]),
            "HEAD_P234_EDGE_P3": ([18, 22, 25], ["RepC2f", "P3EdgeCueFusion", "RepC2f"], [4.0, 8.0, 16.0]),
            "HEAD_P34": ([21, 24], ["RepC2f"] * 2, [8.0, 16.0]),
            "HEAD_P34_EDGE_P3": ([22, 25], ["P3EdgeCueFusion", "RepC2f"], [8.0, 16.0]),
        }
        for name, (indices, types, strides) in expected.items():
            self.assertEqual(cases[name]["detect_from"], indices)
            self.assertEqual(cases[name]["detect_source_types"], types)
            self.assertEqual(cases[name]["expected_detect_strides"], strides)

    def test_graph_trace_rejects_edge_contamination(self):
        for name in ("EDGE_P2_DETECT_ONLY", "EDGE_P3_DETECT_ONLY", "HEAD_P234_EDGE_P3", "HEAD_P34_EDGE_P3"):
            spec = suite.CASES[name]
            layers = suite._layers(suite._load_yaml(spec.config))
            downstream, clean_source = spec.clean_downstream_start
            self.assertEqual(suite._resolved_inputs(layers, downstream), (clean_source,))
            self.assertNotIn(spec.edge_index, suite._ancestors(layers, downstream))
            for source in spec.detect_from:
                if source != spec.edge_index:
                    self.assertNotIn(spec.edge_index, suite._ancestors(layers, source))

    def test_edge_hyperparameters_are_identical(self):
        edge_args = []
        for spec in suite.CASES.values():
            if spec.edge_index is not None:
                edge_args.append(suite._layers(suite._load_yaml(spec.config))[spec.edge_index][3])
        self.assertTrue(edge_args)
        self.assertTrue(all(args == suite.EDGE_ARGS for args in edge_args))

    def test_frozen_training_protocol(self):
        args = suite.parse_args([])
        self.assertEqual(args.seeds, [42, 43, 44])
        self.assertEqual((args.epochs, args.patience, args.imgsz, args.batch_size), (100, 20, 640, 4))
        args.seed_scope = "all"
        kwargs = suite.shared.train_kwargs(args, Path("/tmp/varroa.yaml"), 42, True)
        self.assertEqual(kwargs, {
            "data": "/tmp/varroa.yaml", "epochs": 100, "imgsz": 640, "batch": 4,
            "device": "0", "workers": 4, "patience": 20, "amp": True,
            "plots": False, "seed": 42, "deterministic": True,
        })

    @unittest.skipUnless(importlib.util.find_spec("torch"), "local Python lacks torch")
    def test_generic_edge_api_and_p2_diagnostic_compatibility(self):
        import torch

        suite.shared.local_ultralytics()
        from ultralytics.nn.modules import EdgeCueFusion, P2EdgeCueFusion, P3EdgeCueFusion

        self.assertTrue(issubclass(P2EdgeCueFusion, EdgeCueFusion))
        self.assertTrue(issubclass(P3EdgeCueFusion, EdgeCueFusion))
        p2 = P2EdgeCueFusion(16, 32, 0.25, "constant", 5, 15, "learned")
        p3 = P3EdgeCueFusion(24, 32, 0.25, "constant", 5, 15, "learned")
        image = torch.rand(1, 3, 64, 64)
        p2(torch.rand(1, 16, 16, 16), image)
        p3(torch.rand(1, 24, 8, 8), image)
        p2_metrics = p2.diagnostic_metrics()
        p3_metrics = p3.diagnostic_metrics()
        self.assertEqual(p2_metrics["edge_target_stride"], 4.0)
        self.assertEqual(p3_metrics["edge_target_stride"], 8.0)
        self.assertIn("edge_p2_norm", p2_metrics)
        self.assertIn("edge_residual_p2_norm_ratio", p2_metrics)
        self.assertNotIn("edge_p2_norm", p3_metrics)
        self.assertTrue(torch.equal(p2.projection.weight, torch.zeros_like(p2.projection.weight)))
        # Historical checkpoints retain the same encoder/gate/projection key layout.
        clone = P2EdgeCueFusion(16)
        clone.load_state_dict(copy.deepcopy(p2.state_dict()))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"),
        "local Python lacks torch/cv2",
    )
    def test_all_models_instantiate_on_cpu(self):
        report = suite.model_preflight()
        for name, spec in suite.CASES.items():
            expected = [float(2 ** int(level[1:])) for level in spec.pyramid_levels]
            self.assertEqual(report[name]["detect_strides"], expected)
            self.assertEqual(report[name]["detect_from"], list(spec.detect_from))
            self.assertEqual(report[name]["edge_count"], int(spec.edge_index is not None))


if __name__ == "__main__":
    unittest.main()
