"""Local structural tests for the not-yet-run Varroa FTSC suite."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from train_levir_scripts import train_varroa_ftsc_causal_suite as suite


class VarroaFTSCCausalSuiteTests(unittest.TestCase):
    def test_matrix_and_frozen_protocol(self):
        self.assertEqual(tuple(suite.VARIANTS), ("V0_NOFTSC", "V1_FTSC", "V2_EDGE025", "V3_COLOR"))
        args = suite.parse_args([])
        self.assertEqual(args.seeds, [42, 43, 44])
        self.assertEqual((args.epochs, args.patience, args.imgsz, args.batch_size), (100, 20, 640, 4))
        suite.validate_args(args)

    def test_source_graph_contracts(self):
        variants = suite.source_preflight()["variants"]
        self.assertEqual(variants["V0_NOFTSC"]["detect_from"], [18, 21])
        self.assertEqual(variants["V1_FTSC"]["detect_from"], [18, 21])
        self.assertEqual(variants["V2_EDGE025"]["detect_from"], [19, 22])
        self.assertEqual(variants["V3_COLOR"]["detect_from"], [19, 22])
        self.assertEqual(variants["V2_EDGE025"]["p2_source_type"], "P2EdgeCueFusion")
        self.assertEqual(variants["V3_COLOR"]["p2_source_type"], "P2ColorCueFusion")
        self.assertEqual(variants["V2_EDGE025"]["p3_source_type"], "RepC2f")
        self.assertEqual(variants["V3_COLOR"]["p3_source_type"], "RepC2f")

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"),
        "local Python lacks torch/cv2",
    )
    def test_runtime_models_resolve_to_p2_p3_on_cpu(self):
        report = suite.model_preflight()
        for variant in suite.VARIANTS:
            self.assertEqual(report[variant]["detect_strides"], [4.0, 8.0])
            self.assertEqual(len(report[variant]["detect_from"]), 2)
        self.assertFalse(report["V0_NOFTSC"]["ftsc_enabled"])
        self.assertTrue(report["V1_FTSC"]["ftsc_enabled"])
        self.assertEqual(report["V2_EDGE025"]["detect_source_types"], ["P2EdgeCueFusion", "RepC2f"])
        self.assertEqual(report["V3_COLOR"]["detect_source_types"], ["P2ColorCueFusion", "RepC2f"])

    def test_all_variants_use_identical_established_training_kwargs(self):
        args = suite.parse_args([])
        args.seed_scope = "all"
        data_yaml = Path("/tmp/varroa.yaml")
        kwargs = [suite.shared.train_kwargs(args, data_yaml, 42, True) for _ in suite.VARIANTS]
        self.assertTrue(all(item == kwargs[0] for item in kwargs[1:]))
        self.assertEqual(kwargs[0], {
            "data": str(data_yaml),
            "epochs": 100,
            "imgsz": 640,
            "batch": 4,
            "device": "0",
            "workers": 4,
            "patience": 20,
            "amp": True,
            "plots": False,
            "seed": 42,
            "deterministic": True,
        })

    def test_invalid_protocol_override_fails(self):
        args = suite.parse_args(["--epochs", "99"])
        with self.assertRaisesRegex(RuntimeError, "Frozen Varroa protocol changed"):
            suite.validate_args(args)


if __name__ == "__main__":
    unittest.main()
