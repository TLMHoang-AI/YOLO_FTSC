"""Stdlib-only regression checks for LEVIR FTSC x detection-head factorial."""
from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path

from train_levir_scripts import train_levir_ftsc_head_factorial_v1 as factor


class HeadFactorialTests(unittest.TestCase):

    def test_config_pairs_and_stock_graph(self):
        report = factor.source_preflight()
        self.assertEqual(report["status"], "STATIC_CONFIG_VERIFIED")
        self.assertEqual(set(report["factorial"]), set(factor.CASES))
        for head in ("P345", "P23"):
            a = factor.load(factor.CASES[head + "_NOFTSC"])
            b = factor.load(factor.CASES[head + "_FTSC"])
            self.assertEqual(factor.graph(a), factor.graph(b))
            self.assertFalse(a["ftsc"]["enabled"])
            self.assertTrue(b["ftsc"]["enabled"])
            self.assertEqual(a["head"][-1][0], factor.DETECT[head])

    def test_training_protocol_no_mosaic(self):
        args = argparse.Namespace(epochs=100, imgsz=512, batch_size=8,
                                  patience=20, device="cpu", workers=0)
        values = factor.training(args, Path("fake_dataset.yaml"), 42)
        self.assertEqual(values["optimizer"], "auto")
        self.assertEqual(values["fitness_metric"], "map50_95")
        self.assertEqual(values["mosaic"], 0.0)
        self.assertEqual(values["close_mosaic"], 0)
        self.assertEqual(values["seed"], 42)
        self.assertTrue(values["deterministic"])

    def test_incomplete_sample_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            factor.mean_std([0.4, 0.5])
        self.assertAlmostEqual(factor.mean_std([0.2, 0.4, 0.6])["sample_std"], 0.2)

    def test_orphan_and_contract_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            with self.assertRaisesRegex(RuntimeError, "no contract"):
                factor.current_state(run, {"schema": "x"})
            (run / "factorial_contract.json").write_text(
                '{"schema":"other"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                factor.current_state(run, {"schema": "x"})

    def test_interaction_formula_seed_pairing(self):
        # This synthetic fixture tests arithmetic only; never publish these numbers.
        scores = {}
        values = {
            "P345_NOFTSC": .50,
            "P345_FTSC": .60,
            "P23_NOFTSC": .55,
            "P23_FTSC": .70,
        }
        for case, base in values.items():
            scores[case] = {
                seed: {metric: base + i * .001 for metric in factor.METRICS}
                for i, seed in enumerate(factor.SEEDS)
            }
        effects = factor.factorial_effects(scores)
        self.assertEqual(len(effects), 15)
        ap = {row["effect"]: row for row in effects if row["metric"] == "AP50"}
        self.assertAlmostEqual(ap["ftsc_effect_p345"]["mean"], .1)
        self.assertAlmostEqual(ap["ftsc_effect_p23"]["mean"], .15)
        self.assertAlmostEqual(ap["interaction_p23_vs_p345"]["mean"], .05)
        self.assertAlmostEqual(ap["head_effect_noftsc"]["mean"], .05)
        self.assertAlmostEqual(ap["head_effect_ftsc"]["mean"], .1)
        with self.assertRaisesRegex(RuntimeError, "Incomplete paired seeds"):
            missing = {case: dict(seeds) for case, seeds in scores.items()}
            missing["P345_FTSC"].pop(44)
            factor.factorial_effects(missing)

    def test_default_has_no_training(self):
        args = factor.parse_args([])
        self.assertFalse(args.confirm_run)
        self.assertFalse(args.aggregate)
        self.assertFalse(args.model_preflight)
        self.assertEqual(set(args.cases), set(factor.CASES))
        self.assertEqual(tuple(args.seeds), factor.SEEDS)

    @unittest.skipUnless(os.getenv("RUN_FTSC_MODEL_TEST") == "1",
                         "Opt-in real model graph smoke test")
    def test_real_detection_heads(self):
        report = factor.model_preflight()
        self.assertEqual(report["P345_FTSC"]["stride"], [8.0, 16.0, 32.0])
        self.assertEqual(report["P23_FTSC"]["stride"], [4.0, 8.0])
        for case in factor.CASES:
            self.assertEqual(report[case]["ftsc"], factor.FACTORS[case][1])


if __name__ == "__main__":
    unittest.main()
