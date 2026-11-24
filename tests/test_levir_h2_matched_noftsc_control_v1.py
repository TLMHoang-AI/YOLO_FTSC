"""Offline safeguards for LEVIR control-only reuse runner."""
from __future__ import annotations
import importlib
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

from train_levir_scripts import train_levir_h2_matched_noftsc_control_v1 as runner


class TestMatchedControlSafety(unittest.TestCase):
    def test_only_one_training_arm(self):
        self.assertEqual(runner.CASE, "P23_NOFTSC")
        self.assertEqual(runner.SEEDS, (42, 43, 44))
        self.assertNotEqual(runner.OUT.resolve(), runner.fac.RUNS.resolve())

    def test_default_is_preflight(self):
        args = runner.parse_args([])
        self.assertFalse(args.confirm_run)
        self.assertFalse(args.aggregate)
        self.assertFalse(args.resume)
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.cases, ["P23_NOFTSC"])

    def test_other_arm_is_blocked_in_preflight(self):
        args = runner.parse_args(["--cases", "P23_FTSC"])
        with self.assertRaisesRegex(RuntimeError, "Only P23_NOFTSC"):
            runner.preflight(args)

    def test_no_training_without_explicit_confirm(self):
        source = inspect.getsource(runner.main)
        self.assertIn("elif args.confirm_run:", source)
        self.assertIn("fac.train_and_evaluate(args, CASE", source)
        self.assertIn("else:", source)

    def test_different_generation_remains_explicit(self):
        source = inspect.getsource(runner.aggregate)
        self.assertIn("NOT 2x2 FACTORIAL", source)


if __name__ == "__main__":
    unittest.main()
