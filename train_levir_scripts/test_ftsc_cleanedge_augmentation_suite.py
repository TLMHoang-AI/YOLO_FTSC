"""No-training contract tests for the three LEVIR Clean Edge augmentation cases."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_levir_scripts import train_levir_ftsc_cleanedge_augmentation_suite as runner


class CleanEdgeAugmentationSuiteTest(unittest.TestCase):
    def test_exactly_three_cases_share_one_canonical_clean_edge_model(self):
        args = runner.parse_args([])
        report = runner.source_preflight(args)
        self.assertEqual(runner.CASES, ("CE_A1_OACP_R2", "CE_A2_M5", "CE_A3_CP2"))
        self.assertEqual(args.seeds, [42, 43, 44])
        self.assertEqual(set(report["cases"]), set(runner.CASES))
        self.assertEqual(
            {item["model_config"] for item in report["cases"].values()},
            {str(runner.MODEL_CONFIG)},
        )
        self.assertTrue(runner.MODEL_CONFIG.is_file())
        self.assertEqual(report["status"], "PREPARED — NOT YET RUN")

    def test_clean_edge_graph_is_post_fusion_p2_and_post_c2f_p3(self):
        report = runner.source_preflight(runner.parse_args([]))
        for case in runner.CASES:
            contract = report["cases"][case]
            self.assertEqual(contract["detect_from"], [20, 23])
            self.assertEqual(contract["detect_source_layers"], {"P2": 20, "P3": 23})
            self.assertEqual(contract["detect_source_types"], {"P2": "P2EdgeCueFusion", "P3": "C2f"})
            self.assertEqual(
                contract["edge"],
                {
                    "layer": 20,
                    "residual_scale": 0.25,
                    "residual_schedule": "constant",
                    "orientation_gate_mode": "learned",
                    "hidden": 32,
                },
            )

    def test_ftsc_is_identical_to_h2_for_all_clean_edge_cases(self):
        report = runner.source_preflight(runner.parse_args([]))
        ftsc = [report["cases"][case]["ftsc"] for case in runner.CASES]
        self.assertEqual(ftsc[0], ftsc[1])
        self.assertEqual(ftsc[1], ftsc[2])
        self.assertEqual(ftsc[0]["policy"], "f5")
        self.assertEqual(ftsc[0]["evidence"], ["position_gaussian", "dfl_distribution"])

    def test_only_the_selected_augmentation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "hard_negative_bank.json"
            bank.write_text("[]\n", encoding="utf-8")
            settings = {case: runner.augmentation_for(case, bank) for case in runner.CASES}

            a1 = settings["CE_A1_OACP_R2"]
            self.assertEqual((a1["mosaic"], a1["close_mosaic"], a1["copy_paste_enabled"]), (0.0, 0, False))
            self.assertEqual(
                runner.environment_for("CE_A1_OACP_R2"),
                {
                    "YOLO_CONTEXT_AUG": "oacp",
                    "OACP_PROFILE": "r2",
                    "OACP_VARIANT": "current",
                    "OACP_PLACEMENT": "pre_transform",
                    "YOLO_LEGACY_DOUBLE_OACP": "0",
                },
            )

            a2 = settings["CE_A2_M5"]
            self.assertEqual((a2["mosaic"], a2["close_mosaic"]), (1.0, 10))
            self.assertEqual(
                (a2["mosaic_policy"], a2["hard_negative_tile"], a2["hardneg_mosaic_prob"]),
                ("hard_negative", True, 0.30),
            )
            self.assertFalse(a2["copy_paste_enabled"])
            self.assertEqual(runner.environment_for("CE_A2_M5"), {})

            a3 = settings["CE_A3_CP2"]
            self.assertEqual(a3["copy_paste"], 0.0)  # stock segmentation CopyPaste is off
            self.assertEqual(
                (a3["copy_paste_enabled"], a3["copy_paste_unit"], a3["copy_paste_copies"]),
                (True, "single", 2),
            )
            self.assertEqual(
                (a3["copy_paste_p"], a3["copy_paste_blend"], a3["copy_paste_placement"]),
                (0.5, "hard", "random"),
            )
            self.assertEqual(
                (a3["mosaic"], a3["close_mosaic"], a3["mixup"], a3["cutmix"]),
                (0.0, 0, 0.0, 0.0),
            )

    def test_m5_bank_is_required_only_when_future_training_is_authorized(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = runner.parse_args(
                ["--cases", "CE_A2_M5", "--confirm-run", "--hard-negative-bank", str(Path(directory) / "missing.json")]
            )
            with self.assertRaisesRegex(RuntimeError, "hard-negative bank does not exist"):
                runner.validate_execution_args(missing)

            bank = Path(directory) / "bank.json"
            bank.write_text("[]\n", encoding="utf-8")
            present = runner.parse_args(
                ["--cases", "CE_A2_M5", "--confirm-run", "--hard-negative-bank", str(bank)]
            )
            runner.validate_execution_args(present)

    def test_cp2_uses_the_detection_builder_not_stock_segmentation_copypaste(self):
        runner.workflow.local_ultralytics()
        try:
            from ultralytics.cfg import get_cfg
            from ultralytics.data.augment import Compose, CopyPaste, v8_transforms
        except ModuleNotFoundError as error:
            self.skipTest(f"optional runtime dependency unavailable: {error.name}")

        class Dataset:
            cache = None
            buffer = []
            data = {}
            use_keypoints = False

            def __len__(self):
                return 1

        transforms = v8_transforms(Dataset(), 32, get_cfg(overrides=runner.augmentation_for("CE_A3_CP2")))
        self.assertIsInstance(transforms.transforms[0], Compose)
        self.assertEqual(type(transforms.transforms[1]).__name__, "SmallObjectCopyPaste")
        self.assertFalse(any(isinstance(transform, CopyPaste) for transform in transforms.transforms[0].transforms))

    def test_all_three_runtime_models_instantiate_with_the_same_contract(self):
        try:
            report = runner.model_preflight(runner.parse_args([]))
        except ModuleNotFoundError as error:
            self.skipTest(f"optional runtime dependency unavailable: {error.name}")
        self.assertEqual(set(report), set(runner.CASES))
        for case in runner.CASES:
            self.assertEqual(report[case]["strides"], [4.0, 8.0])
            self.assertEqual(report[case]["detect_source_types"], ["P2EdgeCueFusion", "C2f"])
            self.assertEqual(report[case]["ftsc_policy"], "f5")

    def test_default_entrypoint_never_authorizes_training(self):
        with patch.object(runner, "model_preflight", side_effect=lambda args: {case: {} for case in args.cases}), patch.object(
            runner, "train_one"
        ) as train_one:
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main([])
        train_one.assert_not_called()


if __name__ == "__main__":
    unittest.main()
