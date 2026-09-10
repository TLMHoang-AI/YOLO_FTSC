"""Static and synthetic checks for the seed-42 H2 support screen.

These tests never start training. Tensor checks are skipped when the local
environment does not provide the optional torch/cv2 runtime.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models_related/ultralytics"))
from train_levir_scripts import train_levir_ftsc_h2_support_screen as screen


class H2SupportContractTests(unittest.TestCase):
    def test_exact_cases_and_matched_h2_source(self):
        self.assertEqual(list(screen.VARIANTS), ["S0_H2", "S1_AZ_H2"])
        payloads = {name: yaml.safe_load(path.read_text(encoding="utf-8")) for name, path in screen.VARIANTS.items()}
        self.assertEqual(payloads["S0_H2"]["head"][-1][0], [19, 22])
        self.assertEqual(payloads["S1_AZ_H2"]["head"][-1][0], [19, 22])
        self.assertEqual(payloads["S0_H2"]["backbone"], payloads["S1_AZ_H2"]["backbone"])
        self.assertEqual(payloads["S0_H2"]["head"], payloads["S1_AZ_H2"]["head"])
        self.assertEqual(payloads["S0_H2"]["ftsc"], payloads["S1_AZ_H2"]["ftsc"])
        self.assertEqual(
            {key: value for key, value in payloads["S0_H2"].items() if key != "support_screen"},
            {key: value for key, value in payloads["S1_AZ_H2"].items() if key != "support_screen"},
        )
        self.assertFalse(payloads["S0_H2"]["support_screen"]["adaptive_zoom"])
        self.assertTrue(payloads["S1_AZ_H2"]["support_screen"]["adaptive_zoom"])

    def test_protocol_and_variant_only_zoom_differs(self):
        args = screen.parse_args(["--preflight-only"])
        for variant, expected in (("S0_H2", False), ("S1_AZ_H2", True)):
            args._active_variant = variant
            kwargs = screen.train_kwargs(args, Path("data.yaml"), 42, True)
            self.assertEqual((kwargs["epochs"], kwargs["patience"], kwargs["imgsz"], kwargs["batch"]), (100, 20, 512, 8))
            self.assertEqual((kwargs["mosaic"], kwargs["close_mosaic"]), (0.0, 0))
            self.assertEqual(kwargs["optimizer"], "auto")
            self.assertFalse(kwargs["cos_lr"])
            self.assertEqual(kwargs["lrf"], 0.01)
            self.assertEqual(kwargs["adaptive_zoom"], expected)
        report = screen.source_preflight(args)
        self.assertEqual(report["seed"], 42)
        self.assertEqual(report["same_h2_topology"], True)
        self.assertEqual(report["adaptive_zoom"], {"S0_H2": False, "S1_AZ_H2": True})

    def test_aggregation_uses_actual_late_window_when_run_stops_early(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for variant in screen.VARIANTS:
                run = project / variant / "seed_42"
                run.mkdir(parents=True)
                (run / "results.csv").write_text(
                    "epoch,tal_gt_count,tal_single_positive_gt_fraction,ftsc_activation_rate\n"
                    "1,10,0.5,0.3\n10,10,0.4,0.4\n\n",
                    encoding="utf-8",
                )
            args = screen.parse_args(["--aggregate-only", "--project", str(project)])
            result = screen.aggregate_outputs(args)
            for record in result["summary"]:
                self.assertEqual((record["late_epoch_start"], record["late_epoch_end"]), (0, 9))
            self.assertTrue((project / "support_screen_comparison.csv").is_file())


@unittest.skipUnless(importlib.util.find_spec("torch") is not None and importlib.util.find_spec("cv2") is not None, "tensor runtime dependencies are not installed")
class SupportTensorTests(unittest.TestCase):
    def test_post_tal_histogram_and_h2_level_support(self):
        import torch
        from ultralytics.utils.loss import v8DetectionLoss

        mask_gt = torch.ones(2, 6, 1, dtype=torch.bool)
        mask_gt[1] = False
        strides = torch.tensor([4.0, 8.0])
        flat_stride = torch.tensor([4.0] * 8 + [8.0] * 7)
        fg_mask = torch.zeros(2, 15, dtype=torch.bool)
        target_gt_idx = torch.zeros(2, 15, dtype=torch.long)
        assignments = {
            0: [0],
            1: [1, 8],
            2: [2, 3, 4],
            3: [9, 10, 11, 12],
            4: [5, 6, 7, 13, 14],
        }
        for gt, anchors in assignments.items():
            fg_mask[0, anchors] = True
            target_gt_idx[0, anchors] = gt
        metrics = v8DetectionLoss._post_tal_support_metrics(mask_gt, fg_mask, target_gt_idx, flat_stride, strides)
        self.assertEqual(metrics["_tal_gt_count"], 6.0)
        self.assertEqual(metrics["_tal_positive_count"], 15.0)
        self.assertEqual([metrics[f"_tal_support_bin_{i}"] for i in range(6)], [1.0] * 6)
        self.assertEqual(metrics["_tal_gt_support_p2_only_count"], 2.0)
        self.assertEqual(metrics["_tal_gt_support_p3_only_count"], 1.0)
        self.assertEqual(metrics["_tal_gt_support_p2_p3_count"], 2.0)
        self.assertEqual(metrics["_tal_gt_support_ge2_count"], 4.0)
        self.assertEqual(metrics["_tal_positive_count_p2"], 8.0)
        self.assertEqual(metrics["_tal_positive_count_p3"], 7.0)

    def test_epoch_rates_use_global_counts_not_batch_fraction_average(self):
        from ultralytics.nn.tasks import BaseModel

        model = BaseModel.__new__(BaseModel)
        model._mechanism_epoch_sums = {
            "_batch_count": 2.0,
            "_tal_gt_count": 10.0,
            "_tal_positive_count": 17.0,
            "_tal_support_bin_0": 1.0,
            "_tal_support_bin_1": 1.0,
            "_tal_support_bin_2": 8.0,
            "_tal_support_ge2_count": 8.0,
            "_tal_positive_count_p2": 9.0,
            "_tal_positive_count_p3": 8.0,
        }
        metrics = BaseModel.mechanism_epoch_metrics(model)
        self.assertAlmostEqual(metrics["tal_single_positive_gt_fraction"], 0.1)
        self.assertAlmostEqual(metrics["ftsc_activation_rate"], 0.8)
        self.assertAlmostEqual(metrics["tal_mean_positives_per_gt"], 1.7)
        self.assertAlmostEqual(metrics["tal_median_positives_per_gt"], 2.0)
        self.assertAlmostEqual(metrics["tal_positive_share_p2"], 9.0 / 17.0)

    def test_single_positive_ftsc_group_is_identity(self):
        import torch
        from ultralytics.nn.modules import AnchorFreeFTSCCalibrator

        calibrator = AnchorFreeFTSCCalibrator(
            {"policy": "f5", "evidence": ["position_gaussian"], "per_gt_norm": True}, reg_max=16
        )
        anchor_points = torch.tensor([[5.0, 5.0], [6.0, 5.0], [25.0, 25.0]])
        boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]])
        groups = torch.tensor([[0, 0, 1]])
        fg_mask = torch.ones(1, 3, dtype=torch.bool)
        output = calibrator(anchor_points, boxes, groups, fg_mask, torch.zeros(1, 3, 64), epoch=20)
        self.assertAlmostEqual(float(output["cls"][2]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
