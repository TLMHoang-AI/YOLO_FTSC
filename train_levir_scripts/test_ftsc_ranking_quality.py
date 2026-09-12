"""Focused pure-Python tests for the FTSC ranking audit helpers."""
from __future__ import annotations

import unittest

import numpy as np

from train_levir_scripts.ftsc_ranking_quality import (
    FTSCRankingAccumulator,
    effective_number,
    rankdata_average,
    spearman_tie_aware,
)


class FTSCRankingQualityTests(unittest.TestCase):
    def test_tie_aware_ranks_and_spearman(self):
        np.testing.assert_allclose(rankdata_average([1, 1, 3]), [1.5, 1.5, 3.0])
        self.assertAlmostEqual(spearman_tie_aware([1, 1, 3], [2, 2, 4]), 1.0)
        self.assertIsNone(spearman_tie_aware([1], [1]))
        self.assertIsNone(spearman_tie_aware([1, 1], [2, 3]))

    def test_effective_number_limits(self):
        self.assertAlmostEqual(effective_number([1, 1, 1]), 3.0)
        self.assertAlmostEqual(effective_number([100, 1, 1]), 1.0401919616)

    def test_grouping_and_singletons_are_not_zero_correlations(self):
        snapshot = {
            "image_index": np.array([0, 0, 0, 0]),
            "gt_id": np.array([0, 0, 1, 1]),
            "positive_index": np.array([1, 2, 3, 4]),
            "stride": np.array([4, 4, 8, 8]),
            "gt_positive_count": np.array([2, 2, 2, 2]),
            "tal_target_score": np.array([0.2, 0.8, 0.1, 0.2]),
            "w_cls": np.array([0.2, 0.8, 0.1, 0.2]),
            "w_box": np.ones(4),
            "w_dfl": np.ones(4),
            "w_cls_clipped_low": np.zeros(4),
            "w_cls_clipped_high": np.zeros(4),
            "w_box_clipped_low": np.zeros(4),
            "w_box_clipped_high": np.zeros(4),
            "w_dfl_clipped_low": np.zeros(4),
            "w_dfl_clipped_high": np.zeros(4),
            "position_log_evidence_centered": np.array([-1.0, 1.0, 0.0, 0.0]),
            "dfl_log_evidence_centered": np.array([-1.0, 1.0, 0.0, 0.0]),
            "audit_decoded_iou": np.array([0.1, 0.9, 0.4, 0.5]),
            "dfl_entropy_mean": np.full(4, 0.2),
        }
        accumulator = FTSCRankingAccumulator()
        accumulator.update(snapshot)
        rows = accumulator.summary_rows(variant="Q0", seed=42, split="val")
        position = next(row for row in rows if row["metric"] == "position_vs_iou")
        self.assertEqual(position["valid_gt_count"], 1)
        self.assertEqual(position["undefined_gt_count"], 1)
        self.assertEqual(accumulator._states["all"].groups, 2)
        self.assertEqual(accumulator._states["P2"].groups, 1)
        self.assertEqual(accumulator._states["P3"].groups, 1)


if __name__ == "__main__":
    unittest.main()
