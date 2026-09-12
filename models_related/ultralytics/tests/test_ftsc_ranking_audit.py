"""Optional tensor tests for the observation-only FTSC audit path."""
from __future__ import annotations

import importlib.util
import unittest


_RUNTIME = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(_RUNTIME, "tensor runtime dependencies are not installed")
class FTSCRankingAuditTests(unittest.TestCase):
    def setUp(self):
        import torch
        from ultralytics.nn.modules import AnchorFreeFTSCCalibrator

        self.torch = torch
        self.Calibrator = AnchorFreeFTSCCalibrator

    def _inputs(self):
        torch = self.torch
        anchor = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]])
        boxes = torch.tensor(
            [[[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0]],
             [[0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0], [0.0, 0.0, 8.0, 8.0]]])
        gt_idx = torch.zeros((2, 4), dtype=torch.long)
        fg = torch.tensor([[True, True, False, False], [True, True, True, False]])
        scores = torch.zeros((2, 4, 1))
        scores[..., 0] = torch.tensor([[0.2, 0.8, 0.0, 0.0], [0.1, 0.3, 0.5, 0.0]])
        distri = torch.randn(2, 4, 16, requires_grad=True)
        stride = torch.tensor([4.0, 4.0, 8.0, 8.0])
        anchor_index = torch.arange(4).view(1, -1).expand(2, -1)
        iou = torch.tensor([0.2, 0.8, 0.3, 0.6, 0.4])
        return anchor, boxes, gt_idx, fg, scores, distri, stride, anchor_index, iou

    def test_audit_is_detached_and_assignment_preserving(self):
        config = {
            "policy": "f5",
            "evidence": ["position_gaussian", "dfl_distribution"],
            "fixed_strengths": {"dfl_distribution": 1.0},
            "audit_enabled": True,
        }
        args = self._inputs()
        audited = self.Calibrator(config, reg_max=4)
        baseline = self.Calibrator({**config, "audit_enabled": False}, reg_max=4)
        out_a = audited(args[0], args[1], args[2], args[3], args[5], epoch=0,
                        localization_quality=args[8], audit_target_scores=args[4],
                        audit_stride=args[6], audit_anchor_index=args[7])
        out_b = baseline(args[0], args[1], args[2], args[3], args[5], epoch=0,
                         localization_quality=args[8])
        for task in ("cls", "box", "dfl"):
            self.assertTrue(self.torch.equal(out_a[task], out_b[task]))
        self.assertTrue(args[3].equal(self.torch.tensor([[True, True, False, False], [True, True, True, False]])))
        self.assertTrue(all(not value.requires_grad for value in audited.last_audit.values()))
        self.torch.testing.assert_close(audited.last_audit["audit_decoded_iou"], args[8])
        self.assertEqual(audited.last_audit["image_index"].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(audited.last_audit["stride"].tolist(), [4.0, 4.0, 4.0, 4.0, 8.0])


if __name__ == "__main__":
    unittest.main()
