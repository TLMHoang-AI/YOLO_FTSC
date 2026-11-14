"""CPU-only regression tests for LEVIR small-AP root-cause diagnostics.

No HF/network, PyTorch, GPU or inference; fake _evaluate_coco isolates
confidence filtering and provenance safeguards.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from utils import ftsc_deployment_backend_eval_v1 as deploy
from utils import ftsc_levir_small_ap_rootcause_v1 as small


def fake_eval(gt_file, pred_file):
    gt = json.loads(Path(gt_file).read_text())
    preds = json.loads(Path(pred_file).read_text())
    assert len(gt["images"]) == 788
    value = len(preds) / 100.0
    return {
        "test_size/AP50-Small": value,
        "test_size/AP50": value,
        "test_size/AP-Small": value,
    }


class ReplayTests(unittest.TestCase):
    def test_threshold_sweep_changes_only_prediction_count(self):
        gt = {"images":[{"id":i,"file_name":f"/data/{i}.jpg"} for i in range(1,789)],
              "annotations":[{"id":1,"image_id":1,"bbox":[1,1,22,22],
                              "area":484,"iscrowd":0,"category_id":1}]}
        preds = [
            {"image_id":1,"category_id":1,"bbox":[1,1,22,22],"score":0.001},
            {"image_id":1,"category_id":1,"bbox":[1,1,22,22],"score":0.15},
            {"image_id":1,"category_id":1,"bbox":[1,1,22,22],"score":0.3},
        ]
        result=small._replay(gt,preds,(0.001,0.25),fake_eval,0.03)
        self.assertEqual(len(result),2)
        self.assertEqual([r["retained_predictions"] for r in result],[3,1])
        self.assertEqual([r["AP50_Small_pct"] for r in result],[3.0,1.0])
        with self.assertRaisesRegex(RuntimeError,"REPLAY FAILED"):
            small._replay(gt,preds,(0.001,0.25),fake_eval,0.3)

    def test_replay_provenance_rejects_corrupt_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            folder=deploy.path_for(root,"LEVIR-Ship","FTSC",42,"pt32")
            folder.mkdir(parents=True)
            gt={"images":[{"id":i,"file_name":f"/data/{i}.jpg"} for i in range(1,789)],
                "annotations":[{"id":1,"image_id":1,"bbox":[1,1,22,22],
                                "area":484,"iscrowd":0,"category_id":1}]}
            preds=[{"image_id":1,"category_id":1,"bbox":[1,1,22,22],"score":0.2}]
            gt_path=folder/"test_size_ground_truth.json"
            pr_path=folder/"test_size_predictions.json"
            gt_path.write_text(json.dumps(gt),encoding="utf-8")
            pr_path.write_text(json.dumps(preds),encoding="utf-8")
            expected={"sha256":"a"*64}
            metrics={
                "evaluation_protocol":deploy.RECT_PARITY_PROTOCOL,
                "runtime_revision":deploy.RECT_PARITY_RUNTIME_REVISION,
                "backend":"pt32","case":"FTSC","seed":42,
                "source_sha256":expected["sha256"],
                "gt_sha256":small.sha256(gt_path),
                "predictions_sha256":small.sha256(pr_path),
                "metrics":{"test_size/AP50-Small":0.02},
            }
            (folder/"metrics.json").write_text(json.dumps(metrics),encoding="utf-8")
            record,recovered_gt,recovered_pred=small.read_metric_records(
                root,"FTSC",42,expected)
            self.assertEqual(record["backend"],"pt32")
            self.assertEqual(len(recovered_gt["images"]),788)
            self.assertEqual(len(recovered_pred),1)
            pr_path.write_text("[]",encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError,"were modified"):
                small.read_metric_records(root,"FTSC",42,expected)

    def test_legacy_conf_and_rootcause_controls_are_explicit(self):
        self.assertEqual(small.LEGACY_CONF,0.25)
        self.assertIn(0.001,small.DEFAULT_THRESHOLDS)
        self.assertIn(0.25,small.DEFAULT_THRESHOLDS)
        self.assertIn("FTSC_OFF",small.DEFAULT_CASES)
        self.assertIn("FTSC_YOLOV9T",small.DEFAULT_CASES)
        self.assertIn("FTSC_YOLO11N",small.DEFAULT_CASES)


if __name__ == "__main__":
    unittest.main()
