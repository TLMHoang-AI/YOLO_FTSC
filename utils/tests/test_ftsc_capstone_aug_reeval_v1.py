"""Offline contracts for the HF-first capstone augmentation evaluation runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.ftsc_capstone_aug_reeval_v1 import (
    CONTRACT_VERSION, planned_rows, verify_hf_sources, validated_output,
)
from utils.ftsc_hf_layout import augmentation_checkpoint, augmentation_run


class FakeAPI:
    def __init__(self, paths):
        self.paths = list(paths)
    def repo_info(self, **_kw):
        return SimpleNamespace(sha="a" * 40)
    def list_repo_files(self, **_kw):
        return self.paths


class FakeArchive:
    def __init__(self, paths):
        self.api = FakeAPI(paths)
    def _get_api(self):
        return self.api
    def _retry(self, fn):
        return fn()
    def _source_reference(self, dataset, case, seed):
        return {"best_sha256": "b" * 64}, "a" * 40


class ReevalContracts(unittest.TestCase):
    def test_contract_version_guards_stale_notebook_checkout(self):
        self.assertEqual(CONTRACT_VERSION, "FTSC_CAPSTONE_AUG_HF_CANONICAL_20261110_V2")

    def test_verify_readonly_requires_checkpoint_reference_and_tiny_predictions(self):
        with tempfile.TemporaryDirectory() as d:
            rows = planned_rows("TinyPerson", ("CP2",), (42,), Path(d))
            source = augmentation_checkpoint("TinyPerson", "CP2", 42)
            ref = augmentation_run("TinyPerson", "CP2", 42) + "/checkpoint_reference.json"
            merged = augmentation_run("TinyPerson", "CP2", 42) + "/evaluation/test_merged_predictions.json"
            verified = verify_hf_sources(FakeArchive((source, ref, merged)), rows)
            self.assertEqual(verified[0]["status"], "SOURCE_REFERENCE_VERIFIED")
            self.assertTrue(verified[0]["merged_predictions_available"])
            with self.assertRaisesRegex(RuntimeError, "Missing canonical source artifacts"):
                verify_hf_sources(FakeArchive((source, ref)), rows)

    def test_verify_rejects_noncanonical_levir_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as d:
            rows = planned_rows("LEVIR-Ship", ("FTSC",), (42,), Path(d))
            legacy = "levirship/H2/seed_42/weights/best.pt"
            with self.assertRaisesRegex(RuntimeError, "Missing canonical source"):
                verify_hf_sources(FakeArchive((legacy,)), rows)

    def test_completed_metric_contract_requires_three_merged_metrics_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            f = Path(temp) / "evaluation_metrics_extended.json"
            identity = {
                "evaluation/source_checkpoint": augmentation_checkpoint("TinyPerson", "CP2", 42),
                "evaluation/source_checkpoint_sha256": "c" * 64,
                "evaluation/source_case": "CP2",
                "evaluation/source_seed": 42,
                "evaluation/source_dataset": "TinyPerson",
                "evaluation/source_revision": "a" * 40,
                "evaluation/no_retraining": True,
                "evaluation/evaluator_contract": "tinybenchmark_merged_original_iou50_95_v2",
                "test_merged/AP50": .57,
                "test_merged/mAP50-95": .21,
                "test_merged/AP50-Small": .69,
                "model/parameters": 1821959,
                "model/GFLOPs": 10.587,
            }
            f.write_text(json.dumps(identity), encoding="utf-8")
            self.assertEqual(validated_output(f, "TinyPerson", "CP2", 42), identity)
            del identity["test_merged/mAP50-95"]
            f.write_text(json.dumps(identity), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Missing full six-core"):
                validated_output(f, "TinyPerson", "CP2", 42)


if __name__ == "__main__":
    unittest.main()
