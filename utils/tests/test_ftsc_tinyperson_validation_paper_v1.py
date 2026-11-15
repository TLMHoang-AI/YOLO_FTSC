"""Offline guards for paper TinyPerson validation (no GPU/network needed)."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from utils import ftsc_tinyperson_validation_paper_v1 as tiny


class PaperTinyValidationTests(unittest.TestCase):
    def test_locked_paper_manifest_and_33_jobs(self):
        repo, cases = tiny.manifest_cases()
        self.assertEqual(repo, "TLMHoang/FTSC_results")
        self.assertEqual(len(cases), 11)
        self.assertEqual(set(cases), set(tiny.CASES))
        self.assertEqual(len(tiny.main(["--phase", "plan"])["jobs"]), 33)
        self.assertEqual(cases["FTSC_OFF"]["source_run"],
                         "TinyPerson/original_runs/H2_NOFTSC_NM")
        self.assertEqual(cases["FTSC"]["source_run"],
                         "TinyPerson/original_runs/H2_MOSAIC")

    def test_selection_rejects_unknown_and_duplicate(self):
        with self.assertRaises(RuntimeError):
            tiny.choose(["UNKNOWN"], tiny.CASES, "case")
        with self.assertRaises(RuntimeError):
            tiny.choose([42, 42], tiny.SEEDS, "seed")

    def test_never_use_single_frozen_split(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaises(RuntimeError):
                tiny.data_yaml_for(str(base / "tinyperson.yaml"), 42)
            with self.assertRaises(RuntimeError):
                tiny.data_yaml_for(str(base / "tinyperson_seed_42_corner_sw640_sh512/tinyperson.yaml"), 42)

    def test_split_is_seed_specific_and_original_disjoint(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "tinyperson_seed_42_corner_sw640_sh512"
            for folder in ("images/val", "images/train", "images/test",
                           "labels/val", "labels/train"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            (root / "images/val/img_2.jpg").write_bytes(b"stub")
            (root / "images/train/img_3.jpg").write_bytes(b"stub")
            (root / "labels/val/img_2.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (root / "labels/train/img_3.txt").write_text("")
            (root / "tinyperson.yaml").write_text(
                f"path: {root}\ntrain: images/train\nval: images/val\n"
                "test: images/test\nnames:\n  0: person\n")
            manifest = {
                "val": [{"image_id": 2, "file_name": "original_a.jpg", "corner": [0, 0, 4, 4]}],
                "train": [{"image_id": 3, "file_name": "original_b.jpg", "corner": [0, 0, 4, 4]}],
            }
            (root / "corner_manifest.json").write_text(json.dumps(manifest))
            yaml_path = tiny.data_yaml_for(str(root.parent / "tinyperson_seed_{seed}_corner_sw640_sh512/tinyperson.yaml"), 42)
            split, files = tiny.split_context(yaml_path, 42)
            self.assertEqual(split["val_corner_count"], 1)
            self.assertTrue(split["original_image_disjoint"])
            self.assertEqual([p.name for p in files], ["img_2.jpg"])
            manifest["train"][0]["file_name"] = "original_a.jpg"
            (root / "corner_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "leakage"):
                tiny.split_context(yaml_path, 42)

    def test_cached_record_must_not_claim_merged(self):
        _, cases = tiny.manifest_cases()
        case = "FTSC"
        rec = {
            "contract": tiny.CONTRACT,
            "dataset": "TinyPerson", "case": case, "seed": 42,
            "source/hf_repo": "TLMHoang/FTSC_results",
            "source/checkpoint": cases[case]["seeds"]["42"]["path"],
            "source/checkpoint_sha256": cases[case]["seeds"]["42"]["sha256"],
            "source/evaluator_sha256": {"a": "b"},
            "split_provenance": {"seed": 42},
            "evaluation": {"no_training": True, "split": "val",
                           "coordinate_space": "merged_original",
                           "imgsz": 640, "batch": 8,
                           "native_nms_iou": .5, "small_nms_iou": .5},
            "metrics": {metric: .7 for metric in tiny.METRICS},
        }
        with self.assertRaisesRegex(RuntimeError, "identity"):
            tiny.validate_record(rec, cases, "TLMHoang/FTSC_results",
                                 case, 42, {"seed": 42}, {"a": "b"})
        rec["evaluation"]["coordinate_space"] = "corner_window"
        tiny.validate_record(rec, cases, "TLMHoang/FTSC_results",
                             case, 42, {"seed": 42}, {"a": "b"})
        del rec["metrics"][tiny.METRICS[2]]
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            tiny.validate_record(rec, cases, "TLMHoang/FTSC_results",
                                 case, 42, {"seed": 42}, {"a": "b"})


if __name__ == "__main__":
    unittest.main()
