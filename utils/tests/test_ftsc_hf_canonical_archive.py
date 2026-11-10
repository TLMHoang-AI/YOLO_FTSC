"""Offline, read-only regression checks for canonical FTSC HF paths."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.ftsc_hf_layout import (
    AUGMENTATION_RUNS, SEEDS, HFLayoutError, augmentation_checkpoint,
    dataset_of_namespace, evaluation_root, prefix, training_root,
)
from utils.ftsc_hf_canonical_archive import CanonicalFTSCArchive
from utils.hf_result_archive import HFResultArchiveError


class FakeAPI:
    def __init__(self, paths=()):
        self.paths = set(paths)
        self.uploads = []

    def repo_info(self, **kwargs):
        return type("Info", (), {"sha": "a" * 40})()

    def list_repo_files(self, **kwargs):
        return sorted(self.paths)

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs["path_in_repo"])
        return type("Commit", (), {"oid": "offline"})()


def offline_archive(api=None):
    """Create an uninitialized in-memory transport for path tests only."""
    obj = object.__new__(CanonicalFTSCArchive)
    obj.repo_id = "TLMHoang/FTSC_results"
    obj.repo_type = "dataset"
    obj._api = api or FakeAPI()
    obj._retry = lambda fn: fn()
    obj._assert_repository_access = lambda: None
    return obj


class CanonicalHFTests(unittest.TestCase):
    def test_all_table_source_paths_have_three_seeds(self):
        count = 0
        for dataset, cases in AUGMENTATION_RUNS.items():
            for case in cases:
                for seed in SEEDS:
                    path = augmentation_checkpoint(dataset, case, seed)
                    self.assertEqual(path.split("/")[0], dataset)
                    self.assertTrue(path.endswith("/weights/best.pt"))
                    count += 1
        self.assertEqual(count, 33)

    def test_audited_source_contracts(self):
        self.assertEqual(
            augmentation_checkpoint("LEVIR-Ship", "FTSC", 42),
            "LEVIR-Ship/historical_reconstructions/RECON_H2_V1/seed_42/weights/best.pt",
        )
        self.assertEqual(
            augmentation_checkpoint("TinyPerson", "CP2", 44),
            "TinyPerson/additional_ablations/tinyperson_h2_generalization_completion_v1/"
            "TP_H2_MOSAIC_CP2/seed_44_corner_sw640_sh512/weights/best.pt",
        )
        self.assertEqual(
            augmentation_checkpoint("Varroa", "OACP", 43),
            "Varroa/additional_ablations/varroa_h2_generalization_completion_v1/"
            "VR_H2_MOSAIC_OACP_R2/seed_43/weights/best.pt",
        )

    def test_invalid_paths_are_rejected(self):
        with self.assertRaises(HFLayoutError):
            augmentation_checkpoint("TinyPerson", "CP2", 45)
        with self.assertRaises(HFLayoutError):
            augmentation_checkpoint("TinyPerson", "NC_R4", 42)
        with self.assertRaises(HFLayoutError):
            prefix("tinyperson/legacy/weights")
        with self.assertRaises(HFLayoutError):
            prefix("LEVIR-Ship/../original_runs")
        with self.assertRaises(HFLayoutError):
            dataset_of_namespace("unknown")
        with self.assertRaises(HFLayoutError):
            training_root("tinyperson_ftsc_cross_yolo_portability_v1")

    def test_new_publication_paths(self):
        self.assertEqual(training_root("levir_ftsc_h2_augmentation_suite"),
                         "LEVIR-Ship/original_runs")
        self.assertEqual(training_root("tinyperson_ftsc_position_alpha_h2_v1"),
                         "TinyPerson/journal_experiments/alpha_sensitivity")
        self.assertEqual(training_root("varroa_foo_v2"),
                         "Varroa/additional_ablations/varroa_foo_v2")
        self.assertEqual(evaluation_root("LEVIR-Ship"),
                         "LEVIR-Ship/derived_evaluations/capstone_aug_sixcore_v1")

    def test_cross_yolo_archive_routes_family_and_case(self):
        self.assertEqual(CanonicalFTSCArchive.remote_path(
            "levir_ftsc_cross_yolo_portability_v1",
            "yolov9t__FTSC_H2", "seed_42"),
            "LEVIR-Ship/cross_yolo_portability/yolov9t/FTSC_H2/seed_42",
        )
        self.assertEqual(CanonicalFTSCArchive.remote_path(
            "tinyperson_ftsc_cross_yolo_portability_v1",
            "yolo11n__FTSC_H2", "seed_43_corner_sw640_sh512"),
            "TinyPerson/cross_yolo_portability/yolo11n/FTSC_H2/seed_43_corner_sw640_sh512",
        )

    def test_journal_phase2_routes_to_dataset_journal_or_evaluation_tree(self):
        self.assertEqual(CanonicalFTSCArchive.remote_path(
            "levir_ftsc_journal_phase2_v1", "E_H2_POSTFUSION_V2", "seed_42"),
            "LEVIR-Ship/journal_experiments/phase2/E_H2_POSTFUSION_V2/seed_42",
        )
        self.assertEqual(CanonicalFTSCArchive.remote_path(
            "tinyperson_ftsc_journal_phase2_v1", "historical__ALPHA4_rerun",
            "seed_44_corner_sw640_sh512"),
            "TinyPerson/derived_evaluations/journal_phase2/ALPHA4_rerun/"
            "seed_44_corner_sw640_sh512",
        )

    def test_upload_blocks_existing_canonical_run(self):
        api = FakeAPI(paths=("LEVIR-Ship/original_runs/A3_CP2/seed_42/weights/best.pt",))
        archive = offline_archive(api)
        with self.assertRaisesRegex(HFResultArchiveError, "refusing overwrite"):
            archive.publish_run(
                dataset_slug="levir_ftsc_h2_augmentation_suite",
                variant="A3_CP2", seed=42, run_name="seed_42",
                run_dir=Path("/nonexistent"), required_artifacts=("test.json",))
        self.assertFalse(api.uploads)

    def test_local_notebook_cells_have_valid_python_syntax_and_safe_defaults(self):
        import ast
        workspace = Path(__file__).resolve().parents[3]
        for notebook, dataset, guard in (
            ("current_live_training_notebok_LEV.ipynb", "LEVIR-Ship",
             "RUN_CAPSTONE_AUG_LEVIR = False"),
            ("current_live_training_notebok_TinyPerson.ipynb", "TinyPerson",
             "RUN_CAPSTONE_AUG_TINY = False"),
            ("current_live_training_notebok_Varroa.ipynb", "Varroa",
             "RUN_CAPSTONE_AUG_VARROA = False"),
        ):
            path = workspace / notebook
            if not path.is_file():
                continue  # CI may contain only the codebase, not host notebooks.
            payload = json.loads(path.read_text(encoding="utf-8"))
            cell = payload["cells"][-1]
            source = "".join(cell["source"])
            ast.parse(source, filename=notebook)
            self.assertIn(guard, source)
            self.assertIn(f"'--dataset', '{dataset}'", source)
            self.assertIn("ALLOW_PUBLISH = False", source)
            self.assertIn("ALLOW_EVALUATE = False", source)

    def test_evaluation_dry_run_never_uploads(self):
        with tempfile.TemporaryDirectory() as temp:
            metrics = Path(temp) / "evaluation_metrics_extended.json"
            metrics.write_text(json.dumps({
                "test_merged/AP50": 0.57,
                "test_merged/mAP50-95": 0.21,
                "test_merged/AP50-Small": 0.69,
                "model/parameters": 1821959,
                "model/GFLOPs": 10.5865216,
            }))
            api = FakeAPI()
            archive = offline_archive(api)
            archive.sha256 = lambda _path: "a" * 64
            result = archive.publish_evaluation(
                dataset="TinyPerson", case="CP2", seed=42,
                metric_file=metrics, evaluator_contract="tinybenchmark_merged_original_iou50_95_v2")
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(result["remote_evaluation_path"],
                             "TinyPerson/derived_evaluations/capstone_aug_sixcore_v1/CP2/"
                             "seed_42_corner_sw640_sh512")
            self.assertFalse(api.uploads)
            metrics.write_text(json.dumps({"test_merged/AP50": 0.57}))
            with self.assertRaisesRegex(HFResultArchiveError, "Missing/invalid six-core metric"):
                archive.publish_evaluation(
                    dataset="TinyPerson", case="CP2", seed=42,
                    metric_file=metrics, evaluator_contract="merged")

    def test_merged_contract_and_source_integrity_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evaluation_metrics_extended.json"
            base = {
                "test_merged/AP50": 0.57,
                "test_merged/mAP50-95": 0.21,
                "test_merged/AP50-Small": 0.69,
                "model/parameters": 1821959,
                "model/GFLOPs": 10.5865216,
            }
            archive = offline_archive()
            archive.sha256 = lambda _path: "a" * 64
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(HFResultArchiveError, "original-image merged"):
                archive.publish_evaluation(
                    dataset="TinyPerson", case="CP2", seed=42,
                    metric_file=path, evaluator_contract="cropped")
            base["evaluation/source_checkpoint"] = "TinyPerson/legacy/seed_42/weights/best.pt"
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(HFResultArchiveError, "canonical source"):
                archive.publish_evaluation(
                    dataset="TinyPerson", case="CP2", seed=42,
                    metric_file=path,
                    evaluator_contract="tinybenchmark_merged_original_iou50_95_v2")


if __name__ == "__main__":
    unittest.main()
