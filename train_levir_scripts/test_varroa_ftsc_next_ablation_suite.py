"""CPU-safe contracts for the prepared Varroa next-ablation suite."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_levir_scripts import train_varroa_ftsc_next_ablation_suite as suite


class VarroaNextAblationStaticTests(unittest.TestCase):
    def test_exact_nine_case_registry_and_stage_aliases(self):
        self.assertEqual(len(suite.CASES), 9)
        self.assertEqual(tuple(suite.CASES), suite.ALL_CASES)
        self.assertEqual(suite.STAGES["architecture"], suite.ARCHITECTURE_CASES)
        self.assertEqual(suite.STAGES["augmentation-core"], suite.AUGMENTATION_CORE_CASES)
        self.assertEqual(suite.STAGES["augmentation-edge"], suite.AUGMENTATION_EDGE_CASES)
        self.assertEqual(suite.STAGES["all"], suite.ALL_CASES)
        self.assertTrue(set(suite.CASES).isdisjoint(suite.HISTORICAL_REFERENCES))
        self.assertEqual(len(suite.HISTORICAL_REFERENCES), 5)

    def test_cli_defaults_and_stage_selection(self):
        default = suite.parse_args([])
        suite.validate_args(default)
        self.assertEqual(tuple(default.cases), suite.ALL_CASES)
        self.assertEqual(default.seeds, [42, 43, 44])
        for stage, expected in suite.STAGES.items():
            args = suite.parse_args(["--stage", stage])
            suite.validate_args(args)
            self.assertEqual(tuple(args.cases), expected)
        explicit = suite.parse_args(["--cases", "FTSC_P34", "FTSC_P23_CP2"])
        suite.validate_args(explicit)
        self.assertEqual(explicit.cases, ["FTSC_P34", "FTSC_P23_CP2"])

    def test_frozen_protocol_rejects_overrides(self):
        args = suite.parse_args(["--epochs", "99"])
        with self.assertRaisesRegex(RuntimeError, "frozen Varroa protocol changed"):
            suite.validate_args(args)
        args = suite.parse_args([])
        self.assertEqual((args.epochs, args.imgsz, args.batch_size, args.patience), (100, 640, 4, 20))

    def test_ftsc_text_value_and_fingerprint_parity(self):
        def raw_ftsc(path: Path) -> str:
            text = path.read_text(encoding="utf-8")
            return text[text.index("ftsc:\n"):text.index("nc: 1\n")]

        reference_text = raw_ftsc(suite.CANONICAL_FTSC_CONFIG)
        reference = suite.placement._load_yaml(suite.CANONICAL_FTSC_CONFIG)["ftsc"]
        expected_fingerprint = suite._json_fingerprint(reference)
        for name, spec in suite.CASES.items():
            payload = suite.placement._load_yaml(spec.config)
            self.assertEqual(payload["ftsc"], reference, name)
            self.assertEqual(raw_ftsc(spec.config), reference_text, name)
            self.assertEqual(suite._json_fingerprint(payload["ftsc"]), expected_fingerprint, name)
            self.assertFalse(suite.UNWANTED_TOP_LEVEL & set(payload), name)

    def test_three_architecture_graph_contracts(self):
        report = suite.source_preflight()["cases"]
        expected = {
            "FTSC_P234_EDGE_P2_DETECT_ONLY": (
                [19, 22, 25], ["P2EdgeCueFusion", "RepC2f", "RepC2f"], [4.0, 8.0, 16.0], 1,
            ),
            "FTSC_P34": ([21, 24], ["RepC2f", "RepC2f"], [8.0, 16.0], 0),
            "FTSC_P34_EDGE_P3_DETECT_ONLY": (
                [22, 25], ["P3EdgeCueFusion", "RepC2f"], [8.0, 16.0], 1,
            ),
        }
        for name, (indices, types, strides, edge_count) in expected.items():
            item = report[name]
            self.assertEqual(item["detect_from"], indices)
            self.assertEqual(item["detect_source_types"], types)
            self.assertEqual(item["detect_strides"], strides)
            self.assertEqual(int(item["edge_enabled"]), edge_count)
            self.assertTrue(item["clean_downstream_verified"])

    def test_edge_ancestry_cannot_reach_clean_downstream_features(self):
        for name in ("FTSC_P234_EDGE_P2_DETECT_ONLY", "FTSC_P34_EDGE_P3_DETECT_ONLY"):
            spec = suite.CASES[name]
            layers = suite.placement._layers(suite.placement._load_yaml(spec.config))
            downstream, clean_source = spec.clean_downstream_start
            self.assertEqual(suite.placement._resolved_inputs(layers, downstream), (clean_source,))
            self.assertNotIn(spec.edge_index, suite.placement._ancestors(layers, downstream))
            for source in spec.detect_from:
                if source != spec.edge_index:
                    self.assertNotIn(spec.edge_index, suite.placement._ancestors(layers, source))
        a1_layers = suite.placement._layers(
            suite.placement._load_yaml(suite.CASES["FTSC_P234_EDGE_P2_DETECT_ONLY"].config)
        )
        self.assertEqual(suite.placement._resolved_inputs(a1_layers, 23), (22,))
        self.assertNotIn(19, suite.placement._ancestors(a1_layers, 25))
        a3_layers = suite.placement._layers(
            suite.placement._load_yaml(suite.CASES["FTSC_P34_EDGE_P3_DETECT_ONLY"].config)
        )
        self.assertEqual(suite.placement._resolved_inputs(a3_layers, 23), (21,))
        self.assertNotIn(22, suite.placement._ancestors(a3_layers, 25))

    def test_a1_is_clean_p234_plus_only_prediction_local_edge(self):
        payload = suite.placement._load_yaml(suite.CASES["FTSC_P234_EDGE_P2_DETECT_ONLY"].config)
        normalized = suite._remove_edge_for_comparison(payload, 19, [18, 21, 24])
        clean = suite.placement._load_yaml(suite.P234_REFERENCE_CONFIG)
        self.assertEqual(normalized, clean)

    def test_edge_constructor_parity(self):
        edge_args = []
        for spec in suite.CASES.values():
            if spec.edge_index is not None:
                layers = suite.placement._layers(suite.placement._load_yaml(spec.config))
                edge_args.append(layers[spec.edge_index][3])
        self.assertTrue(edge_args)
        self.assertTrue(all(args == suite.EDGE_ARGS for args in edge_args))

    def test_augmentation_factorial_contract(self):
        expected = {
            "FTSC_P23_NOMOSAIC": (0.0, 0, False),
            "FTSC_P23_CP2": (1.0, 10, True),
            "FTSC_P23_NOMOSAIC_CP2": (0.0, 0, True),
            "EDGE_P2_DETECT_ONLY_NOMOSAIC": (0.0, 0, False),
            "EDGE_P2_DETECT_ONLY_CP2": (1.0, 10, True),
            "EDGE_P2_DETECT_ONLY_NOMOSAIC_CP2": (0.0, 0, True),
        }
        for name, values in expected.items():
            contract = suite.augmentation_contract(name)
            self.assertEqual(
                (contract["mosaic"], contract["close_mosaic"], contract["cp2_enabled"]),
                values,
            )
            self.assertEqual(contract["stock_segmentation_copy_paste"], 0.0)
            self.assertFalse(contract["oacp_enabled"])
            self.assertFalse(contract["m5_enabled"])
            overrides = contract["train_overrides"]
            self.assertEqual(overrides["mosaic_policy"], "standard")
            self.assertFalse(overrides["hard_negative_tile"])
            if values[0] == 1.0:
                self.assertNotIn("mosaic", overrides)
                self.assertNotIn("close_mosaic", overrides)
            else:
                self.assertEqual((overrides["mosaic"], overrides["close_mosaic"]), (0.0, 0))

    def test_cp2_is_exact_levir_a3_contract(self):
        levir = suite.levir_augmentation.augmentation_for("A3_CP2")
        expected = {key: levir[key] for key in suite.CP2_KEYS}
        self.assertEqual(suite.validated_cp2_settings(), expected)
        self.assertEqual((expected["copy_paste_unit"], expected["copy_paste_copies"]), ("single", 2))
        self.assertEqual((expected["copy_paste_p"], expected["copy_paste_scale"]), (0.5, 1.0))
        self.assertEqual((expected["copy_paste_padding"], expected["copy_paste_blend"]), (0.0, "hard"))
        self.assertEqual(expected["copy_paste"], 0.0)

    def test_train_kwargs_change_only_registered_augmentation_keys(self):
        args = suite.parse_args([])
        args.seed_scope = "all"
        data = Path("/tmp/varroa.yaml")
        base = suite.shared.train_kwargs(args, data, 42, True)
        allowed = set(suite.CP2_KEYS) | {
            "mosaic", "close_mosaic", "mosaic_policy", "hard_negative_tile", "hard_negative_bank"
        }
        for case in suite.CASES:
            kwargs = suite.train_kwargs(args, case, data, 42, True)
            changed = set(kwargs) - set(base)
            self.assertTrue(changed <= allowed, (case, changed))
            self.assertEqual({key: kwargs[key] for key in base}, base, case)

    def test_preflight_contains_no_metric_fields(self):
        report = suite.source_preflight()
        self.assertEqual(report["status"], "PREPARED_NOT_RUN")
        serialized = str(report).lower()
        for forbidden in ("test_ap50", "val_ap50", "map50-95", "precision", "recall"):
            self.assertNotIn(forbidden, serialized)
        self.assertFalse(report["references_scheduled"])

    def test_combined_preflight_exposes_auditable_top_level_status(self):
        with patch.object(suite, "model_preflight", return_value={}):
            report = suite.build_preflight()
        self.assertEqual(report["status"], "PREPARED_NOT_RUN")
        self.assertEqual(tuple(report["case_names"]), suite.ALL_CASES)
        self.assertEqual(report["suite"], "varroa_ftsc_next_ablation_suite")

    def test_default_main_fails_closed_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight.json"
            with patch.object(suite, "build_preflight", return_value={"status": "PREPARED_NOT_RUN"}), patch.object(
                suite, "train_one"
            ) as train_one, contextlib.redirect_stdout(io.StringIO()):
                suite.main(["--preflight-json", str(output)])
            train_one.assert_not_called()
            self.assertTrue(output.is_file())


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"),
    "local Python lacks torch/cv2",
)
class VarroaNextAblationRuntimeTests(unittest.TestCase):
    def test_detection_cp2_builder_not_segmentation_copypaste(self):
        suite.shared.local_ultralytics()
        from ultralytics.cfg import get_cfg
        from ultralytics.data.augment import Compose, CopyPaste, v8_transforms
        from project_ultralytics.copy_paste import SmallObjectCopyPaste

        class Dataset:
            cache = None
            buffer = []
            data = {}
            use_keypoints = False
            im_files: list[str] = []
            labels: list[dict] = []

            def __len__(self):
                return 0

        settings = suite.augmentation_overrides("FTSC_P23_NOMOSAIC_CP2")
        transforms = v8_transforms(Dataset(), 32, get_cfg(overrides=settings))
        self.assertIsInstance(transforms.transforms[0], Compose)
        self.assertIsInstance(transforms.transforms[1], SmallObjectCopyPaste)
        self.assertFalse(any(isinstance(transform, CopyPaste) for transform in transforms.transforms[0].transforms))

    def test_cp2_is_deterministic_and_updates_detection_labels(self):
        import cv2
        import numpy as np
        from project_ultralytics.copy_paste import SmallObjectCopyPaste
        from ultralytics.utils.instance import Instances

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[12:20, 12:20] = (17, 101, 233)
            image_path = root / "source.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))

            class Dataset:
                im_files = [str(image_path)]
                labels = [{
                    "bboxes": np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float32),
                    "cls": np.array([[0]], dtype=np.float32),
                    "bbox_format": "xywh",
                    "normalized": True,
                    "shape": (32, 32),
                }]

            def target():
                return {
                    "img": np.zeros((40, 40, 3), dtype=np.uint8),
                    "cls": np.empty((0, 1), dtype=np.float32),
                    "instances": Instances(
                        np.empty((0, 4), dtype=np.float32),
                        segments=np.empty((0, 0, 2), dtype=np.float32),
                        bbox_format="xyxy",
                        normalized=False,
                    ),
                    "im_file": str(root / "target.png"),
                }

            kwargs = dict(
                dataset=Dataset(), p=1.0, unit="single", copies=2, placement="random",
                max_overlap=0.0, padding=0.0, scale=1.0, blend="hard", max_trials=30,
                allow_empty_target=True, allow_same_source=True,
            )
            random.seed(401)
            first = SmallObjectCopyPaste(**kwargs)(target())
            random.seed(401)
            second = SmallObjectCopyPaste(**kwargs)(target())
            np.testing.assert_array_equal(first["img"], second["img"])
            np.testing.assert_array_equal(first["instances"].bboxes, second["instances"].bboxes)
            np.testing.assert_array_equal(first["cls"], second["cls"])
            self.assertEqual(len(first["instances"]), 2)
            self.assertEqual(first["cls"].shape, (2, 1))

    def test_all_unique_models_build_forward_and_profile_on_cpu(self):
        report = suite.model_preflight()
        self.assertEqual(len({spec.config for spec in suite.CASES.values()}), 5)
        self.assertEqual(set(report), set(suite.CASES))
        for name, spec in suite.CASES.items():
            item = report[name]
            self.assertEqual(item["detect_from"], list(spec.detect_from))
            self.assertEqual(item["detect_strides"], list(suite._expected_strides(spec)))
            self.assertEqual(item["edge_count"], int(spec.edge_index is not None))
            self.assertGreater(item["parameters"], 0)
            self.assertEqual(item["forward_input_shape"], [1, 3, 64, 64])
            self.assertIsNotNone(item["forward_output_shapes"])


if __name__ == "__main__":
    unittest.main()
