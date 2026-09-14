from __future__ import annotations

import copy
import inspect
import tempfile
import unittest
from pathlib import Path

import yaml

from train_tinyperson_scripts import train_tinyperson_ftsc_causal_suite as suite


class TinyPersonCausalSuiteSourceTests(unittest.TestCase):
    def test_registry_defaults_and_historical_controls_are_disjoint(self) -> None:
        self.assertEqual(
            list(suite.VARIANTS),
            ["H2_NOFTSC_NM", "CLEAN_EDGE_NM", "H2_MOSAIC", "ES1_MOSAIC"],
        )
        self.assertEqual(set(suite.HISTORICAL_CONTROLS), {"H2_NM", "ES1_NM"})
        self.assertTrue(set(suite.VARIANTS).isdisjoint(suite.HISTORICAL_CONTROLS))
        self.assertEqual(suite.local_ultralytics(), suite.ULTRALYTICS_ROOT.resolve())
        args = suite.parse_args([])
        self.assertEqual(args.variants, list(suite.VARIANTS))
        self.assertEqual(args.seeds, [42, 43, 44])

    def test_historical_yaml_hashes_and_graphs_are_locked(self) -> None:
        args = suite.parse_args([])
        report = suite.source_preflight(args)
        self.assertEqual(suite._sha256(suite.H2_CONFIG), suite.HISTORICAL_CONFIG_SHA256["H2_NM"])
        self.assertEqual(suite._sha256(suite.ES1_CONFIG), suite.HISTORICAL_CONFIG_SHA256["ES1_NM"])
        self.assertEqual(report["historical_controls"]["H2_NM"]["training_default"], False)
        self.assertEqual(report["historical_controls"]["ES1_NM"]["training_default"], False)
        self.assertEqual(yaml.safe_load(suite.H2_CONFIG.read_text())["head"][-1][0], [19, 22])
        self.assertEqual(yaml.safe_load(suite.ES1_CONFIG.read_text())["head"][-1][0], [19, 22])

    def test_h2_noftsc_is_exact_h2_topology_with_only_ftsc_disabled(self) -> None:
        h2 = yaml.safe_load(suite.H2_CONFIG.read_text(encoding="utf-8"))
        control = yaml.safe_load(suite.H2_NOFTSC_CONFIG.read_text(encoding="utf-8"))
        self.assertIs(control["ftsc"]["enabled"], False)
        normalized = copy.deepcopy(control)
        normalized["ftsc"] = copy.deepcopy(h2["ftsc"])
        self.assertEqual(normalized, h2)
        self.assertEqual(control["head"][-1][0], [19, 22])
        self.assertFalse(any(layer[2] == "P2EdgeCueFusion" for layer in control["head"]))

    def test_clean_edge_changes_only_detect_taps_and_uses_final_features(self) -> None:
        es1 = yaml.safe_load(suite.ES1_CONFIG.read_text(encoding="utf-8"))
        clean = yaml.safe_load(suite.CLEAN_EDGE_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(clean["head"][-1][0], [20, 23])
        normalized = copy.deepcopy(clean)
        normalized["head"][-1][0] = [19, 22]
        self.assertEqual(normalized, es1)
        layers = [*clean["backbone"], *clean["head"]]
        self.assertEqual(layers[20][2], "P2EdgeCueFusion")
        self.assertEqual(layers[23][2], "C2f")
        edge = clean["edge_stability"]
        self.assertEqual(
            (edge["residual_scale"], edge["residual_schedule"], edge["orientation_gate_mode"], edge["hidden"]),
            (0.25, "constant", "learned", 32),
        )

    def test_mosaic_is_the_only_training_factor_changed(self) -> None:
        args = suite.parse_args([])
        data_yaml = Path("same-seed-data.yaml")
        h2_nm = suite.train_kwargs(args, "H2_NOFTSC_NM", 42, data_yaml)
        clean_nm = suite.train_kwargs(args, "CLEAN_EDGE_NM", 42, data_yaml)
        h2_mosaic = suite.train_kwargs(args, "H2_MOSAIC", 42, data_yaml)
        es1_mosaic = suite.train_kwargs(args, "ES1_MOSAIC", 42, data_yaml)
        self.assertEqual((h2_nm["mosaic"], h2_nm["close_mosaic"]), (0.0, 0))
        self.assertEqual((clean_nm["mosaic"], clean_nm["close_mosaic"]), (0.0, 0))
        self.assertEqual((h2_mosaic["mosaic"], h2_mosaic["close_mosaic"]), (1.0, 10))
        self.assertEqual((es1_mosaic["mosaic"], es1_mosaic["close_mosaic"]), (1.0, 10))
        for key, value in h2_mosaic.items():
            if key not in {"mosaic", "close_mosaic", "project", "name"}:
                self.assertEqual(value, es1_mosaic[key])

    def test_frozen_training_and_evaluation_protocol(self) -> None:
        args = suite.parse_args([])
        kwargs = suite.train_kwargs(args, "H2_MOSAIC", 42, Path("data.yaml"))
        self.assertEqual(
            {key: kwargs[key] for key in ("epochs", "patience", "imgsz", "batch", "optimizer", "lr0", "lrf")},
            {
                "epochs": 100,
                "patience": 20,
                "imgsz": 640,
                "batch": 8,
                "optimizer": "AdamW",
                "lr0": 0.002,
                "lrf": 0.01,
            },
        )
        self.assertEqual(kwargs["fitness_metric"], "map50_95")
        self.assertEqual(kwargs["warmup_epochs"], 3.0)
        self.assertEqual(kwargs["warmup_bias_lr"], 0.0)
        protocol = suite._frozen_tinyperson_protocol()
        self.assertEqual(protocol["prediction_conf"], 0.001)
        self.assertEqual(protocol["prediction_nms_iou"], 0.5)
        self.assertEqual(protocol["cross_window_nms_iou"], 0.5)
        self.assertEqual(protocol["max_det"], 300)
        self.assertEqual(protocol["tiny_iou_thresholds"], [0.50, 0.55, 0.60, 0.65, 0.70, 0.75])

    def test_preflight_does_not_train_or_load_pretrained(self) -> None:
        source = inspect.getsource(suite.model_preflight)
        self.assertNotIn(".train(", source)
        self.assertNotIn("build_pretrained_model", source)
        self.assertIn("DetectionModel", source)

    def test_pretrained_policy_is_shared_and_smart_transfer_is_explicit(self) -> None:
        source = inspect.getsource(suite.build_pretrained_model)
        self.assertIn("model.load(pretrained, smart_transfer=True)", source)
        self.assertIn("intersect_dicts", source)
        self.assertIn("transferred_backbone_hash", source)

    def test_seed_dataset_contract_rejects_a_different_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.yaml"
            second = root / "second.yaml"
            first.write_text("path: first\n", encoding="utf-8")
            second.write_text("path: second\n", encoding="utf-8")
            args = suite.parse_args([])
            args.project = root / "runs"
            suite.lock_seed_dataset(args, 42, first)
            suite.lock_seed_dataset(args, 42, first)
            with self.assertRaisesRegex(RuntimeError, "dataset contract changed"):
                suite.lock_seed_dataset(args, 42, second)

    def test_summary_reads_historical_results_without_rewriting_them(self) -> None:
        before = suite._sha256(suite.HISTORICAL_RESULTS_DEFAULT)
        with tempfile.TemporaryDirectory() as directory:
            args = suite.parse_args([])
            args.project = Path(directory)
            args.historical_results = suite.HISTORICAL_RESULTS_DEFAULT
            summary = suite.write_summaries(args)
            historical = [row for row in summary["runs"] if row["origin"] == "historical_read_only"]
            self.assertEqual(len(historical), 6)
            self.assertEqual(summary["interactions"], [])
            self.assertTrue(any("interaction" in item for item in summary["missing"]))
            self.assertTrue((args.project / "summary_runs.csv").is_file())
            self.assertTrue((args.project / "summary_aggregate.json").is_file())
            self.assertTrue((args.project / "summary_comparisons.json").is_file())
        self.assertEqual(suite._sha256(suite.HISTORICAL_RESULTS_DEFAULT), before)


class TinyPersonCausalSuiteModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            suite.local_ultralytics()
            suite._tiny_workflow()
            import torch  # noqa: F401
            from ultralytics.nn.tasks import DetectionModel  # noqa: F401
        except (ImportError, ModuleNotFoundError) as error:
            raise unittest.SkipTest(f"model runtime unavailable: {error}") from error

    def test_all_runtime_graph_contracts(self) -> None:
        args = suite.parse_args([])
        report = suite.model_preflight(args)
        self.assertTrue(report["same_initial_backbone"])
        expected_sources = {
            "H2_NOFTSC_NM": ["Identity", "C2f"],
            "CLEAN_EDGE_NM": ["P2EdgeCueFusion", "C2f"],
            "H2_MOSAIC": ["Identity", "C2f"],
            "ES1_MOSAIC": ["Identity", "Concat"],
        }
        for variant, item in report["variants"].items():
            self.assertEqual(item["detect_strides"], [4.0, 8.0])
            self.assertEqual(item["detect_source_types"], expected_sources[variant])
            self.assertEqual(item["ftsc_enabled"], suite.VARIANTS[variant].ftsc_enabled)
            self.assertEqual(item["edge_enabled"], suite.VARIANTS[variant].edge_enabled)
        self.assertIsNone(report["variants"]["H2_NOFTSC_NM"]["ftsc_policy"])
        self.assertTrue(report["variants"]["CLEAN_EDGE_NM"]["edge"]["zero_initialized_residual"])


if __name__ == "__main__":
    unittest.main()
