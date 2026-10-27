"""Static contracts for corrected cues and historical/new Varroa graph separation.

This module never fetches HF artifacts, trains, or alters historical results.
Historical full-neck H1/H2 are explicitly not reusable as structural training runs.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "models_related/models_config/yolov8"


def graph(name: str):
    with (V8 / name).open(encoding="utf-8") as stream:
        obj = yaml.safe_load(stream)
    return obj, obj["backbone"] + obj["head"]


class CrossDatasetColorCueTest(unittest.TestCase):
    def test_historical_levir_color_was_partially_bypassed(self):
        old, graph_old = graph("levir/yolov8n_p2_levir_ftsc_h2_colorcue.yaml")
        self.assertEqual(graph_old[-1][0], [19, 22])
        self.assertEqual(graph_old[19][2], "nn.Identity")
        self.assertEqual(graph_old[20][2], "P2ColorCueFusion")
        self.assertEqual(graph_old[22][2], "Concat")
        self.assertEqual(graph_old[23][2], "C2f")
        # Old P2 detection bypasses color; P3 Concat is downstream of color,
        # so the legacy Color Cue is NOT entirely disconnected.
        self.assertEqual(graph_old[21][0], -1)
        self.assertEqual(graph_old[22][0], [-1, 15])

    def test_three_dataset_cues_use_postfusion_p2_and_postc2f_p3(self):
        cases = [
            ("levir/yolov8n_p2_levir_ftsc_h2_colorcue_postfusion_v2.yaml", 20, 23, "C2f"),
            ("tinyperson/yolov8n_tinyperson_ftsc_color_h2_mosaic.yaml", 20, 23, "C2f"),
            ("varroa/yolov8n_varroa_ftsc_v3_p2p3_color.yaml", 19, 22, "RepC2f"),
        ]
        for name, fusion, p3, p3_type in cases:
            with self.subTest(name=name):
                obj, g = graph(name)
                self.assertEqual(g[fusion][2], "P2ColorCueFusion")
                self.assertEqual(g[p3][2], p3_type)
                self.assertEqual(g[-1][0], [fusion, p3])
                self.assertEqual(g[fusion + 1][0], -1)
                self.assertEqual(g[p3 - 1][0], [-1, 15])
                self.assertEqual(obj["ftsc"]["policy"], "f5")

    def test_levir_corrected_color_only_changes_detect_taps_and_metadata(self):
        legacy, lg = graph("levir/yolov8n_p2_levir_ftsc_h2_colorcue.yaml")
        fixed, fg = graph("levir/yolov8n_p2_levir_ftsc_h2_colorcue_postfusion_v2.yaml")
        self.assertEqual(legacy["ftsc"], fixed["ftsc"])
        self.assertEqual(legacy["backbone"], fixed["backbone"])
        self.assertEqual(lg[:-1], fg[:-1])
        self.assertEqual(lg[-1][1:], fg[-1][1:])
        self.assertEqual(lg[-1][0], [19, 22])
        self.assertEqual(fg[-1][0], [20, 23])
        self.assertNotEqual(legacy["ablation"]["case"], fixed["ablation"]["case"])

    def test_levir_corrected_edge_uses_postfusion_postblock_taps(self):
        legacy, lg = graph("levir/yolov8n_p2_levir_ftsc_h2_edgecue.yaml")
        fixed, fg = graph("levir/yolov8n_p2_levir_ftsc_h2_edgecue_postfusion_v2.yaml")
        self.assertEqual(lg[:-1], fg[:-1])
        self.assertEqual(lg[-1][0], [19, 22])
        self.assertEqual(fg[-1][0], [20, 23])
        self.assertEqual(fg[20][2:], lg[20][2:])


class VarroaHistoricalGraphSeparationTest(unittest.TestCase):
    def test_new_fullneck_p2_only_differs_from_legacy_h2_only_in_detect(self):
        legacy, lg = graph("varroa/yolov8n_varroa_ftsc_v1_p2p3.yaml")
        p2, pg = graph("varroa/yolov8n_varroa_ftsc_head_p2_fullneck_v1.yaml")
        self.assertEqual(legacy["backbone"], p2["backbone"])
        self.assertEqual(legacy["ftsc"], p2["ftsc"])
        self.assertEqual(lg[:-1], pg[:-1])
        self.assertEqual(lg[-1][0], [18, 21])
        self.assertEqual(pg[-1][0], [18])

        # Preserved experimental artifact, deliberately absent from the final registry.
        from train_levir_scripts import train_varroa_ftsc_paper_new_cases_v1 as runner
        self.assertNotIn("VR_HEADSET_H3_P2_FULLNECK", runner.CASES)
        fullneck = V8 / "varroa/yolov8n_varroa_ftsc_head_p2_fullneck_v1.yaml"
        self.assertNotIn(fullneck, [spec.yaml for spec in runner.CASES.values()])

    def test_legacy_fullneck_has_same_active_prefix_as_structural_variant(self):
        controls = [
            ("varroa/yolov8n_varroa_ftsc_head_p234.yaml",
             "varroa/yolov8n_varroa_ftsc_struct_h1_p234.yaml", [18, 21, 24]),
            ("varroa/yolov8n_varroa_ftsc_v1_p2p3.yaml",
             "varroa/yolov8n_varroa_ftsc_struct_h2_p23.yaml", [18, 21]),
        ]
        for legacy, pruned, taps in controls:
            with self.subTest(legacy=legacy):
                old, old_graph = graph(legacy)
                new, new_graph = graph(pruned)
                self.assertEqual(old["ftsc"], new["ftsc"])
                self.assertEqual(old["backbone"], new["backbone"])
                self.assertEqual(old_graph[-1][0], taps)
                self.assertEqual(new_graph[-1][0], taps)
                self.assertEqual(old_graph[:len(new_graph) - 1], new_graph[:-1])
                self.assertGreater(len(old_graph), len(new_graph))
                self.assertTrue(all(i < len(new_graph) - 1 for i in taps))
                # Prefix parity does not make historical training equivalent.
                # Final policy requires independent structural training.


if __name__ == "__main__":
    unittest.main()
