"""CPU-only notebook integrity tests; no pytest, TensorRT, CUDA or network needed."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "FTSC_LEVIR_H2_INT8_RootCause_Seed42_Marimo_Live.ipynb"
ENGINE = ROOT / "models_related/ultralytics/ultralytics/utils/export/engine.py"


class NotebookStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = ["".join(cell["source"]) for cell in cls.document["cells"]
                     if cell["cell_type"] == "code"]
        cls.code = "\n".join(cls.cells)

    def test_notebook_structure_and_python_syntax(self):
        self.assertEqual(self.document["nbformat"], 4)
        self.assertGreaterEqual(len(self.cells), 6)
        for index, source in enumerate(self.cells):
            with self.subTest(index=index):
                compile(source, f"notebook_cell_{index}", "exec", flags=ast.PyCF_ONLY_AST)

    def test_seed42_only_gpu_off(self):
        cfg = self.cells[0]
        self.assertIn("RUN_GPU = False", cfg)
        self.assertIn("RECON_H2_V1/seed_42/weights/best.pt", cfg)
        self.assertNotIn("seed_43/weights", self.code)
        self.assertNotIn("train(", self.code)
        self.assertNotIn("upload_file(", self.code)

    def test_patch_source_is_audited_and_restored(self):
        old = (
            '                    im0s = next(self.data_iter)["img"] / 255.0\n'
            '                    im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s\n'
            '                    return [int(im0s.data_ptr())]'
        )
        prod = ENGINE.read_text(encoding="utf-8")
        self.assertEqual(prod.count(old), 1)
        self.assertEqual(prod.count("else trt.CalibrationAlgoType.MINMAX_CALIBRATION"), 1)
        for text in ("ENTROPY_CALIBRATION_2", "self.current_batch",
                     "engine.onnx2engine = old_engine",
                     "exporter.Exporter.get_int8_calibration_dataloader = old_loader"):
            self.assertIn(text, self.code)

    def test_variant_isolation_and_evaluation_contract(self):
        for arm in ("legacy_minmax", "stable_minmax", "stable_entropy",
                    "stable_minmax_matched", "stable_entropy_matched"):
            with self.subTest(arm=arm):
                self.assertIn(arm, self.code)
        for term in (
            'OUT / "variants" / name',
            "v2.static_engine_for",
            "v2.evaluate_main_pt32",
            "v2.evaluate_main_int8",
            "v2.predict_small",
            "v2.verify_small_parity",
            "len(images) == 788",
            "assert not work.exists()",
            'lock.open("x")',
        ):
            with self.subTest(term=term):
                self.assertIn(term, self.code)
        self.assertNotIn("git reset", self.code)
        self.assertNotIn("pip install", self.code)


if __name__ == "__main__":
    unittest.main()
