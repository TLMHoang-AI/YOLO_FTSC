"""Offline regression checks for LEVIR H2 reproduction audit (no HF/GPU)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "utils/ftsc_levir_h2_reproduction_audit_v1.py"
SPEC = importlib.util.spec_from_file_location("h2_audit", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class H2AuditOffline(unittest.TestCase):
    def test_source_identity(self):
        expected = MOD.canonical_source_sha()
        self.assertEqual(set(expected), {42, 43, 44})
        self.assertEqual(
            expected[42],
            "e658614e617378110a280fe62927eb0df6f708162a5117865fe45efe66612805")
        self.assertEqual(set(MOD.historical()), {42, 43, 44})
        self.assertAlmostEqual(MOD.historical()[42]["AP50"], 81.25051194881348)

    def test_cohorts_are_distinct(self):
        self.assertEqual(len(MOD.SOURCES), 3)
        self.assertEqual(len(set(MOD.SOURCES.values())), 3)
        self.assertNotIn("original_runs/A0_FTSC", " ".join(MOD.SOURCES.values()))

    def test_distinct_selections_cannot_clobber(self):
        a = SimpleNamespace(cases=["RECON_H2_V1"], seeds=[42])
        b = SimpleNamespace(cases=["RECON_H2_V1"], seeds=[42, 43, 44])
        self.assertNotEqual(MOD.selection_id(a), MOD.selection_id(b))

    def test_immutable_json_roundtrip_and_real_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.json"
            data = {"historical_per_seed_pct": {42: {"AP50": 81.25}}}
            MOD.immutable_json(path, data)
            original = path.read_bytes()
            MOD.immutable_json(path, data)
            self.assertEqual(path.read_bytes(), original)
            MOD.immutable_json(path, json.loads(original))
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(RuntimeError, "Existing output differs"):
                MOD.immutable_json(path, {"historical_per_seed_pct": {42: {"AP50": 82.0}}})
            self.assertEqual(path.read_bytes(), original)

    def test_no_training_entrypoint(self):
        import inspect
        code = inspect.getsource(MOD)
        self.assertNotIn(".train(", code)
        self.assertNotIn("upload_file(", code)
        self.assertNotIn("upload_folder(", code)


if __name__ == "__main__":
    unittest.main()
