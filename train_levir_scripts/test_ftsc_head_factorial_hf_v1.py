"""Offline regression tests for the new LEVIR factorial HF-first workflow."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from train_levir_scripts import ftsc_head_factorial_hf_v1 as hf
from train_levir_scripts import train_levir_ftsc_head_factorial_v1 as suite
from utils.hf_result_archive import MARKER_FILENAME


class FakeHub:
    def __init__(self, paths=(), *, fail=False):
        self.paths = tuple(paths)
        self.fail = fail
    def repo_info(self, *, repo_id, repo_type):
        assert repo_id == "TLMHoang/FTSC_results" and repo_type == "dataset"
        if self.fail:
            raise ConnectionError("Remote inventory unavailable")
        return SimpleNamespace(sha="0123456789abcdef" * 3)
    def list_repo_files(self, *, repo_id, repo_type, revision):
        assert revision == "0123456789abcdef" * 3
        return self.paths


class HeadFactorialHubTests(unittest.TestCase):
    def test_canonical_namespace_does_not_overlap_legacy_runs(self):
        path = hf.remote_prefix("P345_FTSC", 42)
        self.assertEqual(path, "LEVIR-Ship/journal_experiments/head_factorial_v1/P345_FTSC/seed_42")
        self.assertNotIn("original_runs", path)

    def test_absent_remote_is_not_mistaken_for_archived(self):
        report = hf.inventory("example-not-real", api=FakeHub())
        self.assertEqual(len(report["rows"]), 12)
        self.assertTrue(all(x["state"] == "ABSENT" for x in report["rows"]))

    def test_complete_marker_and_all_required_files(self):
        prefix = hf.remote_prefix("P345_FTSC", 42)
        paths = [prefix + "/" + name for name in (*hf.REQUIRED, MARKER_FILENAME)]
        report = hf.inventory("example-not-real", api=FakeHub(paths))
        states = {(x["case"], x["seed"]): x["state"] for x in report["rows"]}
        self.assertEqual(states[("P345_FTSC", 42)], "MARKED_PRESENT")
        self.assertEqual(states[("P345_FTSC", 43)], "ABSENT")

    def test_remote_partial_archive_fails_closed(self):
        prefix = hf.remote_prefix("P345_FTSC", 42)
        with self.assertRaisesRegex(RuntimeError, "Partial or incomplete"):
            hf.inventory("example-not-real", api=FakeHub([prefix + "/weights/best.pt"]))

    def test_network_failure_never_triggers_training(self):
        with self.assertRaises(ConnectionError):
            hf.inventory("example-not-real", api=FakeHub(fail=True))

    def test_missing_token_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            hf.inventory("", api=FakeHub())

    def test_invalid_cell_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid factorial cell"):
            hf.remote_prefix("P345_FTSC", 99)


if __name__ == "__main__":
    unittest.main()
